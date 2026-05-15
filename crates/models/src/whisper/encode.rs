//! Whisper encoder forward pass and cross-attention precompute.
//!
//! Composes per encoder layer:
//! `LayerNorm -> self-attention (non-causal) -> residual add ->
//!  LayerNorm -> GELU MLP -> residual add`
//! after two strided Conv1d stages and a positional embedding add.
//!
//! `encode_audio_features_with_timings` also runs the per-decoder-layer
//! cross-attention K/V precompute so the decoder never recomputes those
//! projections per token.
//!
//! GW.4-2B: the per-layer linear, layer-norm, GELU MLP, and residual-add
//! ops are now device-resident `*_d` calls over `DeviceTensor` scratch.
//! Conv1d and the positional-embedding add stay on host (one-shot cold
//! path, two conv1d calls per 30 s window). The only intra-layer host
//! bounce is the scalar attention body — search `to_host_owned()` /
//! `attention_body_host(` in this file to grep every bounce.

use std::time::Instant;

use ocelotl_core::Result;
use ocelotl_kernels::DeviceTensor;

use super::WhisperConfig;
use super::model::{WhisperModel, validate_audio_request};
use super::primitives::{
    add_inplace_d, add_positional_embedding, attention_body_host, conv_output_len, conv1d,
    gelu_inplace, layer_norm_d, linear_d, mlp_gelu_d,
};
use super::state::{WhisperAudioEncodeTimings, WhisperCrossAttentionCache, WhisperEncodedAudio};
use super::{CONV_KERNEL_WIDTH, LAYER_NORM_EPS, invalid_model, invalid_request};

impl WhisperModel {
    pub fn encode_audio_features(
        &self,
        log_mel: &[f32],
        mel_frames: usize,
    ) -> Result<WhisperEncodedAudio> {
        self.encode_audio_features_with_timings(log_mel, mel_frames)
            .map(|(audio, _timings)| audio)
    }

    pub fn encode_audio_features_with_timings(
        &self,
        log_mel: &[f32],
        mel_frames: usize,
    ) -> Result<(WhisperEncodedAudio, WhisperAudioEncodeTimings)> {
        validate_audio_request(&self.config, log_mel, mel_frames)?;

        let encoder_started = Instant::now();
        let (encoded_d, frames) = encode_audio(self, log_mel, mel_frames)?;
        let state_size = self.config.audio_state_size;
        // Read the encoder output back to host for `WhisperEncodedAudio.values`.
        // This is the natural boundary: the host `values` field stays around
        // for inspection / future use, while the device handle is consumed
        // directly by the cross-attention precompute below without a second
        // upload.
        let values = encoded_d.to_host_owned()?;
        let encoder_ms = encoder_started.elapsed().as_millis();
        if values.len() % state_size != 0 {
            return Err(invalid_model(
                "encoded_audio",
                &format!(
                    "encoded audio length {} is not divisible by audio_state_size {state_size}",
                    values.len()
                ),
            ));
        }
        if frames == 0 {
            return Err(invalid_model(
                "encoded_audio",
                "encoder produced zero audio frames",
            ));
        }

        let cross_attention_started = Instant::now();
        let cross_attention = precompute_cross_attention(self, &encoded_d, frames)?;
        let cross_attention_precompute_ms = cross_attention_started.elapsed().as_millis();

        Ok((
            WhisperEncodedAudio {
                frames,
                state_size,
                cross_attention,
                values,
            },
            WhisperAudioEncodeTimings {
                encoder_ms,
                cross_attention_precompute_ms,
            },
        ))
    }
}

/// Run the encoder forward pass. Returns the device-resident encoder output
/// (`audio_seq * audio_state` floats) plus the post-conv frame count. The
/// caller decides whether to read the output back to host.
fn encode_audio(
    model: &WhisperModel,
    log_mel: &[f32],
    mel_frames: usize,
) -> Result<(DeviceTensor, usize)> {
    let config: &WhisperConfig = &model.config;
    let kernels = model.kernels.as_ref();
    // Conv1d + GELU + positional add stays on host: only two convolutions
    // per 30 s window, and the log-mel input arrives on host anyway.
    let conv1 = conv1d(
        log_mel,
        mel_frames,
        config.mel_bins,
        model.weights.get("encoder.conv1.weight"),
        model.weights.get("encoder.conv1.bias"),
        config.audio_state_size,
        CONV_KERNEL_WIDTH,
        1,
        1,
    )?;
    let conv1_frames = conv_output_len(mel_frames, CONV_KERNEL_WIDTH, 1, 1)?;
    let mut conv1 = conv1;
    gelu_inplace(&mut conv1);

    let mut conv2 = conv1d(
        &conv1,
        conv1_frames,
        config.audio_state_size,
        model.weights.get("encoder.conv2.weight"),
        model.weights.get("encoder.conv2.bias"),
        config.audio_state_size,
        CONV_KERNEL_WIDTH,
        2,
        1,
    )?;
    gelu_inplace(&mut conv2);

    let seq = conv_output_len(conv1_frames, CONV_KERNEL_WIDTH, 2, 1)?;
    if seq > config.audio_context_length {
        return Err(invalid_request(
            "mel_frames",
            &format!(
                "convolution output length {seq} exceeds audio_context_length {}",
                config.audio_context_length
            ),
        ));
    }

    add_positional_embedding(
        &mut conv2,
        seq,
        config.audio_state_size,
        model.weights.get("encoder.positional_embedding"),
        config.audio_context_length,
    )?;

    // Upload the post-conv-positional activation onto the device. From here
    // on, every per-layer compute step runs over `DeviceTensor` handles.
    let x_d = kernels.upload(&conv2)?;
    drop(conv2);

    // Per-layer scratch pool: allocated once outside the loop, reused
    // across every encoder layer. Tiny dims (seq=1500, state=384,
    // ffn=1536) put this pool at ~37 MB: attn_ln/q/k/v/proj_out/mlp_ln/
    // mlp_out are each 1500*384 = 576 KB (×7 = ~4 MB) and mlp_hidden is
    // 1500*1536 = 9.2 MB, for ~13 MB device-resident. Compared to the
    // pre-GW.4-2A path that allocated ~110 MB of host `Vec<f32>` per
    // forward pass, this is a >8× reduction in churn even before counting
    // the saved host↔device transfers.
    let state = config.audio_state_size;
    let ffn = config.audio_ffn_size;
    let attn_ln_d = kernels.alloc(seq * state)?;
    let q_proj_d = kernels.alloc(seq * state)?;
    let k_proj_d = kernels.alloc(seq * state)?;
    let v_proj_d = kernels.alloc(seq * state)?;
    let proj_out_d = kernels.alloc(seq * state)?;
    let mlp_ln_d = kernels.alloc(seq * state)?;
    let mlp_hidden_d = kernels.alloc(seq * ffn)?;
    let mlp_out_d = kernels.alloc(seq * state)?;

    for layer in 0..config.audio_layers {
        let prefix = format!("encoder.blocks.{layer}");
        let attn_ln_w = model.device_weight(&format!("{prefix}.attn_ln.weight"))?;
        let attn_ln_b = model.device_weight(&format!("{prefix}.attn_ln.bias"))?;
        layer_norm_d(
            kernels,
            &x_d,
            seq,
            state,
            attn_ln_w,
            attn_ln_b,
            LAYER_NORM_EPS,
            &attn_ln_d,
        )?;

        let q_w = model.device_weight(&format!("{prefix}.attn.query.weight"))?;
        let q_b = model.device_weight(&format!("{prefix}.attn.query.bias"))?;
        let k_w = model.device_weight(&format!("{prefix}.attn.key.weight"))?;
        let v_w = model.device_weight(&format!("{prefix}.attn.value.weight"))?;
        let v_b = model.device_weight(&format!("{prefix}.attn.value.bias"))?;
        linear_d(
            kernels,
            &attn_ln_d,
            seq,
            state,
            q_w,
            state,
            Some(q_b),
            &q_proj_d,
        )?;
        linear_d(
            kernels,
            &attn_ln_d,
            seq,
            state,
            k_w,
            state,
            None,
            &k_proj_d,
        )?;
        linear_d(
            kernels,
            &attn_ln_d,
            seq,
            state,
            v_w,
            state,
            Some(v_b),
            &v_proj_d,
        )?;

        // Host bounce: scalar attention body. Q/K/V are read back to host,
        // run through the existing rayon-parallel attention kernel, and the
        // resulting context is uploaded for the on-device out projection.
        let q_host = q_proj_d.to_host_owned()?;
        let k_host = k_proj_d.to_host_owned()?;
        let v_host = v_proj_d.to_host_owned()?;
        let context_host = attention_body_host(
            kernels,
            &q_host,
            seq,
            &k_host,
            &v_host,
            seq,
            state,
            config.audio_attention_heads,
            false,
        )?;
        // Reuse the q_proj scratch for the uploaded context — saves an
        // allocation. q_host/k_host/v_host live until end of bounce so the
        // overwrite is safe.
        q_proj_d.write_from_host_slice(&context_host)?;
        let out_w = model.device_weight(&format!("{prefix}.attn.out.weight"))?;
        let out_b = model.device_weight(&format!("{prefix}.attn.out.bias"))?;
        linear_d(
            kernels,
            &q_proj_d,
            seq,
            state,
            out_w,
            state,
            Some(out_b),
            &proj_out_d,
        )?;
        add_inplace_d(kernels, &x_d, &proj_out_d)?;

        let mlp_ln_w = model.device_weight(&format!("{prefix}.mlp_ln.weight"))?;
        let mlp_ln_b = model.device_weight(&format!("{prefix}.mlp_ln.bias"))?;
        layer_norm_d(
            kernels,
            &x_d,
            seq,
            state,
            mlp_ln_w,
            mlp_ln_b,
            LAYER_NORM_EPS,
            &mlp_ln_d,
        )?;
        let fc1_w = model.device_weight(&format!("{prefix}.mlp.0.weight"))?;
        let fc1_b = model.device_weight(&format!("{prefix}.mlp.0.bias"))?;
        let fc2_w = model.device_weight(&format!("{prefix}.mlp.2.weight"))?;
        let fc2_b = model.device_weight(&format!("{prefix}.mlp.2.bias"))?;
        mlp_gelu_d(
            kernels,
            &mlp_ln_d,
            seq,
            state,
            ffn,
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
            &mlp_hidden_d,
            &mlp_out_d,
        )?;
        add_inplace_d(kernels, &x_d, &mlp_out_d)?;
    }

    // Final ln_post into a fresh device handle so the result outlives the
    // scratch pool we're about to drop.
    let encoded_d = kernels.alloc(seq * state)?;
    let ln_w = model.device_weight("encoder.ln_post.weight")?;
    let ln_b = model.device_weight("encoder.ln_post.bias")?;
    layer_norm_d(
        kernels,
        &x_d,
        seq,
        state,
        ln_w,
        ln_b,
        LAYER_NORM_EPS,
        &encoded_d,
    )?;

    Ok((encoded_d, seq))
}

fn precompute_cross_attention(
    model: &WhisperModel,
    encoded_d: &DeviceTensor,
    audio_seq: usize,
) -> Result<Vec<WhisperCrossAttentionCache>> {
    let state = model.config.text_state_size;
    let kernels = model.kernels.as_ref();
    let mut caches = Vec::with_capacity(model.config.text_layers);
    for layer in 0..model.config.text_layers {
        let prefix = format!("decoder.blocks.{layer}.cross_attn");
        let key = kernels.alloc(audio_seq * state)?;
        let value = kernels.alloc(audio_seq * state)?;
        let key_w = model.device_weight(&format!("{prefix}.key.weight"))?;
        linear_d(
            kernels,
            encoded_d,
            audio_seq,
            state,
            key_w,
            state,
            None,
            &key,
        )?;
        let value_w = model.device_weight(&format!("{prefix}.value.weight"))?;
        let value_b = model.device_weight(&format!("{prefix}.value.bias"))?;
        linear_d(
            kernels,
            encoded_d,
            audio_seq,
            state,
            value_w,
            state,
            Some(value_b),
            &value,
        )?;
        caches.push(WhisperCrossAttentionCache { key, value });
    }
    Ok(caches)
}
