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

use std::time::Instant;

use ocelotl_core::Result;

use super::WhisperConfig;
use super::model::{WhisperModel, validate_audio_request};
use super::primitives::{
    add_inplace, add_positional_embedding, attention, conv_output_len, conv1d, gelu_inplace,
    layer_norm, linear, mlp_gelu,
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
        let values = encode_audio(self, log_mel, mel_frames)?;
        let encoder_ms = encoder_started.elapsed().as_millis();
        let state_size = self.config.audio_state_size;
        if values.len() % state_size != 0 {
            return Err(invalid_model(
                "encoded_audio",
                &format!(
                    "encoded audio length {} is not divisible by audio_state_size {state_size}",
                    values.len()
                ),
            ));
        }
        let frames = values.len() / state_size;
        if frames == 0 {
            return Err(invalid_model(
                "encoded_audio",
                "encoder produced zero audio frames",
            ));
        }

        let cross_attention_started = Instant::now();
        let cross_attention = precompute_cross_attention(self, &values, frames)?;
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

fn encode_audio(model: &WhisperModel, log_mel: &[f32], mel_frames: usize) -> Result<Vec<f32>> {
    let config: &WhisperConfig = &model.config;
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

    let mut x = conv2;
    for layer in 0..config.audio_layers {
        let prefix = format!("encoder.blocks.{layer}");
        let attn_ln = layer_norm(
            &x,
            seq,
            config.audio_state_size,
            model.weights.get(&format!("{prefix}.attn_ln.weight")),
            model.weights.get(&format!("{prefix}.attn_ln.bias")),
            LAYER_NORM_EPS,
        )?;
        let attn = attention(
            model.kernels.as_ref(),
            &attn_ln,
            seq,
            config.audio_state_size,
            config.audio_attention_heads,
            model.weights.get(&format!("{prefix}.attn.query.weight")),
            model.weights.get(&format!("{prefix}.attn.query.bias")),
            model.weights.get(&format!("{prefix}.attn.key.weight")),
            model.weights.get(&format!("{prefix}.attn.value.weight")),
            model.weights.get(&format!("{prefix}.attn.value.bias")),
            model.weights.get(&format!("{prefix}.attn.out.weight")),
            model.weights.get(&format!("{prefix}.attn.out.bias")),
            None,
            false,
        )?;
        add_inplace(&mut x, &attn);

        let mlp_ln = layer_norm(
            &x,
            seq,
            config.audio_state_size,
            model.weights.get(&format!("{prefix}.mlp_ln.weight")),
            model.weights.get(&format!("{prefix}.mlp_ln.bias")),
            LAYER_NORM_EPS,
        )?;
        let mlp = mlp_gelu(
            model.kernels.as_ref(),
            &mlp_ln,
            seq,
            config.audio_state_size,
            config.audio_ffn_size,
            model.weights.get(&format!("{prefix}.mlp.0.weight")),
            model.weights.get(&format!("{prefix}.mlp.0.bias")),
            model.weights.get(&format!("{prefix}.mlp.2.weight")),
            model.weights.get(&format!("{prefix}.mlp.2.bias")),
        )?;
        add_inplace(&mut x, &mlp);
    }

    layer_norm(
        &x,
        seq,
        config.audio_state_size,
        model.weights.get("encoder.ln_post.weight"),
        model.weights.get("encoder.ln_post.bias"),
        LAYER_NORM_EPS,
    )
}

fn precompute_cross_attention(
    model: &WhisperModel,
    encoded_audio: &[f32],
    audio_seq: usize,
) -> Result<Vec<WhisperCrossAttentionCache>> {
    let state = model.config.text_state_size;
    let mut caches = Vec::with_capacity(model.config.text_layers);
    for layer in 0..model.config.text_layers {
        let prefix = format!("decoder.blocks.{layer}.cross_attn");
        let key = linear(
            model.kernels.as_ref(),
            encoded_audio,
            audio_seq,
            state,
            model.weights.get(&format!("{prefix}.key.weight")),
            state,
            None,
        )?;
        let value = linear(
            model.kernels.as_ref(),
            encoded_audio,
            audio_seq,
            state,
            model.weights.get(&format!("{prefix}.value.weight")),
            state,
            Some(model.weights.get(&format!("{prefix}.value.bias"))),
        )?;
        caches.push(WhisperCrossAttentionCache { key, value });
    }
    Ok(caches)
}
