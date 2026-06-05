//! Whisper decoder forward paths.
//!
//! Two compositions live here:
//!
//! - `decode_tokens_with_self_attention_cache` runs full-context decode over
//!   the prompt and builds the per-layer self-attention K/V cache. Used by
//!   `prepare_decoder_state_from_audio` and by `forward_next_token_logits`.
//! - `decode_appended_token` advances an existing `WhisperDecoderState` by one
//!   token using the prior self-attention cache and the precomputed
//!   cross-attention from `WhisperEncodedAudio`. Used by
//!   `append_decoder_token_from_audio`.
//!
//! `forward_next_token_logits` orchestrates the encode + decode pair for
//! callers that do not want to manage state explicitly.
//!
//! GW.4-2B: the per-layer linear / layer-norm / MLP / residual-add ops run
//! over `DeviceTensor` handles via the `*_d` primitives. Token embedding
//! gather (a host table lookup) and the final logits readback for sampling
//! remain host events.
//!
//! GW.4-5B: self-attention bodies (causal full-context and incremental
//! single-token) are now device-resident via `attention_decoder_causal_d`
//! and `attention_decoder_incremental_cache_append_d`.
//!
//! GW.4-5C: cross-attention is now device-resident via
//! `attention_decoder_cross_d`. Q, K, and V all remain on device through
//! the attention body and the out projection.
//!
//! PostGW.1: decoder self-attention K/V caches are fixed-capacity
//! device tensors. Appended-token decode copies one new row into those
//! tensors and attends over the visible prefix, removing the previous
//! per-layer growing cache upload/readback.

use std::time::Instant;

use ocelotl_core::{Result, TokenId};

use super::model::{
    WhisperModel, validate_decoder_state_for_append, validate_decoder_token,
    validate_decoder_tokens, validate_encoded_audio, validate_forward_request,
};
use super::primitives::{
    add_inplace_d, attention_decoder_causal_d, attention_decoder_cross_d,
    attention_decoder_incremental_cache_append_d, copy_into_d, layer_norm_d, linear_d, mlp_gelu_d,
};
use super::state::{
    WhisperDecoderState, WhisperDecoderStepTimings, WhisperEncodedAudio, WhisperSelfAttentionCache,
};
use super::{LAYER_NORM_EPS, checked_len_product};

impl WhisperModel {
    pub fn forward_next_token_logits(
        &self,
        log_mel: &[f32],
        mel_frames: usize,
        decoder_tokens: &[TokenId],
    ) -> Result<Vec<f32>> {
        validate_forward_request(&self.config, log_mel, mel_frames, decoder_tokens)?;

        let audio = self.encode_audio_features(log_mel, mel_frames)?;
        self.forward_next_token_logits_from_audio(&audio, decoder_tokens)
    }

    pub fn forward_next_token_logits_from_audio(
        &self,
        audio: &WhisperEncodedAudio,
        decoder_tokens: &[TokenId],
    ) -> Result<Vec<f32>> {
        self.prepare_decoder_state_from_audio(audio, decoder_tokens)
            .map(|state| state.next_token_logits)
    }

    pub fn prepare_decoder_state_from_audio(
        &self,
        audio: &WhisperEncodedAudio,
        decoder_tokens: &[TokenId],
    ) -> Result<WhisperDecoderState> {
        self.prepare_decoder_state_from_audio_with_timings(audio, decoder_tokens)
            .map(|(state, _)| state)
    }

    pub fn prepare_decoder_state_from_audio_with_timings(
        &self,
        audio: &WhisperEncodedAudio,
        decoder_tokens: &[TokenId],
    ) -> Result<(WhisperDecoderState, WhisperDecoderStepTimings)> {
        validate_encoded_audio(&self.config, audio)?;
        validate_decoder_tokens(&self.config, decoder_tokens)?;

        let decoder_started = Instant::now();
        let (decoded, self_attention) =
            decode_tokens_with_self_attention_cache(self, decoder_tokens, audio)?;
        let decoder_ms = decoder_started.elapsed().as_millis();
        let state_size = self.config.text_state_size;
        let last_start = (decoder_tokens.len() - 1) * state_size;
        let last = &decoded[last_start..last_start + state_size];
        let logits_started = Instant::now();
        let next_token_logits = project_decoder_logits(self, last)?;
        let logits_project_ms = logits_started.elapsed().as_millis();

        Ok((
            WhisperDecoderState {
                tokens: decoder_tokens.to_vec(),
                self_attention,
                next_token_logits,
            },
            WhisperDecoderStepTimings {
                decoder_ms,
                logits_project_ms,
            },
        ))
    }

    pub fn append_decoder_token_from_audio<'a>(
        &self,
        audio: &WhisperEncodedAudio,
        state: &'a mut WhisperDecoderState,
        token: TokenId,
    ) -> Result<&'a [f32]> {
        self.append_decoder_token_from_audio_with_timings(audio, state, token)
            .map(|(logits, _)| logits)
    }

    pub fn append_decoder_token_from_audio_with_timings<'a>(
        &self,
        audio: &WhisperEncodedAudio,
        state: &'a mut WhisperDecoderState,
        token: TokenId,
    ) -> Result<(&'a [f32], WhisperDecoderStepTimings)> {
        validate_encoded_audio(&self.config, audio)?;
        validate_decoder_state_for_append(&self.config, state)?;
        validate_decoder_token(&self.config, token, state.tokens.len())?;

        let (next_token_logits, timings) =
            decode_appended_token_with_timings(self, audio, state, token)?;
        state.tokens.push(token);
        state.next_token_logits = next_token_logits;
        Ok((state.next_token_logits(), timings))
    }
}

fn decode_tokens_with_self_attention_cache(
    model: &WhisperModel,
    decoder_tokens: &[TokenId],
    audio: &WhisperEncodedAudio,
) -> Result<(Vec<f32>, Vec<WhisperSelfAttentionCache>)> {
    let seq = decoder_tokens.len();
    let text_state = model.config.text_state_size;
    let ffn = model.config.text_ffn_size;
    let heads = model.config.text_attention_heads;
    let audio_seq = audio.frames();
    let kernels = model.kernels.as_ref();
    let mut self_attention = Vec::with_capacity(model.config.text_layers);

    // Token + positional embedding gather stays on host: it's a table lookup
    // keyed by token id with no device-side analogue worth building yet.
    let token_embedding = model.weights.get("decoder.token_embedding.weight");
    let positional_embedding = model.weights.get("decoder.positional_embedding");
    let mut x_host = vec![0.0_f32; seq * text_state];
    for (pos, token) in decoder_tokens.iter().enumerate() {
        let token_start = token.0 as usize * text_state;
        let row_start = pos * text_state;
        for dim in 0..text_state {
            x_host[row_start + dim] =
                token_embedding[token_start + dim] + positional_embedding[row_start + dim];
        }
    }
    let x_d = kernels.upload(&x_host)?;
    drop(x_host);

    // Per-layer device scratch pool, allocated once and reused. Matches the
    // encoder shape exactly: 7 buffers at `seq*state` + 1 at `seq*ffn`.
    let attn_ln_d = kernels.alloc(seq * text_state)?;
    let q_proj_d = kernels.alloc(seq * text_state)?;
    let k_proj_d = kernels.alloc(seq * text_state)?;
    let v_proj_d = kernels.alloc(seq * text_state)?;
    let proj_out_d = kernels.alloc(seq * text_state)?;
    let cross_ln_d = kernels.alloc(seq * text_state)?;
    let cross_q_d = kernels.alloc(seq * text_state)?;
    let cross_out_d = kernels.alloc(seq * text_state)?;
    let mlp_ln_d = kernels.alloc(seq * text_state)?;
    let mlp_hidden_d = kernels.alloc(seq * ffn)?;
    let mlp_out_d = kernels.alloc(seq * text_state)?;

    for layer in 0..model.config.text_layers {
        let prefix = format!("decoder.blocks.{layer}");
        let attn_ln_w = model.device_weight(&format!("{prefix}.attn_ln.weight"))?;
        let attn_ln_b = model.device_weight(&format!("{prefix}.attn_ln.bias"))?;
        layer_norm_d(
            kernels,
            &x_d,
            seq,
            text_state,
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
            text_state,
            q_w,
            text_state,
            Some(q_b),
            &q_proj_d,
        )?;
        linear_d(
            kernels, &attn_ln_d, seq, text_state, k_w, text_state, None, &k_proj_d,
        )?;
        linear_d(
            kernels,
            &attn_ln_d,
            seq,
            text_state,
            v_w,
            text_state,
            Some(v_b),
            &v_proj_d,
        )?;

        // GW.4-5B: device-resident causal self-attention. PostGW.1 keeps
        // the projected K/V cache on device too: copy the prompt rows into
        // fixed-capacity cache tensors so appended decode can mutate one row
        // per layer without a host readback or growing prefix upload.
        let head_dim = text_state / heads;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        attention_decoder_causal_d(
            kernels,
            &q_proj_d,
            &k_proj_d,
            &v_proj_d,
            seq,
            heads,
            head_dim,
            scale,
            &proj_out_d,
        )?;
        let cache_len = checked_len_product(
            "decoder_state.self_attention",
            &[model.config.text_context_length, text_state],
        )?;
        let key_cache_d = kernels.alloc(cache_len)?;
        let value_cache_d = kernels.alloc(cache_len)?;
        copy_into_d(kernels, &k_proj_d, &key_cache_d, 0)?;
        copy_into_d(kernels, &v_proj_d, &value_cache_d, 0)?;
        let out_w = model.device_weight(&format!("{prefix}.attn.out.weight"))?;
        let out_b = model.device_weight(&format!("{prefix}.attn.out.bias"))?;
        linear_d(
            kernels,
            &proj_out_d,
            seq,
            text_state,
            out_w,
            text_state,
            Some(out_b),
            // Reuse q_proj scratch for the output projection result.
            &q_proj_d,
        )?;
        self_attention.push(WhisperSelfAttentionCache {
            key: key_cache_d,
            value: value_cache_d,
            filled_tokens: seq,
            capacity_tokens: model.config.text_context_length,
            row_width: text_state,
        });
        // Out-projection result is in q_proj_d (see linear_d above).
        add_inplace_d(kernels, &x_d, &q_proj_d)?;

        let cross_ln_w = model.device_weight(&format!("{prefix}.cross_attn_ln.weight"))?;
        let cross_ln_b = model.device_weight(&format!("{prefix}.cross_attn_ln.bias"))?;
        layer_norm_d(
            kernels,
            &x_d,
            seq,
            text_state,
            cross_ln_w,
            cross_ln_b,
            LAYER_NORM_EPS,
            &cross_ln_d,
        )?;
        let cross_q_w = model.device_weight(&format!("{prefix}.cross_attn.query.weight"))?;
        let cross_q_b = model.device_weight(&format!("{prefix}.cross_attn.query.bias"))?;
        linear_d(
            kernels,
            &cross_ln_d,
            seq,
            text_state,
            cross_q_w,
            text_state,
            Some(cross_q_b),
            &cross_q_d,
        )?;
        // GW.4-5C: device-resident cross-attention. Q is already on device
        // in cross_q_d. K/V come from the precomputed encoder cache in
        // WhisperEncodedAudio::cross_attention — also device-resident (GW.4-2B).
        // No causal mask; all audio_seq encoder positions are visible.
        let cross_cache = &audio.cross_attention[layer];
        let head_dim_cross = text_state / heads;
        let scale_cross = 1.0_f32 / (head_dim_cross as f32).sqrt();
        attention_decoder_cross_d(
            kernels,
            &cross_q_d,
            &cross_cache.key,
            &cross_cache.value,
            seq,
            audio_seq,
            heads,
            head_dim_cross,
            scale_cross,
            &cross_out_d,
        )?;
        let cross_out_w = model.device_weight(&format!("{prefix}.cross_attn.out.weight"))?;
        let cross_out_b = model.device_weight(&format!("{prefix}.cross_attn.out.bias"))?;
        linear_d(
            kernels,
            &cross_out_d,
            seq,
            text_state,
            cross_out_w,
            text_state,
            Some(cross_out_b),
            &cross_q_d,
        )?;
        add_inplace_d(kernels, &x_d, &cross_q_d)?;

        let mlp_ln_w = model.device_weight(&format!("{prefix}.mlp_ln.weight"))?;
        let mlp_ln_b = model.device_weight(&format!("{prefix}.mlp_ln.bias"))?;
        layer_norm_d(
            kernels,
            &x_d,
            seq,
            text_state,
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
            text_state,
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

    // Final ln, then read the whole decoded sequence back to host so the
    // caller can index the last row for logit projection.
    let decoded_d = kernels.alloc(seq * text_state)?;
    let ln_w = model.device_weight("decoder.ln.weight")?;
    let ln_b = model.device_weight("decoder.ln.bias")?;
    layer_norm_d(
        kernels,
        &x_d,
        seq,
        text_state,
        ln_w,
        ln_b,
        LAYER_NORM_EPS,
        &decoded_d,
    )?;
    let decoded = decoded_d.to_host_owned()?;
    Ok((decoded, self_attention))
}

fn decode_appended_token_with_timings(
    model: &WhisperModel,
    audio: &WhisperEncodedAudio,
    state: &mut WhisperDecoderState,
    token: TokenId,
) -> Result<(Vec<f32>, WhisperDecoderStepTimings)> {
    let decoder_started = Instant::now();
    let text_state = model.config.text_state_size;
    let ffn = model.config.text_ffn_size;
    let heads = model.config.text_attention_heads;
    let pos = state.tokens.len();
    let kernels = model.kernels.as_ref();
    let token_embedding = model.weights.get("decoder.token_embedding.weight");
    let positional_embedding = model.weights.get("decoder.positional_embedding");
    let token_start = token.0 as usize * text_state;
    let row_start = pos * text_state;
    let mut x_host = vec![0.0_f32; text_state];
    for dim in 0..text_state {
        x_host[dim] = token_embedding[token_start + dim] + positional_embedding[row_start + dim];
    }
    let x_d = kernels.upload(&x_host)?;
    drop(x_host);

    // Per-layer device scratch: seq=1, so these are tiny (1*state at tiny =
    // 1.5 KB) but reusing them across the 4-layer loop saves the same
    // allocations the GW.4-1C `mlp_gelu` caller-supplied scratch saves on
    // the host side.
    let attn_ln_d = kernels.alloc(text_state)?;
    let q_proj_d = kernels.alloc(text_state)?;
    let k_proj_d = kernels.alloc(text_state)?;
    let v_proj_d = kernels.alloc(text_state)?;
    let proj_out_d = kernels.alloc(text_state)?;
    let cross_ln_d = kernels.alloc(text_state)?;
    let cross_q_d = kernels.alloc(text_state)?;
    let cross_out_d = kernels.alloc(text_state)?;
    let mlp_ln_d = kernels.alloc(text_state)?;
    let mlp_hidden_d = kernels.alloc(ffn)?;
    let mlp_out_d = kernels.alloc(text_state)?;

    for layer in 0..model.config.text_layers {
        let prefix = format!("decoder.blocks.{layer}");
        let attn_ln_w = model.device_weight(&format!("{prefix}.attn_ln.weight"))?;
        let attn_ln_b = model.device_weight(&format!("{prefix}.attn_ln.bias"))?;
        layer_norm_d(
            kernels,
            &x_d,
            1,
            text_state,
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
            1,
            text_state,
            q_w,
            text_state,
            Some(q_b),
            &q_proj_d,
        )?;
        linear_d(
            kernels, &attn_ln_d, 1, text_state, k_w, text_state, None, &k_proj_d,
        )?;
        linear_d(
            kernels,
            &attn_ln_d,
            1,
            text_state,
            v_w,
            text_state,
            Some(v_b),
            &v_proj_d,
        )?;

        // PostGW.1: append current K/V into the fixed-capacity device cache,
        // then attend the visible prefix in place. This removes the previous
        // per-token upload of `past_seq * state` K/V and the readback of the
        // new row after every layer.
        let head_dim = text_state / heads;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let cache = &mut state.self_attention[layer];
        attention_decoder_incremental_cache_append_d(
            kernels,
            &q_proj_d,
            &cache.key,
            &cache.value,
            &k_proj_d,
            &v_proj_d,
            cache.filled_tokens,
            cache.capacity_tokens,
            heads,
            head_dim,
            scale,
            &proj_out_d,
        )?;
        let out_w = model.device_weight(&format!("{prefix}.attn.out.weight"))?;
        let out_b = model.device_weight(&format!("{prefix}.attn.out.bias"))?;
        linear_d(
            kernels,
            &proj_out_d,
            1,
            text_state,
            out_w,
            text_state,
            Some(out_b),
            &q_proj_d,
        )?;
        // Out-projection result is in q_proj_d (see linear_d above).
        add_inplace_d(kernels, &x_d, &q_proj_d)?;

        let cross_ln_w = model.device_weight(&format!("{prefix}.cross_attn_ln.weight"))?;
        let cross_ln_b = model.device_weight(&format!("{prefix}.cross_attn_ln.bias"))?;
        layer_norm_d(
            kernels,
            &x_d,
            1,
            text_state,
            cross_ln_w,
            cross_ln_b,
            LAYER_NORM_EPS,
            &cross_ln_d,
        )?;
        let cross_q_w = model.device_weight(&format!("{prefix}.cross_attn.query.weight"))?;
        let cross_q_b = model.device_weight(&format!("{prefix}.cross_attn.query.bias"))?;
        linear_d(
            kernels,
            &cross_ln_d,
            1,
            text_state,
            cross_q_w,
            text_state,
            Some(cross_q_b),
            &cross_q_d,
        )?;
        // GW.4-5C: device-resident cross-attention for the single-token
        // incremental path. Q is in cross_q_d (seq=1). K/V are the
        // precomputed per-layer encoder cache in WhisperEncodedAudio —
        // already device-resident from GW.4-2B. No causal mask; all
        // audio.frames() encoder positions are visible.
        let cross_cache = &audio.cross_attention[layer];
        let head_dim_cross = text_state / heads;
        let scale_cross = 1.0_f32 / (head_dim_cross as f32).sqrt();
        let audio_frames = audio.frames();
        attention_decoder_cross_d(
            kernels,
            &cross_q_d,
            &cross_cache.key,
            &cross_cache.value,
            1,
            audio_frames,
            heads,
            head_dim_cross,
            scale_cross,
            &cross_out_d,
        )?;
        let cross_out_w = model.device_weight(&format!("{prefix}.cross_attn.out.weight"))?;
        let cross_out_b = model.device_weight(&format!("{prefix}.cross_attn.out.bias"))?;
        linear_d(
            kernels,
            &cross_out_d,
            1,
            text_state,
            cross_out_w,
            text_state,
            Some(cross_out_b),
            &cross_q_d,
        )?;
        add_inplace_d(kernels, &x_d, &cross_q_d)?;

        let mlp_ln_w = model.device_weight(&format!("{prefix}.mlp_ln.weight"))?;
        let mlp_ln_b = model.device_weight(&format!("{prefix}.mlp_ln.bias"))?;
        layer_norm_d(
            kernels,
            &x_d,
            1,
            text_state,
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
            1,
            text_state,
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

    let decoded_d = kernels.alloc(text_state)?;
    let ln_w = model.device_weight("decoder.ln.weight")?;
    let ln_b = model.device_weight("decoder.ln.bias")?;
    layer_norm_d(
        kernels,
        &x_d,
        1,
        text_state,
        ln_w,
        ln_b,
        LAYER_NORM_EPS,
        &decoded_d,
    )?;
    let decoder_ms = decoder_started.elapsed().as_millis();
    let logits_started = Instant::now();
    let logits = project_decoder_logits_d(model, &decoded_d)?;
    let logits_project_ms = logits_started.elapsed().as_millis();

    for cache in &mut state.self_attention {
        cache.filled_tokens += 1;
    }

    Ok((
        logits,
        WhisperDecoderStepTimings {
            decoder_ms,
            logits_project_ms,
        },
    ))
}

fn project_decoder_logits(model: &WhisperModel, last: &[f32]) -> Result<Vec<f32>> {
    let kernels = model.kernels.as_ref();
    let last_d = kernels.upload(last)?;
    project_decoder_logits_d(model, &last_d)
}

fn project_decoder_logits_d(
    model: &WhisperModel,
    last: &ocelotl_kernels::DeviceTensor,
) -> Result<Vec<f32>> {
    let state = model.config.text_state_size;
    let kernels = model.kernels.as_ref();
    let projection_name = if model.config.tie_word_embeddings {
        "decoder.token_embedding.weight"
    } else {
        "decoder.proj_out.weight"
    };
    let projection = model.device_weight(projection_name)?;
    let logits_d = kernels.alloc(model.config.vocab_size)?;
    linear_d(
        kernels,
        last,
        1,
        state,
        projection,
        model.config.vocab_size,
        None,
        &logits_d,
    )?;
    logits_d.to_host_owned()
}
