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
//! gather (a host table lookup), the self-attention KV cache append
//! (`Vec::extend_from_slice` per Stage 2.5 deferral), the scalar attention
//! bodies, and the final logits readback for sampling remain host events.
//! Search `to_host_owned()` in this file to grep every host bounce.

use ocelotl_core::{Result, TokenId};

use super::LAYER_NORM_EPS;
use super::model::{
    WhisperModel, validate_decoder_state_for_append, validate_decoder_token,
    validate_decoder_tokens, validate_encoded_audio, validate_forward_request,
};
use super::primitives::{
    add_inplace_d, attention_body_host, attention_incremental_body_host, layer_norm_d, linear_d,
    mlp_gelu_d,
};
use super::state::{WhisperDecoderState, WhisperEncodedAudio, WhisperSelfAttentionCache};

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
        validate_encoded_audio(&self.config, audio)?;
        validate_decoder_tokens(&self.config, decoder_tokens)?;

        let (decoded, self_attention) =
            decode_tokens_with_self_attention_cache(self, decoder_tokens, audio)?;
        let state_size = self.config.text_state_size;
        let last_start = (decoder_tokens.len() - 1) * state_size;
        let last = &decoded[last_start..last_start + state_size];
        let next_token_logits = project_decoder_logits(self, last)?;

        Ok(WhisperDecoderState {
            tokens: decoder_tokens.to_vec(),
            self_attention,
            next_token_logits,
        })
    }

    pub fn append_decoder_token_from_audio<'a>(
        &self,
        audio: &WhisperEncodedAudio,
        state: &'a mut WhisperDecoderState,
        token: TokenId,
    ) -> Result<&'a [f32]> {
        validate_encoded_audio(&self.config, audio)?;
        validate_decoder_state_for_append(&self.config, state)?;
        validate_decoder_token(&self.config, token, state.tokens.len())?;

        let next_token_logits = decode_appended_token(self, audio, state, token)?;
        state.tokens.push(token);
        state.next_token_logits = next_token_logits;
        Ok(state.next_token_logits())
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
            kernels,
            &attn_ln_d,
            seq,
            text_state,
            k_w,
            text_state,
            None,
            &k_proj_d,
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

        // Host bounce: read Q/K/V back, run scalar self-attention (causal),
        // push K/V to the host self-attention cache for later token append.
        let q_host = q_proj_d.to_host_owned()?;
        let key_host = k_proj_d.to_host_owned()?;
        let value_host = v_proj_d.to_host_owned()?;
        let context_host = attention_body_host(
            kernels,
            &q_host,
            seq,
            &key_host,
            &value_host,
            seq,
            text_state,
            heads,
            true,
        )?;
        // Reuse q_proj scratch for the uploaded context.
        q_proj_d.write_from_host_slice(&context_host)?;
        let out_w = model.device_weight(&format!("{prefix}.attn.out.weight"))?;
        let out_b = model.device_weight(&format!("{prefix}.attn.out.bias"))?;
        linear_d(
            kernels,
            &q_proj_d,
            seq,
            text_state,
            out_w,
            text_state,
            Some(out_b),
            &proj_out_d,
        )?;
        self_attention.push(WhisperSelfAttentionCache {
            key: key_host,
            value: value_host,
        });
        add_inplace_d(kernels, &x_d, &proj_out_d)?;

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
        // Host bounce: read the cross-Q back and the cross-K/V from the
        // device cache, run scalar cross-attention (non-causal), upload the
        // result for the on-device out projection.
        let cross_cache = &audio.cross_attention[layer];
        let cross_q_host = cross_q_d.to_host_owned()?;
        let cross_k_host = cross_cache.key.to_host_owned()?;
        let cross_v_host = cross_cache.value.to_host_owned()?;
        let cross_ctx_host = attention_body_host(
            kernels,
            &cross_q_host,
            seq,
            &cross_k_host,
            &cross_v_host,
            audio_seq,
            text_state,
            heads,
            false,
        )?;
        cross_q_d.write_from_host_slice(&cross_ctx_host)?;
        let cross_out_w = model.device_weight(&format!("{prefix}.cross_attn.out.weight"))?;
        let cross_out_b = model.device_weight(&format!("{prefix}.cross_attn.out.bias"))?;
        linear_d(
            kernels,
            &cross_q_d,
            seq,
            text_state,
            cross_out_w,
            text_state,
            Some(cross_out_b),
            &cross_out_d,
        )?;
        add_inplace_d(kernels, &x_d, &cross_out_d)?;

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

fn decode_appended_token(
    model: &WhisperModel,
    audio: &WhisperEncodedAudio,
    state: &mut WhisperDecoderState,
    token: TokenId,
) -> Result<Vec<f32>> {
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

    let mut next_self_attention = Vec::with_capacity(model.config.text_layers);
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
            kernels,
            &attn_ln_d,
            1,
            text_state,
            k_w,
            text_state,
            None,
            &k_proj_d,
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

        // Host bounce: incremental self-attention reads past K/V from the
        // host cache plus the new K/V projection, runs scalar attention,
        // uploads the context for the out projection.
        let q_host = q_proj_d.to_host_owned()?;
        let key_host = k_proj_d.to_host_owned()?;
        let value_host = v_proj_d.to_host_owned()?;
        let cache = &state.self_attention[layer];
        let context_host = attention_incremental_body_host(
            &q_host,
            &key_host,
            &value_host,
            &cache.key,
            &cache.value,
            state.tokens.len(),
            text_state,
            heads,
        )?;
        q_proj_d.write_from_host_slice(&context_host)?;
        let out_w = model.device_weight(&format!("{prefix}.attn.out.weight"))?;
        let out_b = model.device_weight(&format!("{prefix}.attn.out.bias"))?;
        linear_d(
            kernels,
            &q_proj_d,
            1,
            text_state,
            out_w,
            text_state,
            Some(out_b),
            &proj_out_d,
        )?;
        next_self_attention.push(WhisperSelfAttentionCache {
            key: key_host,
            value: value_host,
        });
        add_inplace_d(kernels, &x_d, &proj_out_d)?;

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
        // Host bounce: read cross-Q + device-resident cross-K/V cache, run
        // scalar cross-attention. The cross cache being device-resident is
        // the GW.4-2B headline win — it's still bounced here because the
        // attention body itself is host, but the upload-once / read-many
        // shape collapses the per-token re-projection cost.
        let cross_cache = &audio.cross_attention[layer];
        let cross_q_host = cross_q_d.to_host_owned()?;
        let cross_k_host = cross_cache.key.to_host_owned()?;
        let cross_v_host = cross_cache.value.to_host_owned()?;
        let cross_ctx_host = attention_body_host(
            kernels,
            &cross_q_host,
            1,
            &cross_k_host,
            &cross_v_host,
            audio.frames(),
            text_state,
            heads,
            false,
        )?;
        cross_q_d.write_from_host_slice(&cross_ctx_host)?;
        let cross_out_w = model.device_weight(&format!("{prefix}.cross_attn.out.weight"))?;
        let cross_out_b = model.device_weight(&format!("{prefix}.cross_attn.out.bias"))?;
        linear_d(
            kernels,
            &cross_q_d,
            1,
            text_state,
            cross_out_w,
            text_state,
            Some(cross_out_b),
            &cross_out_d,
        )?;
        add_inplace_d(kernels, &x_d, &cross_out_d)?;

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
    let logits = project_decoder_logits_d(model, &decoded_d)?;

    for (cache, next) in state.self_attention.iter_mut().zip(next_self_attention) {
        cache.key.extend_from_slice(&next.key);
        cache.value.extend_from_slice(&next.value);
    }

    Ok(logits)
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
