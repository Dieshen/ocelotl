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

use ocelotl_core::{Result, TokenId};

use super::LAYER_NORM_EPS;
use super::model::{
    WhisperModel, validate_decoder_state_for_append, validate_decoder_token,
    validate_decoder_tokens, validate_encoded_audio, validate_forward_request,
};
use super::primitives::{
    add_inplace, attention_from_projected, attention_incremental_from_projected,
    attention_with_precomputed_kv, layer_norm, linear, mlp_gelu,
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
    let audio_seq = audio.frames();
    let mut x = vec![0.0_f32; seq * text_state];
    let mut self_attention = Vec::with_capacity(model.config.text_layers);
    let token_embedding = model.weights.get("decoder.token_embedding.weight");
    let positional_embedding = model.weights.get("decoder.positional_embedding");

    for (pos, token) in decoder_tokens.iter().enumerate() {
        let token_start = token.0 as usize * text_state;
        let row_start = pos * text_state;
        for dim in 0..text_state {
            x[row_start + dim] =
                token_embedding[token_start + dim] + positional_embedding[row_start + dim];
        }
    }

    for layer in 0..model.config.text_layers {
        let prefix = format!("decoder.blocks.{layer}");
        let attn_ln = layer_norm(
            &x,
            seq,
            text_state,
            model.weights.get(&format!("{prefix}.attn_ln.weight")),
            model.weights.get(&format!("{prefix}.attn_ln.bias")),
            LAYER_NORM_EPS,
        )?;
        let q = linear(
            model.kernels.as_ref(),
            &attn_ln,
            seq,
            text_state,
            model.weights.get(&format!("{prefix}.attn.query.weight")),
            text_state,
            Some(model.weights.get(&format!("{prefix}.attn.query.bias"))),
        )?;
        let key = linear(
            model.kernels.as_ref(),
            &attn_ln,
            seq,
            text_state,
            model.weights.get(&format!("{prefix}.attn.key.weight")),
            text_state,
            None,
        )?;
        let value = linear(
            model.kernels.as_ref(),
            &attn_ln,
            seq,
            text_state,
            model.weights.get(&format!("{prefix}.attn.value.weight")),
            text_state,
            Some(model.weights.get(&format!("{prefix}.attn.value.bias"))),
        )?;
        let self_attn = attention_from_projected(
            model.kernels.as_ref(),
            &q,
            seq,
            &key,
            &value,
            seq,
            text_state,
            model.config.text_attention_heads,
            model.weights.get(&format!("{prefix}.attn.out.weight")),
            model.weights.get(&format!("{prefix}.attn.out.bias")),
            true,
        )?;
        self_attention.push(WhisperSelfAttentionCache { key, value });
        add_inplace(&mut x, &self_attn);

        let cross_ln = layer_norm(
            &x,
            seq,
            text_state,
            model.weights.get(&format!("{prefix}.cross_attn_ln.weight")),
            model.weights.get(&format!("{prefix}.cross_attn_ln.bias")),
            LAYER_NORM_EPS,
        )?;
        let cross_cache = &audio.cross_attention[layer];
        let cross_attn = attention_with_precomputed_kv(
            model.kernels.as_ref(),
            &cross_ln,
            seq,
            text_state,
            model.config.text_attention_heads,
            model
                .weights
                .get(&format!("{prefix}.cross_attn.query.weight")),
            model
                .weights
                .get(&format!("{prefix}.cross_attn.query.bias")),
            &cross_cache.key,
            &cross_cache.value,
            audio_seq,
            model
                .weights
                .get(&format!("{prefix}.cross_attn.out.weight")),
            model.weights.get(&format!("{prefix}.cross_attn.out.bias")),
            false,
        )?;
        add_inplace(&mut x, &cross_attn);

        let mlp_ln = layer_norm(
            &x,
            seq,
            text_state,
            model.weights.get(&format!("{prefix}.mlp_ln.weight")),
            model.weights.get(&format!("{prefix}.mlp_ln.bias")),
            LAYER_NORM_EPS,
        )?;
        let mlp = mlp_gelu(
            model.kernels.as_ref(),
            &mlp_ln,
            seq,
            text_state,
            model.config.text_ffn_size,
            model.weights.get(&format!("{prefix}.mlp.0.weight")),
            model.weights.get(&format!("{prefix}.mlp.0.bias")),
            model.weights.get(&format!("{prefix}.mlp.2.weight")),
            model.weights.get(&format!("{prefix}.mlp.2.bias")),
        )?;
        add_inplace(&mut x, &mlp);
    }

    let decoded = layer_norm(
        &x,
        seq,
        text_state,
        model.weights.get("decoder.ln.weight"),
        model.weights.get("decoder.ln.bias"),
        LAYER_NORM_EPS,
    )?;
    Ok((decoded, self_attention))
}

fn decode_appended_token(
    model: &WhisperModel,
    audio: &WhisperEncodedAudio,
    state: &mut WhisperDecoderState,
    token: TokenId,
) -> Result<Vec<f32>> {
    let text_state = model.config.text_state_size;
    let pos = state.tokens.len();
    let token_embedding = model.weights.get("decoder.token_embedding.weight");
    let positional_embedding = model.weights.get("decoder.positional_embedding");
    let token_start = token.0 as usize * text_state;
    let row_start = pos * text_state;
    let mut x = vec![0.0_f32; text_state];
    for dim in 0..text_state {
        x[dim] = token_embedding[token_start + dim] + positional_embedding[row_start + dim];
    }

    let mut next_self_attention = Vec::with_capacity(model.config.text_layers);
    for layer in 0..model.config.text_layers {
        let prefix = format!("decoder.blocks.{layer}");
        let attn_ln = layer_norm(
            &x,
            1,
            text_state,
            model.weights.get(&format!("{prefix}.attn_ln.weight")),
            model.weights.get(&format!("{prefix}.attn_ln.bias")),
            LAYER_NORM_EPS,
        )?;
        let q = linear(
            model.kernels.as_ref(),
            &attn_ln,
            1,
            text_state,
            model.weights.get(&format!("{prefix}.attn.query.weight")),
            text_state,
            Some(model.weights.get(&format!("{prefix}.attn.query.bias"))),
        )?;
        let key = linear(
            model.kernels.as_ref(),
            &attn_ln,
            1,
            text_state,
            model.weights.get(&format!("{prefix}.attn.key.weight")),
            text_state,
            None,
        )?;
        let value = linear(
            model.kernels.as_ref(),
            &attn_ln,
            1,
            text_state,
            model.weights.get(&format!("{prefix}.attn.value.weight")),
            text_state,
            Some(model.weights.get(&format!("{prefix}.attn.value.bias"))),
        )?;
        let cache = &state.self_attention[layer];
        let self_attn = attention_incremental_from_projected(
            model.kernels.as_ref(),
            &q,
            &key,
            &value,
            &cache.key,
            &cache.value,
            state.tokens.len(),
            text_state,
            model.config.text_attention_heads,
            model.weights.get(&format!("{prefix}.attn.out.weight")),
            model.weights.get(&format!("{prefix}.attn.out.bias")),
        )?;
        next_self_attention.push(WhisperSelfAttentionCache { key, value });
        add_inplace(&mut x, &self_attn);

        let cross_ln = layer_norm(
            &x,
            1,
            text_state,
            model.weights.get(&format!("{prefix}.cross_attn_ln.weight")),
            model.weights.get(&format!("{prefix}.cross_attn_ln.bias")),
            LAYER_NORM_EPS,
        )?;
        let cross_cache = &audio.cross_attention[layer];
        let cross_attn = attention_with_precomputed_kv(
            model.kernels.as_ref(),
            &cross_ln,
            1,
            text_state,
            model.config.text_attention_heads,
            model
                .weights
                .get(&format!("{prefix}.cross_attn.query.weight")),
            model
                .weights
                .get(&format!("{prefix}.cross_attn.query.bias")),
            &cross_cache.key,
            &cross_cache.value,
            audio.frames(),
            model
                .weights
                .get(&format!("{prefix}.cross_attn.out.weight")),
            model.weights.get(&format!("{prefix}.cross_attn.out.bias")),
            false,
        )?;
        add_inplace(&mut x, &cross_attn);

        let mlp_ln = layer_norm(
            &x,
            1,
            text_state,
            model.weights.get(&format!("{prefix}.mlp_ln.weight")),
            model.weights.get(&format!("{prefix}.mlp_ln.bias")),
            LAYER_NORM_EPS,
        )?;
        let mlp = mlp_gelu(
            model.kernels.as_ref(),
            &mlp_ln,
            1,
            text_state,
            model.config.text_ffn_size,
            model.weights.get(&format!("{prefix}.mlp.0.weight")),
            model.weights.get(&format!("{prefix}.mlp.0.bias")),
            model.weights.get(&format!("{prefix}.mlp.2.weight")),
            model.weights.get(&format!("{prefix}.mlp.2.bias")),
        )?;
        add_inplace(&mut x, &mlp);
    }

    let decoded = layer_norm(
        &x,
        1,
        text_state,
        model.weights.get("decoder.ln.weight"),
        model.weights.get("decoder.ln.bias"),
        LAYER_NORM_EPS,
    )?;
    let logits = project_decoder_logits(model, &decoded)?;

    for (cache, next) in state.self_attention.iter_mut().zip(next_self_attention) {
        cache.key.extend_from_slice(&next.key);
        cache.value.extend_from_slice(&next.value);
    }

    Ok(logits)
}

fn project_decoder_logits(model: &WhisperModel, last: &[f32]) -> Result<Vec<f32>> {
    let state = model.config.text_state_size;
    let projection = if model.config.tie_word_embeddings {
        model.weights.get("decoder.token_embedding.weight")
    } else {
        model.weights.get("decoder.proj_out.weight")
    };

    linear(
        model.kernels.as_ref(),
        last,
        1,
        state,
        projection,
        model.config.vocab_size,
        None,
    )
}
