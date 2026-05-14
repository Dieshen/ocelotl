//! Public state types passed between encoder and decoder.
//!
//! `WhisperEncodedAudio` carries the encoder's final hidden states plus
//! per-decoder-layer cross-attention K/V caches so the decoder never recomputes
//! them. `WhisperDecoderState` carries the decoder token history, per-layer
//! self-attention K/V caches, and the most recent next-token logits. Cache
//! payload structs (`WhisperSelfAttentionCache`, `WhisperCrossAttentionCache`)
//! are crate-private but `pub(super)` so siblings inside `real/` can construct
//! and read them without a third indirection module.

use ocelotl_core::TokenId;

#[derive(Debug, Clone, PartialEq)]
pub struct WhisperEncodedAudio {
    pub(super) frames: usize,
    pub(super) state_size: usize,
    pub(super) values: Vec<f32>,
    pub(super) cross_attention: Vec<WhisperCrossAttentionCache>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WhisperAudioEncodeTimings {
    pub encoder_ms: u128,
    pub cross_attention_precompute_ms: u128,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct WhisperCrossAttentionCache {
    pub(super) key: Vec<f32>,
    pub(super) value: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhisperDecoderState {
    pub(super) tokens: Vec<TokenId>,
    pub(super) self_attention: Vec<WhisperSelfAttentionCache>,
    pub(super) next_token_logits: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct WhisperSelfAttentionCache {
    pub(super) key: Vec<f32>,
    pub(super) value: Vec<f32>,
}

impl WhisperEncodedAudio {
    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn state_size(&self) -> usize {
        self.state_size
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

impl WhisperDecoderState {
    pub fn tokens(&self) -> &[TokenId] {
        &self.tokens
    }

    pub fn next_token_logits(&self) -> &[f32] {
        &self.next_token_logits
    }
}
