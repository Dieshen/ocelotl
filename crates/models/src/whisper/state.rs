//! Public state types passed between encoder and decoder.
//!
//! `WhisperEncodedAudio` carries the encoder's final hidden states plus
//! per-decoder-layer cross-attention K/V caches so the decoder never recomputes
//! them. `WhisperDecoderState` carries the decoder token history, per-layer
//! self-attention K/V caches, and the most recent next-token logits. Cache
//! payload structs (`WhisperSelfAttentionCache`, `WhisperCrossAttentionCache`)
//! are crate-private but `pub(super)` so siblings inside `real/` can construct
//! and read them without a third indirection module.
//!
//! GW.4-2B migrated `WhisperCrossAttentionCache` to device-resident
//! `DeviceTensor` handles. The cache is built once per 30 s audio window and
//! read 80× by the decoder, so keeping it on device collapses the largest
//! single host bounce in the forward path (per the GW.4 audit). PostGW.1 keeps
//! decoder self-attention K/V cache buffers device-resident too: they are
//! allocated at text-context capacity and track how many prefix rows are valid.

use ocelotl_core::TokenId;
use ocelotl_kernels::DeviceTensor;

#[derive(Debug, Clone)]
pub struct WhisperEncodedAudio {
    pub(super) frames: usize,
    pub(super) state_size: usize,
    pub(super) values: Vec<f32>,
    pub(super) cross_attention: Vec<WhisperCrossAttentionCache>,
}

// GW.4-2B: `DeviceTensor` does not implement `PartialEq` (it's an opaque
// backend handle that may live on host or GPU). To keep external structs
// like `runtime::WhisperTranscriptionState` that derive `PartialEq` over a
// `WhisperEncodedAudio` field compiling, provide an explicit impl that
// compares the host-resident scalar fields plus a readback of each cache
// pair. The CPU backend keeps caches host-resident so the readback is a
// zero-copy clone; only the GPU backend pays a host transfer here, and
// `PartialEq` is not exercised on the GPU hot path.
impl PartialEq for WhisperEncodedAudio {
    fn eq(&self, other: &Self) -> bool {
        if self.frames != other.frames
            || self.state_size != other.state_size
            || self.values != other.values
            || self.cross_attention.len() != other.cross_attention.len()
        {
            return false;
        }
        self.cross_attention
            .iter()
            .zip(other.cross_attention.iter())
            .all(|(a, b)| a.eq(b))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WhisperAudioEncodeTimings {
    pub encoder_ms: u128,
    pub cross_attention_precompute_ms: u128,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WhisperEncoderTimings {
    pub conv_stack_ms: u128,
    pub device_upload_ms: u128,
    pub qkv_projection_ms: u128,
    pub attention_ms: u128,
    pub attention_out_projection_ms: u128,
    pub mlp_ms: u128,
    pub norm_residual_ms: u128,
    pub final_layer_norm_ms: u128,
    pub readback_ms: u128,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WhisperDecoderStepTimings {
    pub decoder_ms: u128,
    pub logits_project_ms: u128,
}

#[derive(Debug, Clone)]
pub(super) struct WhisperCrossAttentionCache {
    pub(super) key: DeviceTensor,
    pub(super) value: DeviceTensor,
}

impl PartialEq for WhisperCrossAttentionCache {
    fn eq(&self, other: &Self) -> bool {
        // The handles themselves are opaque, so compare by readback. This
        // is only used by external `derive(PartialEq)` on runtime types and
        // by parity tests — neither runs in the per-token hot path.
        match (self.key.to_host_owned(), other.key.to_host_owned()) {
            (Ok(a), Ok(b)) if a != b => return false,
            (Ok(_), Ok(_)) => {}
            // If either readback fails we conservatively call them unequal.
            _ => return false,
        }
        match (self.value.to_host_owned(), other.value.to_host_owned()) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhisperDecoderState {
    pub(super) tokens: Vec<TokenId>,
    pub(super) self_attention: Vec<WhisperSelfAttentionCache>,
    pub(super) next_token_logits: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct WhisperSelfAttentionCache {
    pub(super) key: DeviceTensor,
    pub(super) value: DeviceTensor,
    pub(super) filled_tokens: usize,
    pub(super) capacity_tokens: usize,
    pub(super) row_width: usize,
}

impl PartialEq for WhisperSelfAttentionCache {
    fn eq(&self, other: &Self) -> bool {
        if self.filled_tokens != other.filled_tokens
            || self.capacity_tokens != other.capacity_tokens
            || self.row_width != other.row_width
            || self.key.len() != other.key.len()
            || self.value.len() != other.value.len()
        {
            return false;
        }
        let Some(prefix_len) = self.filled_tokens.checked_mul(self.row_width) else {
            return false;
        };
        match (self.key.to_host_owned(), other.key.to_host_owned()) {
            (Ok(a), Ok(b))
                if a.len() >= prefix_len
                    && b.len() >= prefix_len
                    && a[..prefix_len] == b[..prefix_len] => {}
            _ => return false,
        }
        match (self.value.to_host_owned(), other.value.to_host_owned()) {
            (Ok(a), Ok(b)) if a.len() >= prefix_len && b.len() >= prefix_len => {
                a[..prefix_len] == b[..prefix_len]
            }
            _ => false,
        }
    }
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
