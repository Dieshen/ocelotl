//! Whisper-shaped CPU reference adapter.
//!
//! W-ASR.9's correctness-first CPU path, split across files for legibility:
//!
//! - `weights`     — `WhisperWeights`, `from_loaded_tensors`, tensor-shape rules.
//! - `state`       — public state types returned by the encoder/decoder.
//! - `model`       — `WhisperModel` struct, constructors, validation.
//! - `encode`      — encoder forward pass + cross-attention precomputation.
//! - `decode`      — full and incremental decoder forward passes.
//! - `primitives`  — convolutions, layer norm, attention variants, GELU, linear.
//!
//! Family-level helpers shared by the siblings (error construction, length
//! checks, dtype matching) live here with `pub(super)` visibility.

mod decode;
mod encode;
mod model;
mod primitives;
mod state;
mod weights;

#[cfg(test)]
mod tests;

pub use model::WhisperModel;
pub use state::{WhisperAudioEncodeTimings, WhisperDecoderState, WhisperEncodedAudio};
pub use weights::WhisperWeights;

use ocelotl_core::{DType, InvalidModelError, InvalidRequestError, OcelotlError, Result};
use ocelotl_loader::SupportedDtype;

/// Conv1d kernel width pinned by the OpenAI Whisper architecture.
pub(super) const CONV_KERNEL_WIDTH: usize = 3;

/// LayerNorm epsilon pinned by the OpenAI Whisper architecture.
pub(super) const LAYER_NORM_EPS: f32 = 1.0e-5;

pub(super) fn invalid_model(field: &str, message: &str) -> OcelotlError {
    OcelotlError::from(InvalidModelError {
        path: None,
        field: Some(field.to_string()),
        message: message.to_string(),
    })
}

pub(super) fn invalid_request(field: &str, message: &str) -> OcelotlError {
    OcelotlError::from(InvalidRequestError {
        field: field.to_string(),
        message: message.to_string(),
    })
}

pub(super) fn checked_len_product(label: &str, dims: &[usize]) -> Result<usize> {
    dims.iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| invalid_model(label, &format!("shape product overflows usize: {:?}", dims)))
}

pub(super) fn dtype_matches(actual: SupportedDtype, expected: &DType) -> bool {
    matches!(
        (actual, expected),
        (SupportedDtype::F32, DType::F32)
            | (SupportedDtype::F16, DType::F16)
            | (SupportedDtype::BF16, DType::BF16)
    )
}

pub(super) fn supported_dtype_name(dtype: SupportedDtype) -> &'static str {
    match dtype {
        SupportedDtype::F32 => "F32",
        SupportedDtype::F16 => "F16",
        SupportedDtype::BF16 => "BF16",
    }
}
