//! Whisper-family model semantics.
//!
//! - `audio`      — log-mel preprocessing and audio metadata validation.
//! - `config`     — `WhisperConfig` and HF config parsing.
//! - `tensors`    — required tensor names and safetensors manifest validation.
//! - `weights`    — `WhisperWeights` + safetensors-to-Whisper layout adapter.
//! - `state`      — public encoder/decoder state types.
//! - `model`      — `WhisperModel` struct, constructors, and request validation.
//! - `encode`     — encoder forward pass + cross-attention precompute.
//! - `decode`     — full-context and incremental decoder forward passes.
//! - `primitives` — conv1d, layer norm, attention variants, GELU, linear.
//! - `wer`        — WER scoring utilities for the corpus runner.
//!
//! Family-level helpers shared by `weights`, `model`, `encode`, `decode`, and
//! `primitives` (`invalid_model`, `checked_len_product`, dtype matching) live
//! in this file with `pub(super)` visibility so siblings can use them without
//! a third indirection module.

pub mod audio;
pub mod config;
pub mod tensors;
pub mod wer;

mod decode;
mod encode;
mod model;
mod primitives;
mod state;
mod weights;

#[cfg(test)]
mod tests;

pub use config::{WhisperConfig, parse_whisper_config_json};
pub use model::WhisperModel;
pub use state::{WhisperAudioEncodeTimings, WhisperDecoderState, WhisperEncodedAudio};
pub use tensors::{required_whisper_tensor_names, validate_whisper_tensors};
pub use weights::WhisperWeights;
pub use wer::{
    WerCorpusCase, WerCorpusCaseScore, WerCorpusReport, WerEditCounts, WerScore,
    normalize_transcript, score_wer_corpus, wer_edit_counts, wer_score,
};

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
