//! Whisper-family model semantics.
//!
//! W-ASR.2 starts with audio validation and a scalar preprocessing reference.
//! W-ASR.4 adds a tiny synthetic encoder/decoder path that consumes log-mel
//! frames and decoder token IDs without depending on the tokenizer crate.

pub mod audio;
pub mod config;
pub mod model;
pub mod real;
pub mod tensors;

pub use config::{WhisperConfig, parse_whisper_config_json};
pub use model::{WhisperTinyConfig, WhisperTinyModel, WhisperTinyWeights};
pub use real::{WhisperModel, WhisperWeights};
pub use tensors::{required_whisper_tensor_names, validate_whisper_tensors};
