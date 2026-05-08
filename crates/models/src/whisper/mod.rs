//! Whisper-family model semantics.
//!
//! W-ASR.2 starts with audio validation and a scalar preprocessing reference.
//! W-ASR.4 adds a tiny synthetic encoder/decoder path that consumes log-mel
//! frames and decoder token IDs without depending on the tokenizer crate.

pub mod audio;
pub mod model;

pub use model::{WhisperTinyConfig, WhisperTinyModel, WhisperTinyWeights};
