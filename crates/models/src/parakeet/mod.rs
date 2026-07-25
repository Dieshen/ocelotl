//! Parakeet (NVIDIA FastConformer + TDT) model family.
//!
//! Ported against the exported reference graphs from
//! `istupakov/parakeet-tdt-0.6b-v3-onnx`, which serve as the tensor-level parity
//! oracle. Deliberately independent of [`crate::whisper`]: that path is
//! parity-clean and acts as the regression anchor, so nothing here generalizes
//! or mutates it.

pub mod audio;
pub mod chunk;
pub mod decoder;
pub mod encoder;
pub mod model;
pub mod subsample;
