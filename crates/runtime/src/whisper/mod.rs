//! Whisper-family runtime entry points.
//!
//! - `transcribe` — single-call transcription and streaming-state primitives.
//! - `streaming`  — chunk planning for long-audio transcription.

mod streaming;
mod transcribe;

pub use streaming::{
    ChunkedTranscriptionRequest, TranscriptionChunk, TranscriptionChunkingConfig,
    plan_transcription_chunks,
};
pub use transcribe::{
    TranscriptionRequest, TranscriptionResponse, WhisperDecodeRequest, WhisperTranscriptionRequest,
    WhisperTranscriptionResponse, WhisperTranscriptionState, decode_whisper_transcription,
    prepare_whisper_transcription, transcribe, transcribe_whisper,
};
