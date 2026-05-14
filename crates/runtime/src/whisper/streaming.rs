//! Chunk-planning contract for Whisper transcription.
//!
//! W-ASR.17 defines the runtime surface for chunked transcription planning,
//! not a real-time streaming engine. Phase 7 chunks are decoded independently:
//! every planned chunk starts from the same decoder prompt and no KV cache,
//! audio cache, or model state is reused across chunk boundaries yet.

use ocelotl_core::{InvalidRequestError, OcelotlError, Result, TokenId};
use ocelotl_models::whisper::audio::{AudioMetadata, validate_audio_metadata};

/// Runtime request shape for deterministic chunked Whisper transcription.
///
/// The planner only produces chunk metadata. A future executor can feed each
/// chunk through `transcribe`, but W-ASR.17 intentionally does not implement
/// microphone capture, transcript stitching, or cache/state reuse.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkedTranscriptionRequest {
    pub audio_samples: Vec<f32>,
    pub audio_metadata: AudioMetadata,
    pub decoder_prompt_tokens: Vec<TokenId>,
    pub chunking: TranscriptionChunkingConfig,
}

/// Chunk sizing policy expressed in samples.
///
/// Keeping the public contract sample-based avoids rounding ambiguity. Callers
/// can derive display seconds with `window_seconds` and `overlap_seconds` after
/// the request's audio metadata has been validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscriptionChunkingConfig {
    pub window_samples: usize,
    pub overlap_samples: usize,
}

impl TranscriptionChunkingConfig {
    pub fn window_seconds(&self, sample_rate_hz: u32) -> Result<f64> {
        seconds_for_samples(self.window_samples, sample_rate_hz)
    }

    pub fn overlap_seconds(&self, sample_rate_hz: u32) -> Result<f64> {
        seconds_for_samples(self.overlap_samples, sample_rate_hz)
    }
}

/// Metadata for one planned transcription chunk.
///
/// Sample ranges are half-open: `[start_sample, end_sample)`. Seconds are
/// derived from those sample positions against validated audio metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionChunk {
    pub index: usize,
    pub start_sample: usize,
    pub end_sample: usize,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub is_last: bool,
}

impl TranscriptionChunk {
    pub fn sample_count(&self) -> usize {
        self.end_sample - self.start_sample
    }
}

/// Plan deterministic half-open audio chunks for a chunked Whisper request.
///
/// Phase 7 state semantics are intentionally simple: each returned chunk is an
/// independent decode unit. The overlap only supplies repeated audio context;
/// it does not imply KV/cache reuse or transcript stitching.
pub fn plan_transcription_chunks(
    request: &ChunkedTranscriptionRequest,
) -> Result<Vec<TranscriptionChunk>> {
    validate_audio_metadata(request.audio_metadata)?;
    validate_chunk_request(request)?;

    let sample_rate_hz = request.audio_metadata.sample_rate_hz;
    let audio_len = request.audio_samples.len();
    let step_samples = request.chunking.window_samples - request.chunking.overlap_samples;
    let mut chunks = Vec::new();
    let mut start_sample = 0usize;

    loop {
        let window_end = start_sample
            .checked_add(request.chunking.window_samples)
            .ok_or_else(|| invalid_request("chunking.window_samples", "chunk end overflowed"))?;
        let end_sample = window_end.min(audio_len);
        let is_last = end_sample == audio_len;

        chunks.push(TranscriptionChunk {
            index: chunks.len(),
            start_sample,
            end_sample,
            start_seconds: seconds_for_samples(start_sample, sample_rate_hz)?,
            end_seconds: seconds_for_samples(end_sample, sample_rate_hz)?,
            is_last,
        });

        if is_last {
            break;
        }

        start_sample = start_sample
            .checked_add(step_samples)
            .ok_or_else(|| invalid_request("chunking.window_samples", "chunk start overflowed"))?;
    }

    Ok(chunks)
}

fn validate_chunk_request(request: &ChunkedTranscriptionRequest) -> Result<()> {
    if request.audio_samples.is_empty() {
        return Err(invalid_request(
            "audio_samples",
            "must contain at least one sample",
        ));
    }

    if request.chunking.window_samples == 0 {
        return Err(invalid_request(
            "chunking.window_samples",
            "must be greater than zero",
        ));
    }

    if request.chunking.overlap_samples >= request.chunking.window_samples {
        return Err(invalid_request(
            "chunking.overlap_samples",
            "must be less than window_samples",
        ));
    }

    Ok(())
}

fn seconds_for_samples(samples: usize, sample_rate_hz: u32) -> Result<f64> {
    if sample_rate_hz == 0 {
        return Err(invalid_request(
            "audio_metadata.sample_rate_hz",
            "must not be zero",
        ));
    }

    Ok(samples as f64 / f64::from(sample_rate_hz))
}

fn invalid_request(field: &str, message: &str) -> OcelotlError {
    OcelotlError::InvalidRequest(InvalidRequestError {
        field: field.to_string(),
        message: message.to_string(),
    })
}
