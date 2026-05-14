//! Whisper runtime entry points.

use ocelotl_core::{InvalidRequestError, OcelotlError, Result, RuntimeError, TokenId};
use ocelotl_models::whisper::audio::{AudioMetadata, log_mel_spectrogram, validate_audio_metadata};
use ocelotl_models::whisper::{WhisperEncodedAudio, WhisperModel, WhisperTinyModel};
use ocelotl_tokenizer::{WhisperDecodeMask, WhisperTokenMaskDecision};

use crate::greedy_sample;

/// A Whisper transcription request after audio loading and tokenizer startup
/// policy have already run.
///
/// Runtime accepts raw mono samples and decoder prompt token IDs. Text decoding
/// is deliberately out of scope for W-ASR.6: the tokenizer crate owns Whisper
/// special-token policy and future token-to-text behavior.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionRequest {
    pub audio_samples: Vec<f32>,
    pub audio_metadata: AudioMetadata,
    pub decoder_prompt_tokens: Vec<TokenId>,
}

/// One synthetic Whisper decode step through the runtime API.
///
/// `tokens` is the greedy-selected next token. `logits` is returned alongside
/// it so early ASR tests can pin the model/runtime shape before text decoding
/// exists.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionResponse {
    pub tokens: Vec<TokenId>,
    pub logits: Vec<f32>,
}

/// Real Whisper transcription request for an autoregressive token loop.
///
/// The tokenizer layer still owns startup-token construction and timestamp
/// masking policy. Runtime receives the already-tokenized prompt, decode mask,
/// and stop token, then owns audio preprocessing, encoded-audio state, and the
/// decode lifecycle.
#[derive(Debug, Clone, PartialEq)]
pub struct WhisperTranscriptionRequest {
    pub audio_samples: Vec<f32>,
    pub audio_metadata: AudioMetadata,
    pub decode: WhisperDecodeRequest,
}

/// Real Whisper decoder controls after tokenization and policy selection.
#[derive(Debug, Clone, PartialEq)]
pub struct WhisperDecodeRequest {
    pub decoder_prompt_tokens: Vec<TokenId>,
    pub max_new_tokens: usize,
    pub decode_mask: WhisperDecodeMask,
    pub stop_token: TokenId,
}

/// Runtime-owned Whisper state that is invariant across token decode steps.
#[derive(Debug, Clone, PartialEq)]
pub struct WhisperTranscriptionState {
    encoded_audio: WhisperEncodedAudio,
}

impl WhisperTranscriptionState {
    pub fn encoded_audio(&self) -> &WhisperEncodedAudio {
        &self.encoded_audio
    }
}

/// Tokens produced by a real Whisper autoregressive transcription loop.
///
/// `tokens` contains only newly generated tokens, not the startup prompt. The
/// caller can concatenate `decoder_prompt_tokens + tokens` when it needs the
/// full model sequence for parity fixtures.
#[derive(Debug, Clone, PartialEq)]
pub struct WhisperTranscriptionResponse {
    pub tokens: Vec<TokenId>,
    pub logits: Vec<f32>,
}

/// Run one synthetic Whisper transcription step through the runtime boundary.
///
/// W-ASR.6 keeps this intentionally narrow: runtime validates request-owned
/// audio shape, calls the Whisper log-mel reference path, reaches the
/// `WhisperTinyModel::forward` public model API, and greedily selects one next
/// token. Multi-token decode, timestamp policy, and token-to-text decoding are
/// future tokenizer/runtime work.
pub fn transcribe(
    model: &WhisperTinyModel,
    request: &TranscriptionRequest,
) -> Result<TranscriptionResponse> {
    validate_transcription_request(request)?;
    let mel = log_mel_spectrogram(&request.audio_samples, request.audio_metadata)?;
    let logits = model.forward(&mel, &request.decoder_prompt_tokens)?;
    let token = greedy_sample(&logits)?;

    Ok(TranscriptionResponse {
        tokens: vec![token],
        logits,
    })
}

/// Prepare real Whisper audio state once for a transcription request.
///
/// This is the W-ASR.21 public runtime seam: audio validation and log-mel
/// preprocessing happen once, `WhisperModel::encode_audio_features` produces
/// encoded audio once, and the returned state can be reused by every token
/// decode step for that audio window.
pub fn prepare_whisper_transcription(
    model: &WhisperModel,
    request: &WhisperTranscriptionRequest,
) -> Result<WhisperTranscriptionState> {
    validate_whisper_transcription_request(request)?;
    let mel = log_mel_spectrogram(&request.audio_samples, request.audio_metadata)?;
    let encoded_audio = model.encode_audio_features(&mel.values, mel.frames)?;
    Ok(WhisperTranscriptionState { encoded_audio })
}

/// Decode real Whisper tokens from a prepared encoded-audio state.
///
/// This is the W-ASR.22 runtime path: callers can hold
/// `WhisperTranscriptionState` and avoid recomputing the encoder for each
/// generated token. W-ASR.27 also keeps a decoder state inside this loop so
/// decoder self-attention K/V grows one token at a time instead of recomputing
/// the full decoder prefix for every generated token.
pub fn decode_whisper_transcription(
    model: &WhisperModel,
    state: &WhisperTranscriptionState,
    request: &WhisperDecodeRequest,
) -> Result<WhisperTranscriptionResponse> {
    validate_whisper_decode_request(model, request)?;

    let mut decoder_state = model
        .prepare_decoder_state_from_audio(state.encoded_audio(), &request.decoder_prompt_tokens)?;
    let mut tokens = Vec::with_capacity(request.max_new_tokens);
    let mut logits = Vec::new();

    for _ in 0..request.max_new_tokens {
        logits = decoder_state.next_token_logits().to_vec();
        let next = masked_greedy_sample(&logits, request.decode_mask)?;
        tokens.push(next);
        if next == request.stop_token {
            break;
        }
        if tokens.len() < request.max_new_tokens {
            model.append_decoder_token_from_audio(
                state.encoded_audio(),
                &mut decoder_state,
                next,
            )?;
        }
    }

    Ok(WhisperTranscriptionResponse { tokens, logits })
}

/// Run real Whisper transcription through the runtime boundary.
///
/// This convenience wrapper composes `prepare_whisper_transcription` and
/// `decode_whisper_transcription`, so the public path gets encoded-audio reuse
/// even when the caller does not manage the state directly.
pub fn transcribe_whisper(
    model: &WhisperModel,
    request: &WhisperTranscriptionRequest,
) -> Result<WhisperTranscriptionResponse> {
    validate_whisper_decode_request(model, &request.decode)?;
    let state = prepare_whisper_transcription(model, request)?;
    decode_whisper_transcription(model, &state, &request.decode)
}

fn validate_transcription_request(request: &TranscriptionRequest) -> Result<()> {
    if request.audio_samples.is_empty() {
        return Err(OcelotlError::InvalidRequest(InvalidRequestError {
            field: "audio_samples".to_string(),
            message: "must contain at least one sample".to_string(),
        }));
    }

    validate_audio_metadata(request.audio_metadata)
}

fn validate_whisper_transcription_request(request: &WhisperTranscriptionRequest) -> Result<()> {
    if request.audio_samples.is_empty() {
        return Err(invalid_request(
            "audio_samples",
            "must contain at least one sample",
        ));
    }

    validate_audio_metadata(request.audio_metadata)
}

fn validate_whisper_decode_request(
    model: &WhisperModel,
    request: &WhisperDecodeRequest,
) -> Result<()> {
    if request.decoder_prompt_tokens.is_empty() {
        return Err(invalid_request(
            "decoder_prompt_tokens",
            "must contain at least one token",
        ));
    }
    if request.max_new_tokens == 0 {
        return Err(invalid_request(
            "max_new_tokens",
            "must be greater than zero",
        ));
    }

    let total = request
        .decoder_prompt_tokens
        .len()
        .checked_add(request.max_new_tokens)
        .ok_or_else(|| {
            invalid_request(
                "decoder_context_length",
                "decoder_prompt_tokens + max_new_tokens overflows usize",
            )
        })?;
    if total > model.config().text_context_length {
        return Err(invalid_request(
            "decoder_context_length",
            &format!(
                "decoder_prompt_tokens ({}) + max_new_tokens ({}) = {} exceeds text_context_length ({})",
                request.decoder_prompt_tokens.len(),
                request.max_new_tokens,
                total,
                model.config().text_context_length,
            ),
        ));
    }

    Ok(())
}

fn masked_greedy_sample(logits: &[f32], mask: WhisperDecodeMask) -> Result<TokenId> {
    let mut best = None;
    for (idx, &logit) in logits.iter().enumerate() {
        let token = TokenId(u32::try_from(idx).map_err(|_| {
            OcelotlError::Runtime(RuntimeError {
                message: format!("logit index {idx} does not fit in TokenId"),
            })
        })?);
        if mask.mask_token(token) == WhisperTokenMaskDecision::Suppress {
            continue;
        }
        if best.is_none_or(|(_, best_logit)| logit > best_logit) {
            best = Some((idx, logit));
        }
    }

    let (idx, _) = best.ok_or_else(|| {
        OcelotlError::Runtime(RuntimeError {
            message: "Whisper decode mask suppressed every logit".to_string(),
        })
    })?;
    Ok(TokenId(u32::try_from(idx).map_err(|_| {
        OcelotlError::Runtime(RuntimeError {
            message: format!("logit index {idx} does not fit in TokenId"),
        })
    })?))
}

fn invalid_request(field: &str, message: &str) -> OcelotlError {
    OcelotlError::InvalidRequest(InvalidRequestError {
        field: field.to_string(),
        message: message.to_string(),
    })
}
