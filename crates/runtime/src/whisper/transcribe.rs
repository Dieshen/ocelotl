//! Whisper runtime entry points.

use ocelotl_core::{
    InvalidRequestError, OcelotlError, RequestLimits, Result, RuntimeError, TokenId,
};
use ocelotl_models::whisper::audio::{
    AudioMetadata, log_mel_spectrogram_with_limits, validate_audio_metadata,
    validate_audio_samples_for_context,
};
use ocelotl_models::whisper::{WhisperEncodedAudio, WhisperModel};
use ocelotl_tokenizer::{WhisperDecodeMask, WhisperTokenMaskDecision};

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

impl WhisperTranscriptionRequest {
    /// Prepare encoded audio under deployment-specific request ceilings.
    pub fn prepare_with_limits(
        &self,
        model: &WhisperModel,
        limits: RequestLimits,
    ) -> Result<WhisperTranscriptionState> {
        prepare_whisper_transcription_impl(model, self, limits)
    }

    /// Run transcription under deployment-specific request ceilings.
    pub fn transcribe_with_limits(
        &self,
        model: &WhisperModel,
        limits: RequestLimits,
    ) -> Result<WhisperTranscriptionResponse> {
        transcribe_whisper_impl(model, self, limits)
    }
}

/// Real Whisper decoder controls after tokenization and policy selection.
#[derive(Debug, Clone, PartialEq)]
pub struct WhisperDecodeRequest {
    pub decoder_prompt_tokens: Vec<TokenId>,
    pub max_new_tokens: usize,
    pub decode_mask: WhisperDecodeMask,
    pub stop_token: TokenId,
}

impl WhisperDecodeRequest {
    /// Decode from prepared audio under deployment-specific token ceilings.
    pub fn decode_with_limits(
        &self,
        model: &WhisperModel,
        state: &WhisperTranscriptionState,
        limits: RequestLimits,
    ) -> Result<WhisperTranscriptionResponse> {
        decode_whisper_transcription_impl(model, state, self, limits)
    }
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
    prepare_whisper_transcription_impl(model, request, RequestLimits::default())
}

fn prepare_whisper_transcription_impl(
    model: &WhisperModel,
    request: &WhisperTranscriptionRequest,
    limits: RequestLimits,
) -> Result<WhisperTranscriptionState> {
    validate_whisper_transcription_request(model, request, limits)?;
    let mel =
        log_mel_spectrogram_with_limits(&request.audio_samples, request.audio_metadata, limits)?;
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
    decode_whisper_transcription_impl(model, state, request, RequestLimits::default())
}

fn decode_whisper_transcription_impl(
    model: &WhisperModel,
    state: &WhisperTranscriptionState,
    request: &WhisperDecodeRequest,
    limits: RequestLimits,
) -> Result<WhisperTranscriptionResponse> {
    validate_whisper_decode_request(model, request, limits)?;

    let mut decoder_state = model
        .prepare_decoder_state_from_audio(state.encoded_audio(), &request.decoder_prompt_tokens)?;
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(request.max_new_tokens)
        .map_err(|source| {
            OcelotlError::Runtime(RuntimeError {
                message: format!(
                    "failed to reserve {} Whisper output tokens: {source}",
                    request.max_new_tokens
                ),
            })
        })?;
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
    transcribe_whisper_impl(model, request, RequestLimits::default())
}

fn transcribe_whisper_impl(
    model: &WhisperModel,
    request: &WhisperTranscriptionRequest,
    limits: RequestLimits,
) -> Result<WhisperTranscriptionResponse> {
    validate_whisper_decode_request(model, &request.decode, limits)?;
    let state = prepare_whisper_transcription_impl(model, request, limits)?;
    decode_whisper_transcription_impl(model, &state, &request.decode, limits)
}

fn validate_whisper_transcription_request(
    model: &WhisperModel,
    request: &WhisperTranscriptionRequest,
    limits: RequestLimits,
) -> Result<()> {
    if request.audio_samples.is_empty() {
        return Err(invalid_request(
            "audio_samples",
            "must contain at least one sample",
        ));
    }

    validate_audio_metadata(request.audio_metadata)?;
    validate_whisper_audio_size(
        request.audio_samples.len(),
        model.config().audio_context_length,
        limits,
    )
}

fn validate_whisper_decode_request(
    model: &WhisperModel,
    request: &WhisperDecodeRequest,
    limits: RequestLimits,
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

    let total =
        limits.validate_generation(request.decoder_prompt_tokens.len(), request.max_new_tokens)?;
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

fn validate_whisper_audio_size(
    audio_samples: usize,
    audio_context_length: usize,
    limits: RequestLimits,
) -> Result<()> {
    validate_audio_samples_for_context(audio_samples, audio_context_length)?;
    limits.validate_audio_samples(audio_samples)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_audio_preflight_uses_deployment_and_model_context_limits() {
        let limits = RequestLimits::default();
        validate_whisper_audio_size(480_000, 1_500, limits)
            .expect("standard Whisper audio window must fit");

        let err = validate_whisper_audio_size(480_001, 1_500, limits)
            .expect_err("audio beyond the model context must fail before preprocessing");
        match err {
            OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "audio_samples");
                assert!(invalid.message.contains("audio_context_length"));
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }

        let deployment_limits = RequestLimits {
            max_audio_samples: 400,
            ..RequestLimits::default()
        };
        let err = validate_whisper_audio_size(401, 1_500, deployment_limits)
            .expect_err("a lower deployment ceiling must also fail before preprocessing");
        assert!(format!("{err}").contains("configured limit 400"));
    }
}
