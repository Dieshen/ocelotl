//! Whisper tokenizer startup and decode masking policy.
//!
//! This module is intentionally policy-only. It does not load a Whisper
//! `tokenizer.json`, execute decoding, or expose foreign tokenizer types.

use crate::TokenId;
use ocelotl_core::{OcelotlError, Result, TokenizerError};

/// Whisper timestamp tokens advance in fixed 20 ms steps from
/// `first_timestamp`.
pub const WHISPER_TIMESTAMP_STEP_SECONDS: f32 = 0.02;

/// Ocelotl-owned names for the Whisper startup and decode-control tokens used
/// by the first ASR transcription slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperTokenRole {
    EndOfText,
    StartOfTranscript,
    Language,
    TranscribeTask,
    NoTimestamps,
    FirstTimestamp,
}

/// Token IDs needed to construct the initial Whisper transcription prompt and
/// suppress tokens during decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhisperStartupTokens {
    pub end_of_text: TokenId,
    pub start_of_transcript: TokenId,
    pub language: TokenId,
    pub transcribe_task: TokenId,
    pub no_timestamps: TokenId,
    pub first_timestamp: TokenId,
}

impl WhisperStartupTokens {
    /// Return the token assigned to an explicit Whisper startup role.
    pub fn role_token(self, role: WhisperTokenRole) -> TokenId {
        match role {
            WhisperTokenRole::EndOfText => self.end_of_text,
            WhisperTokenRole::StartOfTranscript => self.start_of_transcript,
            WhisperTokenRole::Language => self.language,
            WhisperTokenRole::TranscribeTask => self.transcribe_task,
            WhisperTokenRole::NoTimestamps => self.no_timestamps,
            WhisperTokenRole::FirstTimestamp => self.first_timestamp,
        }
    }

    /// Construct the initial decode sequence for transcription without
    /// timestamps:
    ///
    /// `<|startoftranscript|>`, language, `<|transcribe|>`, `<|notimestamps|>`.
    pub fn transcribe_no_timestamps_prompt(self) -> Vec<TokenId> {
        vec![
            self.start_of_transcript,
            self.language,
            self.transcribe_task,
            self.no_timestamps,
        ]
    }

    /// Construct the English-only Whisper transcription prompt.
    ///
    /// English-only Whisper does not include language or task tokens in the
    /// startup sequence. Timestamp-enabled transcription omits
    /// `<|notimestamps|>` so the model can emit timestamp boundary tokens.
    pub fn english_transcribe_prompt(self, mode: WhisperTimestampMode) -> Vec<TokenId> {
        match mode {
            WhisperTimestampMode::NoTimestamps => {
                vec![self.start_of_transcript, self.no_timestamps]
            }
            WhisperTimestampMode::Timestamps => vec![self.start_of_transcript],
        }
    }

    /// Convert a Whisper timestamp token into seconds from the start of the
    /// current audio window.
    ///
    /// The OpenAI Whisper timestamp contract uses 0.02 seconds per token offset
    /// from `first_timestamp`: `first_timestamp` is 0.00s,
    /// `first_timestamp + 1` is 0.02s, and so on.
    pub fn timestamp_seconds(self, token: TokenId) -> Option<f32> {
        token
            .0
            .checked_sub(self.first_timestamp.0)
            .map(|offset| offset as f32 * WHISPER_TIMESTAMP_STEP_SECONDS)
    }
}

/// Known OpenAI English-only Whisper special-token IDs for transcription.
///
/// The English-only startup prompt is `<|startoftranscript|>` plus optionally
/// `<|notimestamps|>`. The task/control token IDs are still recorded so decode
/// masks can suppress known Whisper control specials through the whole decode.
pub fn whisper_english_transcribe_tokens() -> WhisperStartupTokens {
    WhisperStartupTokens {
        end_of_text: TokenId(50_256),
        start_of_transcript: TokenId(50_257),
        language: TokenId(50_258),
        transcribe_task: TokenId(50_358),
        no_timestamps: TokenId(50_362),
        first_timestamp: TokenId(50_363),
    }
}

/// Known multilingual Whisper special-token IDs for English transcription.
///
/// These constants mirror the OpenAI Whisper tokenizer special-token layout:
/// EOT 50257, SOT 50258, English language 50259, transcribe 50359,
/// no-timestamps 50363, and the first timestamp token 50364. They are kept as
/// explicit constants instead of fetched artifacts so W-ASR.3 remains offline
/// and policy-focused.
pub fn whisper_multilingual_english_transcribe_tokens() -> WhisperStartupTokens {
    WhisperStartupTokens {
        end_of_text: TokenId(50_257),
        start_of_transcript: TokenId(50_258),
        language: TokenId(50_259),
        transcribe_task: TokenId(50_359),
        no_timestamps: TokenId(50_363),
        first_timestamp: TokenId(50_364),
    }
}

/// Decision returned by a Whisper decode mask for a candidate next token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperTokenMaskDecision {
    Allow,
    Suppress,
}

/// Timestamp behavior for Whisper transcription.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperTimestampMode {
    NoTimestamps,
    Timestamps,
}

/// Decode-time Whisper token suppression policy.
///
/// Non-timestamp prompt special tokens are suppressed for every decode step, not
/// just while consuming the startup prefix. EOT is deliberately not suppressed,
/// because decode must be able to terminate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhisperDecodeMask {
    tokens: WhisperStartupTokens,
    suppress_timestamps: bool,
}

impl WhisperDecodeMask {
    /// Build a transcription decode mask for the selected timestamp mode.
    pub fn transcribe(tokens: WhisperStartupTokens, mode: WhisperTimestampMode) -> Self {
        Self {
            tokens,
            suppress_timestamps: mode == WhisperTimestampMode::NoTimestamps,
        }
    }

    /// Build the decode mask for transcription with `<|notimestamps|>` in the
    /// prompt.
    pub fn transcribe_without_timestamps(tokens: WhisperStartupTokens) -> Self {
        Self::transcribe(tokens, WhisperTimestampMode::NoTimestamps)
    }

    /// Build the decode mask for transcription with timestamp tokens enabled.
    pub fn transcribe_with_timestamps(tokens: WhisperStartupTokens) -> Self {
        Self::transcribe(tokens, WhisperTimestampMode::Timestamps)
    }

    /// Return whether a candidate next token should be available to sampling.
    pub fn mask_token(self, token: TokenId) -> WhisperTokenMaskDecision {
        if self.is_prompt_special(token) {
            return WhisperTokenMaskDecision::Suppress;
        }

        if self.suppress_timestamps && token.0 >= self.tokens.first_timestamp.0 {
            return WhisperTokenMaskDecision::Suppress;
        }

        WhisperTokenMaskDecision::Allow
    }

    fn is_prompt_special(self, token: TokenId) -> bool {
        matches!(
            token,
            t if t == self.tokens.start_of_transcript
                || t == self.tokens.language
                || t == self.tokens.transcribe_task
                || t == self.tokens.no_timestamps
        )
    }
}

/// A timestamp-bounded run of text tokens from a Whisper decode sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct WhisperTimestampedSegment {
    pub start_seconds: f32,
    pub end_seconds: f32,
    pub text_tokens: Vec<TokenId>,
}

/// Parse timestamp-token boundaries into deterministic token-level segments.
///
/// The input is expected to be the generated token stream after the startup
/// prompt. Each segment starts at a timestamp token, collects ordinary text
/// tokens, and ends at the next timestamp token. Consecutive timestamp tokens
/// are treated as boundary updates with no empty segment emitted.
pub fn parse_whisper_timestamped_segments(
    tokens: WhisperStartupTokens,
    sequence: &[TokenId],
) -> Result<Vec<WhisperTimestampedSegment>> {
    let mut segments = Vec::new();
    let mut active_start = None;
    let mut text_tokens = Vec::new();

    for &token in sequence {
        if token == tokens.end_of_text {
            break;
        }

        if tokens.timestamp_seconds(token).is_some() {
            if let Some(start_token) = active_start {
                if !text_tokens.is_empty() {
                    let start_seconds = tokens
                        .timestamp_seconds(start_token)
                        .expect("active start is always a timestamp token");
                    let end_seconds = tokens
                        .timestamp_seconds(token)
                        .expect("current token is known to be a timestamp token");
                    segments.push(WhisperTimestampedSegment {
                        start_seconds,
                        end_seconds,
                        text_tokens: std::mem::take(&mut text_tokens),
                    });
                }
            }
            active_start = Some(token);
            continue;
        }

        if active_start.is_none() {
            return Err(tokenizer_policy_error(format!(
                "timestamped Whisper segment text token {:?} appeared before a start timestamp",
                token
            )));
        }

        text_tokens.push(token);
    }

    if !text_tokens.is_empty() {
        return Err(tokenizer_policy_error(
            "timestamped Whisper segment has text without an end timestamp".to_string(),
        ));
    }

    Ok(segments)
}

fn tokenizer_policy_error(message: String) -> OcelotlError {
    OcelotlError::Tokenizer(TokenizerError {
        message,
        source: None,
    })
}
