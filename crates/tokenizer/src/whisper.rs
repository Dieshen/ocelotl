//! Whisper tokenizer startup and decode masking policy.
//!
//! This module is intentionally policy-only. It does not load a Whisper
//! `tokenizer.json`, execute decoding, or expose foreign tokenizer types.

use crate::TokenId;

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
    /// Build the decode mask for transcription with `<|notimestamps|>` in the
    /// prompt.
    pub fn transcribe_without_timestamps(tokens: WhisperStartupTokens) -> Self {
        Self {
            tokens,
            suppress_timestamps: true,
        }
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
