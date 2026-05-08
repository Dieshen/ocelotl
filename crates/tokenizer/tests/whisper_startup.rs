//! W-ASR.3 - Whisper startup tokens and decode masking policy.

use ocelotl_tokenizer::{
    TokenId, WhisperDecodeMask, WhisperTokenMaskDecision, WhisperTokenRole,
    whisper_multilingual_english_transcribe_tokens,
};

#[test]
fn whisper_transcribe_no_timestamps_startup_sequence_is_explicit() {
    let tokens = whisper_multilingual_english_transcribe_tokens();

    assert_eq!(
        tokens.role_token(WhisperTokenRole::EndOfText),
        TokenId(50_257)
    );
    assert_eq!(
        tokens.role_token(WhisperTokenRole::StartOfTranscript),
        TokenId(50_258)
    );
    assert_eq!(
        tokens.role_token(WhisperTokenRole::Language),
        TokenId(50_259)
    );
    assert_eq!(
        tokens.role_token(WhisperTokenRole::TranscribeTask),
        TokenId(50_359)
    );
    assert_eq!(
        tokens.role_token(WhisperTokenRole::NoTimestamps),
        TokenId(50_363)
    );
    assert_eq!(
        tokens.role_token(WhisperTokenRole::FirstTimestamp),
        TokenId(50_364)
    );

    assert_eq!(
        tokens.transcribe_no_timestamps_prompt(),
        vec![
            TokenId(50_258),
            TokenId(50_259),
            TokenId(50_359),
            TokenId(50_363),
        ],
        "Whisper transcription startup order is <|startoftranscript|>, \
         language, <|transcribe|>, <|notimestamps|>"
    );
}

#[test]
fn whisper_no_timestamps_decode_mask_suppresses_timestamps_and_prompt_specials() {
    let tokens = whisper_multilingual_english_transcribe_tokens();
    let mask = WhisperDecodeMask::transcribe_without_timestamps(tokens);

    for role in [
        WhisperTokenRole::StartOfTranscript,
        WhisperTokenRole::Language,
        WhisperTokenRole::TranscribeTask,
        WhisperTokenRole::NoTimestamps,
    ] {
        let token = tokens.role_token(role);
        assert_eq!(
            mask.mask_token(token),
            WhisperTokenMaskDecision::Suppress,
            "{role:?} must stay suppressed for the whole decode, not only the \
             prompt prefix"
        );
    }

    assert_eq!(
        mask.mask_token(tokens.role_token(WhisperTokenRole::FirstTimestamp)),
        WhisperTokenMaskDecision::Suppress,
        "no-timestamp transcription must suppress timestamp tokens"
    );
    assert_eq!(
        mask.mask_token(TokenId(
            tokens.role_token(WhisperTokenRole::FirstTimestamp).0 + 17
        )),
        WhisperTokenMaskDecision::Suppress,
        "no-timestamp transcription must suppress the timestamp range, not \
         only the first timestamp token"
    );
    assert_eq!(
        mask.mask_token(tokens.role_token(WhisperTokenRole::EndOfText)),
        WhisperTokenMaskDecision::Allow,
        "EOT must remain available so decode can terminate"
    );
    assert_eq!(
        mask.mask_token(TokenId(42)),
        WhisperTokenMaskDecision::Allow,
        "ordinary vocabulary tokens remain available"
    );
}
