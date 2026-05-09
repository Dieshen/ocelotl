//! W-ASR.11 - English Whisper timestamp decode policy.

use ocelotl_tokenizer::{
    TokenId, WhisperDecodeMask, WhisperTimestampMode, WhisperTokenMaskDecision, WhisperTokenRole,
    parse_whisper_timestamped_segments, whisper_english_transcribe_tokens,
    whisper_multilingual_english_transcribe_tokens,
};

#[test]
fn english_timestamp_enabled_startup_prompt_omits_no_timestamps_token() {
    let tokens = whisper_english_transcribe_tokens();

    assert_eq!(
        tokens.english_transcribe_prompt(WhisperTimestampMode::NoTimestamps),
        vec![
            tokens.role_token(WhisperTokenRole::StartOfTranscript),
            tokens.role_token(WhisperTokenRole::NoTimestamps),
        ],
        "English no-timestamps transcription starts with SOT then <|notimestamps|>"
    );

    let timestamp_prompt = tokens.english_transcribe_prompt(WhisperTimestampMode::Timestamps);
    assert_eq!(
        timestamp_prompt,
        vec![tokens.role_token(WhisperTokenRole::StartOfTranscript)],
        "timestamp-enabled English transcription omits <|notimestamps|>"
    );
    assert!(
        !timestamp_prompt.contains(&tokens.role_token(WhisperTokenRole::NoTimestamps)),
        "<|notimestamps|> must not be present when timestamps are enabled"
    );
}

#[test]
fn multilingual_timestamp_enabled_startup_prompt_omits_no_timestamps_token() {
    let tokens = whisper_multilingual_english_transcribe_tokens();

    assert_eq!(
        tokens.transcribe_prompt(WhisperTimestampMode::NoTimestamps),
        vec![
            tokens.role_token(WhisperTokenRole::StartOfTranscript),
            tokens.role_token(WhisperTokenRole::Language),
            tokens.role_token(WhisperTokenRole::TranscribeTask),
            tokens.role_token(WhisperTokenRole::NoTimestamps),
        ],
        "multilingual no-timestamps transcription includes SOT, language, task, then <|notimestamps|>"
    );

    let timestamp_prompt = tokens.transcribe_prompt(WhisperTimestampMode::Timestamps);
    assert_eq!(
        timestamp_prompt,
        vec![
            tokens.role_token(WhisperTokenRole::StartOfTranscript),
            tokens.role_token(WhisperTokenRole::Language),
            tokens.role_token(WhisperTokenRole::TranscribeTask),
        ],
        "timestamp-enabled multilingual transcription omits <|notimestamps|>"
    );
    assert!(
        !timestamp_prompt.contains(&tokens.role_token(WhisperTokenRole::NoTimestamps)),
        "<|notimestamps|> must not be present when timestamps are enabled"
    );
}

#[test]
fn no_timestamps_mode_suppresses_timestamp_tokens() {
    let tokens = whisper_english_transcribe_tokens();
    let mask = WhisperDecodeMask::transcribe(tokens, WhisperTimestampMode::NoTimestamps);

    assert_eq!(
        mask.mask_token(tokens.role_token(WhisperTokenRole::FirstTimestamp)),
        WhisperTokenMaskDecision::Suppress,
        "no-timestamps mode suppresses the first timestamp token"
    );
    assert_eq!(
        mask.mask_token(TokenId(
            tokens.role_token(WhisperTokenRole::FirstTimestamp).0 + 23
        )),
        WhisperTokenMaskDecision::Suppress,
        "no-timestamps mode suppresses the whole timestamp range"
    );
    assert_eq!(
        mask.mask_token(TokenId(42)),
        WhisperTokenMaskDecision::Allow,
        "ordinary text tokens remain available"
    );
}

#[test]
fn timestamp_enabled_mode_allows_timestamp_tokens_and_suppresses_prompt_specials() {
    let tokens = whisper_english_transcribe_tokens();
    let mask = WhisperDecodeMask::transcribe(tokens, WhisperTimestampMode::Timestamps);

    assert_eq!(
        mask.mask_token(tokens.role_token(WhisperTokenRole::FirstTimestamp)),
        WhisperTokenMaskDecision::Allow,
        "timestamp-enabled mode allows timestamp boundary tokens"
    );
    assert_eq!(
        mask.mask_token(TokenId(
            tokens.role_token(WhisperTokenRole::FirstTimestamp).0 + 23
        )),
        WhisperTokenMaskDecision::Allow,
        "timestamp-enabled mode allows later timestamp tokens"
    );

    for role in [
        WhisperTokenRole::StartOfTranscript,
        WhisperTokenRole::NoTimestamps,
    ] {
        assert_eq!(
            mask.mask_token(tokens.role_token(role)),
            WhisperTokenMaskDecision::Suppress,
            "{role:?} must stay suppressed during timestamp-enabled decode"
        );
    }

    assert_eq!(
        mask.mask_token(TokenId(42)),
        WhisperTokenMaskDecision::Allow,
        "ordinary text tokens remain available"
    );
    assert_eq!(
        mask.mask_token(tokens.role_token(WhisperTokenRole::EndOfText)),
        WhisperTokenMaskDecision::Allow,
        "EOT must remain available so decode can terminate"
    );
}

#[test]
fn timestamp_token_to_time_uses_twenty_milliseconds_per_token_offset() {
    let tokens = whisper_english_transcribe_tokens();
    let first_timestamp = tokens.role_token(WhisperTokenRole::FirstTimestamp);

    assert_eq!(
        tokens.timestamp_seconds(TokenId(first_timestamp.0 - 1)),
        None
    );
    assert_seconds(tokens.timestamp_seconds(first_timestamp), 0.0);
    assert_seconds(
        tokens.timestamp_seconds(TokenId(first_timestamp.0 + 1)),
        0.02,
    );
    assert_seconds(
        tokens.timestamp_seconds(TokenId(first_timestamp.0 + 50)),
        1.0,
    );
}

#[test]
fn timestamped_segments_parse_text_between_timestamp_boundaries() {
    let tokens = whisper_english_transcribe_tokens();
    let first_timestamp = tokens.role_token(WhisperTokenRole::FirstTimestamp);

    let segments = parse_whisper_timestamped_segments(
        tokens,
        &[
            TokenId(first_timestamp.0 + 10),
            TokenId(42),
            TokenId(43),
            TokenId(first_timestamp.0 + 25),
            TokenId(first_timestamp.0 + 30),
            TokenId(44),
            TokenId(first_timestamp.0 + 40),
        ],
    )
    .expect("timestamp pairs with text between them should parse");

    assert_eq!(segments.len(), 2);
    assert_seconds(Some(segments[0].start_seconds), 0.20);
    assert_seconds(Some(segments[0].end_seconds), 0.50);
    assert_eq!(segments[0].text_tokens, vec![TokenId(42), TokenId(43)]);

    assert_seconds(Some(segments[1].start_seconds), 0.60);
    assert_seconds(Some(segments[1].end_seconds), 0.80);
    assert_eq!(segments[1].text_tokens, vec![TokenId(44)]);
}

fn assert_seconds(actual: Option<f32>, expected: f32) {
    let actual = actual.expect("expected a timestamp second value");
    assert!(
        (actual - expected).abs() <= f32::EPSILON,
        "expected {expected:.2}s, got {actual:.2}s"
    );
}
