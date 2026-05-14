use ocelotl_core::{OcelotlError, TokenId};
use ocelotl_models::whisper::audio::AudioMetadata;
use ocelotl_runtime::whisper::{
    ChunkedTranscriptionRequest, TranscriptionChunkingConfig, plan_transcription_chunks,
};

#[test]
fn chunk_planner_pins_window_overlap_and_last_partial_chunk() {
    let request = chunked_request(36_000, config_for_1s_window_250ms_overlap());

    let chunks = plan_transcription_chunks(&request).expect("valid chunk plan must succeed");

    assert_eq!(request.chunking.window_samples, 16_000);
    assert_eq!(request.chunking.overlap_samples, 4_000);
    assert_eq!(request.chunking.window_seconds(16_000).unwrap(), 1.0);
    assert_eq!(request.chunking.overlap_seconds(16_000).unwrap(), 0.25);

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].index, 0);
    assert_eq!(chunks[0].start_sample, 0);
    assert_eq!(chunks[0].end_sample, 16_000);
    assert_eq!(chunks[0].start_seconds, 0.0);
    assert_eq!(chunks[0].end_seconds, 1.0);
    assert_eq!(chunks[0].sample_count(), 16_000);
    assert!(!chunks[0].is_last);

    assert_eq!(chunks[1].index, 1);
    assert_eq!(chunks[1].start_sample, 12_000);
    assert_eq!(chunks[1].end_sample, 28_000);
    assert_eq!(chunks[1].start_seconds, 0.75);
    assert_eq!(chunks[1].end_seconds, 1.75);
    assert_eq!(chunks[1].sample_count(), 16_000);
    assert!(!chunks[1].is_last);

    assert_eq!(chunks[2].index, 2);
    assert_eq!(chunks[2].start_sample, 24_000);
    assert_eq!(chunks[2].end_sample, 36_000);
    assert_eq!(chunks[2].start_seconds, 1.5);
    assert_eq!(chunks[2].end_seconds, 2.25);
    assert_eq!(chunks[2].sample_count(), 12_000);
    assert!(chunks[2].is_last);
}

#[test]
fn chunk_planner_keeps_chunk_ranges_monotonic_in_samples_and_seconds() {
    let request = chunked_request(56_000, config_for_1s_window_250ms_overlap());

    let chunks = plan_transcription_chunks(&request).expect("valid chunk plan must succeed");

    let mut previous_start_sample = 0;
    let mut previous_end_sample = 0;
    let mut previous_start_seconds = 0.0;
    let mut previous_end_seconds = 0.0;

    for (idx, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.index, idx);
        assert!(chunk.start_sample < chunk.end_sample);
        assert!(chunk.start_seconds < chunk.end_seconds);

        if idx > 0 {
            assert!(chunk.start_sample > previous_start_sample);
            assert!(chunk.end_sample > previous_end_sample);
            assert!(chunk.start_seconds > previous_start_seconds);
            assert!(chunk.end_seconds > previous_end_seconds);
        }

        previous_start_sample = chunk.start_sample;
        previous_end_sample = chunk.end_sample;
        previous_start_seconds = chunk.start_seconds;
        previous_end_seconds = chunk.end_seconds;
    }
}

#[test]
fn chunk_planner_rejects_zero_window() {
    let request = chunked_request(
        16_000,
        TranscriptionChunkingConfig {
            window_samples: 0,
            overlap_samples: 0,
        },
    );

    let err = plan_transcription_chunks(&request).expect_err("zero window must be rejected");

    match err {
        OcelotlError::InvalidRequest(invalid) => {
            assert_eq!(invalid.field, "chunking.window_samples");
            assert!(invalid.message.contains("greater than zero"));
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn chunk_planner_rejects_overlap_equal_to_window() {
    let request = chunked_request(
        16_000,
        TranscriptionChunkingConfig {
            window_samples: 16_000,
            overlap_samples: 16_000,
        },
    );

    let err = plan_transcription_chunks(&request).expect_err("overlap >= window must fail");

    match err {
        OcelotlError::InvalidRequest(invalid) => {
            assert_eq!(invalid.field, "chunking.overlap_samples");
            assert!(invalid.message.contains("less than window_samples"));
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn chunk_planner_rejects_unsupported_audio_metadata() {
    let request = ChunkedTranscriptionRequest {
        audio_samples: vec![0.0; 16_000],
        audio_metadata: AudioMetadata {
            sample_rate_hz: 44_100,
            channels: 1,
        },
        decoder_prompt_tokens: vec![TokenId(3), TokenId(1)],
        chunking: config_for_1s_window_250ms_overlap(),
    };

    let err = plan_transcription_chunks(&request).expect_err("unsupported sample rate must fail");

    match err {
        OcelotlError::Unsupported(unsupported) => {
            assert_eq!(unsupported.feature, "whisper_audio.sample_rate_hz");
            assert_eq!(unsupported.requested.as_deref(), Some("44100"));
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

fn chunked_request(
    audio_sample_count: usize,
    chunking: TranscriptionChunkingConfig,
) -> ChunkedTranscriptionRequest {
    ChunkedTranscriptionRequest {
        audio_samples: vec![0.0; audio_sample_count],
        audio_metadata: whisper_audio_metadata(),
        decoder_prompt_tokens: vec![TokenId(3), TokenId(1)],
        chunking,
    }
}

fn config_for_1s_window_250ms_overlap() -> TranscriptionChunkingConfig {
    TranscriptionChunkingConfig {
        window_samples: 16_000,
        overlap_samples: 4_000,
    }
}

fn whisper_audio_metadata() -> AudioMetadata {
    AudioMetadata {
        sample_rate_hz: 16_000,
        channels: 1,
    }
}
