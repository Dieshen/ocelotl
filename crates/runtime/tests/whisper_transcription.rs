use ocelotl_core::{OcelotlError, TokenId};
use ocelotl_models::whisper::audio::{AudioMetadata, WHISPER_FFT_SIZE, WHISPER_MEL_BINS};
use ocelotl_models::whisper::{WhisperTinyConfig, WhisperTinyModel, WhisperTinyWeights};
use ocelotl_runtime::whisper::{TranscriptionRequest, transcribe};

#[test]
fn transcribe_rejects_empty_audio_before_preprocessing_or_model_compute() {
    let model = tiny_model();
    let request = TranscriptionRequest {
        audio_samples: Vec::new(),
        audio_metadata: whisper_audio_metadata(),
        decoder_prompt_tokens: vec![TokenId(3), TokenId(1)],
    };

    let err = transcribe(&model, &request).expect_err("empty audio must fail before compute");

    match err {
        OcelotlError::InvalidRequest(invalid) => {
            assert_eq!(invalid.field, "audio_samples");
            assert!(invalid.message.contains("at least one"));
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn transcribe_rejects_unsupported_audio_metadata_before_model_compute() {
    let model = tiny_model();
    let request = TranscriptionRequest {
        audio_samples: tiny_waveform_fixture(),
        audio_metadata: AudioMetadata {
            sample_rate_hz: 44_100,
            channels: 1,
        },
        decoder_prompt_tokens: vec![TokenId(3), TokenId(1)],
    };

    let err = transcribe(&model, &request)
        .expect_err("unsupported sample rate must fail before model compute");

    match err {
        OcelotlError::Unsupported(unsupported) => {
            assert_eq!(unsupported.feature, "whisper_audio.sample_rate_hz");
            assert_eq!(unsupported.requested.as_deref(), Some("44100"));
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn transcribe_runs_tiny_synthetic_whisper_path_and_returns_token_plus_logits() {
    let model = tiny_model();
    let request = TranscriptionRequest {
        audio_samples: tiny_waveform_fixture(),
        audio_metadata: whisper_audio_metadata(),
        decoder_prompt_tokens: vec![TokenId(3), TokenId(1)],
    };

    let response = transcribe(&model, &request).expect("synthetic transcription must run");

    assert_eq!(response.logits.len(), model.config().vocab_size);
    assert_eq!(response.tokens, vec![TokenId(1)]);
    assert!(response.logits.iter().all(|value| value.is_finite()));
}

#[test]
fn transcribe_propagates_model_errors_after_runtime_audio_validation() {
    let model = tiny_model();
    let request = TranscriptionRequest {
        audio_samples: tiny_waveform_fixture(),
        audio_metadata: whisper_audio_metadata(),
        decoder_prompt_tokens: Vec::new(),
    };

    let err = transcribe(&model, &request).expect_err("model validation error must propagate");

    match err {
        OcelotlError::InvalidRequest(invalid) => {
            assert_eq!(invalid.field, "decoder_tokens");
            assert!(invalid.message.contains("at least one"));
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

fn whisper_audio_metadata() -> AudioMetadata {
    AudioMetadata {
        sample_rate_hz: 16_000,
        channels: 1,
    }
}

fn tiny_waveform_fixture() -> Vec<f32> {
    let mut audio = vec![0.0; WHISPER_FFT_SIZE];
    audio[0] = 1.0;
    audio[80] = -0.5;
    audio[160] = 0.25;
    audio[240] = -0.125;
    audio[320] = 0.0625;
    audio
}

fn tiny_model() -> WhisperTinyModel {
    let cfg = tiny_config();
    let weights = tiny_weights(&cfg);
    WhisperTinyModel::new(cfg, weights).expect("tiny synthetic weights should construct")
}

fn tiny_config() -> WhisperTinyConfig {
    WhisperTinyConfig {
        mel_bins: WHISPER_MEL_BINS,
        max_audio_frames: 2,
        decoder_context_length: 4,
        vocab_size: 5,
        hidden_size: 2,
    }
}

fn tiny_weights(cfg: &WhisperTinyConfig) -> WhisperTinyWeights {
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;

    let mut encoder_projection = vec![0.0_f32; cfg.mel_bins * h];
    encoder_projection[0] = 1.0;
    encoder_projection[h + 1] = 1.0;

    let token_embedding = vec![
        0.0, 0.0, // token 0
        1.0, 0.0, // token 1
        0.0, 1.0, // token 2
        1.0, 1.0, // token 3
        -1.0, 1.0, // token 4
    ];

    let lm_head_vocab_by_hidden = vec![
        0.0, 0.0, // token 0
        1.0, 0.0, // token 1 reads hidden[0]
        0.0, 1.0, // token 2 reads hidden[1]
        -1.0, 1.0, // token 3 contrasts both dimensions
        0.5, 0.5, // token 4 averages both dimensions
    ];

    WhisperTinyWeights {
        encoder_projection,
        encoder_bias: vec![0.0; h],
        decoder_token_embedding: token_embedding,
        decoder_query_projection: identity(h),
        cross_attention_key_projection: identity(h),
        cross_attention_value_projection: identity(h),
        cross_attention_output_projection: identity(h),
        token_projection: transpose_2d(&lm_head_vocab_by_hidden, v, h),
    }
}

fn identity(size: usize) -> Vec<f32> {
    let mut values = vec![0.0_f32; size * size];
    for idx in 0..size {
        values[idx * size + idx] = 1.0;
    }
    values
}

fn transpose_2d(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(src.len(), rows * cols);
    let mut dst = vec![0.0_f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            dst[col * rows + row] = src[row * cols + col];
        }
    }
    dst
}
