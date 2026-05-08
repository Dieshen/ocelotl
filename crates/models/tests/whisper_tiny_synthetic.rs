use ocelotl_core::{OcelotlError, TokenId};
use ocelotl_models::whisper::audio::{LogMelSpectrogram, WHISPER_MEL_BINS};
use ocelotl_models::whisper::{WhisperTinyConfig, WhisperTinyModel, WhisperTinyWeights};

#[test]
fn tiny_synthetic_whisper_path_matches_pinned_logits() {
    let model = tiny_model();
    let mel = tiny_log_mel();

    let logits = model
        .forward(&mel, &[TokenId(3), TokenId(1)])
        .expect("tiny synthetic Whisper path should run");

    let expected = [0.0_f32, 1.6697615, 0.33023845, -1.339523, 1.0];
    assert_close(&logits, &expected, 1e-6);

    let chosen = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| TokenId(idx as u32))
        .expect("non-empty vocab");
    assert_eq!(chosen, TokenId(1));
}

#[test]
fn forward_rejects_empty_mel_before_compute() {
    let model = tiny_model();
    let mel = LogMelSpectrogram {
        frames: 0,
        mel_bins: WHISPER_MEL_BINS,
        values: Vec::new(),
    };

    let err = model
        .forward(&mel, &[TokenId(1)])
        .expect_err("empty mel input must fail before compute");

    match err {
        OcelotlError::InvalidRequest(invalid) => {
            assert_eq!(invalid.field, "mel.frames");
            assert!(invalid.message.contains("at least one"));
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn forward_rejects_empty_decoder_tokens_before_compute() {
    let model = tiny_model();
    let mel = tiny_log_mel();

    let err = model
        .forward(&mel, &[])
        .expect_err("empty decoder input must fail before compute");

    match err {
        OcelotlError::InvalidRequest(invalid) => {
            assert_eq!(invalid.field, "decoder_tokens");
            assert!(invalid.message.contains("at least one"));
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn forward_rejects_decoder_token_id_outside_vocab_before_compute() {
    let model = tiny_model();
    let mel = tiny_log_mel();

    let err = model
        .forward(&mel, &[TokenId(5)])
        .expect_err("bad decoder token id must fail before compute");

    match err {
        OcelotlError::InvalidRequest(invalid) => {
            assert_eq!(invalid.field, "decoder_tokens");
            assert!(invalid.message.contains("out of range"));
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn forward_rejects_mel_shape_mismatch_before_compute() {
    let model = tiny_model();
    let mut mel = tiny_log_mel();
    mel.values.pop();

    let err = model
        .forward(&mel, &[TokenId(1)])
        .expect_err("malformed mel shape must fail before compute");

    match err {
        OcelotlError::InvalidRequest(invalid) => {
            assert_eq!(invalid.field, "mel.values");
            assert!(invalid.message.contains("expected length"));
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn forward_rejects_too_many_mel_frames_before_compute() {
    let model = tiny_model();
    let mut mel = tiny_log_mel();
    mel.frames = 3;
    mel.values.resize(3 * WHISPER_MEL_BINS, 0.0);

    let err = model
        .forward(&mel, &[TokenId(1)])
        .expect_err("mel input longer than max_audio_frames must fail before compute");

    match err {
        OcelotlError::InvalidRequest(invalid) => {
            assert_eq!(invalid.field, "mel.frames");
            assert!(invalid.message.contains("max_audio_frames"));
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn new_rejects_invalid_config_before_weight_validation() {
    let mut cfg = tiny_config();
    cfg.hidden_size = 0;
    let weights = tiny_weights(&tiny_config());

    let err = WhisperTinyModel::new(cfg, weights).expect_err("invalid config must fail");

    match err {
        OcelotlError::InvalidModel(invalid) => {
            assert_eq!(invalid.field.as_deref(), Some("hidden_size"));
        }
        other => panic!("expected InvalidModel, got {other:?}"),
    }
}

#[test]
fn new_rejects_audio_frame_shape_product_overflow() {
    let mut cfg = tiny_config();
    cfg.mel_bins = 1;
    cfg.hidden_size = 2;
    cfg.max_audio_frames = (usize::MAX / 2) + 1;
    let weights = WhisperTinyWeights {
        encoder_projection: vec![0.0; cfg.mel_bins * cfg.hidden_size],
        encoder_bias: vec![0.0; cfg.hidden_size],
        decoder_token_embedding: vec![0.0; cfg.vocab_size * cfg.hidden_size],
        decoder_query_projection: identity(cfg.hidden_size),
        cross_attention_key_projection: identity(cfg.hidden_size),
        cross_attention_value_projection: identity(cfg.hidden_size),
        cross_attention_output_projection: identity(cfg.hidden_size),
        token_projection: vec![0.0; cfg.hidden_size * cfg.vocab_size],
    };

    let err = WhisperTinyModel::new(cfg, weights).expect_err("overflowing frame shape must fail");

    match err {
        OcelotlError::InvalidModel(invalid) => {
            assert_eq!(
                invalid.field.as_deref(),
                Some("max_audio_frames*hidden_size")
            );
            assert!(invalid.message.contains("overflows"));
        }
        other => panic!("expected InvalidModel, got {other:?}"),
    }
}

#[test]
fn new_rejects_weight_shape_mismatch_before_compute() {
    let cfg = tiny_config();
    let mut weights = tiny_weights(&cfg);
    weights.cross_attention_key_projection.pop();

    let err = WhisperTinyModel::new(cfg, weights)
        .expect_err("weight length mismatch must fail at model construction");

    match err {
        OcelotlError::InvalidModel(invalid) => {
            assert_eq!(
                invalid.field.as_deref(),
                Some("cross_attention_key_projection")
            );
        }
        other => panic!("expected InvalidModel, got {other:?}"),
    }
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

    // Encoder projection is mostly zeros so the first two mel bins map to
    // hand-checkable encoder states: frame 0 -> [1, 0], frame 1 -> [0, 1].
    let mut encoder_projection = vec![0.0_f32; cfg.mel_bins * h];
    encoder_projection[0] = 1.0;
    encoder_projection[h + 1] = 1.0;

    let token_embedding = vec![
        0.0, 0.0, // token 0
        1.0, 0.0, // token 1: decoder query in the pinned test
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

fn tiny_log_mel() -> LogMelSpectrogram {
    let mut values = vec![0.0_f32; 2 * WHISPER_MEL_BINS];
    values[0] = 1.0;
    values[WHISPER_MEL_BINS + 1] = 1.0;

    LogMelSpectrogram {
        frames: 2,
        mel_bins: WHISPER_MEL_BINS,
        values,
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

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let delta = (actual - expected).abs();
        assert!(
            delta <= tolerance,
            "index {idx}: expected {expected}, got {actual}, delta {delta}"
        );
    }
}
