//! Tests for the Whisper real CPU reference adapter.

use std::{collections::BTreeMap, fs, io::Write, path::PathBuf};

use ocelotl_core::{DType, TokenId};
use ocelotl_kernels::default_kernel_backend;

use super::{WhisperConfig, required_whisper_tensor_names};
use super::WhisperModel;
use super::primitives::{
    attention, attention_incremental_from_projected, attention_with_precomputed_kv, conv1d, gelu,
    layer_norm, mlp_gelu,
};
use super::state::WhisperEncodedAudio;
use super::weights::expected_shape;

#[test]
fn conv1d_applies_padding_and_stride() {
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let weight = [10.0_f32, 1.0, -1.0];
    let bias = [0.5_f32];

    let out = conv1d(&input, 4, 1, &weight, &bias, 1, 3, 2, 1).expect("conv1d");

    assert_eq!(out, vec![-0.5, 19.5]);
}

#[test]
fn layer_norm_applies_weight_bias_and_epsilon_per_row() {
    let input = [1.0_f32, 3.0];
    let weight = [2.0_f32, 0.5];
    let bias = [0.1_f32, -0.2];

    let out = layer_norm(&input, 1, 2, &weight, &bias, 1.0e-5).expect("layer norm");

    assert_close(&out, &[-1.89999, 0.2999975], 1.0e-5);
}

#[test]
fn gelu_mlp_projection_path_matches_hand_checked_fixture() {
    let x = [1.0_f32, 0.0];
    let fc1_w = [1.0_f32, 0.0, 0.0, 1.0];
    let fc1_b = [0.0_f32, 0.0];
    let fc2_w = [1.0_f32, 0.0, 0.0, 1.0];
    let fc2_b = [0.0_f32, 0.0];
    let kernels = default_kernel_backend();

    let out = mlp_gelu(
        kernels.as_ref(),
        &x,
        1,
        2,
        2,
        &fc1_w,
        &fc1_b,
        &fc2_w,
        &fc2_b,
    )
    .expect("mlp");

    assert_close(&out, &[0.841_344_7, 0.0], 1.0e-6);
}

#[test]
fn gelu_pins_exact_erf_variant_used_by_openai_whisper() {
    assert_close(&[gelu(1.0)], &[0.841_344_7], 1.0e-6);
    assert_close(&[gelu(-1.0)], &[-0.158_655_26], 1.0e-6);
}

#[test]
fn encoder_self_attention_does_not_apply_causal_mask() {
    let x = [1.0_f32, 2.0, 7.0];
    let identity = [1.0_f32];
    let zero = [0.0_f32];
    let kernels = default_kernel_backend();
    let out = attention(
        kernels.as_ref(),
        &x,
        3,
        1,
        1,
        &zero,
        &zero,
        &zero,
        &identity,
        &zero,
        &identity,
        &zero,
        None,
        false,
    )
    .expect("encoder self attention");

    assert_close(&out, &[10.0 / 3.0, 10.0 / 3.0, 10.0 / 3.0], 1.0e-5);
}

#[test]
fn decoder_self_attention_applies_causal_mask() {
    let x = [1.0_f32, 2.0, 7.0];
    let identity = [1.0_f32];
    let zero = [0.0_f32];
    let kernels = default_kernel_backend();
    let out = attention(
        kernels.as_ref(),
        &x,
        3,
        1,
        1,
        &zero,
        &zero,
        &zero,
        &identity,
        &zero,
        &identity,
        &zero,
        None,
        true,
    )
    .expect("decoder self attention");

    assert_close(&out, &[1.0, 1.5, 10.0 / 3.0], 1.0e-5);
}

#[test]
fn incremental_self_attention_matches_full_causal_last_row() {
    let x = [1.0_f32, 2.0, 7.0];
    let identity = [1.0_f32];
    let zero = [0.0_f32];
    let kernels = default_kernel_backend();
    let full = attention(
        kernels.as_ref(),
        &x,
        3,
        1,
        1,
        &zero,
        &zero,
        &zero,
        &identity,
        &zero,
        &identity,
        &zero,
        None,
        true,
    )
    .expect("full causal self attention");
    let incremental = attention_incremental_from_projected(
        kernels.as_ref(),
        &[0.0],
        &[0.0],
        &[7.0],
        &[0.0, 0.0],
        &[1.0, 2.0],
        2,
        1,
        1,
        &identity,
        &zero,
    )
    .expect("incremental self attention");

    assert_close(&incremental, &full[2..3], 0.0);
}

#[test]
fn decoder_cross_attention_does_not_apply_causal_mask() {
    let text = [1.0_f32, 1.0];
    let audio = [1.0_f32, 2.0, 7.0];
    let identity = [1.0_f32];
    let zero = [0.0_f32];
    let kernels = default_kernel_backend();
    let out = attention(
        kernels.as_ref(),
        &text,
        2,
        1,
        1,
        &zero,
        &zero,
        &zero,
        &identity,
        &zero,
        &identity,
        &zero,
        Some((&audio, 3)),
        false,
    )
    .expect("decoder cross attention");

    assert_close(&out, &[10.0 / 3.0, 10.0 / 3.0], 1.0e-5);
}

#[test]
fn model_construction_rejects_missing_weight_before_compute() {
    let cfg = tiny_config();
    let mut weights = tiny_weight_tensors(&cfg);
    weights.retain(|tensor| tensor.name != "encoder.conv1.weight");

    let err = WhisperModel::new(cfg, weights).expect_err("missing tensor must fail");

    match err {
        ocelotl_core::OcelotlError::InvalidModel(invalid) => {
            assert_eq!(invalid.field.as_deref(), Some("encoder.conv1.weight"));
        }
        other => panic!("expected InvalidModel, got {other:?}"),
    }
}

#[test]
fn model_construction_rejects_wrong_loaded_shape_before_compute() {
    let cfg = tiny_config();
    let mut weights = tiny_weight_tensors(&cfg);
    let tensor = weights
        .iter_mut()
        .find(|tensor| tensor.name == "decoder.token_embedding.weight")
        .expect("token embedding test tensor");
    tensor.shape = vec![cfg.text_state_size, cfg.vocab_size];

    let err = WhisperModel::new(cfg, weights).expect_err("wrong tensor shape must fail");

    match err {
        ocelotl_core::OcelotlError::InvalidModel(invalid) => {
            assert_eq!(
                invalid.field.as_deref(),
                Some("decoder.token_embedding.weight")
            );
            assert!(invalid.message.contains("expected"));
        }
        other => panic!("expected InvalidModel, got {other:?}"),
    }
}

#[test]
fn cached_audio_logits_match_legacy_forward_path() {
    let cfg = tiny_config();
    let model = WhisperModel::new(cfg.clone(), tiny_weight_tensors(&cfg)).expect("model");
    let mel = vec![0.0_f32; 4 * cfg.mel_bins];
    let tokens = [TokenId(0), TokenId(2)];

    let legacy = model
        .forward_next_token_logits(&mel, 4, &tokens)
        .expect("legacy forward");
    let audio = model.encode_audio_features(&mel, 4).expect("encoded audio");
    let cached = model
        .forward_next_token_logits_from_audio(&audio, &tokens)
        .expect("cached forward");

    assert_eq!(audio.frames(), 2);
    assert_eq!(audio.state_size(), cfg.audio_state_size);
    assert_eq!(audio.values().len(), audio.frames() * audio.state_size());
    assert_eq!(audio.cross_attention.len(), cfg.text_layers);
    for cache in &audio.cross_attention {
        assert_eq!(cache.key.len(), audio.frames() * cfg.text_state_size);
        assert_eq!(cache.value.len(), audio.frames() * cfg.text_state_size);
    }
    assert_close(&cached, &legacy, 0.0);
}

#[test]
fn timed_audio_encode_matches_plain_audio_encode() {
    let cfg = tiny_config();
    let model = WhisperModel::new(cfg.clone(), tiny_weight_tensors(&cfg)).expect("model");
    let mel = vec![0.0_f32; 4 * cfg.mel_bins];

    let plain = model.encode_audio_features(&mel, 4).expect("plain encode");
    let (timed, timings) = model
        .encode_audio_features_with_timings(&mel, 4)
        .expect("timed encode");

    assert_eq!(timed, plain);
    assert!(timings.encoder_ms < 1_000);
    assert!(timings.cross_attention_precompute_ms < 1_000);
}

#[test]
fn load_from_dir_builds_whisper_model_from_local_files_without_downloads() {
    let cfg = tiny_config();
    let dir = tmp_dir("load_from_dir");
    write_whisper_config(&dir.join("config.json"), &cfg);
    write_safetensors_f32(
        &dir.join("model.safetensors"),
        &synthetic_weight_tensors(&cfg),
    );

    let loaded = WhisperModel::load_from_dir(&dir)
        .expect("local Whisper directory must load through family helper");
    let expected = WhisperModel::new(cfg.clone(), tiny_weight_tensors(&cfg)).expect("model");
    let mel = vec![0.0_f32; 4 * cfg.mel_bins];
    let tokens = [TokenId(0), TokenId(2)];

    assert_eq!(loaded.config(), &cfg);
    assert_close(
        &loaded
            .forward_next_token_logits(&mel, 4, &tokens)
            .expect("loaded model logits"),
        &expected
            .forward_next_token_logits(&mel, 4, &tokens)
            .expect("expected model logits"),
        0.0,
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn precomputed_cross_attention_matches_projected_cross_attention() {
    let text = [1.0_f32, 1.0];
    let audio = [1.0_f32, 2.0, 7.0];
    let identity = [1.0_f32];
    let zero = [0.0_f32];
    let kernels = default_kernel_backend();

    let projected = attention(
        kernels.as_ref(),
        &text,
        2,
        1,
        1,
        &zero,
        &zero,
        &zero,
        &identity,
        &zero,
        &identity,
        &zero,
        Some((&audio, 3)),
        false,
    )
    .expect("projected cross attention");
    let precomputed = attention_with_precomputed_kv(
        kernels.as_ref(),
        &text,
        2,
        1,
        1,
        &zero,
        &zero,
        &audio,
        &audio,
        3,
        &identity,
        &zero,
        false,
    )
    .expect("precomputed cross attention");

    assert_close(&precomputed, &projected, 0.0);
}

#[test]
fn decoder_state_append_matches_full_context_logits() {
    let cfg = tiny_config();
    let model = WhisperModel::new(cfg.clone(), tiny_weight_tensors(&cfg)).expect("model");
    let mel = vec![0.0_f32; 4 * cfg.mel_bins];
    let audio = model.encode_audio_features(&mel, 4).expect("encoded audio");
    let prompt = [TokenId(0)];
    let appended = TokenId(2);

    let full_prompt = model
        .forward_next_token_logits_from_audio(&audio, &prompt)
        .expect("full prompt logits");
    let mut state = model
        .prepare_decoder_state_from_audio(&audio, &prompt)
        .expect("decoder state");
    assert_eq!(state.tokens(), &prompt);
    assert_eq!(state.self_attention.len(), cfg.text_layers);
    for cache in &state.self_attention {
        assert_eq!(cache.key.len(), prompt.len() * cfg.text_state_size);
        assert_eq!(cache.value.len(), prompt.len() * cfg.text_state_size);
    }
    assert_close(state.next_token_logits(), &full_prompt, 0.0);

    let full_appended = model
        .forward_next_token_logits_from_audio(&audio, &[prompt[0], appended])
        .expect("full appended logits");
    let incremental_appended = model
        .append_decoder_token_from_audio(&audio, &mut state, appended)
        .expect("incremental appended logits")
        .to_vec();

    assert_eq!(state.tokens(), &[prompt[0], appended]);
    for cache in &state.self_attention {
        assert_eq!(cache.key.len(), state.tokens().len() * cfg.text_state_size);
        assert_eq!(
            cache.value.len(),
            state.tokens().len() * cfg.text_state_size
        );
    }
    assert_close(&incremental_appended, &full_appended, 1.0e-5);
    assert_close(state.next_token_logits(), &full_appended, 1.0e-5);
}

#[test]
fn decoder_state_append_rejects_context_overflow_before_compute() {
    let cfg = tiny_config();
    let model = WhisperModel::new(cfg.clone(), tiny_weight_tensors(&cfg)).expect("model");
    let mel = vec![0.0_f32; 4 * cfg.mel_bins];
    let audio = model.encode_audio_features(&mel, 4).expect("encoded audio");
    let mut state = model
        .prepare_decoder_state_from_audio(&audio, &[TokenId(0), TokenId(1), TokenId(2)])
        .expect("full decoder state");

    let err = model
        .append_decoder_token_from_audio(&audio, &mut state, TokenId(3))
        .expect_err("context overflow must fail");

    match err {
        ocelotl_core::OcelotlError::InvalidRequest(invalid) => {
            assert_eq!(invalid.field, "decoder_context_length");
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn optimized_cpu_backend_preserves_forward_logits() {
    let cfg = tiny_config();
    let scalar = WhisperModel::new(cfg.clone(), tiny_weight_tensors(&cfg)).expect("scalar model");
    let optimized = WhisperModel::with_kernel_backend(
        cfg.clone(),
        tiny_weight_tensors(&cfg),
        ocelotl_kernels::optimized_cpu_kernel_backend(),
    )
    .expect("optimized model");
    let mel = vec![0.0_f32; 4 * cfg.mel_bins];
    let tokens = [TokenId(0), TokenId(2)];

    assert_eq!(optimized.kernel_backend().name(), "cpu");
    let scalar_logits = scalar
        .forward_next_token_logits(&mel, 4, &tokens)
        .expect("scalar logits");
    let optimized_logits = optimized
        .forward_next_token_logits(&mel, 4, &tokens)
        .expect("optimized logits");

    assert_close(&optimized_logits, &scalar_logits, 1.0e-5);
}

#[test]
fn cached_audio_forward_rejects_wrong_state_size_before_compute() {
    let cfg = tiny_config();
    let model = WhisperModel::new(cfg.clone(), tiny_weight_tensors(&cfg)).expect("model");
    let audio = WhisperEncodedAudio {
        frames: 1,
        state_size: cfg.audio_state_size + 1,
        values: vec![0.0; cfg.audio_state_size + 1],
        cross_attention: Vec::new(),
    };

    let err = model
        .forward_next_token_logits_from_audio(&audio, &[TokenId(0)])
        .expect_err("wrong encoded audio shape must fail");

    match err {
        ocelotl_core::OcelotlError::InvalidRequest(invalid) => {
            assert_eq!(invalid.field, "encoded_audio.state_size");
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

fn tiny_config() -> WhisperConfig {
    WhisperConfig {
        vocab_size: 4,
        mel_bins: 2,
        audio_context_length: 2,
        audio_state_size: 2,
        audio_attention_heads: 1,
        audio_layers: 1,
        audio_ffn_size: 2,
        text_context_length: 3,
        text_state_size: 2,
        text_attention_heads: 1,
        text_layers: 1,
        text_ffn_size: 2,
        dtype: DType::F32,
        tie_word_embeddings: true,
    }
}

fn tiny_weight_tensors(cfg: &WhisperConfig) -> Vec<ocelotl_loader::LoadedTensor> {
    synthetic_weight_tensors(cfg)
}

fn synthetic_weight_tensors(cfg: &WhisperConfig) -> Vec<ocelotl_loader::LoadedTensor> {
    required_whisper_tensor_names(cfg)
        .into_iter()
        .map(|name| {
            let shape = expected_shape(&name, cfg).expect("known test tensor shape");
            let len = shape.iter().product();
            let mut values = vec![0.0_f32; len];
            if name == "decoder.token_embedding.weight" {
                values = vec![
                    0.5, 0.0, // token 0
                    0.0, 0.0, // token 1
                    0.25, 0.0, // token 2
                    -0.25, 0.0, // token 3
                ];
            } else if name == "decoder.ln.bias" {
                values = vec![1.0, 0.0];
            }
            ocelotl_loader::LoadedTensor {
                name,
                shape,
                dtype: ocelotl_loader::SupportedDtype::F32,
                values,
            }
        })
        .collect()
}

fn tmp_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("ocelotl_whisper_{}_{}", std::process::id(), name));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).expect("create temp dir");
    path
}

fn write_whisper_config(path: &std::path::Path, cfg: &WhisperConfig) {
    let dtype = match cfg.dtype {
        DType::F32 => "float32",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::Q4 | DType::Q8 => panic!("test config uses unsupported Whisper dtype"),
    };
    let raw = format!(
        r#"{{
          "model_type": "whisper",
          "vocab_size": {vocab_size},
          "num_mel_bins": {mel_bins},
          "d_model": {state},
          "encoder_layers": {audio_layers},
          "encoder_attention_heads": {audio_heads},
          "encoder_ffn_dim": {audio_ffn},
          "decoder_layers": {text_layers},
          "decoder_attention_heads": {text_heads},
          "decoder_ffn_dim": {text_ffn},
          "max_source_positions": {audio_ctx},
          "max_target_positions": {text_ctx},
          "torch_dtype": "{dtype}",
          "tie_word_embeddings": {tie_word_embeddings}
        }}"#,
        vocab_size = cfg.vocab_size,
        mel_bins = cfg.mel_bins,
        state = cfg.audio_state_size,
        audio_layers = cfg.audio_layers,
        audio_heads = cfg.audio_attention_heads,
        audio_ffn = cfg.audio_ffn_size,
        text_layers = cfg.text_layers,
        text_heads = cfg.text_attention_heads,
        text_ffn = cfg.text_ffn_size,
        audio_ctx = cfg.audio_context_length,
        text_ctx = cfg.text_context_length,
        tie_word_embeddings = cfg.tie_word_embeddings,
    );
    fs::write(path, raw).expect("write Whisper config");
}

fn write_safetensors_f32(path: &std::path::Path, tensors: &[ocelotl_loader::LoadedTensor]) {
    let mut header = BTreeMap::new();
    let mut data = Vec::new();
    for tensor in tensors {
        let begin = data.len();
        for value in &tensor.values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let end = data.len();
        header.insert(
            tensor.name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": tensor.shape,
                "data_offsets": [begin, end],
            }),
        );
    }

    let header_json = serde_json::to_string(&header).expect("serialize safetensors header");
    let mut file = fs::File::create(path).expect("create safetensors");
    file.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header length");
    file.write_all(header_json.as_bytes())
        .expect("write header");
    file.write_all(&data).expect("write tensor data");
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
