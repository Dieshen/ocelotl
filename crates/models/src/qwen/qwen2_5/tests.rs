//! Tests for Qwen2.5 model construction, prefill, and weight loading.

use std::{collections::BTreeMap, fs, io::Write, path::PathBuf, sync::Arc};

use ocelotl_core::{DType, OcelotlError, TokenId};
use ocelotl_loader::{LoadedTensor, SupportedDtype};

use super::{Qwen2_5Config, Qwen2_5LayerWeights, Qwen2_5Model, Qwen2_5Weights, transpose_2d};

#[derive(Debug)]
struct TestGpuKernelBackend {
    context: ocelotl_kernels::KernelContext,
}

impl TestGpuKernelBackend {
    fn new() -> Self {
        Self {
            context: ocelotl_kernels::KernelContext {
                device: ocelotl_core::Device::Gpu { ordinal: 7 },
            },
        }
    }
}

impl ocelotl_kernels::KernelBackend for TestGpuKernelBackend {
    fn name(&self) -> &'static str {
        "test-gpu"
    }

    fn context(&self) -> &ocelotl_kernels::KernelContext {
        &self.context
    }

    fn matmul(
        &self,
        _a: &[f32],
        _a_shape: (usize, usize),
        _b: &[f32],
        _b_shape: (usize, usize),
        _out: &mut [f32],
    ) -> ocelotl_core::Result<()> {
        Err(test_gpu_kernel_error(
            "matmul not implemented in test backend",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_out_by_in(
        &self,
        _x: &[f32],
        _rows: usize,
        _in_features: usize,
        _weight_out_by_in: &[f32],
        _out_features: usize,
        _bias: Option<&[f32]>,
        _out: &mut [f32],
    ) -> ocelotl_core::Result<()> {
        Err(test_gpu_kernel_error(
            "linear_out_by_in not implemented in test backend",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn scaled_dot_product_attention(
        &self,
        _q: &[f32],
        _k: &[f32],
        _v: &[f32],
        _seq_len: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _out: &mut [f32],
    ) -> ocelotl_core::Result<()> {
        Err(test_gpu_kernel_error(
            "scaled_dot_product_attention not implemented in test backend",
        ))
    }

    fn rope_apply_inplace(
        &self,
        _x: &mut [f32],
        _head_dim: usize,
        _position: usize,
        _theta: f32,
    ) -> ocelotl_core::Result<()> {
        Err(test_gpu_kernel_error(
            "rope_apply_inplace not implemented in test backend",
        ))
    }

    fn rmsnorm(
        &self,
        _x: &[f32],
        _rows: usize,
        _hidden: usize,
        _weight: &[f32],
        _epsilon: f32,
        _out: &mut [f32],
    ) -> ocelotl_core::Result<()> {
        Err(test_gpu_kernel_error(
            "rmsnorm not implemented in test backend",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn mlp_gated_silu(
        &self,
        _x: &[f32],
        _rows: usize,
        _hidden: usize,
        _intermediate: usize,
        _gate_w: &[f32],
        _up_w: &[f32],
        _down_w: &[f32],
        _gate_buf: &mut [f32],
        _up_buf: &mut [f32],
        _out: &mut [f32],
    ) -> ocelotl_core::Result<()> {
        Err(test_gpu_kernel_error(
            "mlp_gated_silu not implemented in test backend",
        ))
    }

    fn vec_add(&self, _a: &[f32], _b: &[f32], _out: &mut [f32]) -> ocelotl_core::Result<()> {
        Err(test_gpu_kernel_error(
            "vec_add not implemented in test backend",
        ))
    }
}

fn test_gpu_kernel_error(message: &str) -> OcelotlError {
    OcelotlError::Kernel(ocelotl_core::KernelError {
        backend: "test-gpu".to_string(),
        message: message.to_string(),
    })
}

/// A tiny config that satisfies every Qwen2_5Config invariant
/// (head_dim * num_attention_heads == hidden_size, GQA divisibility,
/// positive dims) while keeping arithmetic small enough to debug.
fn tiny_config() -> Qwen2_5Config {
    Qwen2_5Config {
        vocab_size: 32,
        num_hidden_layers: 2,
        hidden_size: 16,
        intermediate_size: 32,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 4,
        context_length: 128,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        dtype: DType::F32,
    }
}

/// Generate deterministic-but-non-trivial weights from a seed via a
/// stateless PRNG-style trig formula. Not random, but indistinguishable
/// for the purposes of an end-to-end smoke test: each tensor entry is a
/// distinct value bounded in [-1, 1], and changing the seed changes
/// every value.
fn synth(seed: u32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (seed as f32 * 0.123) + (i as f32 * 0.0177);
            // small magnitude -> stable across the deep prefill chain;
            // large magnitude would saturate softmax and silu.
            0.05 * x.sin()
        })
        .collect()
}

fn tiny_weights(cfg: &Qwen2_5Config) -> Qwen2_5Weights {
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let q_out = cfg.num_attention_heads * cfg.head_dim;
    let kv_out = cfg.num_key_value_heads * cfg.head_dim;
    let i_size = cfg.intermediate_size;

    let embed = synth(1, v * h);
    // Tied embeddings: lm_head_w is the transpose of embed.
    let lm_head_w = transpose_2d(&embed, v, h);

    let layers = (0..cfg.num_hidden_layers)
        .map(|i| {
            let s = (i as u32) * 100 + 10;
            Qwen2_5LayerWeights {
                q_proj_w: synth(s, h * q_out),
                q_proj_b: synth(s + 1, q_out),
                k_proj_w: synth(s + 2, h * kv_out),
                k_proj_b: synth(s + 3, kv_out),
                v_proj_w: synth(s + 4, h * kv_out),
                v_proj_b: synth(s + 5, kv_out),
                o_proj_w: synth(s + 6, q_out * h),
                // RMSNorm weights initialized to 1.0 (the HF init for
                // these is all-ones; non-trivial values come from
                // training). Keeping them at 1.0 here also sidesteps
                // multiplying small-magnitude noise into every row of
                // the residual stream.
                input_layernorm_w: vec![1.0; h],
                post_attention_layernorm_w: vec![1.0; h],
                gate_proj_w: synth(s + 7, h * i_size),
                up_proj_w: synth(s + 8, h * i_size),
                down_proj_w: synth(s + 9, i_size * h),
            }
        })
        .collect();

    Qwen2_5Weights {
        embed_tokens: embed,
        layers,
        final_norm_w: vec![1.0; h],
        lm_head_w,
        tie_word_embeddings: true,
    }
}

fn loaded_tensor(name: impl Into<String>, shape: Vec<usize>, values: Vec<f32>) -> LoadedTensor {
    LoadedTensor {
        name: name.into(),
        shape,
        dtype: SupportedDtype::F32,
        values,
    }
}

fn tiny_loaded_tensors_from_weights(
    cfg: &Qwen2_5Config,
    weights: &Qwen2_5Weights,
) -> Vec<LoadedTensor> {
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let q_out = cfg.num_attention_heads * cfg.head_dim;
    let kv_out = cfg.num_key_value_heads * cfg.head_dim;
    let i_size = cfg.intermediate_size;

    let mut tensors = Vec::new();
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let prefix = format!("model.layers.{layer_idx}");
        tensors.push(loaded_tensor(
            format!("{prefix}.self_attn.q_proj.weight"),
            vec![q_out, h],
            transpose_2d(&layer.q_proj_w, h, q_out),
        ));
        tensors.push(loaded_tensor(
            format!("{prefix}.self_attn.q_proj.bias"),
            vec![q_out],
            layer.q_proj_b.clone(),
        ));
        tensors.push(loaded_tensor(
            format!("{prefix}.self_attn.k_proj.weight"),
            vec![kv_out, h],
            transpose_2d(&layer.k_proj_w, h, kv_out),
        ));
        tensors.push(loaded_tensor(
            format!("{prefix}.self_attn.k_proj.bias"),
            vec![kv_out],
            layer.k_proj_b.clone(),
        ));
        tensors.push(loaded_tensor(
            format!("{prefix}.self_attn.v_proj.weight"),
            vec![kv_out, h],
            transpose_2d(&layer.v_proj_w, h, kv_out),
        ));
        tensors.push(loaded_tensor(
            format!("{prefix}.self_attn.v_proj.bias"),
            vec![kv_out],
            layer.v_proj_b.clone(),
        ));
        tensors.push(loaded_tensor(
            format!("{prefix}.self_attn.o_proj.weight"),
            vec![h, q_out],
            transpose_2d(&layer.o_proj_w, q_out, h),
        ));
        tensors.push(loaded_tensor(
            format!("{prefix}.mlp.gate_proj.weight"),
            vec![i_size, h],
            transpose_2d(&layer.gate_proj_w, h, i_size),
        ));
        tensors.push(loaded_tensor(
            format!("{prefix}.mlp.up_proj.weight"),
            vec![i_size, h],
            transpose_2d(&layer.up_proj_w, h, i_size),
        ));
        tensors.push(loaded_tensor(
            format!("{prefix}.mlp.down_proj.weight"),
            vec![h, i_size],
            transpose_2d(&layer.down_proj_w, i_size, h),
        ));
        tensors.push(loaded_tensor(
            format!("{prefix}.input_layernorm.weight"),
            vec![h],
            layer.input_layernorm_w.clone(),
        ));
        tensors.push(loaded_tensor(
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![h],
            layer.post_attention_layernorm_w.clone(),
        ));
    }
    tensors.push(loaded_tensor(
        "model.embed_tokens.weight",
        vec![v, h],
        weights.embed_tokens.clone(),
    ));
    tensors.push(loaded_tensor(
        "model.norm.weight",
        vec![h],
        weights.final_norm_w.clone(),
    ));
    if !weights.tie_word_embeddings {
        tensors.push(loaded_tensor(
            "lm_head.weight",
            vec![v, h],
            transpose_2d(&weights.lm_head_w, h, v),
        ));
    }
    tensors
}

fn tmp_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("ocelotl_qwen2_5_{}_{}", std::process::id(), name));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).expect("create temp dir");
    path
}

fn write_qwen_config(path: &std::path::Path, cfg: &Qwen2_5Config, tie_word_embeddings: bool) {
    let dtype = match cfg.dtype {
        DType::F32 => "float32",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::Q4 | DType::Q8 => panic!("test config uses unsupported HF dtype"),
    };
    let raw = format!(
        r#"{{
          "model_type": "qwen2",
          "vocab_size": {vocab_size},
          "hidden_size": {hidden_size},
          "intermediate_size": {intermediate_size},
          "num_hidden_layers": {num_hidden_layers},
          "num_attention_heads": {num_attention_heads},
          "num_key_value_heads": {num_key_value_heads},
          "max_position_embeddings": {context_length},
          "rope_theta": {rope_theta},
          "rms_norm_eps": {rms_norm_eps},
          "torch_dtype": "{dtype}",
          "tie_word_embeddings": {tie_word_embeddings}
        }}"#,
        vocab_size = cfg.vocab_size,
        hidden_size = cfg.hidden_size,
        intermediate_size = cfg.intermediate_size,
        num_hidden_layers = cfg.num_hidden_layers,
        num_attention_heads = cfg.num_attention_heads,
        num_key_value_heads = cfg.num_key_value_heads,
        context_length = cfg.context_length,
        rope_theta = cfg.rope_theta,
        rms_norm_eps = cfg.rms_norm_eps,
    );
    fs::write(path, raw).expect("write config");
}

fn write_safetensors_f32(path: &std::path::Path, tensors: &[LoadedTensor]) {
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

#[test]
fn from_loaded_tensors_maps_hf_layouts_into_qwen_weights() {
    let cfg = tiny_config();
    let expected = tiny_weights(&cfg);
    let loaded = tiny_loaded_tensors_from_weights(&cfg, &expected);

    let actual = Qwen2_5Weights::from_loaded_tensors(&cfg, loaded, expected.tie_word_embeddings)
        .expect("loaded tensors must map into Qwen2.5 weights");

    assert_eq!(actual.embed_tokens, expected.embed_tokens);
    assert_eq!(actual.final_norm_w, expected.final_norm_w);
    assert_eq!(actual.lm_head_w, expected.lm_head_w);
    assert_eq!(actual.tie_word_embeddings, expected.tie_word_embeddings);
    assert_eq!(actual.layers.len(), expected.layers.len());
    assert_eq!(actual.layers[0].q_proj_w, expected.layers[0].q_proj_w);
    assert_eq!(actual.layers[0].k_proj_w, expected.layers[0].k_proj_w);
    assert_eq!(actual.layers[0].v_proj_w, expected.layers[0].v_proj_w);
    assert_eq!(actual.layers[0].o_proj_w, expected.layers[0].o_proj_w);
    assert_eq!(actual.layers[0].gate_proj_w, expected.layers[0].gate_proj_w);
    assert_eq!(actual.layers[0].up_proj_w, expected.layers[0].up_proj_w);
    assert_eq!(actual.layers[0].down_proj_w, expected.layers[0].down_proj_w);
}

#[test]
fn load_from_dir_builds_qwen_model_from_local_files_without_downloads() {
    let cfg = tiny_config();
    let weights = tiny_weights(&cfg);
    let dir = tmp_dir("load_from_dir");
    write_qwen_config(&dir.join("config.json"), &cfg, weights.tie_word_embeddings);
    write_safetensors_f32(
        &dir.join("model.safetensors"),
        &tiny_loaded_tensors_from_weights(&cfg, &weights),
    );

    let loaded = Qwen2_5Model::load_from_dir(&dir)
        .expect("local Qwen2.5 directory must load through family helper");
    let expected = Qwen2_5Model::new(cfg.clone(), weights).expect("expected model");

    assert_eq!(loaded.config(), &cfg);
    assert_eq!(
        loaded.prefill(&[TokenId(3), TokenId(7)]).unwrap(),
        expected.prefill(&[TokenId(3), TokenId(7)]).unwrap()
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn prefill_returns_one_logit_per_vocab_entry_for_single_token_prompt() {
    // Smallest meaningful prefill contract: returns a Vec<f32> of
    // length vocab_size, every entry finite. This exercises the full
    // composition pipeline (embedding, all transformer blocks, final
    // norm, lm_head) without pinning specific numbers yet.
    let cfg = tiny_config();
    let weights = tiny_weights(&cfg);
    let model = Qwen2_5Model::new(cfg.clone(), weights).expect("valid weights must construct");

    let logits = model
        .prefill(&[TokenId(7)])
        .expect("non-empty prompt with valid tokens must succeed");

    assert_eq!(logits.len(), cfg.vocab_size);
    for (i, v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit {i} must be finite, got {v}");
    }
}

#[test]
fn prefill_is_deterministic_for_identical_inputs() {
    let cfg = tiny_config();
    let weights = tiny_weights(&cfg);
    let model = Qwen2_5Model::new(cfg, weights).unwrap();

    let a = model.prefill(&[TokenId(3), TokenId(7)]).unwrap();
    let b = model.prefill(&[TokenId(3), TokenId(7)]).unwrap();

    assert_eq!(a, b, "identical inputs must yield bit-identical logits");
}

#[test]
fn optimized_cpu_backend_preserves_prefill_logits() {
    let cfg = tiny_config();
    let scalar = Qwen2_5Model::new(cfg.clone(), tiny_weights(&cfg)).unwrap();
    let optimized = Qwen2_5Model::with_kernel_backend(
        cfg.clone(),
        tiny_weights(&cfg),
        ocelotl_kernels::optimized_cpu_kernel_backend(),
    )
    .unwrap();

    assert_eq!(optimized.kernel_backend().name(), "cpu");
    let scalar_logits = scalar
        .prefill(&[TokenId(3), TokenId(7), TokenId(11)])
        .unwrap();
    let optimized_logits = optimized
        .prefill(&[TokenId(3), TokenId(7), TokenId(11)])
        .unwrap();

    for (idx, (got, want)) in optimized_logits
        .iter()
        .zip(scalar_logits.iter())
        .enumerate()
    {
        assert!(
            (got - want).abs() <= 1.0e-5,
            "optimized prefill logit {idx} drifted: got {got}, want {want}"
        );
    }
}

#[test]
fn new_uses_cpu_execution_backend_by_default() {
    let cfg = tiny_config();
    let weights = tiny_weights(&cfg);
    let model = Qwen2_5Model::new(cfg, weights).unwrap();

    assert_eq!(model.execution_backend().name(), "cpu");
    assert_eq!(
        model.execution_backend().context().device,
        ocelotl_core::Device::Cpu
    );
}

#[test]
fn model_accepts_non_cpu_kernel_backend_without_naming_concrete_backend() {
    let cfg = tiny_config();
    let model = Qwen2_5Model::with_kernel_backend(
        cfg.clone(),
        tiny_weights(&cfg),
        Arc::new(TestGpuKernelBackend::new()),
    )
    .unwrap();

    assert_eq!(model.execution_backend().name(), "test-gpu");
    assert_eq!(
        model.execution_backend().context().device,
        ocelotl_core::Device::Gpu { ordinal: 7 }
    );
    assert_eq!(model.kernel_backend().name(), "test-gpu");
}

#[test]
fn prefill_responds_to_different_prompt_tokens() {
    // Different prompt content must produce different last-position
    // logits. If this fails, prefill is short-circuiting.
    let cfg = tiny_config();
    let weights = tiny_weights(&cfg);
    let model = Qwen2_5Model::new(cfg, weights).unwrap();

    let a = model.prefill(&[TokenId(7)]).unwrap();
    let b = model.prefill(&[TokenId(8)]).unwrap();

    assert_ne!(a, b, "different prompts must yield different logits");
}

#[test]
fn prefill_rejects_empty_prompt_with_invalid_request() {
    let cfg = tiny_config();
    let weights = tiny_weights(&cfg);
    let model = Qwen2_5Model::new(cfg, weights).unwrap();

    let err = model
        .prefill(&[])
        .expect_err("empty prompt must be rejected");

    match err {
        OcelotlError::InvalidRequest(invalid) => {
            assert_eq!(invalid.field, "tokens");
            assert!(
                invalid.message.contains("at least one"),
                "expected message about minimum tokens, got {:?}",
                invalid.message,
            );
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn prefill_rejects_token_id_outside_vocab_with_invalid_request() {
    let cfg = tiny_config();
    let weights = tiny_weights(&cfg);
    let v = cfg.vocab_size;
    let model = Qwen2_5Model::new(cfg, weights).unwrap();

    // vocab_size is 32; TokenId(40) is out of range.
    let err = model
        .prefill(&[TokenId(v as u32)])
        .expect_err("out-of-range token id must be rejected");

    match err {
        OcelotlError::InvalidRequest(invalid) => {
            assert_eq!(invalid.field, "tokens");
            assert!(invalid.message.contains("out of range"));
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn prefill_rejects_prompt_longer_than_context_with_invalid_request() {
    let cfg = tiny_config();
    let weights = tiny_weights(&cfg);
    let ctx = cfg.context_length;
    let model = Qwen2_5Model::new(cfg, weights).unwrap();

    // ctx + 1 tokens > context_length.
    let prompt: Vec<TokenId> = (0..(ctx + 1) as u32).map(|t| TokenId(t % 32)).collect();
    let err = model
        .prefill(&prompt)
        .expect_err("prompt longer than context must be rejected");

    match err {
        OcelotlError::InvalidRequest(invalid) => {
            assert_eq!(invalid.field, "tokens");
            assert!(invalid.message.contains("context_length"));
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn new_rejects_layer_count_mismatch_with_invalid_model() {
    let cfg = tiny_config();
    let mut weights = tiny_weights(&cfg);
    weights.layers.pop();

    let err = Qwen2_5Model::new(cfg, weights)
        .expect_err("missing layer must be rejected at construction");

    match err {
        OcelotlError::InvalidModel(invalid) => {
            assert_eq!(invalid.field.as_deref(), Some("layers"));
        }
        other => panic!("expected InvalidModel(layers), got {other:?}"),
    }
}

#[test]
fn new_rejects_wrong_embed_tokens_length_with_invalid_model() {
    let cfg = tiny_config();
    let mut weights = tiny_weights(&cfg);
    weights.embed_tokens.pop();

    let err =
        Qwen2_5Model::new(cfg, weights).expect_err("wrong embed_tokens length must be rejected");

    match err {
        OcelotlError::InvalidModel(invalid) => {
            assert_eq!(invalid.field.as_deref(), Some("embed_tokens"));
        }
        other => panic!("expected InvalidModel, got {other:?}"),
    }
}

#[test]
fn new_revalidates_public_config_before_weight_length_math() {
    let mut cfg = tiny_config();
    cfg.rms_norm_eps = f64::NAN;
    let weights = tiny_weights(&tiny_config());

    let err = Qwen2_5Model::new(cfg, weights)
        .expect_err("directly constructed invalid config must be rejected");

    match err {
        OcelotlError::InvalidModel(invalid) => {
            assert_eq!(invalid.field.as_deref(), Some("rms_norm_eps"));
        }
        other => panic!("expected InvalidModel(rms_norm_eps), got {other:?}"),
    }
}

#[test]
fn new_rejects_config_shape_product_overflow() {
    let mut cfg = tiny_config();
    cfg.hidden_size = usize::MAX;
    cfg.num_attention_heads = 1;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = usize::MAX;
    let weights = Qwen2_5Weights {
        embed_tokens: Vec::new(),
        layers: Vec::new(),
        final_norm_w: Vec::new(),
        lm_head_w: Vec::new(),
        tie_word_embeddings: true,
    };

    let err = Qwen2_5Model::new(cfg, weights).expect_err("overflowing config must be rejected");

    match err {
        OcelotlError::InvalidModel(invalid) => {
            assert!(
                invalid.message.contains("overflows")
                    || invalid.field.as_deref() == Some("head_dim"),
                "expected overflow or head_dim diagnostic, got {:?}",
                invalid
            );
        }
        other => panic!("expected InvalidModel, got {other:?}"),
    }
}

#[test]
fn transpose_2d_round_trips() {
    // 2x3 matrix [[1,2,3],[4,5,6]] -> 3x2 [[1,4],[2,5],[3,6]] -> back.
    let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = transpose_2d(&a, 2, 3);
    assert_eq!(b, vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let c = transpose_2d(&b, 3, 2);
    assert_eq!(a, c);
}
