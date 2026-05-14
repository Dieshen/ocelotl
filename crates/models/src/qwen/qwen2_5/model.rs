//! Qwen2.5 model struct, constructors, and cache-layout validation.
//!
//! Forward composition for `prefill` and `decode_token_with_cache` lives in
//! sibling files (`prefill.rs`, `decode.rs`) as separate `impl Qwen2_5Model`
//! blocks. Helpers shared between prefill and decode (`add_bias_per_row`,
//! `validate_cache_layout_for_model`) are declared here with `pub(super)`
//! visibility so the siblings can reuse them without going through a third
//! module.

use std::{path::Path, sync::Arc};

use ocelotl_core::{DType, InvalidRequestError, KvCacheLayout, OcelotlError, Result};
use ocelotl_kernels::{KernelBackend, default_kernel_backend};
use ocelotl_loader::{inspect_safetensors, load_safetensors_tensors_f32, parse_hf_config};

use super::{
    checked_len_product,
    config::Qwen2_5Config,
    invalid_model,
    tensors::{required_tensor_names, validate_qwen2_5_tensors},
    validate_config_for_model,
    weights::Qwen2_5Weights,
};

/// Validated Qwen2.5 model: configuration plus weights.
#[derive(Debug)]
pub struct Qwen2_5Model {
    pub(super) config: Qwen2_5Config,
    pub(super) weights: Qwen2_5Weights,
    pub(super) kernels: Arc<dyn KernelBackend>,
}

impl Qwen2_5Model {
    /// Load a local HF-style Qwen2.5 artifact directory.
    ///
    /// This is intentionally local-only: it reads `config.json` and
    /// `model.safetensors` from `dir`, delegates file parsing to
    /// `ocelotl-loader`, and never downloads artifacts.
    pub fn load_from_dir<P: AsRef<Path>>(dir: P) -> Result<Self> {
        Self::load_from_dir_with_kernel_backend(dir, default_kernel_backend())
    }

    /// Load a local HF-style Qwen2.5 artifact directory with explicit kernel
    /// backend selection.
    pub fn load_from_dir_with_kernel_backend<P: AsRef<Path>>(
        dir: P,
        kernels: Arc<dyn KernelBackend>,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        Self::load_from_paths_with_kernel_backend(
            dir.join("config.json"),
            dir.join("model.safetensors"),
            kernels,
        )
    }

    /// Load a local HF-style Qwen2.5 config/model pair from explicit paths.
    pub fn load_from_paths<P: AsRef<Path>, Q: AsRef<Path>>(
        config_path: P,
        model_path: Q,
    ) -> Result<Self> {
        Self::load_from_paths_with_kernel_backend(config_path, model_path, default_kernel_backend())
    }

    /// Load a local HF-style Qwen2.5 config/model pair from explicit paths
    /// with backend selection.
    pub fn load_from_paths_with_kernel_backend<P: AsRef<Path>, Q: AsRef<Path>>(
        config_path: P,
        model_path: Q,
        kernels: Arc<dyn KernelBackend>,
    ) -> Result<Self> {
        let config_path = config_path.as_ref();
        let model_path = model_path.as_ref();

        let info = parse_hf_config(config_path)?;
        let config = Qwen2_5Config::try_from(&info.metadata)?;
        let manifest = inspect_safetensors(model_path)?;
        validate_qwen2_5_tensors(
            &manifest,
            &config,
            info.tie_word_embeddings,
            Some(model_path),
        )?;

        let names = required_tensor_names(&config, info.tie_word_embeddings);
        let tensors = load_safetensors_tensors_f32(model_path, &names)?;
        let weights =
            Qwen2_5Weights::from_loaded_tensors(&config, tensors, info.tie_word_embeddings)?;
        Self::with_kernel_backend(config, weights, kernels)
    }

    /// Construct a model from validated config and weights.
    ///
    /// Returns `OcelotlError::InvalidModel` if any weight tensor's length
    /// disagrees with the shape implied by `config`. This is the last gate
    /// before the kernel boundary; downstream forward code can assume
    /// every length is correct.
    pub fn new(config: Qwen2_5Config, weights: Qwen2_5Weights) -> Result<Self> {
        Self::with_kernel_backend(config, weights, default_kernel_backend())
    }

    /// Construct a model with an explicit kernel backend.
    pub fn with_kernel_backend(
        config: Qwen2_5Config,
        weights: Qwen2_5Weights,
        kernels: Arc<dyn KernelBackend>,
    ) -> Result<Self> {
        validate_config_for_model(&config)?;
        let h = config.hidden_size;
        let v = config.vocab_size;
        let q_out = checked_len_product(
            "num_attention_heads*head_dim",
            &[config.num_attention_heads, config.head_dim],
        )?;
        let kv_out = checked_len_product(
            "num_key_value_heads*head_dim",
            &[config.num_key_value_heads, config.head_dim],
        )?;
        let i_size = config.intermediate_size;

        let embed_len = checked_len_product("vocab_size*hidden_size", &[v, h])?;
        let lm_head_len = checked_len_product("hidden_size*vocab_size", &[h, v])?;
        let q_proj_len = checked_len_product("hidden_size*q_out", &[h, q_out])?;
        let kv_proj_len = checked_len_product("hidden_size*kv_out", &[h, kv_out])?;
        let o_proj_len = checked_len_product("q_out*hidden_size", &[q_out, h])?;
        let mlp_in_len = checked_len_product("hidden_size*intermediate_size", &[h, i_size])?;
        let mlp_down_len = checked_len_product("intermediate_size*hidden_size", &[i_size, h])?;

        check_len("embed_tokens", weights.embed_tokens.len(), embed_len)?;
        check_len("final_norm_w", weights.final_norm_w.len(), h)?;
        check_len("lm_head_w", weights.lm_head_w.len(), lm_head_len)?;

        if weights.layers.len() != config.num_hidden_layers {
            return Err(invalid_model(
                "layers",
                &format!(
                    "expected {} layer weight bundles, got {}",
                    config.num_hidden_layers,
                    weights.layers.len(),
                ),
            ));
        }

        for (i, layer) in weights.layers.iter().enumerate() {
            check_len(
                &format!("layers[{i}].q_proj_w"),
                layer.q_proj_w.len(),
                q_proj_len,
            )?;
            check_len(
                &format!("layers[{i}].q_proj_b"),
                layer.q_proj_b.len(),
                q_out,
            )?;
            check_len(
                &format!("layers[{i}].k_proj_w"),
                layer.k_proj_w.len(),
                kv_proj_len,
            )?;
            check_len(
                &format!("layers[{i}].k_proj_b"),
                layer.k_proj_b.len(),
                kv_out,
            )?;
            check_len(
                &format!("layers[{i}].v_proj_w"),
                layer.v_proj_w.len(),
                kv_proj_len,
            )?;
            check_len(
                &format!("layers[{i}].v_proj_b"),
                layer.v_proj_b.len(),
                kv_out,
            )?;
            check_len(
                &format!("layers[{i}].o_proj_w"),
                layer.o_proj_w.len(),
                o_proj_len,
            )?;
            check_len(
                &format!("layers[{i}].input_layernorm_w"),
                layer.input_layernorm_w.len(),
                h,
            )?;
            check_len(
                &format!("layers[{i}].post_attention_layernorm_w"),
                layer.post_attention_layernorm_w.len(),
                h,
            )?;
            check_len(
                &format!("layers[{i}].gate_proj_w"),
                layer.gate_proj_w.len(),
                mlp_in_len,
            )?;
            check_len(
                &format!("layers[{i}].up_proj_w"),
                layer.up_proj_w.len(),
                mlp_in_len,
            )?;
            check_len(
                &format!("layers[{i}].down_proj_w"),
                layer.down_proj_w.len(),
                mlp_down_len,
            )?;
        }

        Ok(Self {
            config,
            weights,
            kernels,
        })
    }

    /// Borrow the validated configuration.
    pub fn config(&self) -> &Qwen2_5Config {
        &self.config
    }

    /// Borrow the kernel backend selected for this model instance.
    pub fn kernel_backend(&self) -> &dyn KernelBackend {
        self.kernels.as_ref()
    }

    /// Borrow the active execution backend.
    pub fn execution_backend(&self) -> &dyn KernelBackend {
        self.kernels.as_ref()
    }
}

/// Add a per-feature bias vector to every row of a `[rows, cols]` matrix
/// stored contiguous row-major. Inlined rather than introducing a new
/// kernel: it's a single trivial loop and the only callers are projection
/// paths inside this family.
pub(super) fn add_bias_per_row(x: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
    debug_assert_eq!(x.len(), rows * cols);
    debug_assert_eq!(bias.len(), cols);
    for r in 0..rows {
        let row_start = r * cols;
        for c in 0..cols {
            x[row_start + c] += bias[c];
        }
    }
}

fn check_len(name: &str, got: usize, expected: usize) -> Result<()> {
    if got == expected {
        Ok(())
    } else {
        Err(invalid_model(
            name,
            &format!("expected length {expected}, got {got}"),
        ))
    }
}

pub(super) fn validate_cache_layout_for_model(
    config: &Qwen2_5Config,
    layout: &KvCacheLayout,
    required_tokens: usize,
) -> Result<()> {
    if layout.dtype != DType::F32 {
        return Err(OcelotlError::InvalidRequest(InvalidRequestError {
            field: "kv_cache.dtype".to_string(),
            message: format!(
                "Qwen2.5 cache currently requires F32, got {:?}",
                layout.dtype
            ),
        }));
    }
    if layout.num_layers != config.num_hidden_layers {
        return Err(OcelotlError::InvalidRequest(InvalidRequestError {
            field: "kv_cache.num_layers".to_string(),
            message: format!(
                "expected {}, got {}",
                config.num_hidden_layers, layout.num_layers
            ),
        }));
    }
    if layout.num_key_value_heads != config.num_key_value_heads {
        return Err(OcelotlError::InvalidRequest(InvalidRequestError {
            field: "kv_cache.num_key_value_heads".to_string(),
            message: format!(
                "expected {}, got {}",
                config.num_key_value_heads, layout.num_key_value_heads
            ),
        }));
    }
    if layout.head_dim != config.head_dim {
        return Err(OcelotlError::InvalidRequest(InvalidRequestError {
            field: "kv_cache.head_dim".to_string(),
            message: format!("expected {}, got {}", config.head_dim, layout.head_dim),
        }));
    }
    if required_tokens > layout.capacity_tokens {
        return Err(OcelotlError::InvalidRequest(InvalidRequestError {
            field: "kv_cache.capacity".to_string(),
            message: format!(
                "required {required_tokens} tokens exceeds cache capacity {}",
                layout.capacity_tokens
            ),
        }));
    }
    Ok(())
}
