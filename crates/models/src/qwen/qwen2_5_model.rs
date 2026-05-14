//! Qwen2.5 prefill forward path.
//!
//! `Qwen2_5Model` owns the weights for a Qwen2.5-style dense decoder-only
//! transformer and exposes a `prefill` method that returns the final-token
//! logits over the vocabulary.
//!
//! # Boundary discipline
//!
//! - Composes `ocelotl-kernels` (rmsnorm, rope, attention, mlp, matmul);
//!   does **not** redefine kernels here. Per the M3.7 brief's crate-scope
//!   rule.
//! - Consumes the validated `Qwen2_5Config` from `qwen2_5.rs` (M3.1) so
//!   every shape invariant the kernels rely on (head_dim *
//!   num_attention_heads == hidden_size, GQA divisibility, positive
//!   dimensions) is established at construction time, not at the kernel
//!   boundary.
//! - Tied embeddings (`tie_word_embeddings: true`, the Qwen2.5-0.5B-Instruct
//!   default) are handled by storing the embedding table once and using
//!   the same buffer as the lm_head. The flag is carried by
//!   `ocelotl-loader::HfModelInfo`, not by `ocelotl-core::ModelMetadata`.
//!
//! # Layout & dtype
//!
//! Reference path is `f32`-only. The HF safetensors file ships in `bf16`;
//! converting to `f32` at load time keeps the kernel boundary uniform with
//! the rest of M1/M3 (per M1.7's "f32 contiguous row-major only" rule).
//! Multi-dtype support is M4+/M5 work and is out of scope here.
//!
//! # Weight layouts (HF row-major, [out_features, in_features])
//!
//! Tensor names and shapes are pinned in `qwen2_5_tensors::validate_qwen2_5_tensors`
//! and consumed here directly:
//!
//! - `embed_tokens.weight`: `[vocab_size, hidden_size]`
//! - `model.layers.{i}.self_attn.q_proj.weight`: `[num_q_heads * head_dim, hidden_size]`
//! - `model.layers.{i}.self_attn.q_proj.bias`:   `[num_q_heads * head_dim]`
//! - `model.layers.{i}.self_attn.k_proj.weight`: `[num_kv_heads * head_dim, hidden_size]`
//! - `model.layers.{i}.self_attn.k_proj.bias`:   `[num_kv_heads * head_dim]`
//! - `model.layers.{i}.self_attn.v_proj.weight`: `[num_kv_heads * head_dim, hidden_size]`
//! - `model.layers.{i}.self_attn.v_proj.bias`:   `[num_kv_heads * head_dim]`
//! - `model.layers.{i}.self_attn.o_proj.weight`: `[hidden_size, num_q_heads * head_dim]`
//! - `model.layers.{i}.input_layernorm.weight`:  `[hidden_size]`
//! - `model.layers.{i}.post_attention_layernorm.weight`: `[hidden_size]`
//! - `model.layers.{i}.mlp.gate_proj.weight`: `[intermediate_size, hidden_size]`
//! - `model.layers.{i}.mlp.up_proj.weight`:   `[intermediate_size, hidden_size]`
//! - `model.layers.{i}.mlp.down_proj.weight`: `[hidden_size, intermediate_size]`
//! - `model.norm.weight`: `[hidden_size]`
//! - `lm_head.weight` (only when `!tie_word_embeddings`): `[vocab_size, hidden_size]`
//!
//! # Why a Vec<f32> per tensor and not a TensorView
//!
//! The kernel boundary is contiguous-row-major `&[f32]`. Tensor objects
//! would force a conversion at every kernel call. M3 stays under the
//! M1.7 / M3.3-3.6 boundary contract; M4+ is when a typed tensor view
//! becomes worth its overhead.

use std::{
    collections::{BTreeMap, btree_map::Entry},
    path::Path,
    sync::Arc,
};

use ocelotl_core::{
    DType, InvalidModelError, InvalidRequestError, KvCacheLayout, KvCacheStore, OcelotlError,
    Result, TokenId,
};
use ocelotl_kernels::{KernelBackend, default_kernel_backend};
use ocelotl_loader::{
    LoadedTensor, SupportedDtype, inspect_safetensors, load_safetensors_tensors_f32,
    parse_hf_config,
};

use super::{Qwen2_5Config, required_tensor_names, validate_qwen2_5_tensors};

/// Per-layer weight bundle. All tensors stored as contiguous row-major
/// `Vec<f32>` (HF layout). The model layer transposes weights as needed
/// at the kernel boundary; for matmul-shaped weights the convention is the
/// HF native `[out, in]` shape, used as the `b` argument of an
/// `[seq, in] @ [in, out]` matmul -- which means each kernel call
/// transposes by walking the storage in column-major order. To avoid
/// that hot-path transpose we **pre-transpose** the linear weights into
/// `[in, out]` at load time. See `Qwen2_5Weights::from_hf`.
#[derive(Debug, Clone)]
pub struct Qwen2_5LayerWeights {
    /// `[hidden_size, num_q_heads * head_dim]` after pre-transpose.
    pub q_proj_w: Vec<f32>,
    /// `[num_q_heads * head_dim]`.
    pub q_proj_b: Vec<f32>,
    /// `[hidden_size, num_kv_heads * head_dim]` after pre-transpose.
    pub k_proj_w: Vec<f32>,
    /// `[num_kv_heads * head_dim]`.
    pub k_proj_b: Vec<f32>,
    /// `[hidden_size, num_kv_heads * head_dim]` after pre-transpose.
    pub v_proj_w: Vec<f32>,
    /// `[num_kv_heads * head_dim]`.
    pub v_proj_b: Vec<f32>,
    /// `[num_q_heads * head_dim, hidden_size]` after pre-transpose
    /// (input is `[seq, num_q_heads * head_dim]`, output is `[seq, hidden]`).
    pub o_proj_w: Vec<f32>,
    /// `[hidden_size]`.
    pub input_layernorm_w: Vec<f32>,
    /// `[hidden_size]`.
    pub post_attention_layernorm_w: Vec<f32>,
    /// `[hidden_size, intermediate_size]` after pre-transpose. Matches
    /// `mlp_gated_silu`'s `gate_w` argument layout.
    pub gate_proj_w: Vec<f32>,
    /// `[hidden_size, intermediate_size]` after pre-transpose. Matches
    /// `mlp_gated_silu`'s `up_w` argument layout.
    pub up_proj_w: Vec<f32>,
    /// `[intermediate_size, hidden_size]` after pre-transpose. Matches
    /// `mlp_gated_silu`'s `down_w` argument layout.
    pub down_proj_w: Vec<f32>,
}

/// Top-level model weight bundle.
#[derive(Debug, Clone)]
pub struct Qwen2_5Weights {
    /// `[vocab_size, hidden_size]`. Used as the embedding table directly
    /// (gather rows by token id) and, when `tie_word_embeddings`, as the
    /// lm_head projection weight in its pre-transposed `[hidden_size,
    /// vocab_size]` form (see `lm_head_w`).
    pub embed_tokens: Vec<f32>,
    pub layers: Vec<Qwen2_5LayerWeights>,
    /// `[hidden_size]`.
    pub final_norm_w: Vec<f32>,
    /// `[hidden_size, vocab_size]` after pre-transpose. When the source
    /// model has `tie_word_embeddings: true`, this is built by transposing
    /// `embed_tokens`. Otherwise it is the loaded `lm_head.weight`
    /// pre-transposed.
    pub lm_head_w: Vec<f32>,
    pub tie_word_embeddings: bool,
}

impl Qwen2_5Weights {
    /// Build model-family weights from loader-owned safetensors tensor values.
    ///
    /// This is the model-specific artifact adaptation boundary: `ocelotl-loader`
    /// parses and validates the file format, while this method maps Qwen2.5 HF
    /// tensor names into the layout expected by the Qwen forward path.
    pub fn from_loaded_tensors(
        config: &Qwen2_5Config,
        tensors: Vec<LoadedTensor>,
        tie_word_embeddings: bool,
    ) -> Result<Self> {
        validate_config_for_model(config)?;

        let mut by_name = BTreeMap::new();
        for tensor in tensors {
            match by_name.entry(tensor.name.clone()) {
                Entry::Vacant(entry) => {
                    entry.insert(tensor);
                }
                Entry::Occupied(_) => {
                    return Err(invalid_model(
                        &tensor.name,
                        "duplicate Qwen2.5 tensor supplied",
                    ));
                }
            }
        }

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

        let embed_tokens = take_tensor_values(
            &mut by_name,
            "model.embed_tokens.weight",
            &[v, h],
            &config.dtype,
        )?;
        let final_norm_w =
            take_tensor_values(&mut by_name, "model.norm.weight", &[h], &config.dtype)?;
        let lm_head_w = if tie_word_embeddings {
            transpose_2d(&embed_tokens, v, h)
        } else {
            let lm_head =
                take_tensor_values(&mut by_name, "lm_head.weight", &[v, h], &config.dtype)?;
            transpose_2d(&lm_head, v, h)
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let layer_prefix = format!("model.layers.{layer}");
            let tensor_name = |suffix: &str| format!("{layer_prefix}.{suffix}");

            let q_proj_w = take_tensor_values(
                &mut by_name,
                &tensor_name("self_attn.q_proj.weight"),
                &[q_out, h],
                &config.dtype,
            )?;
            let k_proj_w = take_tensor_values(
                &mut by_name,
                &tensor_name("self_attn.k_proj.weight"),
                &[kv_out, h],
                &config.dtype,
            )?;
            let v_proj_w = take_tensor_values(
                &mut by_name,
                &tensor_name("self_attn.v_proj.weight"),
                &[kv_out, h],
                &config.dtype,
            )?;
            let o_proj_w = take_tensor_values(
                &mut by_name,
                &tensor_name("self_attn.o_proj.weight"),
                &[h, q_out],
                &config.dtype,
            )?;
            let gate_proj_w = take_tensor_values(
                &mut by_name,
                &tensor_name("mlp.gate_proj.weight"),
                &[i_size, h],
                &config.dtype,
            )?;
            let up_proj_w = take_tensor_values(
                &mut by_name,
                &tensor_name("mlp.up_proj.weight"),
                &[i_size, h],
                &config.dtype,
            )?;
            let down_proj_w = take_tensor_values(
                &mut by_name,
                &tensor_name("mlp.down_proj.weight"),
                &[h, i_size],
                &config.dtype,
            )?;

            layers.push(Qwen2_5LayerWeights {
                q_proj_w: transpose_2d(&q_proj_w, q_out, h),
                q_proj_b: take_tensor_values(
                    &mut by_name,
                    &tensor_name("self_attn.q_proj.bias"),
                    &[q_out],
                    &config.dtype,
                )?,
                k_proj_w: transpose_2d(&k_proj_w, kv_out, h),
                k_proj_b: take_tensor_values(
                    &mut by_name,
                    &tensor_name("self_attn.k_proj.bias"),
                    &[kv_out],
                    &config.dtype,
                )?,
                v_proj_w: transpose_2d(&v_proj_w, kv_out, h),
                v_proj_b: take_tensor_values(
                    &mut by_name,
                    &tensor_name("self_attn.v_proj.bias"),
                    &[kv_out],
                    &config.dtype,
                )?,
                o_proj_w: transpose_2d(&o_proj_w, h, q_out),
                input_layernorm_w: take_tensor_values(
                    &mut by_name,
                    &tensor_name("input_layernorm.weight"),
                    &[h],
                    &config.dtype,
                )?,
                post_attention_layernorm_w: take_tensor_values(
                    &mut by_name,
                    &tensor_name("post_attention_layernorm.weight"),
                    &[h],
                    &config.dtype,
                )?,
                gate_proj_w: transpose_2d(&gate_proj_w, i_size, h),
                up_proj_w: transpose_2d(&up_proj_w, i_size, h),
                down_proj_w: transpose_2d(&down_proj_w, h, i_size),
            });
        }

        Ok(Self {
            embed_tokens,
            layers,
            final_norm_w,
            lm_head_w,
            tie_word_embeddings,
        })
    }
}

/// Validated Qwen2.5 model: configuration plus weights.
#[derive(Debug)]
pub struct Qwen2_5Model {
    config: Qwen2_5Config,
    weights: Qwen2_5Weights,
    kernels: Arc<dyn KernelBackend>,
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

    /// Run prefill over the prompt and return the logits at the final
    /// position over the vocabulary.
    ///
    /// Composes (per layer):
    /// `RMSNorm -> q/k/v projections + biases -> RoPE on Q and K ->
    ///  scaled-dot-product attention -> o_proj -> residual add ->
    ///  RMSNorm -> gated SiLU MLP -> residual add`.
    /// Then a final RMSNorm and the lm_head projection on the last
    /// position only.
    ///
    /// Returns a `Vec<f32>` of length `vocab_size`.
    ///
    /// # Errors
    ///
    /// - `InvalidRequest` if `tokens` is empty.
    /// - `InvalidRequest` if any token id is `>= vocab_size`.
    /// - `InvalidRequest` if `tokens.len() > config.context_length`.
    /// - Propagates `KernelError` from the underlying kernels for
    ///   shape/length violations (these should be unreachable when the
    ///   `Qwen2_5Model::new` length checks have run).
    pub fn prefill(&self, tokens: &[TokenId]) -> Result<Vec<f32>> {
        self.prefill_impl(tokens, None)
    }

    /// Run prefill and write each layer's post-RoPE K/V tensors into a
    /// caller-owned cache.
    ///
    /// The cache is runtime-owned but model-written. Its layout must match the
    /// Qwen2.5 config exactly. Cache length is updated only after the full
    /// prefill succeeds, so capacity errors cannot leave a partially advanced
    /// request state.
    pub fn prefill_with_cache(
        &self,
        tokens: &[TokenId],
        cache: &mut dyn KvCacheStore,
    ) -> Result<Vec<f32>> {
        validate_cache_layout_for_model(&self.config, cache.layout(), tokens.len())?;
        cache.set_len_tokens(0)?;
        self.prefill_impl(tokens, Some(cache))
    }

    fn prefill_impl(
        &self,
        tokens: &[TokenId],
        mut cache: Option<&mut dyn KvCacheStore>,
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(OcelotlError::InvalidRequest(InvalidRequestError {
                field: "tokens".to_string(),
                message: "Qwen2_5Model::prefill requires at least one token".to_string(),
            }));
        }
        let cfg = &self.config;
        if tokens.len() > cfg.context_length {
            return Err(OcelotlError::InvalidRequest(InvalidRequestError {
                field: "tokens".to_string(),
                message: format!(
                    "prompt length {} exceeds context_length {}",
                    tokens.len(),
                    cfg.context_length,
                ),
            }));
        }
        for (idx, t) in tokens.iter().enumerate() {
            if (t.0 as usize) >= cfg.vocab_size {
                return Err(OcelotlError::InvalidRequest(InvalidRequestError {
                    field: "tokens".to_string(),
                    message: format!(
                        "token id {} at position {} is out of range for vocab_size {}",
                        t.0, idx, cfg.vocab_size,
                    ),
                }));
            }
        }

        let seq = tokens.len();
        let h = cfg.hidden_size;
        let q_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        let i_size = cfg.intermediate_size;
        let v = cfg.vocab_size;
        let eps = cfg.rms_norm_eps as f32;
        let theta = cfg.rope_theta as f32;

        // Step 1: token embedding lookup.
        // hidden has shape [seq, hidden_size]. embed_tokens is laid out as
        // [vocab, hidden] row-major; row `t` is the embedding for token id t.
        // Inlined rather than introducing a kernel: it's a copy of one row
        // per token, no math, no shape decisions.
        let mut hidden = vec![0.0_f32; seq * h];
        for (t_idx, tok) in tokens.iter().enumerate() {
            let src = (tok.0 as usize) * h;
            let dst = t_idx * h;
            hidden[dst..dst + h].copy_from_slice(&self.weights.embed_tokens[src..src + h]);
        }

        // Scratch buffers reused across layers/projections.
        let mut norm_buf = vec![0.0_f32; seq * h];
        let mut q_buf = vec![0.0_f32; seq * q_out];
        let mut k_buf = vec![0.0_f32; seq * kv_out];
        let mut v_buf = vec![0.0_f32; seq * kv_out];
        let mut attn_out = vec![0.0_f32; seq * q_out];
        let mut o_buf = vec![0.0_f32; seq * h];
        let mut residual_buf = vec![0.0_f32; seq * h];
        let mut gate_buf = vec![0.0_f32; seq * i_size];
        let mut up_buf = vec![0.0_f32; seq * i_size];
        let mut mlp_out = vec![0.0_f32; seq * h];

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            // Save residual before attention.
            residual_buf.copy_from_slice(&hidden);

            // Pre-attention RMSNorm.
            self.kernels.rmsnorm(
                &hidden,
                seq,
                h,
                &layer.input_layernorm_w,
                eps,
                &mut norm_buf,
            )?;

            // Q = norm @ q_proj_w  ([seq,h] @ [h,q_out])
            self.kernels
                .matmul(&norm_buf, (seq, h), &layer.q_proj_w, (h, q_out), &mut q_buf)?;
            add_bias_per_row(&mut q_buf, &layer.q_proj_b, seq, q_out);

            // K = norm @ k_proj_w  ([seq,h] @ [h,kv_out])
            self.kernels.matmul(
                &norm_buf,
                (seq, h),
                &layer.k_proj_w,
                (h, kv_out),
                &mut k_buf,
            )?;
            add_bias_per_row(&mut k_buf, &layer.k_proj_b, seq, kv_out);

            // V = norm @ v_proj_w  ([seq,h] @ [h,kv_out])
            self.kernels.matmul(
                &norm_buf,
                (seq, h),
                &layer.v_proj_w,
                (h, kv_out),
                &mut v_buf,
            )?;
            add_bias_per_row(&mut v_buf, &layer.v_proj_b, seq, kv_out);

            // Apply RoPE to Q and K, per-position. Both have layout
            // [seq, num_heads, head_dim] flattened row-major; rope expects
            // a slice of `num_heads * head_dim` per position.
            for pos in 0..seq {
                let q_start = pos * q_out;
                self.kernels.rope_apply_inplace(
                    &mut q_buf[q_start..q_start + q_out],
                    cfg.head_dim,
                    pos,
                    theta,
                )?;
                let k_start = pos * kv_out;
                self.kernels.rope_apply_inplace(
                    &mut k_buf[k_start..k_start + kv_out],
                    cfg.head_dim,
                    pos,
                    theta,
                )?;
            }
            if let Some(cache) = &mut cache {
                for pos in 0..seq {
                    let start = pos * kv_out;
                    cache.write_layer_kv(
                        layer_idx,
                        pos,
                        &k_buf[start..start + kv_out],
                        &v_buf[start..start + kv_out],
                    )?;
                }
            }

            // Scaled-dot-product attention.
            self.kernels.scaled_dot_product_attention(
                &q_buf,
                &k_buf,
                &v_buf,
                seq,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                &mut attn_out,
            )?;

            // O = attn_out @ o_proj_w  ([seq,q_out] @ [q_out,h])
            self.kernels.matmul(
                &attn_out,
                (seq, q_out),
                &layer.o_proj_w,
                (q_out, h),
                &mut o_buf,
            )?;

            // Residual: hidden = residual + o_buf.
            self.kernels.vec_add(&residual_buf, &o_buf, &mut hidden)?;

            // Save residual before MLP.
            residual_buf.copy_from_slice(&hidden);

            // Post-attention RMSNorm.
            self.kernels.rmsnorm(
                &hidden,
                seq,
                h,
                &layer.post_attention_layernorm_w,
                eps,
                &mut norm_buf,
            )?;

            // MLP: out = down(silu(gate(x)) * up(x)).
            self.kernels.mlp_gated_silu(
                &norm_buf,
                seq,
                h,
                i_size,
                &layer.gate_proj_w,
                &layer.up_proj_w,
                &layer.down_proj_w,
                &mut gate_buf,
                &mut up_buf,
                &mut mlp_out,
            )?;

            // Residual: hidden = residual + mlp_out.
            self.kernels.vec_add(&residual_buf, &mlp_out, &mut hidden)?;
        }

        // Final RMSNorm over the full sequence.
        self.kernels.rmsnorm(
            &hidden,
            seq,
            h,
            &self.weights.final_norm_w,
            eps,
            &mut norm_buf,
        )?;

        // lm_head over the last position only. norm_buf last row is
        // [h] -> a `[1, h] @ [h, v]` matmul into a `[1, v]` output.
        let last_start = (seq - 1) * h;
        let last_row = &norm_buf[last_start..last_start + h];
        let mut logits = vec![0.0_f32; v];
        self.kernels.matmul(
            last_row,
            (1, h),
            &self.weights.lm_head_w,
            (h, v),
            &mut logits,
        )?;

        if let Some(cache) = cache {
            cache.set_len_tokens(seq)?;
        }

        Ok(logits)
    }

    /// Decode a single already-selected token against an existing KV cache and
    /// append that token's K/V at the next cache position.
    ///
    /// This returns logits for the *following* token. The public runtime
    /// cached-decode helper samples from the prior prefill/decode logits, then
    /// calls this method to advance cache state for future decode steps.
    pub fn decode_token_with_cache(
        &self,
        token: TokenId,
        cache: &mut dyn KvCacheStore,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let position = cache.len_tokens();
        let next_len = position.checked_add(1).ok_or_else(|| {
            OcelotlError::InvalidRequest(InvalidRequestError {
                field: "kv_cache.capacity".to_string(),
                message: "cache position overflows usize".to_string(),
            })
        })?;
        validate_cache_layout_for_model(cfg, cache.layout(), next_len)?;
        if (token.0 as usize) >= cfg.vocab_size {
            return Err(OcelotlError::InvalidRequest(InvalidRequestError {
                field: "token".to_string(),
                message: format!(
                    "token id {} is out of range for vocab_size {}",
                    token.0, cfg.vocab_size
                ),
            }));
        }

        let h = cfg.hidden_size;
        let q_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        let i_size = cfg.intermediate_size;
        let v = cfg.vocab_size;
        let eps = cfg.rms_norm_eps as f32;
        let theta = cfg.rope_theta as f32;

        let src = (token.0 as usize) * h;
        let mut hidden = self.weights.embed_tokens[src..src + h].to_vec();

        let mut norm_buf = vec![0.0_f32; h];
        let mut q_buf = vec![0.0_f32; q_out];
        let mut k_buf = vec![0.0_f32; kv_out];
        let mut v_buf = vec![0.0_f32; kv_out];
        let mut k_cache = vec![0.0_f32; next_len * kv_out];
        let mut v_cache = vec![0.0_f32; next_len * kv_out];
        let mut q_context = vec![0.0_f32; next_len * q_out];
        let mut attn_context = vec![0.0_f32; next_len * q_out];
        let mut o_buf = vec![0.0_f32; h];
        let mut residual_buf = vec![0.0_f32; h];
        let mut gate_buf = vec![0.0_f32; i_size];
        let mut up_buf = vec![0.0_f32; i_size];
        let mut mlp_out = vec![0.0_f32; h];

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            residual_buf.copy_from_slice(&hidden);
            self.kernels
                .rmsnorm(&hidden, 1, h, &layer.input_layernorm_w, eps, &mut norm_buf)?;

            self.kernels
                .matmul(&norm_buf, (1, h), &layer.q_proj_w, (h, q_out), &mut q_buf)?;
            add_bias_per_row(&mut q_buf, &layer.q_proj_b, 1, q_out);
            self.kernels
                .matmul(&norm_buf, (1, h), &layer.k_proj_w, (h, kv_out), &mut k_buf)?;
            add_bias_per_row(&mut k_buf, &layer.k_proj_b, 1, kv_out);
            self.kernels
                .matmul(&norm_buf, (1, h), &layer.v_proj_w, (h, kv_out), &mut v_buf)?;
            add_bias_per_row(&mut v_buf, &layer.v_proj_b, 1, kv_out);

            self.kernels
                .rope_apply_inplace(&mut q_buf, cfg.head_dim, position, theta)?;
            self.kernels
                .rope_apply_inplace(&mut k_buf, cfg.head_dim, position, theta)?;

            cache.write_layer_kv(layer_idx, position, &k_buf, &v_buf)?;
            cache.read_layer_keys(layer_idx, next_len, &mut k_cache)?;
            cache.read_layer_values(layer_idx, next_len, &mut v_cache)?;

            q_context.fill(0.0);
            let q_start = position * q_out;
            q_context[q_start..q_start + q_out].copy_from_slice(&q_buf);
            attn_context.fill(0.0);
            self.kernels.scaled_dot_product_attention(
                &q_context,
                &k_cache,
                &v_cache,
                next_len,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                &mut attn_context,
            )?;

            let attn_start = position * q_out;
            self.kernels.matmul(
                &attn_context[attn_start..attn_start + q_out],
                (1, q_out),
                &layer.o_proj_w,
                (q_out, h),
                &mut o_buf,
            )?;

            self.kernels.vec_add(&residual_buf, &o_buf, &mut hidden)?;
            residual_buf.copy_from_slice(&hidden);

            self.kernels.rmsnorm(
                &hidden,
                1,
                h,
                &layer.post_attention_layernorm_w,
                eps,
                &mut norm_buf,
            )?;
            self.kernels.mlp_gated_silu(
                &norm_buf,
                1,
                h,
                i_size,
                &layer.gate_proj_w,
                &layer.up_proj_w,
                &layer.down_proj_w,
                &mut gate_buf,
                &mut up_buf,
                &mut mlp_out,
            )?;
            self.kernels.vec_add(&residual_buf, &mlp_out, &mut hidden)?;
        }

        self.kernels.rmsnorm(
            &hidden,
            1,
            h,
            &self.weights.final_norm_w,
            eps,
            &mut norm_buf,
        )?;

        let mut logits = vec![0.0_f32; v];
        self.kernels.matmul(
            &norm_buf,
            (1, h),
            &self.weights.lm_head_w,
            (h, v),
            &mut logits,
        )?;
        cache.set_len_tokens(next_len)?;

        Ok(logits)
    }
}

/// Add a per-feature bias vector to every row of a `[rows, cols]` matrix
/// stored contiguous row-major. Inlined rather than introducing a new
/// kernel: it's a single trivial loop and the only caller is the
/// projection path inside this module.
fn add_bias_per_row(x: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
    debug_assert_eq!(x.len(), rows * cols);
    debug_assert_eq!(bias.len(), cols);
    for r in 0..rows {
        let row_start = r * cols;
        for c in 0..cols {
            x[row_start + c] += bias[c];
        }
    }
}

/// Transpose a `[rows, cols]` row-major slice into a fresh `[cols, rows]`
/// row-major slice. Used at weight-load time to flip HF's
/// `[out_features, in_features]` storage into the `[in, out]` layout the
/// matmul kernel consumes without re-walking memory in column order.
pub fn transpose_2d(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), rows * cols);
    let mut dst = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
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

fn validate_config_for_model(config: &Qwen2_5Config) -> Result<()> {
    if config.vocab_size < 2 {
        return Err(invalid_model("vocab_size", "must be >= 2"));
    }
    if config.num_hidden_layers == 0 {
        return Err(invalid_model("num_hidden_layers", "must be > 0"));
    }
    if config.hidden_size == 0 {
        return Err(invalid_model("hidden_size", "must be > 0"));
    }
    if config.intermediate_size == 0 {
        return Err(invalid_model("intermediate_size", "must be > 0"));
    }
    if config.num_attention_heads == 0 {
        return Err(invalid_model("num_attention_heads", "must be > 0"));
    }
    if config.num_key_value_heads == 0 {
        return Err(invalid_model("num_key_value_heads", "must be > 0"));
    }
    if config.head_dim == 0 {
        return Err(invalid_model("head_dim", "must be > 0"));
    }
    if config.num_attention_heads % config.num_key_value_heads != 0 {
        return Err(invalid_model(
            "num_attention_heads",
            "must be divisible by num_key_value_heads",
        ));
    }
    let hidden_from_heads = checked_len_product(
        "head_dim*num_attention_heads",
        &[config.head_dim, config.num_attention_heads],
    )?;
    if hidden_from_heads != config.hidden_size {
        return Err(invalid_model(
            "head_dim",
            &format!(
                "head_dim ({}) * num_attention_heads ({}) must equal hidden_size ({})",
                config.head_dim, config.num_attention_heads, config.hidden_size,
            ),
        ));
    }
    if config.head_dim % 2 != 0 {
        return Err(invalid_model("head_dim", "must be even"));
    }
    if config.context_length == 0 {
        return Err(invalid_model("context_length", "must be > 0"));
    }
    if !config.rope_theta.is_finite() || config.rope_theta <= 0.0 {
        return Err(invalid_model("rope_theta", "must be finite and > 0"));
    }
    if !config.rms_norm_eps.is_finite() || config.rms_norm_eps <= 0.0 {
        return Err(invalid_model("rms_norm_eps", "must be finite and > 0"));
    }
    if !matches!(config.dtype, DType::F32 | DType::BF16) {
        return Err(invalid_model("dtype", "must be F32 or BF16"));
    }
    Ok(())
}

fn validate_cache_layout_for_model(
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

fn checked_len_product(label: &str, dims: &[usize]) -> Result<usize> {
    dims.iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| invalid_model(label, &format!("shape product overflows usize: {:?}", dims)))
}

fn invalid_model(field: &str, message: &str) -> OcelotlError {
    OcelotlError::from(InvalidModelError {
        path: None,
        field: Some(field.to_string()),
        message: message.to_string(),
    })
}

fn take_tensor_values(
    by_name: &mut BTreeMap<String, LoadedTensor>,
    name: &str,
    expected_shape: &[usize],
    expected_dtype: &DType,
) -> Result<Vec<f32>> {
    let tensor = by_name
        .remove(name)
        .ok_or_else(|| invalid_model(name, "required Qwen2.5 tensor is missing"))?;
    if tensor.shape != expected_shape {
        return Err(invalid_model(
            name,
            &format!(
                "tensor `{name}` has shape {:?}, expected {:?}",
                tensor.shape, expected_shape
            ),
        ));
    }
    if !supported_dtype_matches(tensor.dtype, expected_dtype) {
        return Err(invalid_model(
            name,
            &format!(
                "tensor `{name}` has dtype {:?}, expected {:?}",
                tensor.dtype, expected_dtype
            ),
        ));
    }
    let expected_len = checked_len_product(name, expected_shape)?;
    if tensor.values.len() != expected_len {
        return Err(invalid_model(
            name,
            &format!(
                "tensor `{name}` has {} values, expected {expected_len}",
                tensor.values.len()
            ),
        ));
    }
    Ok(tensor.values)
}

fn supported_dtype_matches(actual: SupportedDtype, expected: &DType) -> bool {
    matches!(
        (actual, expected),
        (SupportedDtype::F32, DType::F32)
            | (SupportedDtype::F16, DType::F16)
            | (SupportedDtype::BF16, DType::BF16)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ocelotl_core::DType;
    use ocelotl_loader::{LoadedTensor, SupportedDtype};
    use std::{collections::BTreeMap, fs, io::Write, path::PathBuf, sync::Arc};

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
        ) -> Result<()> {
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
        ) -> Result<()> {
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
        ) -> Result<()> {
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
        ) -> Result<()> {
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
        ) -> Result<()> {
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
        ) -> Result<()> {
            Err(test_gpu_kernel_error(
                "mlp_gated_silu not implemented in test backend",
            ))
        }

        fn vec_add(&self, _a: &[f32], _b: &[f32], _out: &mut [f32]) -> Result<()> {
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

        let actual =
            Qwen2_5Weights::from_loaded_tensors(&cfg, loaded, expected.tie_word_embeddings)
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

        let err = Qwen2_5Model::new(cfg, weights)
            .expect_err("wrong embed_tokens length must be rejected");

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
}
