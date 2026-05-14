//! Qwen2.5 weight bundles and the safetensors-to-Qwen layout adapter.
//!
//! Per-layer and top-level weight structs live here, along with the
//! `from_loaded_tensors` adapter that maps Hugging Face tensor names and
//! layouts (`[out_features, in_features]` row-major) into the pre-transposed
//! `[in, out]` form the matmul kernel consumes. Forward composition lives in
//! `model.rs` / `prefill.rs` / `decode.rs`; this file owns layout, not math.
//!
//! # Why pre-transpose at load time
//!
//! HF native weight layout for linear layers is `[out, in]` row-major. The
//! kernel matmul is `[seq, in] @ [in, out]`, so without pre-transposing the
//! `B` matrix the kernel would walk storage in column-major order on every
//! call. We pay the transpose cost once at load and keep the hot path
//! contiguous.

use std::collections::{BTreeMap, btree_map::Entry};

use ocelotl_core::{DType, Result};
use ocelotl_loader::{LoadedTensor, SupportedDtype};

use super::{checked_len_product, config::Qwen2_5Config, invalid_model, validate_config_for_model};

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
