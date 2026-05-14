//! Qwen2.5 model family.
//!
//! Internal organization:
//!
//! - `config`  — `Qwen2_5Config` and `TryFrom<&ModelMetadata>` validation.
//! - `tensors` — required tensor names and safetensors manifest validation.
//! - `weights` — weight structs and the safetensors-to-Qwen layout adapter.
//! - `model`   — `Qwen2_5Model` struct, constructors, and shared cache-layout
//!   validation.
//! - `prefill` — full-prompt forward returning final-token logits.
//! - `decode`  — single-token decode with KV cache reuse.
//!
//! Family-level helpers shared by `weights`, `model`, `prefill`, and `decode`
//! live in this file with `pub(super)` visibility so siblings can use them
//! without going through a third module.

pub mod config;
pub mod decode;
pub mod model;
pub mod prefill;
pub mod tensors;
pub mod weights;

#[cfg(test)]
mod tests;

pub use config::Qwen2_5Config;
pub use model::Qwen2_5Model;
pub use tensors::{required_tensor_names, validate_qwen2_5_tensors};
pub use weights::{Qwen2_5LayerWeights, Qwen2_5Weights, transpose_2d};

use ocelotl_core::{DType, InvalidModelError, OcelotlError, Result};

pub(super) fn invalid_model(field: &str, message: &str) -> OcelotlError {
    OcelotlError::from(InvalidModelError {
        path: None,
        field: Some(field.to_string()),
        message: message.to_string(),
    })
}

pub(super) fn checked_len_product(label: &str, dims: &[usize]) -> Result<usize> {
    dims.iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| invalid_model(label, &format!("shape product overflows usize: {:?}", dims)))
}

pub(super) fn validate_config_for_model(config: &Qwen2_5Config) -> Result<()> {
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
