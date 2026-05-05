//! Qwen2.5 model-family metadata contract.
//!
//! `Qwen2_5Config` is the validated, model-family-specific projection of
//! `ocelotl_core::ModelMetadata` that the Qwen2.5 forward path consumes.
//! The conversion is fallible: it rejects metadata that does not represent
//! a coherent Qwen2.5 shape, so downstream kernel code can assume the
//! invariants hold.
//!
//! # Boundary discipline
//!
//! Per `docs/crate-boundaries.md` and the `crossing-crate-boundaries`
//! workspace concept: shared, generic metadata fields stay in
//! `ocelotl-core::ModelMetadata`; model-specific validation lives here in
//! `ocelotl-models`. We do NOT add Qwen2.5-shaped fields to
//! `ModelMetadata`. We do NOT define a new error type — we map every
//! Qwen2.5 rejection to an existing `OcelotlError` variant
//! (`Unsupported` for "wrong architecture for this model family",
//! `InvalidModel` for "internally inconsistent shape").
//!
//! M3.1 only encodes the shape contract. Tensor name/shape validation is
//! M3.2 territory; kernel hookup is M3.3-M3.6.

use ocelotl_core::{InvalidModelError, ModelMetadata, OcelotlError, UnsupportedError};

/// The single architecture string this model family accepts on
/// `ModelMetadata`. Qwen2.5 model artifacts share the `qwen2` model_type
/// with Qwen2 (the 2.5 release tightens training and extends context but
/// keeps the architecture identifier). The loader's allow-list is the
/// first gate; this is the second — needed because nothing in the
/// Rust type system stops a caller from constructing a `ModelMetadata`
/// directly with any `architecture` string.
const QWEN2_5_ARCHITECTURE: &str = "qwen2";

/// Validated Qwen2.5 model-family configuration.
///
/// Built from `&ModelMetadata` via `TryFrom`. The fields are a refined
/// projection of `ModelMetadata` after the Qwen2.5-specific invariants
/// have been checked. Downstream code (M3.2 tensor validation, M3.3+
/// kernel calls) can rely on:
///
/// - `head_dim * num_attention_heads == hidden_size`
/// - `num_attention_heads % num_key_value_heads == 0` (GQA group sizing)
/// - all positive dimensions
/// - `architecture == "qwen2"`
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen2_5Config {
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub context_length: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub dtype: ocelotl_core::DType,
}

impl TryFrom<&ModelMetadata> for Qwen2_5Config {
    type Error = OcelotlError;

    fn try_from(m: &ModelMetadata) -> Result<Self, Self::Error> {
        // Architecture gate. Qwen2.5 reuses Qwen2's architecture
        // identifier; anything else is a different model family.
        if m.architecture != QWEN2_5_ARCHITECTURE {
            return Err(OcelotlError::from(UnsupportedError {
                feature: "qwen2_5.architecture".to_string(),
                requested: Some(m.architecture.clone()),
                supported: vec![QWEN2_5_ARCHITECTURE.to_string()],
            }));
        }

        // Internal shape consistency. These are not "we don't support
        // it yet" failures — they are "this metadata cannot describe a
        // coherent Qwen2.5 model", which is `InvalidModel`.
        if m.num_attention_heads == 0 {
            return Err(invalid("num_attention_heads", "must be > 0"));
        }
        if m.num_key_value_heads == 0 {
            return Err(invalid("num_key_value_heads", "must be > 0"));
        }
        if m.hidden_size == 0 {
            return Err(invalid("hidden_size", "must be > 0"));
        }
        if m.head_dim == 0 {
            return Err(invalid("head_dim", "must be > 0"));
        }

        // GQA: every KV head groups some number of Q heads. Qwen2.5
        // uses GQA (num_attention_heads >= num_key_value_heads, with
        // exact divisibility).
        if m.num_attention_heads % m.num_key_value_heads != 0 {
            return Err(invalid(
                "num_attention_heads",
                &format!(
                    "must be divisible by num_key_value_heads ({}); got {}",
                    m.num_key_value_heads, m.num_attention_heads,
                ),
            ));
        }

        // head_dim must reconstruct hidden_size given num_attention_heads.
        // Qwen2.5 uses uniform head dims; if these disagree the metadata
        // is internally inconsistent.
        if m.head_dim
            .checked_mul(m.num_attention_heads)
            .map(|p| p != m.hidden_size)
            .unwrap_or(true)
        {
            return Err(invalid(
                "head_dim",
                &format!(
                    "head_dim ({}) * num_attention_heads ({}) must equal hidden_size ({})",
                    m.head_dim, m.num_attention_heads, m.hidden_size,
                ),
            ));
        }

        Ok(Qwen2_5Config {
            vocab_size: m.vocab_size,
            num_hidden_layers: m.num_hidden_layers,
            hidden_size: m.hidden_size,
            intermediate_size: m.intermediate_size,
            num_attention_heads: m.num_attention_heads,
            num_key_value_heads: m.num_key_value_heads,
            head_dim: m.head_dim,
            context_length: m.context_length,
            rope_theta: m.rope_theta,
            rms_norm_eps: m.rms_norm_eps,
            dtype: m.dtype.clone(),
        })
    }
}

fn invalid(field: &str, message: &str) -> OcelotlError {
    OcelotlError::from(InvalidModelError {
        path: None,
        field: Some(field.to_string()),
        message: message.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ocelotl_core::DType;

    /// The real Qwen2.5-0.5B-Instruct config, hand-mirrored from
    /// `fixtures/metadata/qwen2_5_0_5b_instruct_config.json` (pinned at
    /// SHA `7ae5576...` per `fixtures/manifest/qwen2_5_0_5b_instruct.json`).
    /// Built directly here rather than going through `ocelotl-loader` so
    /// the model-family tests don't pull a loader dependency just to
    /// exercise the conversion contract.
    fn qwen2_5_0_5b_metadata() -> ModelMetadata {
        ModelMetadata {
            architecture: "qwen2".to_string(),
            vocab_size: 151_936,
            num_hidden_layers: 24,
            hidden_size: 896,
            intermediate_size: 4_864,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            head_dim: 64, // 896 / 14
            context_length: 32_768,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            dtype: DType::BF16,
            tokenizer_model_hint: None,
        }
    }

    #[test]
    fn try_from_accepts_real_qwen2_5_0_5b_metadata() {
        let meta = qwen2_5_0_5b_metadata();

        let cfg = Qwen2_5Config::try_from(&meta)
            .expect("real Qwen2.5-0.5B metadata must convert into a Qwen2_5Config");

        // Pin every field so a future refactor that reorders or drops a
        // field fails loudly rather than silently producing a wrong
        // model config.
        assert_eq!(cfg.vocab_size, 151_936);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.hidden_size, 896);
        assert_eq!(cfg.intermediate_size, 4_864);
        assert_eq!(cfg.num_attention_heads, 14);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.context_length, 32_768);
        assert!((cfg.rope_theta - 1_000_000.0_f64).abs() < 1e-6);
        assert!((cfg.rms_norm_eps - 1e-6_f64).abs() < 1e-12);
        assert_eq!(cfg.dtype, DType::BF16);
    }
}
