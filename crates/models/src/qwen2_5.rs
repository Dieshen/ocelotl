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

use ocelotl_core::{DType, InvalidModelError, ModelMetadata, OcelotlError, UnsupportedError};

/// The single architecture string this model family accepts on
/// `ModelMetadata`. Qwen2.5 model artifacts share the `qwen2` model_type
/// with Qwen2 (the 2.5 release tightens training and extends context but
/// keeps the architecture identifier). The loader's allow-list is the
/// first gate; this is the second — needed because nothing in the
/// Rust type system stops a caller from constructing a `ModelMetadata`
/// directly with any `architecture` string.
const QWEN2_5_ARCHITECTURE: &str = "qwen2";

/// Dtypes the Qwen2.5 reference forward path accepts at construction time.
///
/// The CPU reference path commits to f32 compute. The on-disk artifact
/// dtype may be `F32` directly, or `BF16` (the published
/// Qwen2.5-0.5B-Instruct dtype, upcast to f32 by the loader/forward path
/// before kernels run). Anything else (`F16`, `Q4`, `Q8`) cannot be
/// executed by the M3 reference path and must be rejected at construction
/// time — runtime should never launch a forward pass for an unsupported
/// dtype combination (per docs/tasks/m3-*.md M3.10 "Done when").
const QWEN2_5_SUPPORTED_DTYPES: &[DType] = &[DType::F32, DType::BF16];

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

        // RoPE head_dim parity (M3.10). RoPE pairs index `i` with
        // `i + head_dim/2` (the upper-half pairing convention used by
        // Qwen2.5/Llama/HF). An odd head_dim has no consistent
        // half-pairing — the rotation is mathematically undefined.
        // Reject here rather than letting the kernel error at compute.
        if m.head_dim % 2 != 0 {
            return Err(invalid(
                "head_dim",
                &format!("must be even for RoPE half-pairing; got {}", m.head_dim,),
            ));
        }

        // RoPE theta gate (M3.10). `rope_theta` is the base of the
        // inverse-frequency formula `1 / theta^(2i/head_dim)`; theta == 0
        // yields division-by-zero, theta < 0 yields complex powers.
        // Both are unrunnable.
        if !(m.rope_theta > 0.0) {
            return Err(invalid(
                "rope_theta",
                &format!("must be > 0; got {}", m.rope_theta),
            ));
        }

        // Dtype gate (M3.10). The CPU reference path can only run F32 and
        // BF16 (BF16 is upcast to F32 for compute). F16 and quantized
        // formats are rejected here so the runtime never attempts a
        // forward pass against an unrunnable dtype.
        if !QWEN2_5_SUPPORTED_DTYPES.contains(&m.dtype) {
            return Err(OcelotlError::from(UnsupportedError {
                feature: "qwen2_5.dtype".to_string(),
                requested: Some(format!("{:?}", m.dtype)),
                supported: QWEN2_5_SUPPORTED_DTYPES
                    .iter()
                    .map(|d| format!("{d:?}"))
                    .collect(),
            }));
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
    fn try_from_rejects_non_qwen2_architecture_with_unsupported_error() {
        // Sanity: even though the loader gates on architecture for its
        // own surface, the model-family layer re-validates because
        // nothing in the type system stops a caller from constructing
        // a `ModelMetadata { architecture: "mistral", ... }` and
        // handing it to the Qwen2.5 model. The rejection is
        // `Unsupported`, not `InvalidModel` — a "mistral" model is a
        // coherent thing in general; it's just not what *this* family
        // implements.
        let mut meta = qwen2_5_0_5b_metadata();
        meta.architecture = "mistral".to_string();

        let err =
            Qwen2_5Config::try_from(&meta).expect_err("non-qwen2 architecture must be rejected");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(
                    unsupported.feature, "qwen2_5.architecture",
                    "expected feature qualified to this model family, got {:?}",
                    unsupported.feature,
                );
                assert_eq!(unsupported.requested.as_deref(), Some("mistral"));
                assert!(
                    unsupported.supported.iter().any(|s| s == "qwen2"),
                    "expected `qwen2` in supported list, got {:?}",
                    unsupported.supported,
                );
            }
            other => panic!("expected Unsupported for foreign architecture, got {other:?}"),
        }
    }

    #[test]
    fn try_from_rejects_non_divisible_gqa_grouping_with_invalid_model_error() {
        // Qwen2.5 uses Grouped-Query Attention: every KV head fans out
        // to an integer number of Q heads. If `num_attention_heads` is
        // not a multiple of `num_key_value_heads`, the metadata cannot
        // describe a runnable Qwen2.5 model, even if every other field
        // is fine. This is a *model-specific* invariant — the loader
        // (M2.6) does not enforce it.
        let mut meta = qwen2_5_0_5b_metadata();
        // 14 Q heads / 2 KV heads is the real Qwen2.5-0.5B group ratio
        // (7 Q per KV). Pick num_key_value_heads=3 to break it: 14 % 3 != 0.
        meta.num_key_value_heads = 3;

        let err = Qwen2_5Config::try_from(&meta)
            .expect_err("non-divisible GQA grouping must be rejected");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(
                    invalid.field.as_deref(),
                    Some("num_attention_heads"),
                    "expected field=num_attention_heads, got {:?}",
                    invalid.field,
                );
                assert!(
                    invalid.message.contains("divisible")
                        && invalid.message.contains("num_key_value_heads"),
                    "expected divisibility detail in message, got {:?}",
                    invalid.message,
                );
            }
            other => panic!("expected InvalidModel for GQA mismatch, got {other:?}"),
        }
    }

    #[test]
    fn try_from_rejects_unsupported_dtype_with_typed_unsupported_error() {
        // M3.10 dtype gate. The CPU reference path commits to producing
        // f32 compute outputs; the artifact-on-disk dtype is allowed to be
        // BF16 (the published Qwen2.5-0.5B-Instruct dtype, upcast to f32
        // for compute) or F32, but anything else (F16, Q4, Q8) cannot be
        // executed by the M3 reference path. The rejection is `Unsupported`
        // — the dtype is a coherent dtype, just not one this family can
        // run today.
        //
        // BF16 is allow-listed because the M3.1 fixture uses it (it's the
        // real Qwen2.5-0.5B artifact dtype, upcast to f32 at compute time).
        // F16 is the cleanest "unsupported but coherent" dtype to pin
        // against; Q4/Q8 cover the quantized-format rejection that M3
        // explicitly defers (see docs/milestones/m3-*.md "Non-Goals").
        let mut meta = qwen2_5_0_5b_metadata();
        meta.dtype = DType::F16;

        let err =
            Qwen2_5Config::try_from(&meta).expect_err("F16 dtype must be rejected at construction");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(
                    unsupported.feature, "qwen2_5.dtype",
                    "expected feature qualified to this model family, got {:?}",
                    unsupported.feature,
                );
                assert_eq!(unsupported.requested.as_deref(), Some("F16"));
                assert!(
                    unsupported.supported.iter().any(|s| s == "F32")
                        && unsupported.supported.iter().any(|s| s == "BF16"),
                    "expected F32 and BF16 in supported list, got {:?}",
                    unsupported.supported,
                );
            }
            other => panic!("expected Unsupported for F16 dtype, got {other:?}"),
        }
    }

    #[test]
    fn try_from_rejects_quantized_dtype_with_typed_unsupported_error() {
        // Q4/Q8 are explicitly out of scope per docs/milestones/m3-*.md
        // ("Non-Goals: Quantized weights"). They must be rejected at
        // construction time so the reference path never tries to compute
        // against a dtype it has no kernel for.
        for bad in [DType::Q4, DType::Q8] {
            let mut meta = qwen2_5_0_5b_metadata();
            meta.dtype = bad.clone();

            let err = Qwen2_5Config::try_from(&meta)
                .expect_err("quantized dtype must be rejected at construction");

            match err {
                OcelotlError::Unsupported(unsupported) => {
                    assert_eq!(unsupported.feature, "qwen2_5.dtype");
                    assert!(
                        unsupported.requested.is_some(),
                        "expected requested dtype name, got None",
                    );
                }
                other => panic!("expected Unsupported for {bad:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn try_from_rejects_odd_head_dim_with_invalid_model_error() {
        // RoPE pairs index `i` with `i + head_dim/2` (the upper-half
        // pairing convention used by Qwen2.5/Llama/HF — see dev-04's
        // M3.4 RoPE learning entry). An odd head_dim has no consistent
        // half-pairing, so the rotation is mathematically undefined.
        // Reject at construction time so the runtime never calls into a
        // RoPE kernel that would either panic or silently produce wrong
        // numbers.
        //
        // Construct a metadata that survives all earlier gates (positive
        // dims, GQA divisibility, head_dim*heads == hidden_size,
        // supported dtype) but has an odd head_dim. Pick head_dim=7,
        // num_attention_heads=14, hidden_size=98, intermediate_size kept
        // at 4*hidden ish.
        let mut meta = qwen2_5_0_5b_metadata();
        meta.head_dim = 7;
        meta.num_attention_heads = 14;
        meta.num_key_value_heads = 2;
        meta.hidden_size = 98; // 7 * 14
        meta.intermediate_size = 256;

        let err = Qwen2_5Config::try_from(&meta)
            .expect_err("odd head_dim must be rejected at construction (RoPE half-pairing)");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(
                    invalid.field.as_deref(),
                    Some("head_dim"),
                    "expected field=head_dim, got {:?}",
                    invalid.field,
                );
                assert!(
                    invalid.message.contains("even") || invalid.message.contains("RoPE"),
                    "expected even/RoPE detail in message, got {:?}",
                    invalid.message,
                );
            }
            other => panic!("expected InvalidModel for odd head_dim, got {other:?}"),
        }
    }

    #[test]
    fn try_from_rejects_zero_rope_theta_with_invalid_model_error() {
        // rope_theta is the base of the RoPE inverse-frequency formula
        // (`1 / theta^(2i/head_dim)`); theta == 0 yields division-by-zero
        // / infinity at the kernel boundary. Reject at construction.
        // Negative theta isn't expressible from a real HF config (the
        // field is always positive in published Qwen2.5 configs) but
        // we still reject it for completeness — `theta <= 0` is the
        // single check.
        let mut meta = qwen2_5_0_5b_metadata();
        meta.rope_theta = 0.0;

        let err = Qwen2_5Config::try_from(&meta)
            .expect_err("zero rope_theta must be rejected at construction");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(
                    invalid.field.as_deref(),
                    Some("rope_theta"),
                    "expected field=rope_theta, got {:?}",
                    invalid.field,
                );
                assert!(
                    invalid.message.contains("> 0") || invalid.message.contains("positive"),
                    "expected positivity detail in message, got {:?}",
                    invalid.message,
                );
            }
            other => panic!("expected InvalidModel for zero rope_theta, got {other:?}"),
        }
    }

    #[test]
    fn try_from_rejects_negative_rope_theta_with_invalid_model_error() {
        let mut meta = qwen2_5_0_5b_metadata();
        meta.rope_theta = -1.0;

        let err = Qwen2_5Config::try_from(&meta)
            .expect_err("negative rope_theta must be rejected at construction");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("rope_theta"));
            }
            other => panic!("expected InvalidModel for negative rope_theta, got {other:?}"),
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
