//! Qwen2.5 tensor-name and shape validation against a safetensors manifest.
//!
//! This module owns the model-family knowledge of *which* tensors a Qwen2.5
//! artifact must contain and *what shapes/dtypes* they must have. The loader's
//! `require_tensors` helper provides generic presence checking; the
//! shape-derivation rules and the canonical name list live here because
//! they are model-specific.
//!
//! # Boundary discipline
//!
//! Per `concepts/external-crate-boundary.md`: `ocelotl-models` consumes the
//! Ocelotl-owned `SafetensorsManifest` from `ocelotl-loader`, not the
//! foreign `safetensors` crate directly. The `safetensors` import stays
//! confined to the loader crate. Per `concepts/crossing-crate-boundaries.md`:
//! every failure path here returns `OcelotlError::InvalidModel`, never a
//! new error type.

use ocelotl_core::{DType, InvalidModelError, OcelotlError, Result};
use ocelotl_loader::{SafetensorsManifest, SupportedDtype, require_tensors};
use std::path::Path;

use super::Qwen2_5Config;

/// Validate a safetensors manifest against the Qwen2.5 tensor contract for
/// the given config.
///
/// Returns `Ok(())` if every required tensor is present with the expected
/// shape. On the first violation, returns `OcelotlError::InvalidModel` with
/// `field` set to the offending tensor name so callers can surface
/// "tensor X is missing" or "tensor X has wrong shape" without parsing
/// the message.
///
/// # The required tensor set
///
/// For each transformer layer `i in 0..num_hidden_layers`:
///
/// - `model.layers.{i}.self_attn.q_proj.weight`
/// - `model.layers.{i}.self_attn.q_proj.bias`
/// - `model.layers.{i}.self_attn.k_proj.weight`
/// - `model.layers.{i}.self_attn.k_proj.bias`
/// - `model.layers.{i}.self_attn.v_proj.weight`
/// - `model.layers.{i}.self_attn.v_proj.bias`
/// - `model.layers.{i}.self_attn.o_proj.weight`
/// - `model.layers.{i}.mlp.gate_proj.weight`
/// - `model.layers.{i}.mlp.up_proj.weight`
/// - `model.layers.{i}.mlp.down_proj.weight`
/// - `model.layers.{i}.input_layernorm.weight`
/// - `model.layers.{i}.post_attention_layernorm.weight`
///
/// Plus the global tensors:
///
/// - `model.embed_tokens.weight`
/// - `model.norm.weight`
/// - `lm_head.weight` (only when `tie_word_embeddings == false`)
///
/// # The tied-embedding parameter
///
/// Qwen2.5-0.5B-Instruct ships with `tie_word_embeddings: true`, meaning the
/// output projection reuses `model.embed_tokens.weight` and `lm_head.weight`
/// is absent from the safetensors file. The HF `config.json` carries the
/// flag; the safetensors header alone does not. Until M3.7 plumbs the flag
/// through `parse_hf_config` into a model-config field, callers must pass
/// it explicitly. (See journal entry for M3.2 for the open question.)
///
/// # Shape conventions (PyTorch Linear: out_features, in_features)
///
/// - `q_proj.weight`: `[num_attention_heads * head_dim, hidden_size]`
/// - `k_proj.weight`: `[num_key_value_heads * head_dim, hidden_size]`
/// - `v_proj.weight`: `[num_key_value_heads * head_dim, hidden_size]`
/// - `o_proj.weight`: `[hidden_size, num_attention_heads * head_dim]`
/// - `q_proj.bias`: `[num_attention_heads * head_dim]`
/// - `k_proj.bias`/`v_proj.bias`: `[num_key_value_heads * head_dim]`
/// - `gate_proj.weight`/`up_proj.weight`: `[intermediate_size, hidden_size]`
/// - `down_proj.weight`: `[hidden_size, intermediate_size]`
/// - `input_layernorm.weight`/`post_attention_layernorm.weight`: `[hidden_size]`
/// - `model.embed_tokens.weight`: `[vocab_size, hidden_size]`
/// - `model.norm.weight`: `[hidden_size]`
/// - `lm_head.weight` (untied): `[vocab_size, hidden_size]`
pub fn validate_qwen2_5_tensors(
    manifest: &SafetensorsManifest,
    config: &Qwen2_5Config,
    tie_word_embeddings: bool,
    path: Option<&Path>,
) -> Result<()> {
    let required = required_tensor_names(config, tie_word_embeddings);
    // Step 1: presence. Use the loader's generic helper so the
    // "first missing tensor" error format stays consistent with M2.5.
    let required_refs: Vec<&str> = required.iter().map(|s| s.as_str()).collect();
    require_tensors(manifest, &required_refs, path)?;

    // Step 2: shapes + dtype. Walk the same list, look each tensor up in
    // the manifest, and compare against the expected shape and dtype derived
    // from config. The presence check above guarantees every name resolves.
    let q_out = checked_dim_product(
        "num_attention_heads*head_dim",
        &[config.num_attention_heads, config.head_dim],
        path,
    )?;
    let kv_out = checked_dim_product(
        "num_key_value_heads*head_dim",
        &[config.num_key_value_heads, config.head_dim],
        path,
    )?;

    for layer in 0..config.num_hidden_layers {
        check_shape(
            manifest,
            &format!("model.layers.{layer}.self_attn.q_proj.weight"),
            &[q_out, config.hidden_size],
            &config.dtype,
            path,
        )?;
        check_shape(
            manifest,
            &format!("model.layers.{layer}.self_attn.q_proj.bias"),
            &[q_out],
            &config.dtype,
            path,
        )?;
        check_shape(
            manifest,
            &format!("model.layers.{layer}.self_attn.k_proj.weight"),
            &[kv_out, config.hidden_size],
            &config.dtype,
            path,
        )?;
        check_shape(
            manifest,
            &format!("model.layers.{layer}.self_attn.k_proj.bias"),
            &[kv_out],
            &config.dtype,
            path,
        )?;
        check_shape(
            manifest,
            &format!("model.layers.{layer}.self_attn.v_proj.weight"),
            &[kv_out, config.hidden_size],
            &config.dtype,
            path,
        )?;
        check_shape(
            manifest,
            &format!("model.layers.{layer}.self_attn.v_proj.bias"),
            &[kv_out],
            &config.dtype,
            path,
        )?;
        check_shape(
            manifest,
            &format!("model.layers.{layer}.self_attn.o_proj.weight"),
            &[config.hidden_size, q_out],
            &config.dtype,
            path,
        )?;
        check_shape(
            manifest,
            &format!("model.layers.{layer}.mlp.gate_proj.weight"),
            &[config.intermediate_size, config.hidden_size],
            &config.dtype,
            path,
        )?;
        check_shape(
            manifest,
            &format!("model.layers.{layer}.mlp.up_proj.weight"),
            &[config.intermediate_size, config.hidden_size],
            &config.dtype,
            path,
        )?;
        check_shape(
            manifest,
            &format!("model.layers.{layer}.mlp.down_proj.weight"),
            &[config.hidden_size, config.intermediate_size],
            &config.dtype,
            path,
        )?;
        check_shape(
            manifest,
            &format!("model.layers.{layer}.input_layernorm.weight"),
            &[config.hidden_size],
            &config.dtype,
            path,
        )?;
        check_shape(
            manifest,
            &format!("model.layers.{layer}.post_attention_layernorm.weight"),
            &[config.hidden_size],
            &config.dtype,
            path,
        )?;
    }

    check_shape(
        manifest,
        "model.embed_tokens.weight",
        &[config.vocab_size, config.hidden_size],
        &config.dtype,
        path,
    )?;
    check_shape(
        manifest,
        "model.norm.weight",
        &[config.hidden_size],
        &config.dtype,
        path,
    )?;

    if !tie_word_embeddings {
        check_shape(
            manifest,
            "lm_head.weight",
            &[config.vocab_size, config.hidden_size],
            &config.dtype,
            path,
        )?;
    }

    Ok(())
}

/// Build the canonical, ordered list of tensor names a valid Qwen2.5
/// artifact must contain for the given config. Pure function — no
/// manifest access, so other M3 tasks (M3.5/M3.6/M3.7) can call this to
/// know which tensors they must read.
pub fn required_tensor_names(config: &Qwen2_5Config, tie_word_embeddings: bool) -> Vec<String> {
    // Capacity: 12 per layer (7 attn weights+biases + 3 mlp + 2 norms) +
    // 2 globals (embed, final norm) + optional lm_head.
    let mut names = Vec::with_capacity(config.num_hidden_layers * 12 + 3);
    for layer in 0..config.num_hidden_layers {
        names.push(format!("model.layers.{layer}.self_attn.q_proj.weight"));
        names.push(format!("model.layers.{layer}.self_attn.q_proj.bias"));
        names.push(format!("model.layers.{layer}.self_attn.k_proj.weight"));
        names.push(format!("model.layers.{layer}.self_attn.k_proj.bias"));
        names.push(format!("model.layers.{layer}.self_attn.v_proj.weight"));
        names.push(format!("model.layers.{layer}.self_attn.v_proj.bias"));
        names.push(format!("model.layers.{layer}.self_attn.o_proj.weight"));
        names.push(format!("model.layers.{layer}.mlp.gate_proj.weight"));
        names.push(format!("model.layers.{layer}.mlp.up_proj.weight"));
        names.push(format!("model.layers.{layer}.mlp.down_proj.weight"));
        names.push(format!("model.layers.{layer}.input_layernorm.weight"));
        names.push(format!(
            "model.layers.{layer}.post_attention_layernorm.weight"
        ));
    }
    names.push("model.embed_tokens.weight".to_string());
    names.push("model.norm.weight".to_string());
    if !tie_word_embeddings {
        names.push("lm_head.weight".to_string());
    }
    names
}

/// Compare the manifest entry's shape against `expected`. Returns
/// `InvalidModel` with `field` set to the tensor name on mismatch. The
/// caller has already verified the name is present via `require_tensors`.
fn check_shape(
    manifest: &SafetensorsManifest,
    name: &str,
    expected: &[usize],
    expected_dtype: &DType,
    path: Option<&Path>,
) -> Result<()> {
    let entry = manifest
        .tensors
        .iter()
        .find(|t| t.name == name)
        // Should be unreachable because validate_qwen2_5_tensors runs
        // require_tensors first; if it does happen, surface as
        // InvalidModel rather than panic.
        .ok_or_else(|| {
            OcelotlError::from(InvalidModelError {
                path: path.map(|p| p.to_path_buf()),
                field: Some(name.to_string()),
                message: format!("tensor `{name}` not found in safetensors header"),
            })
        })?;
    if entry.shape != expected {
        return Err(OcelotlError::from(InvalidModelError {
            path: path.map(|p| p.to_path_buf()),
            field: Some(name.to_string()),
            message: format!(
                "tensor `{name}` has shape {:?}, expected {:?}",
                entry.shape, expected,
            ),
        }));
    }
    if !dtype_matches(entry.dtype, expected_dtype) {
        return Err(OcelotlError::from(InvalidModelError {
            path: path.map(|p| p.to_path_buf()),
            field: Some(name.to_string()),
            message: format!(
                "tensor `{name}` has dtype {}, expected {:?}",
                supported_dtype_name(entry.dtype),
                expected_dtype,
            ),
        }));
    }
    Ok(())
}

fn dtype_matches(actual: SupportedDtype, expected: &DType) -> bool {
    matches!(
        (actual, expected),
        (SupportedDtype::F32, DType::F32)
            | (SupportedDtype::F16, DType::F16)
            | (SupportedDtype::BF16, DType::BF16)
    )
}

fn supported_dtype_name(dtype: SupportedDtype) -> &'static str {
    match dtype {
        SupportedDtype::F32 => "F32",
        SupportedDtype::F16 => "F16",
        SupportedDtype::BF16 => "BF16",
    }
}

fn checked_dim_product(label: &str, dims: &[usize], path: Option<&Path>) -> Result<usize> {
    dims.iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| {
            OcelotlError::from(InvalidModelError {
                path: path.map(|p| p.to_path_buf()),
                field: Some(label.to_string()),
                message: format!("shape product overflows usize: {:?}", dims),
            })
        })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ocelotl_core::DType;
    use ocelotl_loader::{SupportedDtype, TensorEntry};

    /// Build a synthetic Qwen2.5 config sized small enough to enumerate
    /// every tensor by hand. Same divisibility invariants as the real
    /// model (hidden_size = num_attention_heads * head_dim;
    /// num_attention_heads % num_key_value_heads == 0).
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
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            dtype: DType::F32,
        }
    }

    /// Build a synthetic SafetensorsManifest containing exactly the tensors
    /// in `names`, each with the shape returned by `shape_for(name)`.
    /// Byte ranges don't matter for tensor-name/shape/dtype validation, so we
    /// set them to zero. Dtype defaults to F32 to match `tiny_config()`.
    fn manifest_with(
        names: &[String],
        shape_for: impl Fn(&str) -> Vec<usize>,
    ) -> SafetensorsManifest {
        let tensors = names
            .iter()
            .map(|n| TensorEntry {
                name: n.clone(),
                shape: shape_for(n),
                dtype: SupportedDtype::F32,
                byte_range: (0, 0),
            })
            .collect();
        SafetensorsManifest {
            tensors,
            data_len: 0,
        }
    }

    /// Compute the expected shape for a tensor name given a config. Mirrors
    /// the rules in `validate_qwen2_5_tensors` so tests can build "good"
    /// manifests without duplicating the table.
    fn shape_of(name: &str, cfg: &Qwen2_5Config) -> Vec<usize> {
        let q_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
            return vec![cfg.vocab_size, cfg.hidden_size];
        }
        if name == "model.norm.weight" {
            return vec![cfg.hidden_size];
        }
        // model.layers.{i}.{rest}
        let rest = name
            .strip_prefix("model.layers.")
            .expect("tensor name must start with model.layers.");
        // skip "{i}."
        let dot = rest.find('.').expect("layer-index dot");
        let suffix = &rest[dot + 1..];
        match suffix {
            "self_attn.q_proj.weight" => vec![q_out, cfg.hidden_size],
            "self_attn.q_proj.bias" => vec![q_out],
            "self_attn.k_proj.weight" => vec![kv_out, cfg.hidden_size],
            "self_attn.k_proj.bias" => vec![kv_out],
            "self_attn.v_proj.weight" => vec![kv_out, cfg.hidden_size],
            "self_attn.v_proj.bias" => vec![kv_out],
            "self_attn.o_proj.weight" => vec![cfg.hidden_size, q_out],
            "mlp.gate_proj.weight" => vec![cfg.intermediate_size, cfg.hidden_size],
            "mlp.up_proj.weight" => vec![cfg.intermediate_size, cfg.hidden_size],
            "mlp.down_proj.weight" => vec![cfg.hidden_size, cfg.intermediate_size],
            "input_layernorm.weight" => vec![cfg.hidden_size],
            "post_attention_layernorm.weight" => vec![cfg.hidden_size],
            other => panic!("shape_of: unknown tensor suffix `{other}` for `{name}`"),
        }
    }

    #[test]
    fn validate_rejects_missing_tensor_with_invalid_model_error_naming_the_tensor() {
        // The smallest M3.2 contract: a manifest missing exactly one
        // required tensor must fail with InvalidModel and `field` set to
        // the missing tensor name. This mirrors the M2.5 contract on
        // `require_tensors`; the M3.2 layer just adds the model-specific
        // "which tensors are required" knowledge.
        let cfg = tiny_config();
        let mut required = required_tensor_names(&cfg, /* tie */ true);
        // Drop one specific tensor. Pick a layer-0 q_proj.weight so the
        // assertion is unambiguous.
        let dropped = "model.layers.0.self_attn.q_proj.weight".to_string();
        required.retain(|n| n != &dropped);
        let manifest = manifest_with(&required, |n| shape_of(n, &cfg));

        let err = validate_qwen2_5_tensors(&manifest, &cfg, /* tie */ true, None)
            .expect_err("manifest missing a required tensor must be rejected");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(
                    invalid.field.as_deref(),
                    Some(dropped.as_str()),
                    "expected the missing tensor name in field, got {:?}",
                    invalid.field,
                );
                assert!(
                    invalid.message.contains(&dropped),
                    "expected message to mention the missing tensor, got {:?}",
                    invalid.message,
                );
            }
            other => panic!("expected InvalidModel for missing tensor, got {other:?}"),
        }
    }

    #[test]
    fn validate_accepts_complete_manifest_with_tied_embeddings() {
        // When tie_word_embeddings is true, lm_head.weight is NOT required
        // (and the real Qwen2.5-0.5B-Instruct artifact does not contain it).
        // A manifest that has every required tensor with correct shapes
        // must pass.
        let cfg = tiny_config();
        let names = required_tensor_names(&cfg, /* tie */ true);
        let manifest = manifest_with(&names, |n| shape_of(n, &cfg));

        validate_qwen2_5_tensors(&manifest, &cfg, /* tie */ true, None)
            .expect("complete tied-embedding manifest must validate");
    }

    #[test]
    fn validate_accepts_complete_manifest_with_untied_embeddings() {
        // When tie_word_embeddings is false, lm_head.weight IS required.
        // A manifest containing it with the right shape must pass.
        let cfg = tiny_config();
        let names = required_tensor_names(&cfg, /* tie */ false);
        let manifest = manifest_with(&names, |n| shape_of(n, &cfg));

        validate_qwen2_5_tensors(&manifest, &cfg, /* tie */ false, None)
            .expect("complete untied-embedding manifest must validate");
    }

    #[test]
    fn validate_rejects_missing_lm_head_only_when_embeddings_are_untied() {
        // The tied/untied switch is the contract that distinguishes
        // Qwen2.5-0.5B-Instruct (tied) from larger Qwen2.5 variants
        // (untied). Same manifest contents, different validator outcome
        // depending on the flag.
        let cfg = tiny_config();
        let names_no_lm = required_tensor_names(&cfg, /* tie */ true);
        let manifest = manifest_with(&names_no_lm, |n| shape_of(n, &cfg));

        // tied: passes.
        validate_qwen2_5_tensors(&manifest, &cfg, true, None)
            .expect("manifest without lm_head must pass under tied embeddings");

        // untied: fails on lm_head.weight specifically.
        let err = validate_qwen2_5_tensors(&manifest, &cfg, false, None)
            .expect_err("manifest without lm_head must fail under untied embeddings");
        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("lm_head.weight"));
            }
            other => panic!("expected InvalidModel for missing lm_head, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_wrong_shape_with_invalid_model_error_naming_the_tensor() {
        // Per the M3 plan ("shape and dtype mismatches fail explicitly"),
        // validation must distinguish "tensor present but wrong shape"
        // from "tensor missing". Both are InvalidModel with the offending
        // tensor name, but the message must mention the shape so callers
        // can render "expected X, got Y".
        let cfg = tiny_config();
        let names = required_tensor_names(&cfg, true);
        // Mutate one tensor's shape: swap rows and columns of q_proj.weight
        // for layer 0. With hidden_size=16 and q_out=16 they happen to be
        // square, so use embed_tokens (vocab × hidden = 32 × 16, not square).
        let bad = "model.embed_tokens.weight".to_string();
        let manifest = manifest_with(&names, |n| {
            if n == bad {
                vec![cfg.hidden_size, cfg.vocab_size] // swapped on purpose
            } else {
                shape_of(n, &cfg)
            }
        });

        let err = validate_qwen2_5_tensors(&manifest, &cfg, true, None)
            .expect_err("wrong-shape tensor must be rejected");
        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some(bad.as_str()));
                assert!(
                    invalid.message.contains("expected"),
                    "expected message to mention `expected`, got {:?}",
                    invalid.message,
                );
            }
            other => panic!("expected InvalidModel for wrong shape, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_wrong_dtype_with_invalid_model_error_naming_the_tensor() {
        let cfg = tiny_config();
        let names = required_tensor_names(&cfg, true);
        let bad = "model.embed_tokens.weight".to_string();
        let mut manifest = manifest_with(&names, |n| shape_of(n, &cfg));
        let entry = manifest
            .tensors
            .iter_mut()
            .find(|t| t.name == bad)
            .expect("bad tensor must exist in manifest");
        entry.dtype = SupportedDtype::F16;

        let err = validate_qwen2_5_tensors(&manifest, &cfg, true, None)
            .expect_err("wrong-dtype tensor must be rejected");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some(bad.as_str()));
                assert!(
                    invalid.message.contains("dtype") && invalid.message.contains("F16"),
                    "expected dtype mismatch detail, got {:?}",
                    invalid.message,
                );
            }
            other => panic!("expected InvalidModel for wrong dtype, got {other:?}"),
        }
    }

    #[test]
    fn validate_accepts_bf16_manifest_when_config_dtype_is_bf16() {
        let mut cfg = tiny_config();
        cfg.dtype = DType::BF16;
        let names = required_tensor_names(&cfg, true);
        let mut manifest = manifest_with(&names, |n| shape_of(n, &cfg));
        for entry in &mut manifest.tensors {
            entry.dtype = SupportedDtype::BF16;
        }

        validate_qwen2_5_tensors(&manifest, &cfg, true, None)
            .expect("BF16 config must accept BF16 tensor headers");
    }

    #[test]
    fn validate_rejects_shape_product_overflow_with_invalid_model() {
        let mut cfg = tiny_config();
        cfg.num_hidden_layers = 0;
        cfg.num_attention_heads = usize::MAX;
        cfg.head_dim = 2;
        let names = required_tensor_names(&cfg, true);
        let manifest = manifest_with(&names, |n| shape_of(n, &tiny_config()));

        let err = validate_qwen2_5_tensors(&manifest, &cfg, true, None)
            .expect_err("overflowing q_out shape product must be rejected");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert!(
                    invalid.message.contains("overflows"),
                    "expected overflow diagnostic, got {:?}",
                    invalid.message
                );
            }
            other => panic!("expected InvalidModel for shape overflow, got {other:?}"),
        }
    }

    #[test]
    fn required_tensor_names_enumerates_full_set_for_tied_and_untied() {
        // Pin the exact count: 12 per layer (q/k/v weights+biases + o_proj
        // + 3 mlp + 2 norms = 7+3+2 = 12) + 2 globals (embed, norm); add 1
        // when not tied (lm_head).
        let cfg = tiny_config();
        let tied = required_tensor_names(&cfg, true);
        assert_eq!(tied.len(), cfg.num_hidden_layers * 12 + 2);
        let untied = required_tensor_names(&cfg, false);
        assert_eq!(untied.len(), cfg.num_hidden_layers * 12 + 3);
        assert!(untied.contains(&"lm_head.weight".to_string()));
        assert!(!tied.contains(&"lm_head.weight".to_string()));
    }
}
