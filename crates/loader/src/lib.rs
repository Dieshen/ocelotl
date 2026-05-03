//! Model artifact loading and validation.

use std::path::Path;

use ocelotl_core::{
    DType, InvalidModelError, IoError, ModelMetadata, OcelotlError, Result, UnsupportedError,
};
use serde::Deserialize;

pub mod safetensors_inspect;
pub use safetensors_inspect::{
    SafetensorsManifest, SupportedDtype, TensorEntry, inspect_safetensors, require_tensors,
};

/// Architectures the loader currently accepts. Anything outside this list is
/// rejected with `OcelotlError::Unsupported` before any further validation.
const SUPPORTED_ARCHITECTURES: &[&str] = &["qwen2"];

/// Dtypes the loader currently accepts. Anything outside this list is rejected
/// with `OcelotlError::Unsupported` before the full metadata parse, so the
/// rejection happens with a typed error rather than a generic serde
/// "unknown variant" InvalidModel.
const SUPPORTED_DTYPES: &[&str] = &["f32"];

/// Top-level fixture envelope: `{ "model": { ... }, ... }`. Only the `model`
/// field is meaningful for loading; the rest is fixture metadata.
#[derive(Debug, Deserialize)]
struct MetadataEnvelope {
    model: ModelInspect,
}

/// Minimal projection of the model object used to gate on architecture and
/// dtype before committing to a full `ModelMetadata` deserialize. Keeping
/// these as `String` (not the typed `DType` enum) is intentional: serde would
/// reject unknown enum variants at parse time and surface them as
/// `InvalidModel`, when we want a typed `Unsupported` instead.
#[derive(Debug, Deserialize)]
struct ModelInspect {
    architecture: String,
    dtype: String,
}

/// Load and validate a model metadata document from disk.
///
/// Returns `OcelotlError::Unsupported` when the architecture is recognized
/// but not yet implemented (e.g. anything outside `SUPPORTED_ARCHITECTURES`).
pub fn load_metadata(path: &Path) -> Result<ModelMetadata> {
    // File-read failures map to `Io`, not `InvalidModel` — see the matching
    // comment in `safetensors_inspect::inspect_safetensors` and
    // docs/design/loader.md "Error Mapping" for the rationale.
    let json = std::fs::read_to_string(path).map_err(|source| {
        OcelotlError::from(IoError {
            path: Some(path.to_path_buf()),
            source,
        })
    })?;

    let envelope: MetadataEnvelope = serde_json::from_str(&json).map_err(|source| {
        OcelotlError::from(InvalidModelError {
            path: Some(path.to_path_buf()),
            field: None,
            message: format!("failed to parse metadata JSON: {source}"),
        })
    })?;

    if !SUPPORTED_ARCHITECTURES.contains(&envelope.model.architecture.as_str()) {
        return Err(OcelotlError::from(UnsupportedError {
            feature: "architecture".to_string(),
            requested: Some(envelope.model.architecture),
            supported: SUPPORTED_ARCHITECTURES
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }));
    }

    if !SUPPORTED_DTYPES.contains(&envelope.model.dtype.as_str()) {
        return Err(OcelotlError::from(UnsupportedError {
            feature: "dtype".to_string(),
            requested: Some(envelope.model.dtype),
            supported: SUPPORTED_DTYPES.iter().map(|s| s.to_string()).collect(),
        }));
    }

    // Architecture and dtype are supported; deserialize the full metadata struct.
    #[derive(Debug, Deserialize)]
    struct FullEnvelope {
        model: ModelMetadata,
    }
    let full: FullEnvelope = serde_json::from_str(&json).map_err(|source| {
        let message = format!("failed to parse metadata JSON: {source}");
        let field = extract_missing_field(&message);
        OcelotlError::from(InvalidModelError {
            path: Some(path.to_path_buf()),
            field,
            message,
        })
    })?;
    Ok(full.model)
}

/// Best-effort extraction of the field name from serde's standard
/// "missing field `<name>`" error message. Returns `None` when the message
/// does not match that pattern; callers should still surface the full message
/// in the error so nothing is lost when extraction fails.
fn extract_missing_field(message: &str) -> Option<String> {
    let needle = "missing field `";
    let start = message.find(needle)? + needle.len();
    let rest = &message[start..];
    let end = rest.find('`')?;
    Some(rest[..end].to_string())
}

// ---------------------------------------------------------------------------
// M2.6 — Hugging Face config.json -> ocelotl-core::ModelMetadata
// ---------------------------------------------------------------------------
//
// `parse_hf_config` parses a real Hugging Face `config.json` (as published in
// transformers-format model repositories) and produces an
// `ocelotl-core::ModelMetadata`. The mapping is deliberately explicit because
// HF field names diverge from Ocelotl's normalized names (e.g.
// `max_position_embeddings` -> `context_length`, `model_type` -> `architecture`,
// `torch_dtype` -> `dtype`), and `head_dim` is absent in the HF config and
// must be derived from `hidden_size / num_attention_heads`.
//
// Architecture and torch_dtype are validated against the loader's
// `SUPPORTED_*` allow-lists *before* deeper parsing so unsupported artifacts
// fail with `Unsupported`, not `InvalidModel`.
//
// `load_metadata` (above) is the M1 path that consumes Ocelotl-shaped fixture
// envelopes; the two functions exist side by side for now because the
// existing M1 fixtures use the Ocelotl shape and we don't want to invalidate
// them.

/// `torch_dtype` strings that map to a supported Ocelotl `DType`. Anything
/// outside this list is rejected with `OcelotlError::Unsupported`.
const SUPPORTED_TORCH_DTYPES: &[&str] = &["float32", "float16", "bfloat16"];

/// Subset of HF `config.json` fields used to construct `ModelMetadata`. We
/// list only what we need; serde silently ignores the rest of the document
/// (initializer_range, attention_dropout, transformers_version, etc.) so the
/// struct doesn't drift every time HF adds a field.
#[derive(Debug, Deserialize)]
struct HfConfig {
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    /// HF's field is `max_position_embeddings`; we rename it to
    /// `context_length` in `ModelMetadata`.
    max_position_embeddings: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
    /// HF's field is `torch_dtype` (e.g. `"bfloat16"`); we map it to
    /// `core::DType` via the `SUPPORTED_TORCH_DTYPES` allow-list.
    torch_dtype: String,
}

/// Parse a Hugging Face `config.json` and return the equivalent Ocelotl
/// `ModelMetadata`.
///
/// Errors:
/// - `Io` if the file cannot be read.
/// - `InvalidModel` if the JSON is malformed or required fields are missing.
/// - `Unsupported` if `model_type` or `torch_dtype` is outside the allow-list.
/// - `InvalidModel` (with `field = "head_dim"`) if `hidden_size` is not
///   divisible by `num_attention_heads`.
pub fn parse_hf_config(path: &Path) -> Result<ModelMetadata> {
    let json = std::fs::read_to_string(path).map_err(|source| {
        OcelotlError::from(IoError {
            path: Some(path.to_path_buf()),
            source,
        })
    })?;

    // Two-stage parse: first project just the strings we want to allow-list
    // on, so unsupported architectures/dtypes surface as `Unsupported`
    // rather than as `InvalidModel(unknown variant ...)` from a strongly
    // typed enum deserialize. Mirrors the load_metadata strategy above.
    #[derive(Debug, Deserialize)]
    struct Gate {
        model_type: String,
        torch_dtype: String,
    }
    let gate: Gate = serde_json::from_str(&json).map_err(|source| {
        OcelotlError::from(InvalidModelError {
            path: Some(path.to_path_buf()),
            field: None,
            message: format!("failed to parse HF config.json: {source}"),
        })
    })?;

    if !SUPPORTED_ARCHITECTURES.contains(&gate.model_type.as_str()) {
        return Err(OcelotlError::from(UnsupportedError {
            feature: "architecture".to_string(),
            requested: Some(gate.model_type),
            supported: SUPPORTED_ARCHITECTURES
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }));
    }
    if !SUPPORTED_TORCH_DTYPES.contains(&gate.torch_dtype.as_str()) {
        return Err(OcelotlError::from(UnsupportedError {
            feature: "torch_dtype".to_string(),
            requested: Some(gate.torch_dtype),
            supported: SUPPORTED_TORCH_DTYPES
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }));
    }

    let cfg: HfConfig = serde_json::from_str(&json).map_err(|source| {
        let message = format!("failed to parse HF config.json: {source}");
        let field = extract_missing_field(&message);
        OcelotlError::from(InvalidModelError {
            path: Some(path.to_path_buf()),
            field,
            message,
        })
    })?;

    // Derive head_dim. HF Qwen2 configs don't carry head_dim explicitly;
    // the convention is hidden_size / num_attention_heads. If the division
    // has a remainder, the artifact is internally inconsistent.
    if cfg.num_attention_heads == 0 || cfg.hidden_size % cfg.num_attention_heads != 0 {
        return Err(OcelotlError::from(InvalidModelError {
            path: Some(path.to_path_buf()),
            field: Some("head_dim".to_string()),
            message: format!(
                "cannot derive head_dim: hidden_size ({}) is not divisible by num_attention_heads ({})",
                cfg.hidden_size, cfg.num_attention_heads,
            ),
        }));
    }
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;

    let dtype = map_torch_dtype(&cfg.torch_dtype);

    Ok(ModelMetadata {
        architecture: cfg.model_type,
        vocab_size: cfg.vocab_size,
        num_hidden_layers: cfg.num_hidden_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim,
        context_length: cfg.max_position_embeddings,
        rope_theta: cfg.rope_theta,
        rms_norm_eps: cfg.rms_norm_eps,
        dtype,
        // HF config.json doesn't carry a tokenizer model hint; tokenizer
        // discovery is the loader's responsibility (separate file, separate
        // path), so we leave this None here. Setting it from a sibling
        // tokenizer file is a future task.
        tokenizer_model_hint: None,
    })
}

/// Map a `torch_dtype` string (validated upstream against
/// `SUPPORTED_TORCH_DTYPES`) to the corresponding `core::DType`. Panics on
/// unknown input — callers MUST validate first. Kept private so the panic
/// surface stays inside this module.
fn map_torch_dtype(s: &str) -> DType {
    match s {
        "float32" => DType::F32,
        "float16" => DType::F16,
        "bfloat16" => DType::BF16,
        other => unreachable!(
            "map_torch_dtype called with unsupported value `{other}`; \
             SUPPORTED_TORCH_DTYPES gate must run first",
        ),
    }
}

// ---------------------------------------------------------------------------
// M2.6 design decision #2: SupportedDtype <-> core::DType conversion
// ---------------------------------------------------------------------------
//
// Loader-owned `SupportedDtype { F32, F16, BF16 }` and core-owned
// `DType { F32, F16, BF16, Q4, Q8 }` stay separate by design:
//   - `SupportedDtype` is the *artifact-read* surface (what we accept reading
//     from a safetensors header today).
//   - `core::DType` is the *compute* surface (what kernels can dispatch to,
//     including future quantized formats).
// They will diverge further as quantization lands. Until they do, every
// loader-discovered dtype maps cleanly into a core dtype, so a total
// `From<SupportedDtype> for DType` impl is the right bridge. Kept in the
// loader crate to preserve the inward-only crate dependency direction
// (core stays ignorant of loader's existence).

impl From<SupportedDtype> for DType {
    fn from(value: SupportedDtype) -> Self {
        match value {
            SupportedDtype::F32 => DType::F32,
            SupportedDtype::F16 => DType::F16,
            SupportedDtype::BF16 => DType::BF16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ocelotl_core::OcelotlError;
    use ocelotl_core::test_fixtures::metadata_fixture_path;

    #[test]
    fn load_metadata_rejects_unknown_architecture_with_typed_unsupported_error() {
        let path = metadata_fixture_path("unsupported_unknown_architecture.json");

        let err = load_metadata(&path)
            .expect_err("loading an unknown architecture must fail with a typed error");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(
                    unsupported.feature, "architecture",
                    "expected feature == \"architecture\", got {:?}",
                    unsupported.feature
                );
                assert_eq!(
                    unsupported.requested.as_deref(),
                    Some("unknown-transformer"),
                    "expected requested arch from fixture, got {:?}",
                    unsupported.requested
                );
                assert!(
                    unsupported.supported.iter().any(|s| s == "qwen2"),
                    "expected `qwen2` in supported list, got {:?}",
                    unsupported.supported
                );
            }
            other => {
                panic!("expected OcelotlError::Unsupported for unknown architecture, got {other:?}")
            }
        }
    }

    #[test]
    fn load_metadata_rejects_unknown_dtype_with_typed_unsupported_error() {
        let path = metadata_fixture_path("unsupported_dtype.json");

        let err = load_metadata(&path)
            .expect_err("loading an unknown dtype must fail with a typed error");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(
                    unsupported.feature, "dtype",
                    "expected feature == \"dtype\", got {:?}",
                    unsupported.feature
                );
                assert_eq!(
                    unsupported.requested.as_deref(),
                    Some("f8"),
                    "expected requested dtype from fixture, got {:?}",
                    unsupported.requested
                );
                assert!(
                    unsupported.supported.iter().any(|s| s == "f32"),
                    "expected `f32` in supported list, got {:?}",
                    unsupported.supported
                );
            }
            other => {
                panic!("expected OcelotlError::Unsupported for unknown dtype, got {other:?}")
            }
        }
    }

    #[test]
    fn load_metadata_rejects_missing_required_field_with_invalid_model_error() {
        let path = metadata_fixture_path("invalid_missing_vocab_size.json");

        let err =
            load_metadata(&path).expect_err("loading metadata missing a required field must fail");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(
                    invalid.path.as_deref(),
                    Some(path.as_path()),
                    "expected fixture path on the InvalidModel error, got {:?}",
                    invalid.path
                );
                assert_eq!(
                    invalid.field.as_deref(),
                    Some("vocab_size"),
                    "expected extracted field name == vocab_size, got {:?}",
                    invalid.field
                );
                assert!(
                    invalid.message.contains("vocab_size"),
                    "expected message to mention the missing field, got {:?}",
                    invalid.message
                );
            }
            other => {
                panic!("expected OcelotlError::InvalidModel for missing field, got {other:?}")
            }
        }
    }

    // -----------------------------------------------------------------------
    // M2.6 — parse_hf_config: real Qwen2.5 config.json -> ocelotl ModelMetadata
    // -----------------------------------------------------------------------

    #[test]
    fn parse_hf_config_maps_real_qwen2_5_config_into_model_metadata() {
        // Real config.json from Qwen/Qwen2.5-0.5B-Instruct at the M2.1-pinned
        // SHA 7ae557604adf67be50417f59c2c2f167def9a775. Field names differ
        // from the M1 synthetic fixture (max_position_embeddings vs
        // context_length, model_type vs architecture, torch_dtype vs dtype,
        // and head_dim is absent and must be derived).
        let path = metadata_fixture_path("qwen2_5_0_5b_instruct_config.json");
        let m = parse_hf_config(&path).expect("real Qwen2.5 config must parse");

        // architecture from model_type (NOT from architectures[0], which is
        // the python class name "Qwen2ForCausalLM").
        assert_eq!(m.architecture, "qwen2");
        assert_eq!(m.vocab_size, 151_936);
        assert_eq!(m.num_hidden_layers, 24);
        assert_eq!(m.hidden_size, 896);
        assert_eq!(m.intermediate_size, 4_864);
        assert_eq!(m.num_attention_heads, 14);
        assert_eq!(m.num_key_value_heads, 2);
        // head_dim is absent in HF config; derived as hidden_size /
        // num_attention_heads = 896 / 14 = 64. Pin both inputs and result so
        // a future refactor that breaks the derivation rule fails loudly.
        assert_eq!(m.head_dim, 64);
        // context_length comes from max_position_embeddings.
        assert_eq!(m.context_length, 32_768);
        assert!((m.rope_theta - 1_000_000.0_f64).abs() < 1e-6);
        assert!((m.rms_norm_eps - 1e-6_f64).abs() < 1e-12);
        // torch_dtype "bfloat16" maps to ocelotl-core DType::BF16.
        assert_eq!(m.dtype, ocelotl_core::DType::BF16);
        // tokenizer hint isn't carried in config.json; should be absent.
        assert_eq!(m.tokenizer_model_hint, None);
    }

    #[test]
    fn parse_hf_config_rejects_unknown_model_type_with_unsupported() {
        // Build a minimal config.json with an unknown model_type. The
        // contract: rejection happens *before* attempting the full parse so
        // callers get a typed `Unsupported` rather than an
        // `InvalidModel(missing field ...)` if the rest of the doc is
        // shaped differently.
        let mut path = std::env::temp_dir();
        path.push(format!(
            "ocelotl_m2_6_unknown_arch_{}.json",
            std::process::id()
        ));
        std::fs::write(
            &path,
            r#"{ "model_type": "definitely-not-real",
                 "vocab_size": 1, "hidden_size": 1, "intermediate_size": 1,
                 "num_hidden_layers": 1, "num_attention_heads": 1,
                 "num_key_value_heads": 1, "max_position_embeddings": 1,
                 "rope_theta": 1.0, "rms_norm_eps": 1e-6,
                 "torch_dtype": "float32" }"#,
        )
        .expect("write fixture");

        let err = parse_hf_config(&path).expect_err("unknown model_type must fail");
        match err {
            OcelotlError::Unsupported(u) => {
                assert_eq!(u.feature, "architecture");
                assert_eq!(u.requested.as_deref(), Some("definitely-not-real"));
                assert!(u.supported.iter().any(|s| s == "qwen2"));
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn parse_hf_config_rejects_unknown_torch_dtype_with_unsupported() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "ocelotl_m2_6_unknown_dtype_{}.json",
            std::process::id()
        ));
        std::fs::write(
            &path,
            r#"{ "model_type": "qwen2",
                 "vocab_size": 1, "hidden_size": 1, "intermediate_size": 1,
                 "num_hidden_layers": 1, "num_attention_heads": 1,
                 "num_key_value_heads": 1, "max_position_embeddings": 1,
                 "rope_theta": 1.0, "rms_norm_eps": 1e-6,
                 "torch_dtype": "float64" }"#,
        )
        .expect("write fixture");

        let err = parse_hf_config(&path).expect_err("unknown torch_dtype must fail");
        match err {
            OcelotlError::Unsupported(u) => {
                assert_eq!(u.feature, "torch_dtype");
                assert_eq!(u.requested.as_deref(), Some("float64"));
                assert!(u.supported.iter().any(|s| s == "float32"));
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn parse_hf_config_rejects_non_divisible_head_dim_with_invalid_model() {
        // head_dim is derived as hidden_size / num_attention_heads. If the
        // division has a remainder, the artifact is internally inconsistent
        // for the kind of attention we know how to run -- InvalidModel, not
        // Unsupported, because the artifact doesn't represent a coherent
        // model shape.
        let mut path = std::env::temp_dir();
        path.push(format!(
            "ocelotl_m2_6_bad_head_dim_{}.json",
            std::process::id()
        ));
        std::fs::write(
            &path,
            r#"{ "model_type": "qwen2",
                 "vocab_size": 1, "hidden_size": 100, "intermediate_size": 1,
                 "num_hidden_layers": 1, "num_attention_heads": 7,
                 "num_key_value_heads": 1, "max_position_embeddings": 1,
                 "rope_theta": 1.0, "rms_norm_eps": 1e-6,
                 "torch_dtype": "float32" }"#,
        )
        .expect("write fixture");

        let err = parse_hf_config(&path).expect_err("non-divisible head dim must fail");
        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("head_dim"));
                assert!(
                    invalid.message.contains("100") && invalid.message.contains('7'),
                    "expected derivation context in message, got {:?}",
                    invalid.message
                );
            }
            other => panic!("expected InvalidModel, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // M2.6 design decision #2: SupportedDtype <-> core::DType conversion
    // -----------------------------------------------------------------------

    #[test]
    fn supported_dtype_converts_into_core_dtype() {
        // The two enums stay separate (loader's SupportedDtype is the
        // artifact-read surface; core::DType is the compute surface that
        // also carries quantized variants), but conversion must be total
        // and lossless for the values that overlap.
        use ocelotl_core::DType;
        assert_eq!(DType::from(SupportedDtype::F32), DType::F32);
        assert_eq!(DType::from(SupportedDtype::F16), DType::F16);
        assert_eq!(DType::from(SupportedDtype::BF16), DType::BF16);
    }

    #[test]
    fn load_metadata_returns_io_error_when_file_does_not_exist() {
        // Per docs/design/errors.md, missing-file failures are `Io`, not
        // `InvalidModel`. This test pins that contract for `load_metadata`
        // alongside the matching test on `inspect_safetensors`.
        let mut path = std::env::temp_dir();
        path.push(format!("ocelotl_m2_6_missing_{}.json", std::process::id()));
        // Intentionally never create the file.

        let err = load_metadata(&path).expect_err("missing metadata file must fail");
        match err {
            OcelotlError::Io(io) => {
                assert_eq!(
                    io.path.as_deref(),
                    Some(path.as_path()),
                    "expected the missing path on the Io error, got {:?}",
                    io.path,
                );
                assert_eq!(
                    io.source.kind(),
                    std::io::ErrorKind::NotFound,
                    "expected NotFound, got {:?}",
                    io.source.kind(),
                );
            }
            other => panic!("expected OcelotlError::Io for missing file, got {other:?}"),
        }
    }

    #[test]
    fn extract_missing_field_returns_none_for_unrelated_message() {
        assert_eq!(extract_missing_field("some other error text"), None);
    }

    #[test]
    fn extract_missing_field_extracts_name_from_serde_message() {
        let message =
            "failed to parse metadata JSON: missing field `vocab_size` at line 1 column 50";
        assert_eq!(
            extract_missing_field(message).as_deref(),
            Some("vocab_size")
        );
    }
}
