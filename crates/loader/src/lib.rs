//! Model artifact loading and validation.

use std::path::Path;

use ocelotl_core::{InvalidModelError, ModelMetadata, OcelotlError, Result, UnsupportedError};
use serde::Deserialize;

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
    let json = std::fs::read_to_string(path).map_err(|source| {
        OcelotlError::from(InvalidModelError {
            path: Some(path.to_path_buf()),
            field: None,
            message: format!("failed to read metadata file: {source}"),
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
