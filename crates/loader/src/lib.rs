//! Model artifact loading and validation.

use std::path::{Path, PathBuf};

use ocelotl_core::{
    DType, InvalidModelError, ModelMetadata, OcelotlError, Result, UnsupportedError,
};
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
        OcelotlError::from(InvalidModelError {
            path: Some(path.to_path_buf()),
            field: None,
            message: format!("failed to parse metadata JSON: {source}"),
        })
    })?;
    Ok(full.model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ocelotl_core::OcelotlError;

    fn fixture_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/metadata")
            .join(name)
    }

    #[test]
    fn load_metadata_rejects_unknown_architecture_with_typed_unsupported_error() {
        let path = fixture_path("unsupported_unknown_architecture.json");

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
        let path = fixture_path("unsupported_dtype.json");

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
}

#[derive(Debug, Clone)]
pub struct ModelArtifact {
    pub path: PathBuf,
    pub metadata: ModelMetadata,
}

pub fn inspect_model(path: impl AsRef<Path>) -> Result<ModelArtifact> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(OcelotlError::from(InvalidModelError {
            path: Some(path.to_path_buf()),
            field: None,
            message: "path does not exist".to_string(),
        }));
    }

    Ok(ModelArtifact {
        path: path.to_path_buf(),
        metadata: ModelMetadata {
            architecture: "unknown".to_string(),
            vocab_size: 0,
            num_hidden_layers: 0,
            hidden_size: 0,
            intermediate_size: 0,
            num_attention_heads: 0,
            num_key_value_heads: 0,
            head_dim: 0,
            context_length: 0,
            rope_theta: 0.0,
            rms_norm_eps: 0.0,
            dtype: DType::F32,
            tokenizer_model_hint: None,
        },
    })
}
