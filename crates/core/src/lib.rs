//! Core types and contracts shared across Ocelotl crates.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, OcelotlError>;

// ---------------------------------------------------------------------------
// Top-level error enum
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum OcelotlError {
    #[error(transparent)]
    InvalidModel(#[from] InvalidModelError),
    #[error(transparent)]
    InvalidRequest(#[from] InvalidRequestError),
    #[error(transparent)]
    Unsupported(#[from] UnsupportedError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error(transparent)]
    Kernel(#[from] KernelError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Io(#[from] IoError),
}

// ---------------------------------------------------------------------------
// Error structs
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
#[error(
    "invalid model artifact{}{}: {message}",
    self.path.as_ref().map(|p| format!(" at `{}`", p.display())).unwrap_or_default(),
    self.field.as_deref().map(|f| format!(" (field `{f}`)")).unwrap_or_default(),
)]
pub struct InvalidModelError {
    pub path: Option<PathBuf>,
    pub field: Option<String>,
    pub message: String,
}

#[derive(Debug, Error)]
#[error("invalid request field `{field}`: {message}")]
pub struct InvalidRequestError {
    pub field: String,
    pub message: String,
}

#[derive(Debug, Error)]
#[error(
    "unsupported feature `{feature}`{}: supported = [{}]",
    self.requested.as_deref().map(|r| format!(" (requested `{r}`)")).unwrap_or_default(),
    self.supported.join(", ")
)]
pub struct UnsupportedError {
    pub feature: String,
    pub requested: Option<String>,
    pub supported: Vec<String>,
}

#[derive(Debug, Error)]
#[error("tokenizer error: {message}")]
pub struct TokenizerError {
    pub message: String,
    #[source]
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

#[derive(Debug, Error)]
#[error("kernel error on backend `{backend}`: {message}")]
pub struct KernelError {
    pub backend: String,
    pub message: String,
}

#[derive(Debug, Error)]
#[error("runtime error: {message}")]
pub struct RuntimeError {
    pub message: String,
}

#[derive(Debug, Error)]
#[error("IO error{}: {source}", self.path.as_ref().map(|p| format!(" on `{}`", p.display())).unwrap_or_default())]
pub struct IoError {
    pub path: Option<PathBuf>,
    #[source]
    pub source: std::io::Error,
}

impl From<std::io::Error> for IoError {
    fn from(source: std::io::Error) -> Self {
        Self { path: None, source }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    // --- UnsupportedError ---

    #[test]
    fn unsupported_error_display_mentions_feature_requested_and_supported() {
        let err = OcelotlError::Unsupported(UnsupportedError {
            feature: "rope_scaling".to_string(),
            requested: Some("yarn".to_string()),
            supported: vec!["linear".to_string(), "dynamic".to_string()],
        });

        let rendered = format!("{err}");

        assert!(
            rendered.contains("rope_scaling"),
            "expected feature name in display, got {rendered:?}"
        );
        assert!(
            rendered.contains("yarn"),
            "expected requested value in display, got {rendered:?}"
        );
        assert!(
            rendered.contains("linear") && rendered.contains("dynamic"),
            "expected supported list in display, got {rendered:?}"
        );
    }

    #[test]
    fn unsupported_error_can_be_constructed_with_no_requested_value() {
        let err = OcelotlError::Unsupported(UnsupportedError {
            feature: "paged_kv_cache".to_string(),
            requested: None,
            supported: vec!["contiguous".to_string()],
        });

        let rendered = format!("{err}");

        assert!(rendered.contains("paged_kv_cache"));
        assert!(rendered.contains("contiguous"));
        assert!(
            !rendered.contains("requested"),
            "should not mention requested when None, got {rendered:?}"
        );
    }

    // --- InvalidModelError ---

    #[test]
    fn invalid_model_error_display_includes_path_field_and_message() {
        let err = OcelotlError::InvalidModel(InvalidModelError {
            path: Some(PathBuf::from("/models/qwen.safetensors")),
            field: Some("num_hidden_layers".to_string()),
            message: "expected positive integer".to_string(),
        });

        let rendered = format!("{err}");

        assert!(
            rendered.contains("qwen.safetensors"),
            "expected path in display, got {rendered:?}"
        );
        assert!(
            rendered.contains("num_hidden_layers"),
            "expected field in display, got {rendered:?}"
        );
        assert!(
            rendered.contains("expected positive integer"),
            "expected message in display, got {rendered:?}"
        );
    }

    #[test]
    fn invalid_model_error_works_with_no_path_or_field() {
        let err = OcelotlError::InvalidModel(InvalidModelError {
            path: None,
            field: None,
            message: "malformed header".to_string(),
        });

        let rendered = format!("{err}");

        assert!(
            rendered.contains("malformed header"),
            "expected message in display, got {rendered:?}"
        );
    }

    // --- InvalidRequestError ---

    #[test]
    fn invalid_request_error_display_includes_field_and_message() {
        let err = OcelotlError::InvalidRequest(InvalidRequestError {
            field: "max_new_tokens".to_string(),
            message: "must be greater than zero".to_string(),
        });

        let rendered = format!("{err}");

        assert!(
            rendered.contains("max_new_tokens"),
            "expected field in display, got {rendered:?}"
        );
        assert!(
            rendered.contains("must be greater than zero"),
            "expected message in display, got {rendered:?}"
        );
    }

    // --- TokenizerError ---

    #[test]
    fn tokenizer_error_display_includes_message() {
        let err = OcelotlError::Tokenizer(TokenizerError {
            message: "unknown special token <|im_start|>".to_string(),
            source: None,
        });

        let rendered = format!("{err}");

        assert!(
            rendered.contains("unknown special token"),
            "expected message in display, got {rendered:?}"
        );
    }

    #[test]
    fn tokenizer_error_preserves_source_when_present() {
        let inner = std::io::Error::new(std::io::ErrorKind::InvalidData, "bad utf8");
        let err = TokenizerError {
            message: "failed to load tokenizer".to_string(),
            source: Some(Box::new(inner)),
        };

        assert!(err.source().is_some(), "expected source to be preserved");
    }

    // --- KernelError ---

    #[test]
    fn kernel_error_display_includes_backend_and_message() {
        let err = OcelotlError::Kernel(KernelError {
            backend: "cpu".to_string(),
            message: "unsupported stride layout".to_string(),
        });

        let rendered = format!("{err}");

        assert!(
            rendered.contains("cpu"),
            "expected backend in display, got {rendered:?}"
        );
        assert!(
            rendered.contains("unsupported stride layout"),
            "expected message in display, got {rendered:?}"
        );
    }

    // --- RuntimeError ---

    #[test]
    fn runtime_error_display_includes_message() {
        let err = OcelotlError::Runtime(RuntimeError {
            message: "KV cache allocation failed".to_string(),
        });

        let rendered = format!("{err}");

        assert!(
            rendered.contains("KV cache allocation failed"),
            "expected message in display, got {rendered:?}"
        );
    }

    // --- IoError ---

    #[test]
    fn io_error_display_includes_path() {
        let err = OcelotlError::Io(IoError {
            path: Some(PathBuf::from("/weights/model.safetensors")),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "file missing"),
        });

        let rendered = format!("{err}");

        assert!(
            rendered.contains("model.safetensors"),
            "expected path in display, got {rendered:?}"
        );
    }

    #[test]
    fn io_error_preserves_source() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let err = OcelotlError::Io(IoError {
            path: None,
            source: io_err,
        });

        assert!(
            err.source().is_some(),
            "expected io::Error to be reachable via source()"
        );
    }

    // --- M1.2 newtype tests ---

    #[test]
    fn token_id_wraps_and_unwraps_correctly() {
        let id = TokenId(1234);
        assert_eq!(id.0, 1234);
    }

    #[test]
    fn seq_len_wraps_and_unwraps_correctly() {
        let s = SeqLen(512);
        assert_eq!(s.0, 512);
    }

    #[test]
    fn batch_size_wraps_and_unwraps_correctly() {
        let b = BatchSize(8);
        assert_eq!(b.0, 8);
    }

    #[test]
    fn context_len_wraps_and_unwraps_correctly() {
        let c = ContextLen(4096);
        assert_eq!(c.0, 4096);
    }

    // Type safety is enforced at compile time — TokenId and SeqLen are distinct
    // types; passing one where the other is expected is a type error.

    // --- M1.3 metadata fixture test ---

    #[test]
    fn qwen2_5_tiny_synthetic_fixture_deserializes_correctly() {
        #[derive(Debug, serde::Deserialize)]
        struct MetadataFixture {
            model: ModelMetadata,
        }

        let fixture_path =
            crate::test_fixtures::metadata_fixture_path("qwen2_5_tiny_synthetic.json");

        let json = std::fs::read_to_string(&fixture_path)
            .expect("fixture file must be readable from workspace root");

        let fixture: MetadataFixture =
            serde_json::from_str(&json).expect("fixture must deserialize without error");

        let m = fixture.model;
        assert_eq!(m.architecture, "qwen2");
        assert_eq!(m.vocab_size, 32);
        assert_eq!(m.num_hidden_layers, 2);
        assert_eq!(m.hidden_size, 16);
        assert_eq!(m.intermediate_size, 32);
        assert_eq!(m.num_attention_heads, 4);
        assert_eq!(m.num_key_value_heads, 2);
        assert_eq!(m.head_dim, 4);
        assert_eq!(m.context_length, 128);
        assert_eq!(m.dtype, DType::F32);
        assert!((m.rope_theta - 1_000_000.0_f64).abs() < 1e-6);
        assert!((m.rms_norm_eps - 1e-6_f64).abs() < 1e-12);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Gpu { ordinal: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    F16,
    BF16,
    Q4,
    Q8,
}

// ---------------------------------------------------------------------------
// Domain newtypes (M1.2)
// ---------------------------------------------------------------------------

/// A vocabulary token identifier. Wraps `u32` to prevent mixing with other integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TokenId(pub u32);

/// Length of a sequence (number of tokens).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SeqLen(pub usize);

/// Number of concurrent requests in a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BatchSize(pub usize);

/// Maximum context length supported by a model or requested by a caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContextLen(pub usize);

// ---------------------------------------------------------------------------
// Model metadata (M1.3)
// ---------------------------------------------------------------------------

/// Full model configuration as loaded from a metadata artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub architecture: String,
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
    pub dtype: DType,
    #[serde(default)]
    pub tokenizer_model_hint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationOptions {
    pub max_new_tokens: usize,
    pub temperature: Option<f32>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Test-fixture helpers (M1.5)
// ---------------------------------------------------------------------------

/// Test-only helpers shared across crates. Gated behind the `test-fixtures`
/// feature so the helpers stay out of the production API surface.
///
/// Enable from another crate's `[dev-dependencies]`:
/// ```toml
/// ocelotl-core = { workspace = true, features = ["test-fixtures"] }
/// ```
#[cfg(any(test, feature = "test-fixtures"))]
pub mod test_fixtures {
    use std::path::PathBuf;

    /// Resolve a fixture under `<repo>/fixtures/metadata/` by name.
    ///
    /// Both `ocelotl-core` and `ocelotl-loader` live at `crates/<name>/` and
    /// therefore share the same `../../fixtures/metadata/` relative path from
    /// their `CARGO_MANIFEST_DIR`. `CARGO_MANIFEST_DIR` here resolves to the
    /// `ocelotl-core` crate, but the absolute target path is identical for
    /// any sibling crate at the same depth.
    pub fn metadata_fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/metadata")
            .join(name)
    }
}
