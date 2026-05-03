//! Core types and contracts shared across Ocelotl crates.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, OcelotlError>;

#[derive(Debug, Error)]
pub enum OcelotlError {
    #[error("invalid model artifact: {0}")]
    InvalidModel(String),
    #[error(transparent)]
    Unsupported(#[from] UnsupportedError),
    #[error("runtime error: {0}")]
    Runtime(String),
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

#[cfg(test)]
mod tests {
    use super::*;

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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Gpu { ordinal: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q4,
    Q8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub architecture: String,
    pub parameter_count: Option<u64>,
    pub context_length: usize,
    pub dtype: DType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationOptions {
    pub max_new_tokens: usize,
    pub temperature: Option<u32>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: None,
        }
    }
}
