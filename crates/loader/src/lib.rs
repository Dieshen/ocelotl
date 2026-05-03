//! Model artifact loading and validation.

use std::path::{Path, PathBuf};

use ocelotl_core::{DType, InvalidModelError, ModelMetadata, OcelotlError, Result};

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
