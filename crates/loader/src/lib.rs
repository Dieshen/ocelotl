//! Model artifact loading and validation.

use std::path::{Path, PathBuf};

use ocelotl_core::{DType, ModelInfo, OcelotlError, Result};

#[derive(Debug, Clone)]
pub struct ModelArtifact {
    pub path: PathBuf,
    pub info: ModelInfo,
}

pub fn inspect_model(path: impl AsRef<Path>) -> Result<ModelArtifact> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(OcelotlError::InvalidModel(format!(
            "model path does not exist: {}",
            path.display()
        )));
    }

    Ok(ModelArtifact {
        path: path.to_path_buf(),
        info: ModelInfo {
            architecture: "unknown".to_string(),
            parameter_count: None,
            context_length: 0,
            dtype: DType::F32,
        },
    })
}
