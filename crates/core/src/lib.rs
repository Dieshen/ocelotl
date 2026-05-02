//! Core types and contracts shared across Ocelotl crates.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, OcelotlError>;

#[derive(Debug, Error)]
pub enum OcelotlError {
    #[error("invalid model artifact: {0}")]
    InvalidModel(String),
    #[error("unsupported feature: {0}")]
    Unsupported(String),
    #[error("runtime error: {0}")]
    Runtime(String),
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
