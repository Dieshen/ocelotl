//! Request lifecycle and generation runtime.

use ocelotl_core::{GenerationOptions, OcelotlError, Result, UnsupportedError};
use ocelotl_kernels::{CpuKernelBackend, KernelBackend};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub options: GenerationOptions,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub text: String,
}

pub struct Runtime<B: KernelBackend = CpuKernelBackend> {
    backend: B,
}

impl Runtime<CpuKernelBackend> {
    pub fn cpu() -> Self {
        Self {
            backend: CpuKernelBackend::default(),
        }
    }
}

impl<B: KernelBackend> Runtime<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn generate(&self, _request: GenerateRequest) -> Result<GenerateResponse> {
        Err(OcelotlError::Unsupported(UnsupportedError {
            feature: "generate".to_string(),
            requested: None,
            supported: vec![],
        }))
    }
}
