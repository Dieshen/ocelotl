//! Portable kernel dispatch boundary.

use ocelotl_core::{Device, OcelotlError, Result, UnsupportedError};

#[derive(Debug, Clone)]
pub struct KernelContext {
    pub device: Device,
}

pub trait KernelBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn context(&self) -> &KernelContext;
}

#[derive(Debug, Clone)]
pub struct CpuKernelBackend {
    context: KernelContext,
}

impl Default for CpuKernelBackend {
    fn default() -> Self {
        Self {
            context: KernelContext {
                device: Device::Cpu,
            },
        }
    }
}

impl KernelBackend for CpuKernelBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn context(&self) -> &KernelContext {
        &self.context
    }
}

pub fn require_gpu(backend: &dyn KernelBackend) -> Result<()> {
    match backend.context().device {
        Device::Gpu { .. } => Ok(()),
        Device::Cpu => Err(OcelotlError::Unsupported(UnsupportedError {
            feature: "gpu_backend".to_string(),
            requested: Some("gpu".to_string()),
            supported: vec!["cpu".to_string()],
        })),
    }
}
