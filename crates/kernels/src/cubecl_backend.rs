//! CubeCL backend spike.
//!
//! This module is intentionally narrow: it proves that an Ocelotl-owned kernel
//! contract can cross into CubeCL without changing the CPU reference path. RoPE
//! is the first spike because it exercises layout-sensitive indexing while
//! avoiding a premature GEMM or attention-library decision.

use std::mem::size_of_val;

use cubecl::prelude::*;
use ocelotl_core::{DType, Device, KernelError, OcelotlError, Result};

use crate::rope::{rope_trig_tables, validate_rope_shape};
use crate::{KernelBackend, KernelContext};

const CUBECL_BACKEND: &str = "cubecl";

/// Layout contract for the CubeCL RoPE spike.
///
/// M4 intentionally keeps device buffers out of the public model/runtime API,
/// but the backend still needs an explicit shape/dtype/stride contract before
/// launch. The first supported layout is contiguous f32 rows:
/// `[num_heads, head_dim]`, row-major, with `row_stride == head_dim`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CubeClRopeLayout {
    pub dtype: DType,
    pub head_dim: usize,
    pub row_stride: usize,
}

impl CubeClRopeLayout {
    pub fn contiguous_f32(head_dim: usize) -> Self {
        Self {
            dtype: DType::F32,
            head_dim,
            row_stride: head_dim,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CubeClKernelBackend {
    context: KernelContext,
}

impl CubeClKernelBackend {
    pub fn new_gpu(ordinal: usize) -> Self {
        Self {
            context: KernelContext {
                device: Device::Gpu { ordinal },
            },
        }
    }

    #[cfg(feature = "cubecl-wgpu")]
    pub fn rope_apply_inplace(
        &self,
        x: &mut [f32],
        head_dim: usize,
        position: usize,
        theta: f32,
    ) -> Result<()> {
        rope_apply_inplace_wgpu(x, head_dim, position, theta)
    }
}

impl KernelBackend for CubeClKernelBackend {
    fn name(&self) -> &'static str {
        CUBECL_BACKEND
    }

    fn context(&self) -> &KernelContext {
        &self.context
    }

    fn matmul(
        &self,
        a: &[f32],
        a_shape: (usize, usize),
        b: &[f32],
        b_shape: (usize, usize),
        out: &mut [f32],
    ) -> Result<()> {
        crate::matmul(a, a_shape, b, b_shape, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_out_by_in(
        &self,
        x: &[f32],
        rows: usize,
        in_features: usize,
        weight_out_by_in: &[f32],
        out_features: usize,
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        crate::linear_out_by_in(
            x,
            rows,
            in_features,
            weight_out_by_in,
            out_features,
            bias,
            out,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn scaled_dot_product_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        out: &mut [f32],
    ) -> Result<()> {
        crate::attention::scaled_dot_product_attention(
            q,
            k,
            v,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            out,
        )
    }

    fn rope_apply_inplace(
        &self,
        x: &mut [f32],
        head_dim: usize,
        position: usize,
        theta: f32,
    ) -> Result<()> {
        #[cfg(feature = "cubecl-wgpu")]
        {
            return rope_apply_inplace_wgpu(x, head_dim, position, theta);
        }

        #[cfg(not(feature = "cubecl-wgpu"))]
        {
            crate::rope_apply_inplace(x, head_dim, position, theta)
        }
    }

    fn rmsnorm(
        &self,
        x: &[f32],
        rows: usize,
        hidden: usize,
        weight: &[f32],
        epsilon: f32,
        out: &mut [f32],
    ) -> Result<()> {
        crate::rmsnorm::rmsnorm(x, rows, hidden, weight, epsilon, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn mlp_gated_silu(
        &self,
        x: &[f32],
        rows: usize,
        hidden: usize,
        intermediate: usize,
        gate_w: &[f32],
        up_w: &[f32],
        down_w: &[f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        out: &mut [f32],
    ) -> Result<()> {
        crate::mlp::mlp_gated_silu(
            x,
            rows,
            hidden,
            intermediate,
            gate_w,
            up_w,
            down_w,
            gate_buf,
            up_buf,
            out,
        )
    }

    fn vec_add(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        crate::vec_add(a, b, out)
    }
}

/// Apply RoPE through CubeCL.
///
/// This is a portability spike, not a performance path. It copies one CPU
/// slice to the selected CubeCL runtime, launches a simple f32 kernel, then
/// copies the result back. The CPU RoPE kernel remains the parity oracle.
pub fn rope_apply_inplace_cubecl<R: Runtime>(
    device: &R::Device,
    x: &mut [f32],
    head_dim: usize,
    position: usize,
    theta: f32,
) -> Result<()> {
    rope_apply_inplace_cubecl_with_layout::<R>(
        device,
        x,
        CubeClRopeLayout::contiguous_f32(head_dim),
        position,
        theta,
    )
}

/// Apply RoPE through CubeCL with an explicit layout contract.
///
/// Unsupported dtype and non-contiguous row stride are rejected before the
/// CubeCL runtime client is requested, so invalid layouts fail even on hosts
/// without a working WGPU device.
pub fn rope_apply_inplace_cubecl_with_layout<R: Runtime>(
    device: &R::Device,
    x: &mut [f32],
    layout: CubeClRopeLayout,
    position: usize,
    theta: f32,
) -> Result<()> {
    if layout.dtype != DType::F32 {
        return Err(cubecl_err(format!(
            "CubeCL RoPE supports only F32 input, got {:?}",
            layout.dtype
        )));
    }
    if layout.row_stride != layout.head_dim {
        return Err(cubecl_err(format!(
            "CubeCL RoPE requires contiguous rows (row_stride == head_dim), got row_stride {} for head_dim {}",
            layout.row_stride, layout.head_dim
        )));
    }

    let head_dim = layout.head_dim;
    let _half = validate_rope_shape(CUBECL_BACKEND, x.len(), head_dim)?;
    u32::try_from(head_dim).map_err(|_| {
        cubecl_err(format!(
            "head_dim {head_dim} exceeds CubeCL u32 launch limit"
        ))
    })?;
    let pair_count = u32::try_from(x.len() / 2).map_err(|_| {
        cubecl_err(format!(
            "rope pair count {} exceeds CubeCL u32 launch limit",
            x.len() / 2
        ))
    })?;

    let (cos, sin) = rope_trig_tables(head_dim, position, theta);

    let client = R::client(device);
    let input = client.create_from_slice(f32::as_bytes(x));
    let cos = client.create_from_slice(f32::as_bytes(&cos));
    let sin = client.create_from_slice(f32::as_bytes(&sin));
    let output = client.empty(size_of_val(x));

    unsafe {
        rope_apply_f32::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(pair_count),
            ArrayArg::from_raw_parts(input, x.len()),
            ArrayArg::from_raw_parts(cos, head_dim / 2),
            ArrayArg::from_raw_parts(sin, head_dim / 2),
            ArrayArg::from_raw_parts(output.clone(), x.len()),
            head_dim,
        );
    }

    let bytes = client
        .read_one(output)
        .map_err(|err| cubecl_err(format!("failed to read CubeCL RoPE output: {err:?}")))?;
    let output = f32::from_bytes(&bytes);
    if output.len() != x.len() {
        return Err(cubecl_err(format!(
            "CubeCL RoPE output length {} did not match input length {}",
            output.len(),
            x.len()
        )));
    }
    x.copy_from_slice(output);

    Ok(())
}

#[cfg(feature = "cubecl-wgpu")]
pub fn rope_apply_inplace_wgpu(
    x: &mut [f32],
    head_dim: usize,
    position: usize,
    theta: f32,
) -> Result<()> {
    rope_apply_inplace_cubecl::<cubecl::wgpu::WgpuRuntime>(
        &Default::default(),
        x,
        head_dim,
        position,
        theta,
    )
}

#[cube(launch_unchecked)]
fn rope_apply_f32(
    input: &Array<f32>,
    cos: &Array<f32>,
    sin: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] head_dim: usize,
) {
    let half = head_dim / 2;
    let pair = ABSOLUTE_POS;
    let head = pair / half;
    let i = pair - head * half;
    let base = head * head_dim;
    let lo = base + i;
    let hi = lo + half;

    let x_lo = input[lo];
    let x_hi = input[hi];
    let c = cos[i];
    let s = sin[i];

    output[lo] = x_lo * c - x_hi * s;
    output[hi] = x_lo * s + x_hi * c;
}

fn cubecl_err(message: impl Into<String>) -> OcelotlError {
    OcelotlError::Kernel(KernelError {
        backend: CUBECL_BACKEND.to_string(),
        message: message.into(),
    })
}

#[cfg(test)]
mod tests {
    use crate::require_gpu;
    #[cfg(feature = "cubecl-wgpu")]
    use crate::rope_apply_inplace;

    use super::*;

    #[test]
    fn cubecl_backend_advertises_gpu_device() {
        let backend = CubeClKernelBackend::new_gpu(0);

        assert_eq!(backend.name(), "cubecl");
        assert_eq!(backend.context().device, Device::Gpu { ordinal: 0 });
        require_gpu(&backend).unwrap();
    }

    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn wgpu_rope_rejects_invalid_shape_before_launch() {
        let mut actual = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let err = rope_apply_inplace_wgpu(&mut actual, 3, 1, 10_000.0)
            .expect_err("odd head_dim must be rejected before CubeCL launch");

        match err {
            OcelotlError::Kernel(KernelError { backend, message }) => {
                assert_eq!(backend, "cubecl");
                assert!(
                    message.contains("head_dim must be even"),
                    "unexpected diagnostic: {message}"
                );
            }
            other => panic!("expected CubeCL KernelError, got {other:?}"),
        }
    }

    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn wgpu_rope_rejects_non_f32_dtype_before_launch() {
        let mut actual = [1.0_f32, 2.0, 3.0, 4.0];
        let layout = CubeClRopeLayout {
            dtype: DType::BF16,
            head_dim: 4,
            row_stride: 4,
        };

        let err = rope_apply_inplace_cubecl_with_layout::<cubecl::wgpu::WgpuRuntime>(
            &Default::default(),
            &mut actual,
            layout,
            1,
            10_000.0,
        )
        .expect_err("unsupported dtype must be rejected before CubeCL launch");

        match err {
            OcelotlError::Kernel(KernelError { backend, message }) => {
                assert_eq!(backend, "cubecl");
                assert!(message.contains("F32"), "unexpected diagnostic: {message}");
            }
            other => panic!("expected CubeCL KernelError, got {other:?}"),
        }
    }

    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn wgpu_rope_rejects_non_contiguous_stride_before_launch() {
        let mut actual = [1.0_f32, 2.0, 3.0, 4.0];
        let layout = CubeClRopeLayout {
            dtype: DType::F32,
            head_dim: 4,
            row_stride: 8,
        };

        let err = rope_apply_inplace_cubecl_with_layout::<cubecl::wgpu::WgpuRuntime>(
            &Default::default(),
            &mut actual,
            layout,
            1,
            10_000.0,
        )
        .expect_err("non-contiguous layout must be rejected before CubeCL launch");

        match err {
            OcelotlError::Kernel(KernelError { backend, message }) => {
                assert_eq!(backend, "cubecl");
                assert!(
                    message.contains("contiguous"),
                    "unexpected diagnostic: {message}"
                );
            }
            other => panic!("expected CubeCL KernelError, got {other:?}"),
        }
    }

    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    #[ignore = "requires a CubeCL WGPU-capable local runtime"]
    fn wgpu_rope_matches_cpu_reference_for_position_one() {
        let mut expected = [1.0_f32, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        let mut actual = expected;

        rope_apply_inplace(&mut expected, 4, 1, 10_000.0).unwrap();
        rope_apply_inplace_wgpu(&mut actual, 4, 1, 10_000.0).unwrap();

        for (idx, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1.0e-5,
                "idx {idx}: got {got}, want {want}"
            );
        }
    }
}
