//! CubeCL backend spike.
//!
//! This module is intentionally narrow: it proves that an Ocelotl-owned kernel
//! contract can cross into CubeCL without changing the CPU reference path. RoPE
//! is the first spike because it exercises layout-sensitive indexing while
//! avoiding a premature GEMM or attention-library decision.

use std::mem::size_of_val;
#[cfg(feature = "cubecl-wgpu")]
use std::sync::Mutex;

use cubecl::prelude::*;
use ocelotl_core::{DType, Device, KernelError, OcelotlError, Result};

use crate::rope::{rope_trig_tables, validate_rope_shape};
#[cfg(feature = "cubecl-wgpu")]
use crate::tensor::{DeviceBuffer, DeviceTensor};
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
        #[cfg(feature = "cubecl-wgpu")]
        {
            linear_out_by_in_wgpu(
                x,
                rows,
                in_features,
                weight_out_by_in,
                out_features,
                bias,
                out,
            )
        }

        #[cfg(not(feature = "cubecl-wgpu"))]
        {
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
            rope_apply_inplace_wgpu(x, head_dim, position, theta)
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

    fn upload(&self, host: &[f32]) -> Result<DeviceTensor> {
        #[cfg(feature = "cubecl-wgpu")]
        {
            self.upload_wgpu(host)
        }
        #[cfg(not(feature = "cubecl-wgpu"))]
        {
            Ok(DeviceTensor::from_host(host.to_vec()))
        }
    }

    fn alloc(&self, len: usize) -> Result<DeviceTensor> {
        #[cfg(feature = "cubecl-wgpu")]
        {
            self.alloc_wgpu(len)
        }
        #[cfg(not(feature = "cubecl-wgpu"))]
        {
            Ok(DeviceTensor::host_zeros(len))
        }
    }

    /// Device-resident linear projection. When every operand is already a
    /// `WgpuDeviceBuffer`, launch the cube kernel against the existing
    /// handles — no `create_from_slice`, no `read_one`. The output buffer
    /// is updated in place by swapping its handle to the kernel's output
    /// handle (same swap pattern as `WgpuDeviceBuffer::write_from_host`).
    /// If any operand isn't `cubecl-wgpu` device-resident, fall back to
    /// the trait default so the slow-but-correct host bounce still works.
    #[allow(clippy::too_many_arguments)]
    fn linear_d(
        &self,
        x: &DeviceTensor,
        rows: usize,
        in_features: usize,
        weight: &DeviceTensor,
        out_features: usize,
        bias: Option<&DeviceTensor>,
        out: &DeviceTensor,
    ) -> Result<()> {
        #[cfg(feature = "cubecl-wgpu")]
        {
            if let Some((x_buf, w_buf, bias_buf, out_buf)) =
                Self::try_extract_wgpu_operands(x, weight, bias, out)
            {
                return run_linear_d_wgpu(
                    x_buf,
                    rows,
                    in_features,
                    w_buf,
                    out_features,
                    bias_buf,
                    out_buf,
                );
            }
        }
        // Fall through to the default impl, which forces host readback
        // and uploads. This is the rare/wrong-residency path — Stage 2
        // is what makes it cold.
        let x_host = x.to_host_owned()?;
        let weight_host = weight.to_host_owned()?;
        let bias_host = bias.map(DeviceTensor::to_host_owned).transpose()?;
        let mut out_buf = vec![0.0_f32; rows * out_features];
        KernelBackend::linear_out_by_in(
            self,
            &x_host,
            rows,
            in_features,
            &weight_host,
            out_features,
            bias_host.as_deref(),
            &mut out_buf,
        )?;
        out.write_from_host_slice(&out_buf)
    }
}

/// Run a fully device-resident `linear_out_by_in` on the WGPU runtime.
///
/// All four operands already own cubecl handles on the same client. We
/// derive a dummy bias handle when `bias_buf` is `None` (the kernel takes
/// a length-1 dummy and a `has_bias` comptime flag, matching the
/// slice-based path). The kernel's output handle is then swapped into
/// `out_buf`, so subsequent device-resident reads pick up the result
/// without a host bounce.
#[cfg(feature = "cubecl-wgpu")]
#[allow(clippy::too_many_arguments)]
fn run_linear_d_wgpu(
    x_buf: &WgpuDeviceBuffer,
    rows: usize,
    in_features: usize,
    w_buf: &WgpuDeviceBuffer,
    out_features: usize,
    bias_buf: Option<&WgpuDeviceBuffer>,
    out_buf: &WgpuDeviceBuffer,
) -> Result<()> {
    // Length-level invariants. We can't call `validate_linear_out_by_in`
    // here because it wants `&[f32]`; replicate its checks against the
    // device buffer lengths.
    let x_expected = rows
        .checked_mul(in_features)
        .ok_or_else(|| cubecl_wgpu_err("linear_d rows*in_features overflowed usize"))?;
    let w_expected = out_features
        .checked_mul(in_features)
        .ok_or_else(|| cubecl_wgpu_err("linear_d out_features*in_features overflowed usize"))?;
    let out_expected = rows
        .checked_mul(out_features)
        .ok_or_else(|| cubecl_wgpu_err("linear_d rows*out_features overflowed usize"))?;
    if x_buf.len_f32() != x_expected {
        return Err(cubecl_wgpu_err(format!(
            "linear_d x len {} != rows*in_features {}",
            x_buf.len_f32(),
            x_expected
        )));
    }
    if w_buf.len_f32() != w_expected {
        return Err(cubecl_wgpu_err(format!(
            "linear_d weight len {} != out_features*in_features {}",
            w_buf.len_f32(),
            w_expected
        )));
    }
    if out_buf.len_f32() != out_expected {
        return Err(cubecl_wgpu_err(format!(
            "linear_d out len {} != rows*out_features {}",
            out_buf.len_f32(),
            out_expected
        )));
    }
    if let Some(b) = bias_buf {
        if b.len_f32() != out_features {
            return Err(cubecl_wgpu_err(format!(
                "linear_d bias len {} != out_features {}",
                b.len_f32(),
                out_features
            )));
        }
    }

    let client = x_buf.client();
    let (bias_handle, bias_len, has_bias) = match bias_buf {
        Some(b) => (b.clone_handle(), b.len_f32(), true),
        None => {
            let dummy = [0.0_f32];
            (client.create_from_slice(f32::as_bytes(&dummy)), 1, false)
        }
    };

    // The fully-on-device hot path: launch into a fresh output handle,
    // then swap it into the caller-supplied `out_buf`. We could also try
    // to launch directly against `out_buf.clone_handle()`, but the kernel
    // reads from `x`/`weight` and writes to `output` — running them in
    // parallel through the same client is safe because cubecl serialises
    // submissions on the stream. We still issue a fresh allocation here
    // to keep clear separation between "the buffer the caller still owns"
    // and "the buffer we just wrote into".
    let output_handle = client.empty(out_buf.len_f32() * std::mem::size_of::<f32>());

    launch_linear_out_by_in_kernel::<cubecl::wgpu::WgpuRuntime>(
        client,
        x_buf.clone_handle(),
        x_buf.len_f32(),
        w_buf.clone_handle(),
        w_buf.len_f32(),
        bias_handle,
        bias_len,
        has_bias,
        output_handle.clone(),
        out_buf.len_f32(),
        rows,
        in_features,
        out_features,
    )?;

    // Swap the freshly-written handle into `out_buf`. The previous handle
    // (which the kernel did not write to) is dropped, freeing storage.
    *out_buf
        .handle
        .lock()
        .expect("WgpuDeviceBuffer handle mutex poisoned") = output_handle;
    Ok(())
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

/// Compute a single `[row, out_dim]` cell of `output = x * weight^T + bias`.
///
/// `weight` is row-major `[out_features, in_features]` matching the CPU
/// `linear_out_by_in` contract. `bias` is `[out_features]` when `has_bias`
/// and a length-1 dummy buffer (ignored) when `has_bias == false`. Each
/// invocation maps `ABSOLUTE_POS` to one `(row, out_dim)` cell so a launch
/// of `rows * out_features` threads covers the full output.
#[cube(launch_unchecked)]
fn linear_out_by_in_f32(
    x: &Array<f32>,
    weight: &Array<f32>,
    bias: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] in_features: usize,
    #[comptime] out_features: usize,
    #[comptime] has_bias: bool,
) {
    let cell = ABSOLUTE_POS;
    // `ABSOLUTE_POS` can run past the real output length because we round the
    // launch grid up to whole workgroups. Bail out for those tail threads.
    if cell >= output.len() {
        terminate!();
    }

    let row = cell / out_features;
    let out_dim = cell - row * out_features;

    let mut acc = f32::new(0.0);
    if has_bias {
        acc = bias[out_dim];
    }

    let x_base = row * in_features;
    let w_base = out_dim * in_features;
    for k in 0..in_features {
        acc += x[x_base + k] * weight[w_base + k];
    }
    output[row * out_features + out_dim] = acc;
}

/// Derive the workgroup grid for a `linear_out_by_in_f32` launch and
/// validate shape invariants that the kernel itself relies on (u32 indexing,
/// non-zero cell count). Caller is expected to have already validated buffer
/// lengths via `crate::validate_linear_out_by_in`.
fn prepare_linear_launch(
    rows: usize,
    in_features: usize,
    out_features: usize,
) -> Result<(u32, u32)> {
    u32::try_from(in_features).map_err(|_| {
        cubecl_err(format!(
            "in_features {in_features} exceeds CubeCL u32 launch limit"
        ))
    })?;
    u32::try_from(out_features).map_err(|_| {
        cubecl_err(format!(
            "out_features {out_features} exceeds CubeCL u32 launch limit"
        ))
    })?;
    let total_cells = u32::try_from(rows * out_features).map_err(|_| {
        cubecl_err(format!(
            "linear_out_by_in cell count {} exceeds CubeCL u32 launch limit",
            rows * out_features
        ))
    })?;
    // WGPU caps a single workgroup at 256 invocations on most adapters, so
    // a single-workgroup launch (`CubeCount::Static(1, 1, 1)`) only computes
    // the first 256 output cells of any larger problem. Spread the work
    // across multiple workgroups using a fixed 1-D CubeDim. The kernel
    // includes a bounds check for the rounded-up tail.
    const WORKGROUP_SIZE: u32 = 256;
    let workgroup_count = total_cells.div_ceil(WORKGROUP_SIZE).max(1);
    Ok((workgroup_count, WORKGROUP_SIZE))
}

/// Launch the bare `linear_out_by_in_f32` kernel against pre-existing CubeCL
/// handles. This is the device-resident entry point: no `create_from_slice`,
/// no `read_one`, no host bounce. Both the slice-based legacy
/// `linear_out_by_in_cubecl` (which still does the host round trip) and the
/// `DeviceTensor`-based `linear_d` override call into this helper.
#[allow(clippy::too_many_arguments)]
fn launch_linear_out_by_in_kernel<R: Runtime>(
    client: &ComputeClient<R>,
    x_handle: cubecl::server::Handle,
    x_len: usize,
    weight_handle: cubecl::server::Handle,
    weight_len: usize,
    bias_handle: cubecl::server::Handle,
    bias_len: usize,
    has_bias: bool,
    output_handle: cubecl::server::Handle,
    out_len: usize,
    rows: usize,
    in_features: usize,
    out_features: usize,
) -> Result<()> {
    let (workgroup_count, workgroup_size) =
        prepare_linear_launch(rows, in_features, out_features)?;

    unsafe {
        linear_out_by_in_f32::launch_unchecked::<R>(
            client,
            CubeCount::Static(workgroup_count, 1, 1),
            CubeDim::new_1d(workgroup_size),
            ArrayArg::from_raw_parts(x_handle, x_len),
            ArrayArg::from_raw_parts(weight_handle, weight_len),
            ArrayArg::from_raw_parts(bias_handle, bias_len),
            ArrayArg::from_raw_parts(output_handle, out_len),
            in_features,
            out_features,
            has_bias,
        );
    }
    Ok(())
}

/// Launch `linear_out_by_in_f32` through a CubeCL runtime.
///
/// The CPU reference (`crate::linear_out_by_in`) remains the parity oracle.
/// Buffers cross the runtime boundary as contiguous row-major f32 slices,
/// matching the M1 layout contract — no strides, no quantized weights.
///
/// This is the slice-based legacy entry: it uploads inputs, launches via
/// [`launch_linear_out_by_in_kernel`], then reads the output back. The
/// `DeviceTensor`-based `CubeClKernelBackend::linear_d` override skips the
/// upload/readback when the caller already holds device-resident handles.
#[allow(clippy::too_many_arguments)]
pub fn linear_out_by_in_cubecl<R: Runtime>(
    device: &R::Device,
    x: &[f32],
    rows: usize,
    in_features: usize,
    weight_out_by_in: &[f32],
    out_features: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> Result<()> {
    crate::validate_linear_out_by_in(
        x,
        rows,
        in_features,
        weight_out_by_in,
        out_features,
        bias,
        out,
    )?;

    let client = R::client(device);
    let x_handle = client.create_from_slice(f32::as_bytes(x));
    let weight_handle = client.create_from_slice(f32::as_bytes(weight_out_by_in));
    let (bias_handle, bias_len, has_bias) = match bias {
        Some(b) => (client.create_from_slice(f32::as_bytes(b)), b.len(), true),
        None => {
            let dummy = [0.0_f32];
            (client.create_from_slice(f32::as_bytes(&dummy)), 1, false)
        }
    };
    let output_handle = client.empty(size_of_val(out));

    launch_linear_out_by_in_kernel::<R>(
        &client,
        x_handle,
        x.len(),
        weight_handle,
        weight_out_by_in.len(),
        bias_handle,
        bias_len,
        has_bias,
        output_handle.clone(),
        out.len(),
        rows,
        in_features,
        out_features,
    )?;

    let bytes = client.read_one(output_handle).map_err(|err| {
        cubecl_err(format!(
            "failed to read CubeCL linear_out_by_in output: {err:?}"
        ))
    })?;
    let read = f32::from_bytes(&bytes);
    if read.len() != out.len() {
        return Err(cubecl_err(format!(
            "CubeCL linear_out_by_in output length {} did not match expected {}",
            read.len(),
            out.len()
        )));
    }
    out.copy_from_slice(read);

    Ok(())
}

#[cfg(feature = "cubecl-wgpu")]
#[allow(clippy::too_many_arguments)]
pub fn linear_out_by_in_wgpu(
    x: &[f32],
    rows: usize,
    in_features: usize,
    weight_out_by_in: &[f32],
    out_features: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> Result<()> {
    linear_out_by_in_cubecl::<cubecl::wgpu::WgpuRuntime>(
        &Default::default(),
        x,
        rows,
        in_features,
        weight_out_by_in,
        out_features,
        bias,
        out,
    )
}

fn cubecl_err(message: impl Into<String>) -> OcelotlError {
    OcelotlError::Kernel(KernelError {
        backend: CUBECL_BACKEND.to_string(),
        message: message.into(),
    })
}

#[cfg(feature = "cubecl-wgpu")]
fn cubecl_wgpu_err(message: impl Into<String>) -> OcelotlError {
    OcelotlError::Kernel(KernelError {
        backend: CUBECL_WGPU_BACKEND.to_string(),
        message: message.into(),
    })
}

/// Stable backend id for `WgpuDeviceBuffer`. Distinct from the generic
/// `"cubecl"` name so future CUDA/HIP buffers can share the trait without
/// being mistaken for each other in `linear_d` downcasts.
#[cfg(feature = "cubecl-wgpu")]
pub const CUBECL_WGPU_BACKEND: &str = "cubecl-wgpu";

/// A f32 buffer that lives on a CubeCL WGPU runtime. The `client` is the
/// `ComputeClient<WgpuRuntime>` that owns the underlying memory; the
/// `handle` is the cubecl `Handle` (Clone is cheap — it's an Arc bump).
/// The handle sits behind a `Mutex` because `write_from_host` has to swap
/// it (cubecl 0.10 exposes no public "write into existing handle" path on
/// `ComputeClient`, so we recreate the handle via `create_from_slice`).
#[cfg(feature = "cubecl-wgpu")]
pub struct WgpuDeviceBuffer {
    client: ComputeClient<cubecl::wgpu::WgpuRuntime>,
    handle: Mutex<cubecl::server::Handle>,
    len_f32: usize,
}

// `ComputeClient` doesn't derive `Debug`. Hand-roll a Debug that just
// shows the residency-relevant fields so `DeviceTensor`'s Debug still
// compiles.
#[cfg(feature = "cubecl-wgpu")]
impl std::fmt::Debug for WgpuDeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuDeviceBuffer")
            .field("backend_id", &CUBECL_WGPU_BACKEND)
            .field("len_f32", &self.len_f32)
            .finish()
    }
}

#[cfg(feature = "cubecl-wgpu")]
impl WgpuDeviceBuffer {
    /// Build a new buffer by uploading `host` to the default WGPU device.
    pub fn upload_default(host: &[f32]) -> Self {
        let device = cubecl::wgpu::WgpuDevice::default();
        let client = cubecl::wgpu::WgpuRuntime::client(&device);
        let handle = client.create_from_slice(f32::as_bytes(host));
        Self {
            client,
            handle: Mutex::new(handle),
            len_f32: host.len(),
        }
    }

    /// Allocate a new zero-initialised buffer of `len` `f32` elements on
    /// the default WGPU device. The contents are whatever the runtime
    /// initialises freshly-issued memory to (in practice zero for the
    /// wgpu backend, but callers shouldn't rely on a specific bit pattern
    /// until the kernel writes into it).
    pub fn alloc_default(len: usize) -> Self {
        let device = cubecl::wgpu::WgpuDevice::default();
        let client = cubecl::wgpu::WgpuRuntime::client(&device);
        let handle = client.empty(len * std::mem::size_of::<f32>());
        Self {
            client,
            handle: Mutex::new(handle),
            len_f32: len,
        }
    }

    /// Borrow a clone of the underlying cubecl handle. Cheap (Arc bump).
    fn clone_handle(&self) -> cubecl::server::Handle {
        self.handle
            .lock()
            .expect("WgpuDeviceBuffer handle mutex poisoned")
            .clone()
    }

    fn client(&self) -> &ComputeClient<cubecl::wgpu::WgpuRuntime> {
        &self.client
    }
}

#[cfg(feature = "cubecl-wgpu")]
impl DeviceBuffer for WgpuDeviceBuffer {
    fn backend_id(&self) -> &'static str {
        CUBECL_WGPU_BACKEND
    }

    fn len_f32(&self) -> usize {
        self.len_f32
    }

    fn to_host(&self) -> Result<Vec<f32>> {
        let handle = self.clone_handle();
        let bytes = self.client.read_one(handle).map_err(|err| {
            cubecl_wgpu_err(format!("WgpuDeviceBuffer::to_host read failed: {err:?}"))
        })?;
        let read = f32::from_bytes(&bytes);
        if read.len() != self.len_f32 {
            return Err(cubecl_wgpu_err(format!(
                "WgpuDeviceBuffer::to_host length {} did not match expected {}",
                read.len(),
                self.len_f32
            )));
        }
        Ok(read.to_vec())
    }

    fn write_from_host(&self, src: &[f32]) -> Result<()> {
        if src.len() != self.len_f32 {
            return Err(cubecl_wgpu_err(format!(
                "WgpuDeviceBuffer::write_from_host src.len()={} != buffer len {}",
                src.len(),
                self.len_f32
            )));
        }
        // cubecl 0.10 has no "write into existing handle" on
        // `ComputeClient`. Swap the handle for a fresh `create_from_slice`
        // upload; the old handle is dropped, freeing the storage.
        let new_handle = self.client.create_from_slice(f32::as_bytes(src));
        *self
            .handle
            .lock()
            .expect("WgpuDeviceBuffer handle mutex poisoned") = new_handle;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(feature = "cubecl-wgpu")]
impl CubeClKernelBackend {
    fn upload_wgpu(&self, host: &[f32]) -> Result<DeviceTensor> {
        Ok(DeviceTensor::from_device(Box::new(
            WgpuDeviceBuffer::upload_default(host),
        )))
    }

    fn alloc_wgpu(&self, len: usize) -> Result<DeviceTensor> {
        Ok(DeviceTensor::from_device(Box::new(
            WgpuDeviceBuffer::alloc_default(len),
        )))
    }

    /// Attempt to extract the four `WgpuDeviceBuffer` operands from
    /// `DeviceTensor` handles. Returns `None` if any operand is not
    /// device-resident on `cubecl-wgpu`, in which case the caller must
    /// fall back to the host-bouncing default `linear_d`. `bias` is
    /// optional — when the model has no bias this returns
    /// `Some((x, w, None, out))`.
    fn try_extract_wgpu_operands<'a>(
        x: &'a DeviceTensor,
        weight: &'a DeviceTensor,
        bias: Option<&'a DeviceTensor>,
        out: &'a DeviceTensor,
    ) -> Option<(
        &'a WgpuDeviceBuffer,
        &'a WgpuDeviceBuffer,
        Option<&'a WgpuDeviceBuffer>,
        &'a WgpuDeviceBuffer,
    )> {
        let x_buf = extract_wgpu_buf(x)?;
        let w_buf = extract_wgpu_buf(weight)?;
        let out_buf = extract_wgpu_buf(out)?;
        let bias_buf = match bias {
            Some(b) => Some(extract_wgpu_buf(b)?),
            None => None,
        };
        Some((x_buf, w_buf, bias_buf, out_buf))
    }
}

#[cfg(feature = "cubecl-wgpu")]
fn extract_wgpu_buf(t: &DeviceTensor) -> Option<&WgpuDeviceBuffer> {
    t.try_as_device_buffer()?.as_any().downcast_ref()
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
    fn wgpu_linear_out_by_in_matches_scalar_within_tolerance() {
        use crate::CpuKernelBackend;

        let rows = 17;
        let in_features = 23;
        let out_features = 13;
        let x: Vec<f32> = (0..rows * in_features)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let w: Vec<f32> = (0..out_features * in_features)
            .map(|i| ((i as f32) * 0.019).cos())
            .collect();
        let b: Vec<f32> = (0..out_features).map(|i| (i as f32) * 0.05).collect();

        let mut scalar = vec![0.0_f32; rows * out_features];
        CpuKernelBackend::scalar()
            .linear_out_by_in(&x, rows, in_features, &w, out_features, Some(&b), &mut scalar)
            .expect("scalar linear_out_by_in must succeed");

        let mut gpu = vec![0.0_f32; rows * out_features];
        if let Err(err) = linear_out_by_in_wgpu(
            &x,
            rows,
            in_features,
            &w,
            out_features,
            Some(&b),
            &mut gpu,
        ) {
            // No usable WGPU adapter on this host — skip rather than fail. This
            // mirrors `wgpu_rope_matches_cpu_reference_for_position_one`, which
            // is `#[ignore]` for the same reason but in a coarser way; here we
            // would rather run the test when an adapter is present and skip
            // cleanly when it is not.
            eprintln!(
                "skipping wgpu_linear_out_by_in_matches_scalar_within_tolerance: {err:?}"
            );
            return;
        }

        for (idx, (s, g)) in scalar.iter().zip(gpu.iter()).enumerate() {
            let abs = (s - g).abs();
            let rel = if s.abs() > 1e-6 { abs / s.abs() } else { abs };
            assert!(
                abs <= 1e-4 || rel <= 1e-4,
                "GPU drifted at idx {idx}: scalar={s} gpu={g} abs={abs} rel={rel}"
            );
        }
    }

    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn wgpu_linear_d_with_device_handles_matches_scalar_within_tolerance() {
        use crate::CpuKernelBackend;

        let rows = 17;
        let in_features = 23;
        let out_features = 13;
        let x: Vec<f32> = (0..rows * in_features)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let w: Vec<f32> = (0..out_features * in_features)
            .map(|i| ((i as f32) * 0.019).cos())
            .collect();
        let b: Vec<f32> = (0..out_features).map(|i| (i as f32) * 0.05).collect();

        // Reference: scalar CPU path.
        let mut scalar = vec![0.0_f32; rows * out_features];
        CpuKernelBackend::scalar()
            .linear_out_by_in(&x, rows, in_features, &w, out_features, Some(&b), &mut scalar)
            .expect("scalar linear_out_by_in must succeed");

        // Skip cleanly when there's no usable WGPU adapter — same pattern
        // as `wgpu_linear_out_by_in_matches_scalar_within_tolerance`. We
        // probe with the simpler `linear_out_by_in_wgpu` first because
        // it returns an error rather than panicking on adapter failure.
        let mut probe = vec![0.0_f32; rows * out_features];
        if let Err(err) = linear_out_by_in_wgpu(
            &x,
            rows,
            in_features,
            &w,
            out_features,
            Some(&b),
            &mut probe,
        ) {
            eprintln!(
                "skipping wgpu_linear_d_with_device_handles_matches_scalar_within_tolerance: {err:?}"
            );
            return;
        }

        // Device-resident path: upload everything, run `linear_d`, read
        // the result back. This is the hot path Stage 1B is meant to
        // unlock — no host bounce between upload and the kernel launch.
        let backend = CubeClKernelBackend::new_gpu(0);
        let x_d = backend.upload(&x).expect("upload x");
        let w_d = backend.upload(&w).expect("upload weight");
        let b_d = backend.upload(&b).expect("upload bias");
        let out_d = backend
            .alloc(rows * out_features)
            .expect("alloc output");

        assert!(matches!(
            x_d.residency(),
            crate::Residency::Device(CUBECL_WGPU_BACKEND)
        ));
        assert!(matches!(
            out_d.residency(),
            crate::Residency::Device(CUBECL_WGPU_BACKEND)
        ));

        backend
            .linear_d(&x_d, rows, in_features, &w_d, out_features, Some(&b_d), &out_d)
            .expect("device-resident linear_d must succeed");

        let gpu = out_d.to_host_owned().expect("readback output");
        assert_eq!(gpu.len(), scalar.len());
        for (idx, (s, g)) in scalar.iter().zip(gpu.iter()).enumerate() {
            let abs = (s - g).abs();
            let rel = if s.abs() > 1e-6 { abs / s.abs() } else { abs };
            assert!(
                abs <= 1e-4 || rel <= 1e-4,
                "GPU drifted at idx {idx}: scalar={s} gpu={g} abs={abs} rel={rel}"
            );
        }
    }

    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn wgpu_device_buffer_round_trips_through_to_host() {
        let host = vec![1.0_f32, 2.0, -3.0, 4.5];

        // Skip if no adapter is available. We mirror the parity-test skip:
        // the simplest probe is to ask the legacy slice path to do *any*
        // GPU work — if that errors we know we can't run the round-trip.
        let mut probe = vec![0.0_f32; 1];
        if let Err(err) = linear_out_by_in_wgpu(
            &[1.0_f32],
            1,
            1,
            &[1.0_f32],
            1,
            None,
            &mut probe,
        ) {
            eprintln!("skipping wgpu_device_buffer_round_trips_through_to_host: {err:?}");
            return;
        }

        let buf = WgpuDeviceBuffer::upload_default(&host);
        assert_eq!(buf.len_f32(), host.len());
        assert_eq!(buf.backend_id(), CUBECL_WGPU_BACKEND);

        let back = buf.to_host().expect("readback");
        assert_eq!(back, host);

        // Overwrite, then verify the new bytes come back.
        let next = vec![10.0_f32, 20.0, 30.0, 40.0];
        buf.write_from_host(&next).expect("write_from_host");
        let back2 = buf.to_host().expect("readback after write");
        assert_eq!(back2, next);
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
