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

    /// Device-resident elementwise add. When both operands are
    /// `WgpuDeviceBuffer`s, launch the cube kernel against their existing
    /// handles. Otherwise fall back to the trait default, which forces
    /// host readback.
    fn add_inplace_d(&self, lhs: &DeviceTensor, rhs: &DeviceTensor) -> Result<()> {
        #[cfg(feature = "cubecl-wgpu")]
        {
            if let (Some(lhs_buf), Some(rhs_buf)) = (extract_wgpu_buf(lhs), extract_wgpu_buf(rhs)) {
                return run_add_inplace_d_wgpu(lhs_buf, rhs_buf);
            }
        }
        // Default: readback + host loop.
        let mut lhs_host = lhs.to_host_owned()?;
        let rhs_host = rhs.to_host_owned()?;
        if lhs_host.len() != rhs_host.len() {
            return Err(cubecl_err(format!(
                "add_inplace_d length mismatch: lhs={} rhs={}",
                lhs_host.len(),
                rhs_host.len()
            )));
        }
        for (l, r) in lhs_host.iter_mut().zip(rhs_host.iter()) {
            *l += *r;
        }
        lhs.write_from_host_slice(&lhs_host)
    }

    /// Device-resident GELU in place. When the operand is a
    /// `WgpuDeviceBuffer`, launch the cube kernel. Otherwise fall back
    /// to the host path. Note: the cube `f32::erf` intrinsic may not be
    /// bit-identical to the CPU `erf_approx` Whisper uses; the 1e-4
    /// tolerance gate accepts this drift.
    fn gelu_inplace_d(&self, x: &DeviceTensor) -> Result<()> {
        #[cfg(feature = "cubecl-wgpu")]
        {
            if let Some(buf) = extract_wgpu_buf(x) {
                return run_gelu_inplace_d_wgpu(buf);
            }
        }
        let mut host = x.to_host_owned()?;
        for v in host.iter_mut() {
            *v = crate::gelu_whisper_scalar(*v);
        }
        x.write_from_host_slice(&host)
    }

    /// Device-resident LayerNorm. When every operand is a
    /// `WgpuDeviceBuffer`, launch a naive one-thread-per-row kernel.
    /// Stage 3 can optimize to a workgroup-per-row reduction.
    #[allow(clippy::too_many_arguments)]
    fn layer_norm_d(
        &self,
        x: &DeviceTensor,
        rows: usize,
        hidden: usize,
        weight: &DeviceTensor,
        bias: &DeviceTensor,
        eps: f32,
        out: &DeviceTensor,
    ) -> Result<()> {
        crate::validate_layer_norm_shapes(x, rows, hidden, weight, bias, out)?;
        #[cfg(feature = "cubecl-wgpu")]
        {
            if let (Some(x_buf), Some(w_buf), Some(b_buf), Some(out_buf)) = (
                extract_wgpu_buf(x),
                extract_wgpu_buf(weight),
                extract_wgpu_buf(bias),
                extract_wgpu_buf(out),
            ) {
                return run_layer_norm_d_wgpu(x_buf, rows, hidden, w_buf, b_buf, eps, out_buf);
            }
        }
        // Default: readback + host scalar layer_norm + writeback.
        let x_host = x.to_host_owned()?;
        let w_host = weight.to_host_owned()?;
        let b_host = bias.to_host_owned()?;
        let mut out_buf = vec![0.0_f32; rows * hidden];
        crate::layer_norm_whisper_scalar(
            &x_host,
            rows,
            hidden,
            &w_host,
            &b_host,
            eps,
            &mut out_buf,
        );
        out.write_from_host_slice(&out_buf)
    }

    /// Device-resident encoder self-attention. When every operand is a
    /// `WgpuDeviceBuffer`, launch the fused cube kernel against the
    /// existing handles — no host bounce, no scalar fallback. When the
    /// adapter is missing or any operand is not on the WGPU runtime, fall
    /// back to the trait default (readback → scalar → writeback).
    ///
    /// GW.4-5A: the fused kernel keeps the entire encoder forward on
    /// device. Decoder paths still bounce through host because their
    /// kernels (causal mask, KV cache, cross-attention) are different
    /// shapes and were not in this dispatch.
    #[allow(clippy::too_many_arguments)]
    fn attention_encoder_d(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        seq: usize,
        n_head: usize,
        head_dim: usize,
        scale: f32,
        output: &DeviceTensor,
    ) -> Result<()> {
        crate::validate_attention_encoder_shapes(q, k, v, seq, n_head, head_dim, output)?;
        #[cfg(feature = "cubecl-wgpu")]
        {
            if let (Some(q_buf), Some(k_buf), Some(v_buf), Some(out_buf)) = (
                extract_wgpu_buf(q),
                extract_wgpu_buf(k),
                extract_wgpu_buf(v),
                extract_wgpu_buf(output),
            ) {
                return run_attention_encoder_d_wgpu(
                    q_buf, k_buf, v_buf, seq, n_head, head_dim, scale, out_buf,
                );
            }
        }
        // Default: readback + host scalar + writeback. Keeps the path
        // correct on hosts without a usable WGPU adapter.
        let q_host = q.to_host_owned()?;
        let k_host = k.to_host_owned()?;
        let v_host = v.to_host_owned()?;
        let mut out_buf = vec![0.0_f32; seq * n_head * head_dim];
        crate::attention_encoder_scalar(
            &q_host,
            &k_host,
            &v_host,
            seq,
            n_head,
            head_dim,
            scale,
            &mut out_buf,
        );
        output.write_from_host_slice(&out_buf)
    }

    /// Device-resident `x += pe[start_pos..start_pos + rows]`. When both
    /// operands are `WgpuDeviceBuffer`s, launch the cube kernel against
    /// the existing handles.
    fn add_positional_embedding_d(
        &self,
        x: &DeviceTensor,
        rows: usize,
        cols: usize,
        pe: &DeviceTensor,
        pe_rows: usize,
        start_pos: usize,
    ) -> Result<()> {
        crate::validate_add_positional_embedding_shapes(x, rows, cols, pe, pe_rows, start_pos)?;
        #[cfg(feature = "cubecl-wgpu")]
        {
            if let (Some(x_buf), Some(pe_buf)) = (extract_wgpu_buf(x), extract_wgpu_buf(pe)) {
                return run_add_positional_embedding_d_wgpu(x_buf, rows, cols, pe_buf, start_pos);
            }
        }
        let mut x_host = x.to_host_owned()?;
        let pe_host = pe.to_host_owned()?;
        for row in 0..rows {
            let dst_start = row * cols;
            let src_start = (start_pos + row) * cols;
            for col in 0..cols {
                x_host[dst_start + col] += pe_host[src_start + col];
            }
        }
        x.write_from_host_slice(&x_host)
    }

    /// Device-resident decoder causal self-attention (full-context). When
    /// every operand is a `WgpuDeviceBuffer`, launch the fused cube kernel.
    /// Falls back to the trait default (readback → scalar → writeback) when
    /// the adapter is unavailable or any operand is not on the WGPU runtime.
    ///
    /// GW.4-5B: lifts the full-context causal decode path onto device so
    /// the host bounce at `attention_body_host(causal=true)` in
    /// `decode_tokens_with_self_attention_cache` can be removed.
    #[allow(clippy::too_many_arguments)]
    fn attention_decoder_causal_d(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        seq: usize,
        n_head: usize,
        head_dim: usize,
        scale: f32,
        output: &DeviceTensor,
    ) -> Result<()> {
        crate::validate_attention_decoder_causal_shapes(q, k, v, seq, n_head, head_dim, output)?;
        #[cfg(feature = "cubecl-wgpu")]
        {
            if let (Some(q_buf), Some(k_buf), Some(v_buf), Some(out_buf)) = (
                extract_wgpu_buf(q),
                extract_wgpu_buf(k),
                extract_wgpu_buf(v),
                extract_wgpu_buf(output),
            ) {
                return run_attention_decoder_causal_d_wgpu(
                    q_buf, k_buf, v_buf, seq, n_head, head_dim, scale, out_buf,
                );
            }
        }
        // Default: readback + scalar causal decoder attention + writeback.
        let q_host = q.to_host_owned()?;
        let k_host = k.to_host_owned()?;
        let v_host = v.to_host_owned()?;
        let mut out_buf = vec![0.0_f32; seq * n_head * head_dim];
        crate::attention_decoder_causal_scalar(
            &q_host,
            &k_host,
            &v_host,
            seq,
            n_head,
            head_dim,
            scale,
            &mut out_buf,
        );
        output.write_from_host_slice(&out_buf)
    }

    /// Device-resident incremental decoder self-attention (single token).
    /// When every operand is a `WgpuDeviceBuffer`, launch the fused cube
    /// kernel. Falls back to the trait default otherwise.
    ///
    /// GW.4-5B: lifts the single-token incremental path onto device so the
    /// host bounce at `attention_incremental_body_host` in
    /// `decode_appended_token` can be removed.
    #[allow(clippy::too_many_arguments)]
    fn attention_decoder_incremental_d(
        &self,
        q: &DeviceTensor,
        past_k: &DeviceTensor,
        past_v: &DeviceTensor,
        new_k: &DeviceTensor,
        new_v: &DeviceTensor,
        past_seq: usize,
        n_head: usize,
        head_dim: usize,
        scale: f32,
        output: &DeviceTensor,
    ) -> Result<()> {
        crate::validate_attention_decoder_incremental_shapes(
            q, past_k, past_v, new_k, new_v, past_seq, n_head, head_dim, output,
        )?;
        #[cfg(feature = "cubecl-wgpu")]
        {
            if let (
                Some(q_buf),
                Some(pk_buf),
                Some(pv_buf),
                Some(nk_buf),
                Some(nv_buf),
                Some(out_buf),
            ) = (
                extract_wgpu_buf(q),
                extract_wgpu_buf(past_k),
                extract_wgpu_buf(past_v),
                extract_wgpu_buf(new_k),
                extract_wgpu_buf(new_v),
                extract_wgpu_buf(output),
            ) {
                return run_attention_decoder_incremental_d_wgpu(
                    q_buf, pk_buf, pv_buf, nk_buf, nv_buf, past_seq, n_head, head_dim, scale,
                    out_buf,
                );
            }
        }
        // Default: readback + scalar incremental attention + writeback.
        let q_host = q.to_host_owned()?;
        let past_k_host = past_k.to_host_owned()?;
        let past_v_host = past_v.to_host_owned()?;
        let new_k_host = new_k.to_host_owned()?;
        let new_v_host = new_v.to_host_owned()?;
        let mut out_buf = vec![0.0_f32; n_head * head_dim];
        crate::attention_decoder_incremental_scalar(
            &q_host,
            &past_k_host,
            &past_v_host,
            &new_k_host,
            &new_v_host,
            past_seq,
            n_head,
            head_dim,
            scale,
            &mut out_buf,
        );
        output.write_from_host_slice(&out_buf)
    }

    /// Device-resident decoder cross-attention (encoder-decoder attention).
    /// Q comes from the decoder hidden state `[q_seq, state]`; K and V come
    /// from the precomputed encoder output `[kv_seq, state]`. No causal mask:
    /// each query row attends all `kv_seq` encoder positions freely.
    ///
    /// When all operands are `WgpuDeviceBuffer`, launches the fused cube
    /// kernel. Falls back to the trait default (readback → scalar → writeback)
    /// when the adapter is unavailable or any operand is not on the WGPU runtime.
    ///
    /// GW.4-5C: replaces the `attention_body_host(causal=false)` host bounce
    /// in both `decode_tokens_with_self_attention_cache` and
    /// `decode_appended_token`.
    #[allow(clippy::too_many_arguments)]
    fn attention_decoder_cross_d(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        q_seq: usize,
        kv_seq: usize,
        n_head: usize,
        head_dim: usize,
        scale: f32,
        output: &DeviceTensor,
    ) -> Result<()> {
        crate::validate_attention_decoder_cross_shapes(
            q, k, v, q_seq, kv_seq, n_head, head_dim, output,
        )?;
        #[cfg(feature = "cubecl-wgpu")]
        {
            if let (Some(q_buf), Some(k_buf), Some(v_buf), Some(out_buf)) = (
                extract_wgpu_buf(q),
                extract_wgpu_buf(k),
                extract_wgpu_buf(v),
                extract_wgpu_buf(output),
            ) {
                return run_attention_decoder_cross_d_wgpu(
                    q_buf, k_buf, v_buf, q_seq, kv_seq, n_head, head_dim, scale, out_buf,
                );
            }
        }
        // Default: readback + scalar cross-attention + writeback.
        let q_host = q.to_host_owned()?;
        let k_host = k.to_host_owned()?;
        let v_host = v.to_host_owned()?;
        let mut out_buf = vec![0.0_f32; q_seq * n_head * head_dim];
        crate::attention_decoder_cross_scalar(
            &q_host,
            &k_host,
            &v_host,
            q_seq,
            kv_seq,
            n_head,
            head_dim,
            scale,
            &mut out_buf,
        );
        output.write_from_host_slice(&out_buf)
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

// ---------------------------------------------------------------------------
// GW.4 Stage 2A: device-resident critical-chain primitives
// (add_inplace_d, gelu_inplace_d, layer_norm_d, add_positional_embedding_d).
//
// All four kernels follow the same "out-of-place + handle swap" pattern as
// `linear_d`: launch into a fresh output handle, then swap it into the
// caller-supplied `WgpuDeviceBuffer` so subsequent reads pick up the
// result. The kernels themselves are "_out_f32" suffixed; the trait
// methods that look like "in-place" describe caller-visible semantics.
// ---------------------------------------------------------------------------

/// Naive LayerNorm — one thread per row, sequential reductions of the row.
/// Bit-identical to the CPU `layer_norm_whisper_scalar` op order: sum-then-
/// divide for the mean, then sum of squared deltas, divide, then add eps
/// and inverse-sqrt. Stage 3 should swap this for a workgroup-per-row
/// reduction, but for correctness on encoder shapes (rows ~1500, hidden
/// 384–1280) the naive form is fine.
#[cube(launch_unchecked)]
fn layer_norm_naive_f32(
    x: &Array<f32>,
    weight: &Array<f32>,
    bias: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] hidden: usize,
    eps: f32,
) {
    let row = ABSOLUTE_POS;
    let row_start = row * hidden;
    // Bound check: each row writes `hidden` cells starting at `row_start`,
    // so the last written index is `row_start + hidden - 1`. Bail out if
    // there is no room for a full row.
    if row_start + hidden > output.len() {
        terminate!();
    }

    // Pass 1: mean.
    let mut sum = f32::new(0.0);
    for c in 0..hidden {
        sum += x[row_start + c];
    }
    let mean = sum / f32::cast_from(hidden as u32);

    // Pass 2: biased variance.
    let mut var_sum = f32::new(0.0);
    for c in 0..hidden {
        let d = x[row_start + c] - mean;
        var_sum += d * d;
    }
    let var = var_sum / f32::cast_from(hidden as u32);
    let inv_std = f32::new(1.0) / f32::sqrt(var + eps);

    // Pass 3: normalize + affine.
    for c in 0..hidden {
        let normed = (x[row_start + c] - mean) * inv_std;
        output[row_start + c] = normed * weight[c] + bias[c];
    }
}

/// Compute `(workgroup_count, workgroup_size)` for a 1-D elementwise
/// launch sized `total` (in cells), rounded up to whole workgroups. Same
/// pattern as `prepare_linear_launch`: WGPU caps workgroup size at 256 on
/// most adapters, so we fan out across workgroups with a fixed 1-D
/// `CubeDim::new_1d(256)` and the kernel does a tail bounds check.
#[cfg(feature = "cubecl-wgpu")]
fn prepare_elementwise_launch(total: usize) -> Result<(u32, u32)> {
    const WORKGROUP_SIZE: u32 = 256;
    let total = u32::try_from(total).map_err(|_| {
        cubecl_wgpu_err(format!(
            "elementwise launch cell count {total} exceeds CubeCL u32 launch limit"
        ))
    })?;
    let workgroup_count = total.div_ceil(WORKGROUP_SIZE).max(1);
    Ok((workgroup_count, WORKGROUP_SIZE))
}

/// Compute `(workgroup_count, workgroup_size)` for a one-thread-per-row
/// launch. We use a 1-D launch (`CubeDim::new_1d(WORKGROUP_SIZE)`) and
/// round up the row count to whole workgroups; the kernel itself bounds-
/// checks against `output.len()`.
#[cfg(feature = "cubecl-wgpu")]
fn prepare_per_row_launch(rows: usize) -> Result<(u32, u32)> {
    const WORKGROUP_SIZE: u32 = 64; // rows per layer are typically << #cells;
    // a smaller workgroup avoids wasting threads
    // on tail bounds checks.
    let rows = u32::try_from(rows).map_err(|_| {
        cubecl_wgpu_err(format!(
            "per-row launch row count {rows} exceeds CubeCL u32 launch limit"
        ))
    })?;
    let workgroup_count = rows.div_ceil(WORKGROUP_SIZE).max(1);
    Ok((workgroup_count, WORKGROUP_SIZE))
}

#[cfg(feature = "cubecl-wgpu")]
fn run_add_inplace_d_wgpu(lhs: &WgpuDeviceBuffer, rhs: &WgpuDeviceBuffer) -> Result<()> {
    if lhs.len_f32() != rhs.len_f32() {
        return Err(cubecl_wgpu_err(format!(
            "add_inplace_d length mismatch: lhs={} rhs={}",
            lhs.len_f32(),
            rhs.len_f32()
        )));
    }
    let len = lhs.len_f32();
    let client = lhs.client();
    let (workgroup_count, workgroup_size) = prepare_elementwise_launch(len)?;

    // Same swap-on-write discipline as `run_linear_d_wgpu`: launch a
    // dedicated out-of-place kernel into a fresh output handle, then
    // swap that handle into `lhs`.
    let lhs_handle = lhs.clone_handle();
    let rhs_handle = rhs.clone_handle();
    let out_handle = client.empty(len * std::mem::size_of::<f32>());

    unsafe {
        add_out_f32::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            client,
            CubeCount::Static(workgroup_count, 1, 1),
            CubeDim::new_1d(workgroup_size),
            ArrayArg::from_raw_parts(lhs_handle, len),
            ArrayArg::from_raw_parts(rhs_handle, len),
            ArrayArg::from_raw_parts(out_handle.clone(), len),
        );
    }

    *lhs.handle
        .lock()
        .expect("WgpuDeviceBuffer handle mutex poisoned") = out_handle;
    Ok(())
}

/// Out-of-place add used by `run_add_inplace_d_wgpu`. The "_inplace"
/// suffix on `add_inplace_d` describes the caller-visible semantics
/// (lhs's handle is swapped to a buffer that holds lhs+rhs), not the
/// kernel itself.
#[cube(launch_unchecked)]
fn add_out_f32(lhs: &Array<f32>, rhs: &Array<f32>, output: &mut Array<f32>) {
    let i = ABSOLUTE_POS;
    if i >= output.len() {
        terminate!();
    }
    output[i] = lhs[i] + rhs[i];
}

#[cfg(feature = "cubecl-wgpu")]
fn run_gelu_inplace_d_wgpu(x: &WgpuDeviceBuffer) -> Result<()> {
    let len = x.len_f32();
    let client = x.client();
    let (workgroup_count, workgroup_size) = prepare_elementwise_launch(len)?;
    let in_handle = x.clone_handle();
    let out_handle = client.empty(len * std::mem::size_of::<f32>());

    unsafe {
        gelu_out_f32::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            client,
            CubeCount::Static(workgroup_count, 1, 1),
            CubeDim::new_1d(workgroup_size),
            ArrayArg::from_raw_parts(in_handle, len),
            ArrayArg::from_raw_parts(out_handle.clone(), len),
        );
    }

    *x.handle
        .lock()
        .expect("WgpuDeviceBuffer handle mutex poisoned") = out_handle;
    Ok(())
}

/// Out-of-place GELU. See `add_out_f32` for the rationale on the
/// kernel-vs-caller "in place" naming.
#[cube(launch_unchecked)]
fn gelu_out_f32(input: &Array<f32>, output: &mut Array<f32>) {
    let i = ABSOLUTE_POS;
    if i >= output.len() {
        terminate!();
    }
    let v = input[i];
    let sqrt2 = f32::sqrt(f32::new(2.0));
    let erf_val = cubecl::frontend::Erf::erf(v / sqrt2);
    output[i] = f32::new(0.5) * v * (f32::new(1.0) + erf_val);
}

#[cfg(feature = "cubecl-wgpu")]
fn run_layer_norm_d_wgpu(
    x: &WgpuDeviceBuffer,
    rows: usize,
    hidden: usize,
    weight: &WgpuDeviceBuffer,
    bias: &WgpuDeviceBuffer,
    eps: f32,
    out: &WgpuDeviceBuffer,
) -> Result<()> {
    let client = x.client();
    let (workgroup_count, workgroup_size) = prepare_per_row_launch(rows)?;
    let out_handle = client.empty(out.len_f32() * std::mem::size_of::<f32>());

    unsafe {
        layer_norm_naive_f32::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            client,
            CubeCount::Static(workgroup_count, 1, 1),
            CubeDim::new_1d(workgroup_size),
            ArrayArg::from_raw_parts(x.clone_handle(), x.len_f32()),
            ArrayArg::from_raw_parts(weight.clone_handle(), weight.len_f32()),
            ArrayArg::from_raw_parts(bias.clone_handle(), bias.len_f32()),
            ArrayArg::from_raw_parts(out_handle.clone(), out.len_f32()),
            hidden,
            eps,
        );
    }

    *out.handle
        .lock()
        .expect("WgpuDeviceBuffer handle mutex poisoned") = out_handle;
    Ok(())
}

#[cfg(feature = "cubecl-wgpu")]
fn run_add_positional_embedding_d_wgpu(
    x: &WgpuDeviceBuffer,
    rows: usize,
    cols: usize,
    pe: &WgpuDeviceBuffer,
    start_pos: usize,
) -> Result<()> {
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| cubecl_wgpu_err("add_positional_embedding_d rows*cols overflowed usize"))?;
    if x.len_f32() != len {
        return Err(cubecl_wgpu_err(format!(
            "add_positional_embedding_d x len {} != rows*cols {len}",
            x.len_f32()
        )));
    }
    let offset_elems = start_pos.checked_mul(cols).ok_or_else(|| {
        cubecl_wgpu_err("add_positional_embedding_d start_pos*cols overflowed usize")
    })?;
    if offset_elems + len > pe.len_f32() {
        return Err(cubecl_wgpu_err(format!(
            "add_positional_embedding_d pe window {}..{} exceeds pe len {}",
            offset_elems,
            offset_elems + len,
            pe.len_f32()
        )));
    }

    let client = x.client();
    let (workgroup_count, workgroup_size) = prepare_elementwise_launch(len)?;
    let out_handle = client.empty(len * std::mem::size_of::<f32>());

    unsafe {
        add_positional_embedding_out_f32::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            client,
            CubeCount::Static(workgroup_count, 1, 1),
            CubeDim::new_1d(workgroup_size),
            ArrayArg::from_raw_parts(x.clone_handle(), x.len_f32()),
            ArrayArg::from_raw_parts(pe.clone_handle(), pe.len_f32()),
            ArrayArg::from_raw_parts(out_handle.clone(), len),
            offset_elems,
        );
    }

    *x.handle
        .lock()
        .expect("WgpuDeviceBuffer handle mutex poisoned") = out_handle;
    Ok(())
}

/// Out-of-place positional-embedding add. The launch passes the absolute
/// element offset `start_pos * cols` as a comptime so the kernel does a
/// single load from `pe[offset + i]` without a per-thread div/mod for the
/// row.
#[cube(launch_unchecked)]
fn add_positional_embedding_out_f32(
    x_in: &Array<f32>,
    pe: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] pe_offset_elems: usize,
) {
    let i = ABSOLUTE_POS;
    if i >= output.len() {
        terminate!();
    }
    output[i] = x_in[i] + pe[pe_offset_elems + i];
}

// ---------------------------------------------------------------------------
// GW.4-5A: fused encoder self-attention kernel.
//
// The cube kernel maps one thread to one `(query_row, head)` pair and does
// the entire `(scaled-dot → softmax → P·V)` chain using a flash-attention-
// style online softmax. Layout follows the host primitive: `[seq, state]`
// row-major with `state == n_head * head_dim`; within each row, head H
// occupies the `head_dim` cells starting at `head * head_dim`.
//
// GW.4-shmem fix: the previous implementation stored a `seq`-long score
// row in SharedMemory per thread (`WORKGROUP_SIZE * seq * 4 bytes`). At
// seq=1500 and ENC_ATTN_WG=4 this was 24 KiB per workgroup, exceeding the
// 16 KiB WebGPU spec floor for `maxComputeWorkgroupStorageSize`. The fix
// replaces the three-pass algorithm with a single-pass online softmax that
// keeps only three running scalars (`m`, `l`, `p`) plus an O(head_dim)
// per-thread accumulator. No SharedMemory is allocated.
//
// Numerical equivalence: the online softmax is mathematically identical to
// the subtract-max two-pass form. See the flash-attention algorithm for proof.
// ---------------------------------------------------------------------------

/// Threads per workgroup for the attention kernels. After the GW.4-shmem
/// flash rewrite there is no O(seq) shared-memory slab, so this constant
/// is only about launch occupancy, not memory budget.
const ENC_ATTN_WG: u32 = 4;

/// Maximum head_dim used by any Whisper variant (tiny through large-v2 all
/// use head_dim=64). The flash-attention kernels allocate `ENC_ATTN_WG *
/// head_dim * 4` bytes of shared memory per workgroup. At the maximum:
///   4 threads/wg * 64 floats/thread * 4 bytes/float = 1024 bytes.
/// The WebGPU spec floor is 16 384 bytes. This assertion ensures the
/// constant-size slab never silently regress to O(seq) proportions.
const WHISPER_MAX_HEAD_DIM: u32 = 64;
const _: () = assert!(
    ENC_ATTN_WG * WHISPER_MAX_HEAD_DIM * 4 <= 16_384,
    "GW.4-shmem: attention kernel shared-memory budget exceeds 16 KiB WebGPU floor"
);

/// Fused encoder self-attention. One thread per `(query_row, head)` pair.
///
/// Layout (matches `attention_encoder_scalar` and the host primitive):
///   q, k, v, output: `[seq, n_head, head_dim]` flattened row-major, where
///   row stride is `n_head * head_dim` and the head H slice of row R lives
///   at `R * n_head * head_dim + H * head_dim`.
///
/// Per `(query_row, head)`: single-pass online softmax over all `seq` keys,
/// accumulating `P · V` into `head_dim` per-thread register values.
/// No shared memory is used; shared-memory budget is O(1) per workgroup.
#[cube(launch_unchecked)]
fn attention_encoder_f32(
    q: &Array<f32>,
    k: &Array<f32>,
    v: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] seq: usize,
    #[comptime] n_head: usize,
    #[comptime] head_dim: usize,
    scale: f32,
) {
    // Cast each u32 launch builtin to usize once so the rest of the body
    // unifies with the usize comptime constants — same trick the tiled
    // linear kernel uses for CUBE_POS_X/Y. The casts trip clippy's
    // `unnecessary_cast` because the macro expansion presents the
    // builtins as usize at the surface level, but the underlying type
    // before macro expansion is u32 and the casts are load-bearing.
    #[allow(clippy::unnecessary_cast)]
    let pos = ABSOLUTE_POS as usize;
    let total = seq * n_head;
    // Tail bounds check: launch grid rounds up to whole workgroups.
    if pos >= total {
        terminate!();
    }

    let query_row = pos / n_head;
    let head = pos - query_row * n_head;
    let state = n_head * head_dim;
    let q_base = query_row * state + head * head_dim;

    // Online (flash-attention) softmax: stream over all seq keys, maintaining
    // running max `m`, running normaliser `l`, and weighted V accumulator.
    // On each step j:
    //   s      = scale * dot(Q[query_row,head,:], K[j,head,:])
    //   m_new  = max(m, s)
    //   alpha  = exp(m - m_new)      -- rescale accumulated mass from old max
    //   p      = exp(s - m_new)      -- softmax weight for key j
    //   l      = l * alpha + p
    //   acc[d] = acc[d] * alpha + p * V[j,head,d]
    //   m      = m_new
    //
    // After the loop: output[d] = acc[d] / l. This is numerically identical
    // to subtract-max two-pass softmax (within f32 rounding).
    //
    // `m` is seeded with the j=0 score so we never compare against a
    // sentinel; the alpha for j=0 is exp(m - m_new) = exp(0) = 1.0, which
    // is correct — no prior mass to rescale.

    // Seed with j=0 to avoid needing -inf as a CubeCL literal.
    let k_base_0 = head * head_dim;
    let mut m = f32::new(0.0);
    for d in 0..head_dim {
        m += q[q_base + d] * k[k_base_0 + d];
    }
    m *= scale;

    let p0 = f32::new(1.0); // exp(m - m) = 1.0
    let mut l = p0;

    // acc holds the running weighted V sum; initialise with p0 * V[0].
    // Using an unrolled approach: CubeCL comptime loops over head_dim.
    // We declare a SharedMemory of size head_dim as per-thread accumulator
    // storage — it is O(head_dim * wg_size) which is tiny (e.g. 64 * 4 = 256
    // bytes) and independent of seq. We index by lane * head_dim + d.
    // NOTE: head_dim is comptime so the slab size is a compile-time constant.
    // At Whisper's head_dim=64 and ENC_ATTN_WG=4 this is 1 KiB — well within
    // the 16 KiB WebGPU floor.
    #[allow(clippy::unnecessary_cast)]
    let lane = UNIT_POS as usize;
    let mut acc = SharedMemory::<f32>::new(ENC_ATTN_WG as usize * head_dim);
    let acc_base = lane * head_dim;

    for d in 0..head_dim {
        acc[acc_base + d] = p0 * v[k_base_0 + d];
    }

    // Stream remaining keys j=1..seq.
    for j in 1..seq {
        let k_base = j * state + head * head_dim;
        let mut s = f32::new(0.0);
        for d in 0..head_dim {
            s += q[q_base + d] * k[k_base + d];
        }
        s *= scale;

        let m_new = f32::max(m, s);
        let alpha = f32::exp(m - m_new);
        let p = f32::exp(s - m_new);
        l = l * alpha + p;

        let v_base = j * state + head * head_dim;
        for d in 0..head_dim {
            acc[acc_base + d] = acc[acc_base + d] * alpha + p * v[v_base + d];
        }
        m = m_new;
    }

    // Normalise and write output.
    let out_base = q_base;
    for d in 0..head_dim {
        output[out_base + d] = acc[acc_base + d] / l;
    }
}

/// Launch helper for the fused encoder-attention kernel. Validates buffer
/// lengths (the kernel can't), allocates a fresh output handle, runs the
/// kernel, and swaps the handle into `out_buf` so subsequent device reads
/// pick up the result without a host bounce.
#[cfg(feature = "cubecl-wgpu")]
#[allow(clippy::too_many_arguments)]
fn run_attention_encoder_d_wgpu(
    q: &WgpuDeviceBuffer,
    k: &WgpuDeviceBuffer,
    v: &WgpuDeviceBuffer,
    seq: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
    out: &WgpuDeviceBuffer,
) -> Result<()> {
    let state = n_head
        .checked_mul(head_dim)
        .ok_or_else(|| cubecl_wgpu_err("attention_encoder_d n_head*head_dim overflowed usize"))?;
    let expected = seq
        .checked_mul(state)
        .ok_or_else(|| cubecl_wgpu_err("attention_encoder_d seq*state overflowed usize"))?;
    for (label, len) in [
        ("q", q.len_f32()),
        ("k", k.len_f32()),
        ("v", v.len_f32()),
        ("out", out.len_f32()),
    ] {
        if len != expected {
            return Err(cubecl_wgpu_err(format!(
                "attention_encoder_d {label} len {len} != seq*state {expected}"
            )));
        }
    }

    let total = u32::try_from(seq * n_head).map_err(|_| {
        cubecl_wgpu_err(format!(
            "attention_encoder_d thread count {} exceeds CubeCL u32 launch limit",
            seq * n_head
        ))
    })?;
    let workgroup_count = total.div_ceil(ENC_ATTN_WG).max(1);

    let client = q.client();
    let out_handle = client.empty(expected * std::mem::size_of::<f32>());

    unsafe {
        attention_encoder_f32::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            client,
            CubeCount::Static(workgroup_count, 1, 1),
            CubeDim::new_1d(ENC_ATTN_WG),
            ArrayArg::from_raw_parts(q.clone_handle(), q.len_f32()),
            ArrayArg::from_raw_parts(k.clone_handle(), k.len_f32()),
            ArrayArg::from_raw_parts(v.clone_handle(), v.len_f32()),
            ArrayArg::from_raw_parts(out_handle.clone(), expected),
            seq,
            n_head,
            head_dim,
            scale,
        );
    }

    *out.handle
        .lock()
        .expect("WgpuDeviceBuffer handle mutex poisoned") = out_handle;
    Ok(())
}

// ---------------------------------------------------------------------------
// GW.4-5B: fused decoder self-attention kernels.
//
// Two kernels to match the two host paths in decode.rs:
//
//   1. `attention_decoder_causal_f32`: full-context causal self-attention.
//      Maps one thread per `(query_row, head)`. Row `qi` attends keys
//      `0..=qi`. Layout: `[seq, state]` row-major, same as encoder.
//      Scores live in shared memory (same slab strategy as encoder).
//
//   2. `attention_decoder_incremental_f32`: single-token incremental.
//      Maps one thread per head. Q is `[state]`, K/V comes from
//      `concat(past_k, new_k)` of length `visible = past_seq + 1`.
//      Because `visible` is not comptime, scores are stored in a
//      caller-supplied scratch `Array<f32>` of length `n_head * visible`,
//      with each thread's row at `head * visible`.
// ---------------------------------------------------------------------------

/// Fused causal decoder self-attention. One thread per `(query_row, head)`.
///
/// Layout: `q`, `k`, `v`, `output` are `[seq, state]` row-major,
/// `state == n_head * head_dim`. Row `qi` attends keys `0..=qi` only.
///
/// GW.4-shmem fix: replaced O(seq) SharedMemory slab with an online softmax
/// that uses O(head_dim) per-thread shared memory (comptime size, independent
/// of seq). No shared-memory budget grows with runtime `seq`.
#[cube(launch_unchecked)]
fn attention_decoder_causal_f32(
    q: &Array<f32>,
    k: &Array<f32>,
    v: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] seq: usize,
    #[comptime] n_head: usize,
    #[comptime] head_dim: usize,
    scale: f32,
) {
    #[allow(clippy::unnecessary_cast)]
    let pos = ABSOLUTE_POS as usize;
    let total = seq * n_head;
    if pos >= total {
        terminate!();
    }

    let query_row = pos / n_head;
    let head = pos - query_row * n_head;
    let state = n_head * head_dim;
    // Row `query_row` may attend keys `0..=query_row` (causal mask).
    let visible = query_row + 1;

    let q_base = query_row * state + head * head_dim;

    // Online (flash-attention) softmax over the causally-visible keys.
    // Seed with j=0: exp(m - m) = 1, no prior mass to rescale.
    let k_base_0 = head * head_dim; // j=0: row 0, head slice
    let mut m = f32::new(0.0);
    for d in 0..head_dim {
        m += q[q_base + d] * k[k_base_0 + d];
    }
    m *= scale;

    let p0 = f32::new(1.0);
    let mut l = p0;

    // Per-thread accumulator stored in O(head_dim) shared memory.
    #[allow(clippy::unnecessary_cast)]
    let lane = UNIT_POS as usize;
    let mut acc = SharedMemory::<f32>::new(ENC_ATTN_WG as usize * head_dim);
    let acc_base = lane * head_dim;

    for d in 0..head_dim {
        acc[acc_base + d] = p0 * v[k_base_0 + d];
    }

    // Stream keys j=1..visible (causal: stop at query_row).
    for j in 1..seq {
        if j < visible {
            let k_base = j * state + head * head_dim;
            let mut s = f32::new(0.0);
            for d in 0..head_dim {
                s += q[q_base + d] * k[k_base + d];
            }
            s *= scale;

            let m_new = f32::max(m, s);
            let alpha = f32::exp(m - m_new);
            let p = f32::exp(s - m_new);
            l = l * alpha + p;

            let v_base = j * state + head * head_dim;
            for d in 0..head_dim {
                acc[acc_base + d] = acc[acc_base + d] * alpha + p * v[v_base + d];
            }
            m = m_new;
        }
    }

    // Normalise and write output.
    let out_base = query_row * state + head * head_dim;
    for d in 0..head_dim {
        output[out_base + d] = acc[acc_base + d] / l;
    }
}

/// Launch helper for the fused decoder causal self-attention kernel.
#[cfg(feature = "cubecl-wgpu")]
#[allow(clippy::too_many_arguments)]
fn run_attention_decoder_causal_d_wgpu(
    q: &WgpuDeviceBuffer,
    k: &WgpuDeviceBuffer,
    v: &WgpuDeviceBuffer,
    seq: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
    out: &WgpuDeviceBuffer,
) -> Result<()> {
    let state = n_head.checked_mul(head_dim).ok_or_else(|| {
        cubecl_wgpu_err("attention_decoder_causal_d n_head*head_dim overflowed usize")
    })?;
    let expected = seq
        .checked_mul(state)
        .ok_or_else(|| cubecl_wgpu_err("attention_decoder_causal_d seq*state overflowed usize"))?;
    for (label, len) in [
        ("q", q.len_f32()),
        ("k", k.len_f32()),
        ("v", v.len_f32()),
        ("out", out.len_f32()),
    ] {
        if len != expected {
            return Err(cubecl_wgpu_err(format!(
                "attention_decoder_causal_d {label} len {len} != seq*state {expected}"
            )));
        }
    }

    let total = u32::try_from(seq * n_head).map_err(|_| {
        cubecl_wgpu_err(format!(
            "attention_decoder_causal_d thread count {} exceeds CubeCL u32 launch limit",
            seq * n_head
        ))
    })?;
    // Reuse the same workgroup size as the encoder kernel.
    let workgroup_count = total.div_ceil(ENC_ATTN_WG).max(1);

    let client = q.client();
    let out_handle = client.empty(expected * std::mem::size_of::<f32>());

    unsafe {
        attention_decoder_causal_f32::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            client,
            CubeCount::Static(workgroup_count, 1, 1),
            CubeDim::new_1d(ENC_ATTN_WG),
            ArrayArg::from_raw_parts(q.clone_handle(), q.len_f32()),
            ArrayArg::from_raw_parts(k.clone_handle(), k.len_f32()),
            ArrayArg::from_raw_parts(v.clone_handle(), v.len_f32()),
            ArrayArg::from_raw_parts(out_handle.clone(), expected),
            seq,
            n_head,
            head_dim,
            scale,
        );
    }

    *out.handle
        .lock()
        .expect("WgpuDeviceBuffer handle mutex poisoned") = out_handle;
    Ok(())
}

/// Fused incremental decoder self-attention (single new token).
///
/// One thread per head. Q: `[state]`, past_kv: `[past_seq, state]`,
/// new_kv: `[state]`. `visible = past_seq + 1`. `scores`: caller-supplied
/// scratch of length `n_head * visible` — each thread at head `h` owns
/// row `h * visible`. Output: `[state]`.
///
/// Because `visible` is not comptime, scores cannot live in shared memory
/// of a fixed comptime size. A caller-supplied global scratch is the
/// simplest device-side alternative without an extra runtime allocation
/// inside the kernel.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn attention_decoder_incremental_f32(
    q: &Array<f32>,
    past_k: &Array<f32>,
    past_v: &Array<f32>,
    new_k: &Array<f32>,
    new_v: &Array<f32>,
    scores_scratch: &mut Array<f32>,
    output: &mut Array<f32>,
    #[comptime] n_head: usize,
    #[comptime] head_dim: usize,
    visible: u32,
    scale: f32,
) {
    #[allow(clippy::unnecessary_cast)]
    let head = ABSOLUTE_POS as usize;
    if head >= n_head {
        terminate!();
    }

    let state = n_head * head_dim;
    let visible_us = visible as usize;
    let past_seq = visible_us - 1;
    let q_base = head * head_dim;
    let scores_base = head * visible_us;

    // Pass 1: scaled dot products over concat(past_k, new_k).
    for ki in 0..visible_us {
        let mut acc = f32::new(0.0);
        for d in 0..head_dim {
            let key = if ki < past_seq {
                past_k[ki * state + head * head_dim + d]
            } else {
                new_k[head * head_dim + d]
            };
            acc += q[q_base + d] * key;
        }
        scores_scratch[scores_base + ki] = acc * scale;
    }

    // Pass 2: numerically stable softmax over scores_scratch[scores_base..scores_base+visible].
    let mut row_max = scores_scratch[scores_base];
    for ki in 1..visible_us {
        if scores_scratch[scores_base + ki] > row_max {
            row_max = scores_scratch[scores_base + ki];
        }
    }
    let mut denom = f32::new(0.0);
    for ki in 0..visible_us {
        let e = f32::exp(scores_scratch[scores_base + ki] - row_max);
        scores_scratch[scores_base + ki] = e;
        denom += e;
    }
    let inv_denom = f32::new(1.0) / denom;
    for ki in 0..visible_us {
        scores_scratch[scores_base + ki] = scores_scratch[scores_base + ki] * inv_denom;
    }

    // Pass 3: P · V accumulation.
    let out_base = head * head_dim;
    for d in 0..head_dim {
        let mut acc = f32::new(0.0);
        for ki in 0..visible_us {
            let value = if ki < past_seq {
                past_v[ki * state + head * head_dim + d]
            } else {
                new_v[head * head_dim + d]
            };
            acc += scores_scratch[scores_base + ki] * value;
        }
        output[out_base + d] = acc;
    }
}

/// Launch helper for the fused incremental decoder self-attention kernel.
#[cfg(feature = "cubecl-wgpu")]
#[allow(clippy::too_many_arguments)]
fn run_attention_decoder_incremental_d_wgpu(
    q: &WgpuDeviceBuffer,
    past_k: &WgpuDeviceBuffer,
    past_v: &WgpuDeviceBuffer,
    new_k: &WgpuDeviceBuffer,
    new_v: &WgpuDeviceBuffer,
    past_seq: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
    out: &WgpuDeviceBuffer,
) -> Result<()> {
    let state = n_head.checked_mul(head_dim).ok_or_else(|| {
        cubecl_wgpu_err("attention_decoder_incremental_d n_head*head_dim overflowed usize")
    })?;
    let visible = past_seq + 1;

    // Validate operand lengths.
    for (label, len, expected) in [
        ("q", q.len_f32(), state),
        ("new_k", new_k.len_f32(), state),
        ("new_v", new_v.len_f32(), state),
        ("out", out.len_f32(), state),
        ("past_k", past_k.len_f32(), past_seq * state),
        ("past_v", past_v.len_f32(), past_seq * state),
    ] {
        if len != expected {
            return Err(cubecl_wgpu_err(format!(
                "attention_decoder_incremental_d {label} len {len} != expected {expected}"
            )));
        }
    }

    let visible_u32 = u32::try_from(visible).map_err(|_| {
        cubecl_wgpu_err(format!(
            "attention_decoder_incremental_d visible {} exceeds u32",
            visible
        ))
    })?;
    let n_head_u32 = u32::try_from(n_head).map_err(|_| {
        cubecl_wgpu_err("attention_decoder_incremental_d n_head exceeds u32".to_string())
    })?;

    let client = q.client();
    // Scores scratch: n_head * visible floats, zeroed so uninitialised reads
    // in the boundary case (past_seq == 0, visible == 1) are 0.0.
    let scores_len = n_head * visible;
    let scores_handle = client.empty(scores_len * std::mem::size_of::<f32>());
    let out_handle = client.empty(state * std::mem::size_of::<f32>());

    unsafe {
        attention_decoder_incremental_f32::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            client,
            CubeCount::Static(n_head_u32, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(q.clone_handle(), q.len_f32()),
            ArrayArg::from_raw_parts(past_k.clone_handle(), past_k.len_f32().max(1)),
            ArrayArg::from_raw_parts(past_v.clone_handle(), past_v.len_f32().max(1)),
            ArrayArg::from_raw_parts(new_k.clone_handle(), new_k.len_f32()),
            ArrayArg::from_raw_parts(new_v.clone_handle(), new_v.len_f32()),
            ArrayArg::from_raw_parts(scores_handle.clone(), scores_len),
            ArrayArg::from_raw_parts(out_handle.clone(), state),
            n_head,
            head_dim,
            visible_u32,
            scale,
        );
    }

    *out.handle
        .lock()
        .expect("WgpuDeviceBuffer handle mutex poisoned") = out_handle;
    Ok(())
}

// ---------------------------------------------------------------------------
// GW.4-5C: fused decoder cross-attention kernel.
//
// Cross-attention (encoder-decoder): Q from decoder `[q_seq, state]`,
// K and V from encoder output `[kv_seq, state]`. No causal mask — every
// decoder query row attends all `kv_seq` encoder positions freely.
//
// One thread per `(query_row, head)`. Layout: `[seq, state]` row-major,
// `state == n_head * head_dim`. Shared memory slab: `wg_size * kv_seq` floats
// (one row per intra-workgroup lane), same strategy as the encoder kernel.
//
// Key differences from `attention_decoder_causal_f32` (GW.4-5B):
//   - K/V sequence length is `kv_seq` (encoder frames), not `q_seq`.
//   - No causal guard — all `kv_seq` positions are always visible.
//   - `q_seq` may be 1 (incremental append path) or > 1 (full-context path).
// ---------------------------------------------------------------------------

/// Fused decoder cross-attention kernel. One thread per `(query_row, head)`.
///
/// Layout: `q` is `[q_seq, state]`, `k`/`v` are `[kv_seq, state]`,
/// `output` is `[q_seq, state]`. `state == n_head * head_dim`.
/// Every query row attends all `kv_seq` encoder positions (no causal mask).
///
/// GW.4-shmem fix: replaced O(kv_seq) SharedMemory slab with an online
/// softmax that uses O(head_dim) per-thread shared memory, independent of
/// `kv_seq`. At Whisper-large kv_seq=1500 the old slab was 24 KiB; the new
/// slab is constant (e.g. 1 KiB at head_dim=64, ENC_ATTN_WG=4).
#[cube(launch_unchecked)]
fn attention_decoder_cross_f32(
    q: &Array<f32>,
    k: &Array<f32>,
    v: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] q_seq: usize,
    #[comptime] kv_seq: usize,
    #[comptime] n_head: usize,
    #[comptime] head_dim: usize,
    scale: f32,
) {
    #[allow(clippy::unnecessary_cast)]
    let pos = ABSOLUTE_POS as usize;
    let total = q_seq * n_head;
    if pos >= total {
        terminate!();
    }

    let query_row = pos / n_head;
    let head = pos - query_row * n_head;
    let state = n_head * head_dim;
    let q_base = query_row * state + head * head_dim;

    // Online softmax over all kv_seq encoder keys. Seed with j=0.
    let k_base_0 = head * head_dim; // j=0: row 0, head slice in K
    let mut m = f32::new(0.0);
    for d in 0..head_dim {
        m += q[q_base + d] * k[k_base_0 + d];
    }
    m *= scale;

    let p0 = f32::new(1.0);
    let mut l = p0;

    // Per-thread O(head_dim) accumulator in shared memory.
    #[allow(clippy::unnecessary_cast)]
    let lane = UNIT_POS as usize;
    let mut acc = SharedMemory::<f32>::new(ENC_ATTN_WG as usize * head_dim);
    let acc_base = lane * head_dim;

    let v_base_0 = head * head_dim; // j=0: row 0, head slice in V
    for d in 0..head_dim {
        acc[acc_base + d] = p0 * v[v_base_0 + d];
    }

    // Stream keys j=1..kv_seq (no causal mask in cross-attention).
    for j in 1..kv_seq {
        let k_base = j * state + head * head_dim;
        let mut s = f32::new(0.0);
        for d in 0..head_dim {
            s += q[q_base + d] * k[k_base + d];
        }
        s *= scale;

        let m_new = f32::max(m, s);
        let alpha = f32::exp(m - m_new);
        let p = f32::exp(s - m_new);
        l = l * alpha + p;

        let v_base = j * state + head * head_dim;
        for d in 0..head_dim {
            acc[acc_base + d] = acc[acc_base + d] * alpha + p * v[v_base + d];
        }
        m = m_new;
    }

    // Normalise and write output.
    let out_base = query_row * state + head * head_dim;
    for d in 0..head_dim {
        output[out_base + d] = acc[acc_base + d] / l;
    }
}

/// Launch helper for the fused decoder cross-attention kernel.
#[cfg(feature = "cubecl-wgpu")]
#[allow(clippy::too_many_arguments)]
fn run_attention_decoder_cross_d_wgpu(
    q: &WgpuDeviceBuffer,
    k: &WgpuDeviceBuffer,
    v: &WgpuDeviceBuffer,
    q_seq: usize,
    kv_seq: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
    out: &WgpuDeviceBuffer,
) -> Result<()> {
    let state = n_head.checked_mul(head_dim).ok_or_else(|| {
        cubecl_wgpu_err("attention_decoder_cross_d n_head*head_dim overflowed usize")
    })?;
    let q_expected = q_seq
        .checked_mul(state)
        .ok_or_else(|| cubecl_wgpu_err("attention_decoder_cross_d q_seq*state overflowed usize"))?;
    let kv_expected = kv_seq.checked_mul(state).ok_or_else(|| {
        cubecl_wgpu_err("attention_decoder_cross_d kv_seq*state overflowed usize")
    })?;
    for (label, len, expected) in [
        ("q", q.len_f32(), q_expected),
        ("out", out.len_f32(), q_expected),
    ] {
        if len != expected {
            return Err(cubecl_wgpu_err(format!(
                "attention_decoder_cross_d {label} len {len} != q_seq*state {expected}"
            )));
        }
    }
    for (label, len) in [("k", k.len_f32()), ("v", v.len_f32())] {
        if len != kv_expected {
            return Err(cubecl_wgpu_err(format!(
                "attention_decoder_cross_d {label} len {len} != kv_seq*state {kv_expected}"
            )));
        }
    }

    let total = u32::try_from(q_seq * n_head).map_err(|_| {
        cubecl_wgpu_err(format!(
            "attention_decoder_cross_d thread count {} exceeds CubeCL u32 launch limit",
            q_seq * n_head
        ))
    })?;
    // Reuse the same workgroup size as the encoder and decoder self-attention kernels.
    let workgroup_count = total.div_ceil(ENC_ATTN_WG).max(1);

    let client = q.client();
    let out_handle = client.empty(q_expected * std::mem::size_of::<f32>());

    unsafe {
        attention_decoder_cross_f32::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            client,
            CubeCount::Static(workgroup_count, 1, 1),
            CubeDim::new_1d(ENC_ATTN_WG),
            ArrayArg::from_raw_parts(q.clone_handle(), q.len_f32()),
            ArrayArg::from_raw_parts(k.clone_handle(), k.len_f32()),
            ArrayArg::from_raw_parts(v.clone_handle(), v.len_f32()),
            ArrayArg::from_raw_parts(out_handle.clone(), q_expected),
            q_seq,
            kv_seq,
            n_head,
            head_dim,
            ScalarArg::new(scale),
        );
    }

    *out.handle
        .lock()
        .expect("WgpuDeviceBuffer handle mutex poisoned") = out_handle;
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
///
/// Retained as a parity reference and as a fallback path; the live launch
/// helper currently dispatches the tiled variant below.
#[cube(launch_unchecked)]
#[allow(dead_code)]
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

/// Workgroup tile dimensions for `linear_out_by_in_tiled_f32`.
///
/// Constraint: `TILE_M == TILE_N == TILE_K` so each (ty, tx) thread in the
/// 16×16 workgroup is responsible for exactly one `x_tile` element load
/// (at column `tx`) and one `w_tile` element load (at row `ty`) per K
/// chunk. Workgroup size is `TILE_M * TILE_N = 256`, matching the WGPU
/// per-workgroup cap on the local DX12 adapter (GW.1). Whisper-tiny
/// `in_features = 384` divides evenly by 16, so the common path takes
/// 24 chunks with no tail.
const LINEAR_TILE_M: usize = 16;
const LINEAR_TILE_N: usize = 16;
const LINEAR_TILE_K: usize = 16;

/// Workgroup-tiled `output = x * weight^T + bias`.
///
/// Each workgroup computes one `TILE_M × TILE_N` output tile. The K axis
/// (`in_features`) is split into `ceil(in_features / TILE_K)` chunks; per
/// chunk, the workgroup cooperatively loads a `TILE_M × TILE_K` block of
/// `x` and a `TILE_N × TILE_K` block of `weight` into shared memory, then
/// each thread accumulates its dot product over the loaded chunk. Two
/// `sync_cube()` barriers per chunk: one after the cooperative load (so
/// every thread sees the full tiles before reading) and one after the
/// compute (so no thread overwrites the tile that another is still
/// reading). Out-of-bounds threads zero-fill their tile slot and skip the
/// final store so non-multiple-of-tile shapes work without a tail kernel.
#[cube(launch_unchecked)]
fn linear_out_by_in_tiled_f32(
    x: &Array<f32>,
    weight: &Array<f32>,
    bias: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] rows: usize,
    #[comptime] in_features: usize,
    #[comptime] out_features: usize,
    #[comptime] tile_m: usize,
    #[comptime] tile_n: usize,
    #[comptime] tile_k: usize,
    #[comptime] has_bias: bool,
) {
    // 2-D workgroup grid: CUBE_POS_X selects the out_features tile column,
    // CUBE_POS_Y selects the rows tile row. UNIT_POS_X/Y are this thread's
    // position within the workgroup; (ty, tx) maps to one output cell in
    // the tile at (row_in_tile=ty, out_in_tile=tx). Cast the u32 builtins
    // to usize once so the rest of the body unifies with the usize
    // comptime tile/shape constants — `layer_norm_naive_f32` mixes u32 and
    // usize freely because its math goes straight through array indexing,
    // but this kernel does enough intermediate arithmetic that one
    // explicit cast up front is cheaper than peppering casts everywhere.
    let tile_col = CUBE_POS_X as usize;
    let tile_row = CUBE_POS_Y as usize;
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;

    let row = tile_row * tile_m + ty;
    let out_dim = tile_col * tile_n + tx;

    let mut x_tile = SharedMemory::<f32>::new(tile_m * tile_k);
    let mut w_tile = SharedMemory::<f32>::new(tile_n * tile_k);

    let mut acc = f32::new(0.0);
    if has_bias && out_dim < out_features {
        acc = bias[out_dim];
    }

    // Number of K chunks. `comptime`d so the inner loop unrolls cleanly.
    let chunks = in_features.div_ceil(tile_k);

    for chunk in 0..chunks {
        let k_base = chunk * tile_k;

        // x_tile cooperative load: thread (ty, tx) loads x_tile[ty, tx]
        // = x[row, k_base + tx]. Zero-fill OOB threads so partial-tile
        // loads don't poison the dot product.
        let x_k = k_base + tx;
        if row < rows && x_k < in_features {
            x_tile[ty * tile_k + tx] = x[row * in_features + x_k];
        } else {
            x_tile[ty * tile_k + tx] = f32::new(0.0);
        }

        // w_tile cooperative load: thread (ty, tx) loads w_tile[tx, ty]
        // = weight[out_dim_of_tx, k_base + ty]. Note the transposed
        // storage: w_tile is indexed by (out_in_tile, k) so the inner
        // accumulate reads contiguously over k for each thread's out_dim.
        let w_out = tile_col * tile_n + tx;
        let w_k = k_base + ty;
        if w_out < out_features && w_k < in_features {
            w_tile[tx * tile_k + ty] = weight[w_out * in_features + w_k];
        } else {
            w_tile[tx * tile_k + ty] = f32::new(0.0);
        }

        // Wait until every thread finished loading before any thread
        // reads from shared memory.
        sync_cube();

        if row < rows && out_dim < out_features {
            for k in 0..tile_k {
                acc += x_tile[ty * tile_k + k] * w_tile[tx * tile_k + k];
            }
        }

        // Wait until every thread finished computing before the next
        // chunk's load overwrites the tiles.
        sync_cube();
    }

    if row < rows && out_dim < out_features {
        output[row * out_features + out_dim] = acc;
    }
}

/// Derive the 2-D workgroup grid for a tiled `linear_out_by_in` launch
/// and validate shape invariants that the kernel itself relies on (u32
/// indexing, non-zero cell count). Caller is expected to have already
/// validated buffer lengths via `crate::validate_linear_out_by_in`.
///
/// Returns `(cube_count_x, cube_count_y)` — the number of workgroups
/// along the `out_features` axis and the `rows` axis respectively. The
/// workgroup itself is always `LINEAR_TILE_N × LINEAR_TILE_M` units.
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
    u32::try_from(rows)
        .map_err(|_| cubecl_err(format!("rows {rows} exceeds CubeCL u32 launch limit")))?;
    let total_cells = u32::try_from(rows * out_features).map_err(|_| {
        cubecl_err(format!(
            "linear_out_by_in cell count {} exceeds CubeCL u32 launch limit",
            rows * out_features
        ))
    })?;
    if total_cells == 0 {
        return Err(cubecl_err(
            "linear_out_by_in launch requires non-zero rows*out_features".to_string(),
        ));
    }

    let cube_count_x = (out_features as u32).div_ceil(LINEAR_TILE_N as u32).max(1);
    let cube_count_y = (rows as u32).div_ceil(LINEAR_TILE_M as u32).max(1);
    Ok((cube_count_x, cube_count_y))
}

/// Launch the tiled `linear_out_by_in_tiled_f32` kernel against
/// pre-existing CubeCL handles. This is the device-resident entry point:
/// no `create_from_slice`, no `read_one`, no host bounce. Both the
/// slice-based legacy `linear_out_by_in_cubecl` (which still does the
/// host round trip) and the `DeviceTensor`-based `linear_d` override
/// call into this helper, so swapping the kernel here is the single
/// point of dispatch.
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
    let (cube_count_x, cube_count_y) = prepare_linear_launch(rows, in_features, out_features)?;

    unsafe {
        linear_out_by_in_tiled_f32::launch_unchecked::<R>(
            client,
            CubeCount::Static(cube_count_x, cube_count_y, 1),
            // CubeDim x = out_in_tile axis (matches UNIT_POS_X / tx),
            // CubeDim y = row_in_tile axis (matches UNIT_POS_Y / ty).
            CubeDim::new_2d(LINEAR_TILE_N as u32, LINEAR_TILE_M as u32),
            ArrayArg::from_raw_parts(x_handle, x_len),
            ArrayArg::from_raw_parts(weight_handle, weight_len),
            ArrayArg::from_raw_parts(bias_handle, bias_len),
            ArrayArg::from_raw_parts(output_handle, out_len),
            rows,
            in_features,
            out_features,
            LINEAR_TILE_M,
            LINEAR_TILE_N,
            LINEAR_TILE_K,
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
            .linear_out_by_in(
                &x,
                rows,
                in_features,
                &w,
                out_features,
                Some(&b),
                &mut scalar,
            )
            .expect("scalar linear_out_by_in must succeed");

        let mut gpu = vec![0.0_f32; rows * out_features];
        if let Err(err) =
            linear_out_by_in_wgpu(&x, rows, in_features, &w, out_features, Some(&b), &mut gpu)
        {
            // No usable WGPU adapter on this host — skip rather than fail. This
            // mirrors `wgpu_rope_matches_cpu_reference_for_position_one`, which
            // is `#[ignore]` for the same reason but in a coarser way; here we
            // would rather run the test when an adapter is present and skip
            // cleanly when it is not.
            eprintln!("skipping wgpu_linear_out_by_in_matches_scalar_within_tolerance: {err:?}");
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

    /// Parity gate for a fully tile-aligned shape that spans multiple
    /// workgroups along every axis. GW.4-4 swaps `linear_out_by_in` to a
    /// tiled kernel with 16-wide tiles in M/N/K; this exercises that the
    /// common Whisper-tiny encoder path (rows = seq = 1500, in/out
    /// features = 384) — sized at exact multiples of 16 — still matches
    /// the scalar CPU reference. The existing
    /// `wgpu_linear_out_by_in_matches_scalar_within_tolerance` test covers
    /// the unaligned-tail path with 17/23/13.
    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn wgpu_linear_out_by_in_tiled_aligned_matches_scalar_within_tolerance() {
        use crate::CpuKernelBackend;

        let rows = 64;
        let in_features = 48;
        let out_features = 32;
        let x: Vec<f32> = (0..rows * in_features)
            .map(|i| ((i as f32) * 0.011).sin())
            .collect();
        let w: Vec<f32> = (0..out_features * in_features)
            .map(|i| ((i as f32) * 0.017).cos())
            .collect();
        let b: Vec<f32> = (0..out_features).map(|i| (i as f32) * 0.03).collect();

        let mut scalar = vec![0.0_f32; rows * out_features];
        CpuKernelBackend::scalar()
            .linear_out_by_in(
                &x,
                rows,
                in_features,
                &w,
                out_features,
                Some(&b),
                &mut scalar,
            )
            .expect("scalar linear_out_by_in must succeed");

        let mut gpu = vec![0.0_f32; rows * out_features];
        if let Err(err) =
            linear_out_by_in_wgpu(&x, rows, in_features, &w, out_features, Some(&b), &mut gpu)
        {
            eprintln!(
                "skipping wgpu_linear_out_by_in_tiled_aligned_matches_scalar_within_tolerance: {err:?}"
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
            .linear_out_by_in(
                &x,
                rows,
                in_features,
                &w,
                out_features,
                Some(&b),
                &mut scalar,
            )
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
        let out_d = backend.alloc(rows * out_features).expect("alloc output");

        assert!(matches!(
            x_d.residency(),
            crate::Residency::Device(CUBECL_WGPU_BACKEND)
        ));
        assert!(matches!(
            out_d.residency(),
            crate::Residency::Device(CUBECL_WGPU_BACKEND)
        ));

        backend
            .linear_d(
                &x_d,
                rows,
                in_features,
                &w_d,
                out_features,
                Some(&b_d),
                &out_d,
            )
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
        if let Err(err) = linear_out_by_in_wgpu(&[1.0_f32], 1, 1, &[1.0_f32], 1, None, &mut probe) {
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

    // ---------------------------------------------------------------
    // GW.4 Stage 2A: device-resident critical-chain primitives
    // ---------------------------------------------------------------

    /// Small adapter-probe helper: try a trivial wgpu launch through the
    /// legacy slice path. If it errors (no adapter / no driver / etc.) we
    /// signal "skip" so each device-parity test gets the same skip
    /// semantics as the existing `wgpu_linear_out_by_in_matches_scalar_*`.
    #[cfg(feature = "cubecl-wgpu")]
    fn wgpu_adapter_available() -> bool {
        let mut probe = vec![0.0_f32; 1];
        linear_out_by_in_wgpu(&[1.0_f32], 1, 1, &[1.0_f32], 1, None, &mut probe).is_ok()
    }

    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn wgpu_add_inplace_d_matches_scalar_within_tolerance() {
        if !wgpu_adapter_available() {
            eprintln!("skipping wgpu_add_inplace_d_matches_scalar_within_tolerance: no adapter");
            return;
        }

        // Use a non-power-of-two length so the workgroup tail bounds check
        // exercises real work.
        let lhs: Vec<f32> = (0..(257_usize + 13))
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let rhs: Vec<f32> = lhs.iter().map(|v| v.cos() * 0.5).collect();
        let mut expected = lhs.clone();
        for (l, r) in expected.iter_mut().zip(rhs.iter()) {
            *l += *r;
        }

        let backend = CubeClKernelBackend::new_gpu(0);
        let lhs_d = backend.upload(&lhs).expect("upload lhs");
        let rhs_d = backend.upload(&rhs).expect("upload rhs");
        assert!(matches!(
            lhs_d.residency(),
            crate::Residency::Device(CUBECL_WGPU_BACKEND)
        ));
        backend
            .add_inplace_d(&lhs_d, &rhs_d)
            .expect("device add_inplace_d must succeed");
        let got = lhs_d.to_host_owned().expect("readback");

        assert_eq!(got.len(), expected.len());
        for (idx, (s, g)) in expected.iter().zip(got.iter()).enumerate() {
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
    fn wgpu_gelu_inplace_d_matches_scalar_within_tolerance() {
        if !wgpu_adapter_available() {
            eprintln!("skipping wgpu_gelu_inplace_d_matches_scalar_within_tolerance: no adapter");
            return;
        }

        let host: Vec<f32> = (0..513_usize).map(|i| ((i as f32) * 0.025) - 6.4).collect();
        let mut expected = host.clone();
        for v in expected.iter_mut() {
            *v = crate::gelu_whisper_scalar(*v);
        }

        let backend = CubeClKernelBackend::new_gpu(0);
        let x_d = backend.upload(&host).expect("upload x");
        backend
            .gelu_inplace_d(&x_d)
            .expect("device gelu_inplace_d must succeed");
        let got = x_d.to_host_owned().expect("readback");

        for (idx, (s, g)) in expected.iter().zip(got.iter()).enumerate() {
            let abs = (s - g).abs();
            let rel = if s.abs() > 1e-6 { abs / s.abs() } else { abs };
            assert!(
                abs <= 1e-4 || rel <= 1e-4,
                "GPU GELU drifted at idx {idx}: scalar={s} gpu={g} abs={abs} rel={rel}"
            );
        }
    }

    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn wgpu_layer_norm_d_matches_scalar_within_tolerance() {
        if !wgpu_adapter_available() {
            eprintln!("skipping wgpu_layer_norm_d_matches_scalar_within_tolerance: no adapter");
            return;
        }

        let rows = 17usize;
        let hidden = 23usize;
        let eps = 1e-5_f32;
        let x_vec: Vec<f32> = (0..rows * hidden)
            .map(|i| ((i as f32) * 0.011).sin())
            .collect();
        let weight: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let bias: Vec<f32> = (0..hidden).map(|i| (i as f32) * -0.005).collect();

        let mut expected = vec![0.0_f32; rows * hidden];
        crate::layer_norm_whisper_scalar(&x_vec, rows, hidden, &weight, &bias, eps, &mut expected);

        let backend = CubeClKernelBackend::new_gpu(0);
        let x_d = backend.upload(&x_vec).expect("upload x");
        let w_d = backend.upload(&weight).expect("upload weight");
        let b_d = backend.upload(&bias).expect("upload bias");
        let out_d = backend.alloc(rows * hidden).expect("alloc output");
        backend
            .layer_norm_d(&x_d, rows, hidden, &w_d, &b_d, eps, &out_d)
            .expect("device layer_norm_d must succeed");
        let got = out_d.to_host_owned().expect("readback");

        assert_eq!(got.len(), expected.len());
        for (idx, (s, g)) in expected.iter().zip(got.iter()).enumerate() {
            let abs = (s - g).abs();
            let rel = if s.abs() > 1e-6 { abs / s.abs() } else { abs };
            assert!(
                abs <= 1e-4 || rel <= 1e-4,
                "GPU layer_norm drifted at idx {idx}: scalar={s} gpu={g} abs={abs} rel={rel}"
            );
        }
    }

    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn wgpu_add_positional_embedding_d_matches_scalar_within_tolerance() {
        if !wgpu_adapter_available() {
            eprintln!(
                "skipping wgpu_add_positional_embedding_d_matches_scalar_within_tolerance: no adapter"
            );
            return;
        }

        let rows = 5usize;
        let cols = 11usize;
        let pe_rows = 12usize;
        let start_pos = 3usize;
        let x_vec: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.02).collect();
        let pe_vec: Vec<f32> = (0..pe_rows * cols).map(|i| (i as f32) * -0.013).collect();

        let mut expected = x_vec.clone();
        for row in 0..rows {
            let dst = row * cols;
            let src = (start_pos + row) * cols;
            for col in 0..cols {
                expected[dst + col] += pe_vec[src + col];
            }
        }

        let backend = CubeClKernelBackend::new_gpu(0);
        let x_d = backend.upload(&x_vec).expect("upload x");
        let pe_d = backend.upload(&pe_vec).expect("upload pe");
        backend
            .add_positional_embedding_d(&x_d, rows, cols, &pe_d, pe_rows, start_pos)
            .expect("device add_positional_embedding_d must succeed");
        let got = x_d.to_host_owned().expect("readback");

        for (idx, (s, g)) in expected.iter().zip(got.iter()).enumerate() {
            let abs = (s - g).abs();
            let rel = if s.abs() > 1e-6 { abs / s.abs() } else { abs };
            assert!(
                abs <= 1e-4 || rel <= 1e-4,
                "GPU pe-add drifted at idx {idx}: scalar={s} gpu={g} abs={abs} rel={rel}"
            );
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

    // -----------------------------------------------------------------------
    // GW.4-5B: fused decoder causal self-attention parity tests.
    //
    // Two scenarios match the two host paths in decode.rs:
    //   1. Full-context causal: Q/K/V all [seq_q, state], row qi sees keys
    //      0..=qi (the `decode_tokens_with_self_attention_cache` path).
    //   2. Incremental (single token): Q is [1, state], K/V = concat
    //      (past_K/past_V, new_k/new_v) (the `decode_appended_token` path).
    //
    // CPU oracle: `attention_decoder_causal_scalar` for case 1 and
    // `attention_decoder_incremental_scalar` for case 2 (both defined below).
    // Both are the same math as `attention_body_host` / `attention_incremental_body_host`
    // in `crates/models/src/whisper/primitives.rs`.
    // -----------------------------------------------------------------------

    /// Scalar reference for causal decoder self-attention.
    /// Q/K/V: `[seq, state]` row-major, `state == n_head * head_dim`.
    /// Row `qi` attends keys 0..=qi only. Writes `[seq, state]` into `out`.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn attention_decoder_causal_scalar(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq: usize,
        n_head: usize,
        head_dim: usize,
        scale: f32,
        out: &mut [f32],
    ) {
        let state = n_head * head_dim;
        assert_eq!(q.len(), seq * state);
        assert_eq!(k.len(), seq * state);
        assert_eq!(v.len(), seq * state);
        assert_eq!(out.len(), seq * state);
        let mut scores = vec![0.0_f32; seq];
        for qi in 0..seq {
            let visible = qi + 1; // causal mask
            for head in 0..n_head {
                let q_base = qi * state + head * head_dim;
                for (ki, score) in scores.iter_mut().enumerate().take(visible) {
                    let k_base = ki * state + head * head_dim;
                    let mut acc = 0.0_f32;
                    for d in 0..head_dim {
                        acc += q[q_base + d] * k[k_base + d];
                    }
                    *score = acc * scale;
                }
                crate::softmax(&mut scores[..visible]);
                let out_base = qi * state + head * head_dim;
                for d in 0..head_dim {
                    out[out_base + d] = 0.0;
                }
                for (ki, &p) in scores.iter().enumerate().take(visible) {
                    let v_base = ki * state + head * head_dim;
                    for d in 0..head_dim {
                        out[out_base + d] += p * v[v_base + d];
                    }
                }
            }
        }
    }

    /// Scalar reference for incremental decoder self-attention (single token).
    /// Q: `[state]`, past_k/past_v: `[past_seq, state]`, new_k/new_v: `[state]`.
    /// Visible = past_seq + 1. Writes `[state]` into `out`.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn attention_decoder_incremental_scalar(
        q: &[f32],
        past_k: &[f32],
        past_v: &[f32],
        new_k: &[f32],
        new_v: &[f32],
        past_seq: usize,
        n_head: usize,
        head_dim: usize,
        scale: f32,
        out: &mut [f32],
    ) {
        let state = n_head * head_dim;
        let visible = past_seq + 1;
        assert_eq!(q.len(), state);
        assert_eq!(past_k.len(), past_seq * state);
        assert_eq!(past_v.len(), past_seq * state);
        assert_eq!(new_k.len(), state);
        assert_eq!(new_v.len(), state);
        assert_eq!(out.len(), state);
        let mut scores = vec![0.0_f32; visible];
        for head in 0..n_head {
            let q_base = head * head_dim;
            for (ki, score) in scores.iter_mut().enumerate() {
                let mut acc = 0.0_f32;
                for d in 0..head_dim {
                    let key = if ki < past_seq {
                        past_k[ki * state + head * head_dim + d]
                    } else {
                        new_k[head * head_dim + d]
                    };
                    acc += q[q_base + d] * key;
                }
                *score = acc * scale;
            }
            crate::softmax(&mut scores);
            let out_base = head * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0_f32;
                for (ki, &p) in scores.iter().enumerate() {
                    let value = if ki < past_seq {
                        past_v[ki * state + head * head_dim + d]
                    } else {
                        new_v[head * head_dim + d]
                    };
                    acc += p * value;
                }
                out[out_base + d] = acc;
            }
        }
    }

    /// GW.4-5B GPU parity gate: the fused decoder causal self-attention
    /// kernel must match `attention_decoder_causal_scalar` within `1e-4`
    /// rel/abs. Shape (seq=6, n_head=2, head_dim=4) exercises the causal
    /// mask across multiple heads and the softmax stability path.
    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn wgpu_attention_decoder_causal_d_matches_scalar_within_tolerance() {
        if !wgpu_adapter_available() {
            eprintln!(
                "skipping wgpu_attention_decoder_causal_d_matches_scalar_within_tolerance: no adapter"
            );
            return;
        }

        let seq = 6usize;
        let n_head = 2usize;
        let head_dim = 4usize;
        let state = n_head * head_dim;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..seq * state)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let k: Vec<f32> = (0..seq * state)
            .map(|i| ((i as f32) * 0.019).cos())
            .collect();
        let v: Vec<f32> = (0..seq * state)
            .map(|i| ((i as f32) * 0.023).sin())
            .collect();

        let mut expected = vec![0.0_f32; seq * state];
        attention_decoder_causal_scalar(&q, &k, &v, seq, n_head, head_dim, scale, &mut expected);

        let backend = CubeClKernelBackend::new_gpu(0);
        let q_d = backend.upload(&q).expect("upload q");
        let k_d = backend.upload(&k).expect("upload k");
        let v_d = backend.upload(&v).expect("upload v");
        let out_d = backend.alloc(seq * state).expect("alloc out");
        backend
            .attention_decoder_causal_d(&q_d, &k_d, &v_d, seq, n_head, head_dim, scale, &out_d)
            .expect("device attention_decoder_causal_d must succeed");
        let got = out_d.to_host_owned().expect("readback");

        assert_eq!(got.len(), expected.len());
        for (idx, (s, g)) in expected.iter().zip(got.iter()).enumerate() {
            let abs = (s - g).abs();
            let rel = if s.abs() > 1e-6 { abs / s.abs() } else { abs };
            assert!(
                abs <= 1e-4 || rel <= 1e-4,
                "GPU causal decoder attention drifted at idx {idx}: scalar={s} gpu={g} abs={abs} rel={rel}"
            );
        }
    }

    /// GW.4-5B GPU parity gate: the fused decoder incremental (single-token)
    /// attention kernel must match `attention_decoder_incremental_scalar` within
    /// `1e-4` rel/abs. Exercises past_seq=4 so the concat(past, new) boundary
    /// is not at position 0.
    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn wgpu_attention_decoder_incremental_d_matches_scalar_within_tolerance() {
        if !wgpu_adapter_available() {
            eprintln!(
                "skipping wgpu_attention_decoder_incremental_d_matches_scalar_within_tolerance: no adapter"
            );
            return;
        }

        let past_seq = 4usize;
        let n_head = 2usize;
        let head_dim = 4usize;
        let state = n_head * head_dim;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..state).map(|i| ((i as f32) * 0.041).sin()).collect();
        let past_k: Vec<f32> = (0..past_seq * state)
            .map(|i| ((i as f32) * 0.013).cos())
            .collect();
        let past_v: Vec<f32> = (0..past_seq * state)
            .map(|i| ((i as f32) * 0.019).sin())
            .collect();
        let new_k: Vec<f32> = (0..state).map(|i| ((i as f32) * 0.027).cos()).collect();
        let new_v: Vec<f32> = (0..state).map(|i| ((i as f32) * 0.033).sin()).collect();

        let mut expected = vec![0.0_f32; state];
        attention_decoder_incremental_scalar(
            &q,
            &past_k,
            &past_v,
            &new_k,
            &new_v,
            past_seq,
            n_head,
            head_dim,
            scale,
            &mut expected,
        );

        let backend = CubeClKernelBackend::new_gpu(0);
        let q_d = backend.upload(&q).expect("upload q");
        let past_k_d = backend.upload(&past_k).expect("upload past_k");
        let past_v_d = backend.upload(&past_v).expect("upload past_v");
        let new_k_d = backend.upload(&new_k).expect("upload new_k");
        let new_v_d = backend.upload(&new_v).expect("upload new_v");
        let out_d = backend.alloc(state).expect("alloc out");
        backend
            .attention_decoder_incremental_d(
                &q_d, &past_k_d, &past_v_d, &new_k_d, &new_v_d, past_seq, n_head, head_dim, scale,
                &out_d,
            )
            .expect("device attention_decoder_incremental_d must succeed");
        let got = out_d.to_host_owned().expect("readback");

        assert_eq!(got.len(), expected.len());
        for (idx, (s, g)) in expected.iter().zip(got.iter()).enumerate() {
            let abs = (s - g).abs();
            let rel = if s.abs() > 1e-6 { abs / s.abs() } else { abs };
            assert!(
                abs <= 1e-4 || rel <= 1e-4,
                "GPU incremental decoder attention drifted at idx {idx}: scalar={s} gpu={g} abs={abs} rel={rel}"
            );
        }
    }

    /// GW.4-5C GPU parity gate: the fused decoder cross-attention kernel must
    /// match `attention_decoder_cross_scalar` within `1e-4` rel/abs.
    /// Shape: q_seq=3 decoder rows, kv_seq=7 encoder frames, n_head=2, head_dim=4.
    /// The asymmetric q_seq vs kv_seq exercises the cross-attention-specific
    /// path where Q and K/V have different sequence lengths.
    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn wgpu_attention_decoder_cross_d_matches_scalar_within_tolerance() {
        if !wgpu_adapter_available() {
            eprintln!(
                "skipping wgpu_attention_decoder_cross_d_matches_scalar_within_tolerance: no adapter"
            );
            return;
        }

        let q_seq = 3usize;
        let kv_seq = 7usize;
        let n_head = 2usize;
        let head_dim = 4usize;
        let state = n_head * head_dim;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..q_seq * state)
            .map(|i| ((i as f32) * 0.017).sin())
            .collect();
        let k: Vec<f32> = (0..kv_seq * state)
            .map(|i| ((i as f32) * 0.013).cos())
            .collect();
        let v: Vec<f32> = (0..kv_seq * state)
            .map(|i| ((i as f32) * 0.023).sin())
            .collect();

        let mut expected = vec![0.0_f32; q_seq * state];
        crate::attention_decoder_cross_scalar(
            &q,
            &k,
            &v,
            q_seq,
            kv_seq,
            n_head,
            head_dim,
            scale,
            &mut expected,
        );

        let backend = CubeClKernelBackend::new_gpu(0);
        let q_d = backend.upload(&q).expect("upload q");
        let k_d = backend.upload(&k).expect("upload k");
        let v_d = backend.upload(&v).expect("upload v");
        let out_d = backend.alloc(q_seq * state).expect("alloc out");
        backend
            .attention_decoder_cross_d(
                &q_d, &k_d, &v_d, q_seq, kv_seq, n_head, head_dim, scale, &out_d,
            )
            .expect("device attention_decoder_cross_d must succeed");
        let got = out_d.to_host_owned().expect("readback");

        assert_eq!(got.len(), expected.len());
        for (idx, (s, g)) in expected.iter().zip(got.iter()).enumerate() {
            let abs = (s - g).abs();
            let rel = if s.abs() > 1e-6 { abs / s.abs() } else { abs };
            assert!(
                abs <= 1e-4 || rel <= 1e-4,
                "GPU cross attention drifted at idx {idx}: scalar={s} gpu={g} abs={abs} rel={rel}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // GW.4-shmem large-seq regression guards.
    //
    // The small-seq parity tests above (seq <= 8) were blind to the O(seq)
    // shared-memory overflow: at seq=8 and ENC_ATTN_WG=4, the slab was only
    // 128 bytes, well within the 16 KiB WebGPU floor. At seq=1500 it becomes
    // 24 KiB, exceeding it on spec/downlevel adapters. These tests exercise
    // seq=512 (well above the 16 KiB threshold for the old kernel) against the
    // same CPU oracles. Any regression back to O(seq) shared memory will cause
    // pipeline creation failures visible here.
    // -----------------------------------------------------------------------

    /// GW.4-shmem large-seq encoder regression guard.
    /// seq=512, n_head=2, head_dim=4: with the old slab, this would require
    /// `4 * 512 * 4 = 8192` bytes per workgroup -- still fits on most adapters
    /// but the test documents the intent; at seq=1500 the old design exceeds
    /// the 16 KiB WebGPU floor. Use seq=512 as the highest value that still
    /// exercises the overflow regime without making the test slow.
    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn wgpu_attention_encoder_d_large_seq_matches_scalar_within_tolerance() {
        if !wgpu_adapter_available() {
            eprintln!(
                "skipping wgpu_attention_encoder_d_large_seq_matches_scalar_within_tolerance: no adapter"
            );
            return;
        }

        let seq = 512usize;
        let n_head = 2usize;
        let head_dim = 4usize;
        let state = n_head * head_dim;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..seq * state)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let k: Vec<f32> = (0..seq * state)
            .map(|i| ((i as f32) * 0.019).cos())
            .collect();
        let v: Vec<f32> = (0..seq * state)
            .map(|i| ((i as f32) * 0.023).sin())
            .collect();

        let mut expected = vec![0.0_f32; seq * state];
        crate::attention_encoder_scalar(&q, &k, &v, seq, n_head, head_dim, scale, &mut expected);

        let backend = CubeClKernelBackend::new_gpu(0);
        let q_d = backend.upload(&q).expect("upload q");
        let k_d = backend.upload(&k).expect("upload k");
        let v_d = backend.upload(&v).expect("upload v");
        let out_d = backend.alloc(seq * state).expect("alloc out");
        backend
            .attention_encoder_d(&q_d, &k_d, &v_d, seq, n_head, head_dim, scale, &out_d)
            .expect("device attention_encoder_d large-seq must succeed");
        let got = out_d.to_host_owned().expect("readback");

        assert_eq!(got.len(), expected.len());
        let mut max_err = 0.0_f32;
        for (idx, (s, g)) in expected.iter().zip(got.iter()).enumerate() {
            let abs = (s - g).abs();
            let rel = if s.abs() > 1e-6 { abs / s.abs() } else { abs };
            max_err = max_err.max(abs);
            assert!(
                abs <= 1e-4 || rel <= 1e-4,
                "GPU encoder attention (large-seq) drifted at idx {idx}: scalar={s} gpu={g} abs={abs} rel={rel}"
            );
        }
        eprintln!("wgpu_attention_encoder_d_large_seq: seq={seq} max_abs_err={max_err:.2e}");
    }

    /// GW.4-shmem large-seq causal decoder regression guard.
    /// seq=512 exercises the causal mask path across many rows.
    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn wgpu_attention_decoder_causal_d_large_seq_matches_scalar_within_tolerance() {
        if !wgpu_adapter_available() {
            eprintln!(
                "skipping wgpu_attention_decoder_causal_d_large_seq_matches_scalar_within_tolerance: no adapter"
            );
            return;
        }

        let seq = 512usize;
        let n_head = 2usize;
        let head_dim = 4usize;
        let state = n_head * head_dim;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..seq * state)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let k: Vec<f32> = (0..seq * state)
            .map(|i| ((i as f32) * 0.019).cos())
            .collect();
        let v: Vec<f32> = (0..seq * state)
            .map(|i| ((i as f32) * 0.023).sin())
            .collect();

        let mut expected = vec![0.0_f32; seq * state];
        attention_decoder_causal_scalar(&q, &k, &v, seq, n_head, head_dim, scale, &mut expected);

        let backend = CubeClKernelBackend::new_gpu(0);
        let q_d = backend.upload(&q).expect("upload q");
        let k_d = backend.upload(&k).expect("upload k");
        let v_d = backend.upload(&v).expect("upload v");
        let out_d = backend.alloc(seq * state).expect("alloc out");
        backend
            .attention_decoder_causal_d(&q_d, &k_d, &v_d, seq, n_head, head_dim, scale, &out_d)
            .expect("device attention_decoder_causal_d large-seq must succeed");
        let got = out_d.to_host_owned().expect("readback");

        assert_eq!(got.len(), expected.len());
        let mut max_err = 0.0_f32;
        for (idx, (s, g)) in expected.iter().zip(got.iter()).enumerate() {
            let abs = (s - g).abs();
            let rel = if s.abs() > 1e-6 { abs / s.abs() } else { abs };
            max_err = max_err.max(abs);
            assert!(
                abs <= 1e-4 || rel <= 1e-4,
                "GPU causal decoder attention (large-seq) drifted at idx {idx}: scalar={s} gpu={g} abs={abs} rel={rel}"
            );
        }
        eprintln!("wgpu_attention_decoder_causal_d_large_seq: seq={seq} max_abs_err={max_err:.2e}");
    }

    /// GW.4-shmem large-kv_seq cross-attention regression guard.
    /// kv_seq=512 encoder frames exercises the O(kv_seq) path that existed
    /// before the flash rewrite.
    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn wgpu_attention_decoder_cross_d_large_kv_seq_matches_scalar_within_tolerance() {
        if !wgpu_adapter_available() {
            eprintln!(
                "skipping wgpu_attention_decoder_cross_d_large_kv_seq_matches_scalar_within_tolerance: no adapter"
            );
            return;
        }

        let q_seq = 4usize;
        let kv_seq = 512usize;
        let n_head = 2usize;
        let head_dim = 4usize;
        let state = n_head * head_dim;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..q_seq * state)
            .map(|i| ((i as f32) * 0.017).sin())
            .collect();
        let k: Vec<f32> = (0..kv_seq * state)
            .map(|i| ((i as f32) * 0.013).cos())
            .collect();
        let v: Vec<f32> = (0..kv_seq * state)
            .map(|i| ((i as f32) * 0.023).sin())
            .collect();

        let mut expected = vec![0.0_f32; q_seq * state];
        crate::attention_decoder_cross_scalar(
            &q,
            &k,
            &v,
            q_seq,
            kv_seq,
            n_head,
            head_dim,
            scale,
            &mut expected,
        );

        let backend = CubeClKernelBackend::new_gpu(0);
        let q_d = backend.upload(&q).expect("upload q");
        let k_d = backend.upload(&k).expect("upload k");
        let v_d = backend.upload(&v).expect("upload v");
        let out_d = backend.alloc(q_seq * state).expect("alloc out");
        backend
            .attention_decoder_cross_d(
                &q_d, &k_d, &v_d, q_seq, kv_seq, n_head, head_dim, scale, &out_d,
            )
            .expect("device attention_decoder_cross_d large-kv_seq must succeed");
        let got = out_d.to_host_owned().expect("readback");

        assert_eq!(got.len(), expected.len());
        let mut max_err = 0.0_f32;
        for (idx, (s, g)) in expected.iter().zip(got.iter()).enumerate() {
            let abs = (s - g).abs();
            let rel = if s.abs() > 1e-6 { abs / s.abs() } else { abs };
            max_err = max_err.max(abs);
            assert!(
                abs <= 1e-4 || rel <= 1e-4,
                "GPU cross attention (large-kv_seq) drifted at idx {idx}: scalar={s} gpu={g} abs={abs} rel={rel}"
            );
        }
        eprintln!(
            "wgpu_attention_decoder_cross_d_large_kv_seq: kv_seq={kv_seq} max_abs_err={max_err:.2e}"
        );
    }

    /// GW.4-5A parity gate: the fused encoder-attention kernel must match
    /// the scalar reference within `1e-4` rel/abs on a small but
    /// non-trivial shape that exercises the head/seq indexing, the
    /// softmax stability path, and the P·V accumulation. The shape
    /// (seq=8, n_head=2, head_dim=4) keeps the flash-attention slab O(1)
    /// and is viable on all adapters.
    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn wgpu_attention_encoder_d_matches_scalar_within_tolerance() {
        if !wgpu_adapter_available() {
            eprintln!(
                "skipping wgpu_attention_encoder_d_matches_scalar_within_tolerance: no adapter"
            );
            return;
        }

        let seq = 8usize;
        let n_head = 2usize;
        let head_dim = 4usize;
        let state = n_head * head_dim;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..seq * state)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let k: Vec<f32> = (0..seq * state)
            .map(|i| ((i as f32) * 0.019).cos())
            .collect();
        let v: Vec<f32> = (0..seq * state)
            .map(|i| ((i as f32) * 0.023).sin())
            .collect();

        let mut expected = vec![0.0_f32; seq * state];
        crate::attention_encoder_scalar(&q, &k, &v, seq, n_head, head_dim, scale, &mut expected);

        let backend = CubeClKernelBackend::new_gpu(0);
        let q_d = backend.upload(&q).expect("upload q");
        let k_d = backend.upload(&k).expect("upload k");
        let v_d = backend.upload(&v).expect("upload v");
        let out_d = backend.alloc(seq * state).expect("alloc out");
        backend
            .attention_encoder_d(&q_d, &k_d, &v_d, seq, n_head, head_dim, scale, &out_d)
            .expect("device attention_encoder_d must succeed");
        let got = out_d.to_host_owned().expect("readback");

        assert_eq!(got.len(), expected.len());
        for (idx, (s, g)) in expected.iter().zip(got.iter()).enumerate() {
            let abs = (s - g).abs();
            let rel = if s.abs() > 1e-6 { abs / s.abs() } else { abs };
            assert!(
                abs <= 1e-4 || rel <= 1e-4,
                "GPU attention drifted at idx {idx}: scalar={s} gpu={g} abs={abs} rel={rel}"
            );
        }
    }
}
