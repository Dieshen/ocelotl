//! Portable kernel dispatch boundary.
//!
//! # M1 layout & stride contract
//!
//! M1 kernels are **CPU reference-only** and accept **contiguous row-major
//! layout only**. Strides are not supported in M1 and will be added when GPU
//! kernels need them (M4+). When that change happens it will be deliberate and
//! breaking — every call site should be updated together. Do not silently
//! accept a stride argument that gets ignored.
//!
//! These kernels exist to make the rest of the inference path testable on a
//! laptop with no GPU. They are not optimized.
//!
//! # Pair-design notes (M1.7, 2026-05-02)
//!
//! Locked by James (driver) + Rick (reviewer):
//!
//! 1. **Boundary type:** raw slices `&[f32]` / `&mut [f32]` plus shape tuples.
//!    No `TensorView`, no `ndarray` dependency. Revisit when ≥3 kernels share
//!    the same parameter pattern.
//! 2. **Layout:** contiguous row-major only (see above).
//! 3. **Validation:** at the launch boundary, inside each kernel. Length and
//!    shape mismatches that the caller might plausibly hit at runtime become
//!    `KernelError`. Pure programmer-error invariants (e.g. an output buffer
//!    that can never be the wrong size on a contiguous layout) use
//!    `debug_assert!`. We do not extract a shared validation helper until ≥3
//!    kernels share the same pattern.

use std::{fmt::Debug, sync::Arc};

pub mod recurrent;
pub mod relpos;
pub mod rope;
pub use rope::{rope_apply_inplace, rope_apply_inplace_with_factors};

use ocelotl_core::{Device, KernelError, OcelotlError, Result, UnsupportedError};

pub mod activation;
pub mod attention;
pub mod conv;
#[cfg(target_arch = "x86_64")]
mod cpu_avx2;
mod cpu_backend;
pub mod pooling;
pub use cpu_backend::CpuKernelBackend;
#[cfg(feature = "cubecl")]
pub mod cubecl_backend;
#[cfg(feature = "_gpu")]
pub use cubecl_backend::{
    CUBECL_WGPU_BACKEND, WgpuDeviceBuffer, linear_out_by_in_wgpu, rope_apply_inplace_wgpu,
};
#[cfg(feature = "cubecl")]
pub use cubecl_backend::{CubeClKernelBackend, linear_out_by_in_cubecl, rope_apply_inplace_cubecl};
pub mod k_quant;
pub use k_quant::{GgmlKQuantKind, GgmlKQuantMatrixRef, linear_q8_k_k_quant};
pub mod layout;
pub mod mlp;
pub mod rmsnorm;
pub mod tensor;
pub use layout::transpose_2d;
pub use tensor::{DeviceBuffer, DeviceTensor, HostBorrow, HostBorrowMut, Residency};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CpuKernelMode {
    /// Original correctness-first CPU loops. This remains the default and the
    /// parity oracle for optimized CPU, GPU, and quantized kernels.
    #[default]
    Scalar,
    /// CPU loops with cache-friendlier accumulation order for hot matrix work.
    /// This path stays safe Rust and keeps the same slice/shape contract.
    Optimized,
    /// AVX2 + FMA path for `linear_out_by_in` only (other kernels still use
    /// the scalar implementation). x86_64-only at runtime; constructing a
    /// backend in this mode on a non-AVX2 CPU returns a typed Kernel error.
    /// Scalar mode remains the parity oracle — AVX2 output is validated
    /// against Scalar within a pinned tolerance on every test run.
    Avx2,
}

impl CpuKernelMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Optimized => "optimized",
            Self::Avx2 => "avx2",
        }
    }
}

#[derive(Debug, Clone)]
pub struct KernelContext {
    pub device: Device,
}

pub trait KernelBackend: Debug + Send + Sync {
    fn name(&self) -> &'static str;
    fn context(&self) -> &KernelContext;

    /// Borrow the backend's CPU thread pool, if any. Default `None`. CPU-side
    /// helpers (e.g. Whisper's attention outer loop) can use this to launch a
    /// parallel walk on the same pool the backend uses internally. GPU
    /// backends keep the default `None` because no host-side parallelism
    /// applies.
    fn cpu_thread_pool(&self) -> Option<&rayon::ThreadPool> {
        None
    }

    fn matmul(
        &self,
        a: &[f32],
        a_shape: (usize, usize),
        b: &[f32],
        b_shape: (usize, usize),
        out: &mut [f32],
    ) -> Result<()>;

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
    ) -> Result<()>;

    fn linear_q8_k_k_quant(
        &self,
        x: &[f32],
        rows: usize,
        matrix: GgmlKQuantMatrixRef<'_>,
        out: &mut [f32],
    ) -> Result<()> {
        k_quant::linear_q8_k_k_quant(x, rows, matrix, out)
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
    ) -> Result<()>;

    /// Decode-time GQA attention for one query row over a visible K/V prefix.
    /// Backends may override this to keep cache reads and attention resident;
    /// the default CPU reference avoids constructing a full causal Q matrix.
    #[allow(clippy::too_many_arguments)]
    fn scaled_dot_product_attention_incremental(
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
        attention::scaled_dot_product_attention_incremental(
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

    #[allow(clippy::too_many_arguments)]
    fn scaled_dot_product_attention_with_scale(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        out: &mut [f32],
    ) -> Result<()> {
        attention::scaled_dot_product_attention_with_scale(
            q,
            k,
            v,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            out,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn scaled_dot_product_attention_windowed(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        out: &mut [f32],
    ) -> Result<()> {
        attention::scaled_dot_product_attention_windowed(
            q,
            k,
            v,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            out,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn scaled_dot_product_attention_windowed_with_scale(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        scale: f32,
        out: &mut [f32],
    ) -> Result<()> {
        attention::scaled_dot_product_attention_windowed_with_scale(
            q,
            k,
            v,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            scale,
            out,
        )
    }

    fn rope_apply_inplace(
        &self,
        x: &mut [f32],
        head_dim: usize,
        position: usize,
        theta: f32,
    ) -> Result<()>;

    fn rope_apply_inplace_with_factors(
        &self,
        x: &mut [f32],
        head_dim: usize,
        position: usize,
        theta: f32,
        freq_factors: &[f32],
    ) -> Result<()> {
        rope_apply_inplace_with_factors(x, head_dim, position, theta, freq_factors)
    }

    fn rmsnorm(
        &self,
        x: &[f32],
        rows: usize,
        hidden: usize,
        weight: &[f32],
        epsilon: f32,
        out: &mut [f32],
    ) -> Result<()>;

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
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn mlp_gated_gelu(
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
        mlp::mlp_gated_gelu(
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

    fn vec_add(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()>;

    // ---- device-tensor surface (GW.4 Stage 1) -----------------------------
    //
    // These methods build the GPU forward path: backends that can keep
    // activations on-device should override them. Defaults wrap a host
    // `Vec<f32>` so the CPU backend works without overrides, and so any
    // GPU backend that hasn't yet implemented a primitive can fall back
    // to "readback → CPU kernel → upload" without breaking parity.

    /// Upload host data into a backend-preferred buffer. CPU returns a
    /// `Host` variant zero-copy; GPU backends override to upload.
    fn upload(&self, host: &[f32]) -> Result<DeviceTensor> {
        Ok(DeviceTensor::from_host(host.to_vec()))
    }

    /// Allocate a zero-filled buffer of `len` `f32` elements in the
    /// backend-preferred location.
    fn alloc(&self, len: usize) -> Result<DeviceTensor> {
        Ok(DeviceTensor::host_zeros(len))
    }

    /// Copy all values from `src` into `dst[dst_offset..]`. The default
    /// implementation bounces through host memory; device backends can override
    /// this to keep cache appends resident.
    fn copy_into_d(&self, src: &DeviceTensor, dst: &DeviceTensor, dst_offset: usize) -> Result<()> {
        validate_copy_into_shapes(src, dst, dst_offset)?;
        if let (Ok(src_host), Ok(mut dst_host)) =
            (src.borrow_host_slice(), dst.borrow_host_slice_mut())
        {
            dst_host[dst_offset..dst_offset + src_host.len()].copy_from_slice(&src_host);
            return Ok(());
        }
        let src_host = src.to_host_owned()?;
        let mut dst_host = dst.to_host_owned()?;
        dst_host[dst_offset..dst_offset + src_host.len()].copy_from_slice(&src_host);
        dst.write_from_host_slice(&dst_host)
    }

    /// Device-resident linear projection. `out` is caller-supplied so the
    /// caller can recycle scratch across loop iterations. The default
    /// implementation forces host readback through `to_host_owned` and
    /// then calls the existing slice-based `linear_out_by_in`, so any
    /// backend that does not override pays the round-trip but stays
    /// correct.
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
        let x_host = x.to_host_owned()?;
        let weight_host = weight.to_host_owned()?;
        let bias_host = bias.map(DeviceTensor::to_host_owned).transpose()?;
        let mut out_buf = vec![0.0_f32; rows * out_features];
        self.linear_out_by_in(
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

    /// Elementwise `lhs[i] += rhs[i]`. Lengths must match. Default impl
    /// forces a host readback of both operands and a write-back of the
    /// sum — backends override to keep the work on-device.
    fn add_inplace_d(&self, lhs: &DeviceTensor, rhs: &DeviceTensor) -> Result<()> {
        let mut lhs_host = lhs.to_host_owned()?;
        let rhs_host = rhs.to_host_owned()?;
        if lhs_host.len() != rhs_host.len() {
            return Err(kernel_err(format!(
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

    /// Elementwise GELU. Must match
    /// `crates/models/src/whisper/primitives.rs::gelu_inplace` bit-for-bit
    /// on CPU (same exact-erf approximation) and within `1e-4` rel/abs on
    /// GPU. Default impl forces readback + host compute.
    fn gelu_inplace_d(&self, x: &DeviceTensor) -> Result<()> {
        let mut host = x.to_host_owned()?;
        for v in host.iter_mut() {
            *v = gelu_whisper_scalar(*v);
        }
        x.write_from_host_slice(&host)
    }

    /// Elementwise SiLU `x = x * sigmoid(x)` (SwiGLU activation). Default
    /// reads back and runs the host scalar; GPU backends override on device.
    fn silu_inplace_d(&self, x: &DeviceTensor) -> Result<()> {
        let mut host = x.to_host_owned()?;
        mlp::silu_inplace(&mut host);
        x.write_from_host_slice(&host)
    }

    /// Elementwise in-place product `lhs *= rhs` (the gated-MLP combine).
    /// Default reads back and multiplies on host; GPU backends override.
    fn mul_inplace_d(&self, lhs: &DeviceTensor, rhs: &DeviceTensor) -> Result<()> {
        let mut lhs_host = lhs.to_host_owned()?;
        let rhs_host = rhs.to_host_owned()?;
        if lhs_host.len() != rhs_host.len() {
            return Err(kernel_err(format!(
                "mul_inplace_d length mismatch: lhs={} rhs={}",
                lhs_host.len(),
                rhs_host.len()
            )));
        }
        for (l, r) in lhs_host.iter_mut().zip(rhs_host.iter()) {
            *l *= *r;
        }
        lhs.write_from_host_slice(&lhs_host)
    }

    /// Per-row LayerNorm with affine. `weight` and `bias` are length
    /// `hidden`; `x` and `out` are length `rows * hidden`. Variance uses
    /// the biased estimator (divide by `hidden`, not `hidden - 1`) so this
    /// matches `crates/models/src/whisper/primitives.rs::layer_norm`
    /// bit-for-bit on CPU.
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
        validate_layer_norm_shapes(x, rows, hidden, weight, bias, out)?;
        let x_host = x.to_host_owned()?;
        let weight_host = weight.to_host_owned()?;
        let bias_host = bias.to_host_owned()?;
        let mut out_buf = vec![0.0_f32; rows * hidden];
        layer_norm_whisper_scalar(
            &x_host,
            rows,
            hidden,
            &weight_host,
            &bias_host,
            eps,
            &mut out_buf,
        );
        out.write_from_host_slice(&out_buf)
    }

    /// Device-resident **RMSNorm** (`out = x / sqrt(mean(x²) + eps) * weight`;
    /// no mean subtraction, no bias) — the normalization Gemma/Qwen use. The
    /// default reads back and runs the host scalar `rmsnorm`; GPU backends
    /// override it to stay on device. Distinct from [`Self::layer_norm_d`],
    /// which is standard LayerNorm and not interchangeable here.
    fn rmsnorm_d(
        &self,
        x: &DeviceTensor,
        rows: usize,
        hidden: usize,
        weight: &DeviceTensor,
        eps: f32,
        out: &DeviceTensor,
    ) -> Result<()> {
        let x_host = x.to_host_owned()?;
        let weight_host = weight.to_host_owned()?;
        let mut out_buf = vec![0.0_f32; rows * hidden];
        rmsnorm::rmsnorm(&x_host, rows, hidden, &weight_host, eps, &mut out_buf)?;
        out.write_from_host_slice(&out_buf)
    }

    /// Whisper-style encoder self-attention on device handles. `q`, `k`, and
    /// `v` are pre-projected activations of shape `[seq, state]` row-major
    /// where `state == n_head * head_dim`; inside each row the heads are
    /// laid out contiguously (head 0 occupies `head_dim` cells, head 1 the
    /// next `head_dim`, and so on). `output` has the same `[seq, state]`
    /// layout. `scale` is typically `1.0 / sqrt(head_dim as f32)`.
    ///
    /// This is encoder-only: there is no causal mask, no GQA, and the
    /// query/key/value sequence lengths are all equal (`seq`). Decoder
    /// attention (causal + KV cache + cross-attention) is a separate
    /// surface; for now those paths stay on host.
    ///
    /// The default implementation forces host readback through
    /// `to_host_owned` and runs the scalar reference, so any backend that
    /// hasn't overridden this method stays correct but pays the round-trip.
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
        validate_attention_encoder_shapes(q, k, v, seq, n_head, head_dim, output)?;
        let q_host = q.to_host_owned()?;
        let k_host = k.to_host_owned()?;
        let v_host = v.to_host_owned()?;
        let mut out_buf = vec![0.0_f32; seq * n_head * head_dim];
        attention_encoder_scalar(
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

    /// Add a slice of a positional-embedding table into `x` in place:
    /// `x[r * cols + c] += pe[(start_pos + r) * cols + c]`. `pe` has shape
    /// `[pe_rows, cols]`; `start_pos + rows <= pe_rows` must hold.
    fn add_positional_embedding_d(
        &self,
        x: &DeviceTensor,
        rows: usize,
        cols: usize,
        pe: &DeviceTensor,
        pe_rows: usize,
        start_pos: usize,
    ) -> Result<()> {
        validate_add_positional_embedding_shapes(x, rows, cols, pe, pe_rows, start_pos)?;
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

    /// Whisper decoder causal self-attention on device handles (full context).
    ///
    /// `q`, `k`, `v`: `[seq, state]` row-major where `state == n_head * head_dim`.
    /// Row `qi` attends only keys `0..=qi` (causal mask). `output` has the
    /// same `[seq, state]` shape. `scale` is typically `1 / sqrt(head_dim)`.
    ///
    /// This is the full-context path used by `decode_tokens_with_self_attention_cache`.
    /// The caller extracts K/V from the result to build the host self-attention
    /// cache; that extraction stays on host because the cache is host-resident.
    ///
    /// The default implementation forces host readback and runs
    /// `attention_decoder_causal_scalar`, so any backend that has not yet
    /// overridden this method stays correct but pays the round-trip.
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
        validate_attention_decoder_causal_shapes(q, k, v, seq, n_head, head_dim, output)?;
        let q_host = q.to_host_owned()?;
        let k_host = k.to_host_owned()?;
        let v_host = v.to_host_owned()?;
        let mut out_buf = vec![0.0_f32; seq * n_head * head_dim];
        attention_decoder_causal_scalar(
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

    /// Whisper decoder incremental self-attention on device handles (single token).
    ///
    /// `q`: `[state]`, `past_k`/`past_v`: `[past_seq, state]`,
    /// `new_k`/`new_v`: `[state]`. Visible = `past_seq + 1`. `output`: `[state]`.
    /// `scale` is typically `1 / sqrt(head_dim)`.
    ///
    /// This is the append path used by `decode_appended_token`. The KV-cache
    /// append (`past_k` grow by one row) is performed by the caller on host
    /// before or after this call; the kernel only reads the past cache.
    ///
    /// The default implementation forces host readback and runs
    /// `attention_decoder_incremental_scalar`.
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
        validate_attention_decoder_incremental_shapes(
            q, past_k, past_v, new_k, new_v, past_seq, n_head, head_dim, output,
        )?;
        let q_host = q.to_host_owned()?;
        let past_k_host = past_k.to_host_owned()?;
        let past_v_host = past_v.to_host_owned()?;
        let new_k_host = new_k.to_host_owned()?;
        let new_v_host = new_v.to_host_owned()?;
        let mut out_buf = vec![0.0_f32; n_head * head_dim];
        attention_decoder_incremental_scalar(
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

    /// Whisper decoder incremental self-attention over a fixed-capacity K/V
    /// cache. `visible_seq` rows at the beginning of `key_cache`/`value_cache`
    /// are visible, and `q` is the query for the final visible row.
    ///
    /// This is equivalent to `attention_decoder_incremental_d` after the caller
    /// has already copied the new K/V row into the cache. The default
    /// implementation reads the cache prefix back and computes the scalar
    /// reference path.
    #[allow(clippy::too_many_arguments)]
    fn attention_decoder_incremental_cache_d(
        &self,
        q: &DeviceTensor,
        key_cache: &DeviceTensor,
        value_cache: &DeviceTensor,
        visible_seq: usize,
        cache_capacity: usize,
        n_head: usize,
        head_dim: usize,
        scale: f32,
        output: &DeviceTensor,
    ) -> Result<()> {
        validate_attention_decoder_incremental_cache_shapes(
            q,
            key_cache,
            value_cache,
            visible_seq,
            cache_capacity,
            n_head,
            head_dim,
            output,
        )?;
        let state = n_head * head_dim;
        let visible_len = visible_seq * state;
        let key_host = key_cache.to_host_owned()?;
        let value_host = value_cache.to_host_owned()?;
        let q_host = q.to_host_owned()?;
        let mut out_buf = vec![0.0_f32; state];
        attention_decoder_incremental_cache_scalar(
            &q_host,
            &key_host[..visible_len],
            &value_host[..visible_len],
            visible_seq,
            n_head,
            head_dim,
            scale,
            &mut out_buf,
        );
        output.write_from_host_slice(&out_buf)
    }

    /// Append the current K/V row into a fixed-capacity cache, then run
    /// incremental decoder self-attention over the visible prefix. Backends can
    /// fuse the row write with the attention kernel to avoid separate copy
    /// launches on autoregressive decode.
    #[allow(clippy::too_many_arguments)]
    fn attention_decoder_incremental_cache_append_d(
        &self,
        q: &DeviceTensor,
        key_cache: &DeviceTensor,
        value_cache: &DeviceTensor,
        new_k: &DeviceTensor,
        new_v: &DeviceTensor,
        past_seq: usize,
        cache_capacity: usize,
        n_head: usize,
        head_dim: usize,
        scale: f32,
        output: &DeviceTensor,
    ) -> Result<()> {
        validate_attention_decoder_incremental_cache_append_shapes(
            q,
            key_cache,
            value_cache,
            new_k,
            new_v,
            past_seq,
            cache_capacity,
            n_head,
            head_dim,
            output,
        )?;
        let state = n_head * head_dim;
        let dst_offset = past_seq.checked_mul(state).ok_or_else(|| {
            kernel_err(
                "attention_decoder_incremental_cache_append_d past_seq*state overflowed usize",
            )
        })?;
        self.copy_into_d(new_k, key_cache, dst_offset)?;
        self.copy_into_d(new_v, value_cache, dst_offset)?;
        self.attention_decoder_incremental_cache_d(
            q,
            key_cache,
            value_cache,
            past_seq + 1,
            cache_capacity,
            n_head,
            head_dim,
            scale,
            output,
        )
    }

    /// Whisper decoder cross-attention on device handles.
    ///
    /// Q comes from the decoder hidden state: shape `[q_seq, state]` where
    /// `state == n_head * head_dim`. K and V come from the encoder output
    /// (precomputed per-sequence in `WhisperEncodedAudio`): shape
    /// `[kv_seq, state]`. There is **no causal mask** — each decoder query
    /// row attends all `kv_seq` encoder positions freely. `output`: `[q_seq, state]`.
    ///
    /// `q_seq` is the number of decoder tokens being processed (may be 1 for
    /// the incremental path or > 1 for the full-context path).
    /// `kv_seq` is the number of encoder frames (audio context length).
    ///
    /// The default implementation forces host readback and runs
    /// `attention_decoder_cross_scalar`.
    ///
    /// GW.4-5C: replaces the `attention_body_host(causal=false)` host bounce
    /// in both `decode_tokens_with_self_attention_cache` and
    /// `decode_appended_token` so cross-attention stays on device.
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
        validate_attention_decoder_cross_shapes(q, k, v, q_seq, kv_seq, n_head, head_dim, output)?;
        let q_host = q.to_host_owned()?;
        let k_host = k.to_host_owned()?;
        let v_host = v.to_host_owned()?;
        let mut out_buf = vec![0.0_f32; q_seq * n_head * head_dim];
        attention_decoder_cross_scalar(
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

/// Whisper's exact-erf GELU. Mirrors
/// `crates/models/src/whisper/primitives.rs::gelu` bit-for-bit so the
/// kernels-crate `gelu_inplace_d` CPU path is a parity oracle for any
/// GPU implementation.
#[inline]
pub(crate) fn gelu_whisper_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_whisper_scalar(x / std::f32::consts::SQRT_2))
}

#[inline]
pub(crate) fn erf_whisper_scalar(x: f32) -> f32 {
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp());
    sign * y
}

/// Scalar LayerNorm matching `whisper::primitives::layer_norm` op-for-op
/// (biased variance, `1.0 / sqrt(var + eps)`, then `(x - mean) * inv_std
/// * weight + bias`).
pub(crate) fn layer_norm_whisper_scalar(
    x: &[f32],
    rows: usize,
    hidden: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    out: &mut [f32],
) {
    for row in 0..rows {
        let start = row * hidden;
        let values = &x[start..start + hidden];
        let mean = values.iter().sum::<f32>() / hidden as f32;
        let variance = values
            .iter()
            .map(|v| {
                let delta = *v - mean;
                delta * delta
            })
            .sum::<f32>()
            / hidden as f32;
        let inv_std = 1.0_f32 / (variance + eps).sqrt();
        for col in 0..hidden {
            out[start + col] = ((x[start + col] - mean) * inv_std) * weight[col] + bias[col];
        }
    }
}

/// Scalar Whisper encoder self-attention. Must produce the same numerical
/// result as `crates/models/src/whisper/primitives.rs::attention_body_host`
/// when invoked with `q_seq == kv_seq == seq` and `causal == false`. Layout
/// (`[seq, state]` with state == n_head * head_dim) is the parity oracle
/// for both the CPU `attention_encoder_d` override and the GPU cube kernel.
///
/// Math: per `(query_row, head)`:
///   1. dot product `Q · K^T` scaled by `scale`
///   2. numerically-stable softmax across all `seq` keys
///   3. probability-weighted sum of `V`
///
/// Same operation order (subtract row max → exp → sum → divide → P·V) as
/// the per-row host body so the encoder forward stays bit-stable.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_encoder_scalar(
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
    debug_assert_eq!(q.len(), seq * state);
    debug_assert_eq!(k.len(), seq * state);
    debug_assert_eq!(v.len(), seq * state);
    debug_assert_eq!(out.len(), seq * state);

    let mut scores = vec![0.0_f32; seq];
    for qi in 0..seq {
        for head in 0..n_head {
            let q_base = qi * state + head * head_dim;
            // Pass 1: scaled dot products into scores.
            for (ki, score) in scores.iter_mut().enumerate() {
                let k_base = ki * state + head * head_dim;
                let mut acc = 0.0_f32;
                for d in 0..head_dim {
                    acc += q[q_base + d] * k[k_base + d];
                }
                *score = acc * scale;
            }
            // Pass 2: numerically stable softmax — subtract row max,
            // exp, then normalize. Mirrors `softmax(&mut scores)`.
            softmax(&mut scores);
            // Pass 3: probability-weighted accumulation of V into the
            // output row's head slice.
            let out_base = qi * state + head * head_dim;
            for d in 0..head_dim {
                out[out_base + d] = 0.0;
            }
            for (ki, &p) in scores.iter().enumerate() {
                let v_base = ki * state + head * head_dim;
                for d in 0..head_dim {
                    out[out_base + d] += p * v[v_base + d];
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn attention_encoder_parallel(
    pool: &rayon::ThreadPool,
    mode: CpuKernelMode,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
) {
    use rayon::prelude::*;

    if seq == 0 {
        return;
    }

    let state = n_head * head_dim;
    debug_assert_eq!(q.len(), seq * state);
    debug_assert_eq!(k.len(), seq * state);
    debug_assert_eq!(v.len(), seq * state);
    debug_assert_eq!(out.len(), seq * state);

    let threads = pool.current_num_threads().max(1);
    let rows_per_chunk = seq.div_ceil(threads).max(1);
    let chunk_out_len = rows_per_chunk * state;

    pool.install(|| {
        out.par_chunks_mut(chunk_out_len)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let qi_start = chunk_idx * rows_per_chunk;
                let chunk_rows = out_chunk.len() / state;
                let mut scores = vec![0.0_f32; seq];

                for local_qi in 0..chunk_rows {
                    let qi = qi_start + local_qi;
                    for head in 0..n_head {
                        let q_base = qi * state + head * head_dim;
                        for (ki, score) in scores.iter_mut().enumerate() {
                            let k_base = ki * state + head * head_dim;
                            let acc = match mode {
                                CpuKernelMode::Scalar | CpuKernelMode::Optimized => {
                                    let mut acc = 0.0_f32;
                                    for d in 0..head_dim {
                                        acc += q[q_base + d] * k[k_base + d];
                                    }
                                    acc
                                }
                                CpuKernelMode::Avx2 => {
                                    // SAFETY: AVX2 mode is accepted only after
                                    // `validate_mode_supported` verifies AVX2
                                    // and FMA support. Slices are equal-length
                                    // head windows inside prevalidated Q/K.
                                    #[cfg(target_arch = "x86_64")]
                                    unsafe {
                                        cpu_avx2::dot_f32_avx2(
                                            &q[q_base..q_base + head_dim],
                                            &k[k_base..k_base + head_dim],
                                        )
                                    }
                                    #[cfg(not(target_arch = "x86_64"))]
                                    unreachable!(
                                        "Avx2 mode rejected at construction on non-x86_64"
                                    );
                                }
                            };
                            *score = acc * scale;
                        }

                        softmax(&mut scores);

                        let out_base = local_qi * state + head * head_dim;
                        match mode {
                            CpuKernelMode::Scalar | CpuKernelMode::Optimized => {
                                for d in 0..head_dim {
                                    out_chunk[out_base + d] = 0.0;
                                }
                                for (ki, &p) in scores.iter().enumerate() {
                                    let v_base = ki * state + head * head_dim;
                                    for d in 0..head_dim {
                                        out_chunk[out_base + d] += p * v[v_base + d];
                                    }
                                }
                            }
                            CpuKernelMode::Avx2 => {
                                // SAFETY: AVX2 mode is accepted only after
                                // `validate_mode_supported` verifies AVX2
                                // and FMA support. Slices are prevalidated
                                // `[seq, state]` buffers, and the output
                                // head slice is exactly `head_dim` wide.
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    cpu_avx2::attention_value_weighted_sum_avx2(
                                        &scores,
                                        v,
                                        seq,
                                        state,
                                        head * head_dim,
                                        head_dim,
                                        &mut out_chunk[out_base..out_base + head_dim],
                                    );
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                unreachable!("Avx2 mode rejected at construction on non-x86_64");
                            }
                        }
                    }
                }
            });
    });
}

#[allow(clippy::too_many_arguments)]
fn attention_decoder_cross_host(
    mode: CpuKernelMode,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_seq: usize,
    kv_seq: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
) {
    let state = n_head * head_dim;
    debug_assert_eq!(q.len(), q_seq * state);
    debug_assert_eq!(k.len(), kv_seq * state);
    debug_assert_eq!(v.len(), kv_seq * state);
    debug_assert_eq!(out.len(), q_seq * state);

    if mode != CpuKernelMode::Avx2 {
        attention_decoder_cross_scalar(q, k, v, q_seq, kv_seq, n_head, head_dim, scale, out);
        return;
    }

    let mut scores = vec![0.0_f32; kv_seq];
    for qi in 0..q_seq {
        for head in 0..n_head {
            attention_scores_avx2(
                q,
                k,
                qi * state + head * head_dim,
                kv_seq,
                state,
                head * head_dim,
                head_dim,
                scale,
                &mut scores,
            );
            softmax(&mut scores);
            let out_base = qi * state + head * head_dim;
            attention_values_avx2(
                &scores,
                v,
                kv_seq,
                state,
                head * head_dim,
                head_dim,
                &mut out[out_base..out_base + head_dim],
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn attention_decoder_cross_parallel(
    pool: &rayon::ThreadPool,
    mode: CpuKernelMode,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_seq: usize,
    kv_seq: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
) {
    use rayon::prelude::*;

    let state = n_head * head_dim;
    debug_assert_eq!(q.len(), q_seq * state);
    debug_assert_eq!(k.len(), kv_seq * state);
    debug_assert_eq!(v.len(), kv_seq * state);
    debug_assert_eq!(out.len(), q_seq * state);

    pool.install(|| {
        out.par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(chunk_idx, out_head)| {
                let qi = chunk_idx / n_head;
                let head = chunk_idx % n_head;
                let q_base = qi * state + head * head_dim;
                let head_offset = head * head_dim;
                let mut scores = vec![0.0_f32; kv_seq];

                match mode {
                    CpuKernelMode::Scalar | CpuKernelMode::Optimized => {
                        for (ki, score) in scores.iter_mut().enumerate() {
                            let k_base = ki * state + head_offset;
                            let mut acc = 0.0_f32;
                            for d in 0..head_dim {
                                acc += q[q_base + d] * k[k_base + d];
                            }
                            *score = acc * scale;
                        }
                        softmax(&mut scores);
                        out_head.fill(0.0);
                        for (ki, &p) in scores.iter().enumerate() {
                            let v_base = ki * state + head_offset;
                            for d in 0..head_dim {
                                out_head[d] += p * v[v_base + d];
                            }
                        }
                    }
                    CpuKernelMode::Avx2 => {
                        attention_scores_avx2(
                            q,
                            k,
                            q_base,
                            kv_seq,
                            state,
                            head_offset,
                            head_dim,
                            scale,
                            &mut scores,
                        );
                        softmax(&mut scores);
                        attention_values_avx2(
                            &scores,
                            v,
                            kv_seq,
                            state,
                            head_offset,
                            head_dim,
                            out_head,
                        );
                    }
                }
            });
    });
}

#[allow(clippy::too_many_arguments)]
fn attention_decoder_incremental_cache_host(
    mode: CpuKernelMode,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    visible_seq: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
) {
    let state = n_head * head_dim;
    debug_assert_eq!(q.len(), state);
    debug_assert_eq!(k.len(), visible_seq * state);
    debug_assert_eq!(v.len(), visible_seq * state);
    debug_assert_eq!(out.len(), state);

    if mode != CpuKernelMode::Avx2 {
        attention_decoder_incremental_cache_scalar(
            q,
            k,
            v,
            visible_seq,
            n_head,
            head_dim,
            scale,
            out,
        );
        return;
    }

    let mut scores = vec![0.0_f32; visible_seq];
    for head in 0..n_head {
        let head_offset = head * head_dim;
        attention_scores_avx2(
            q,
            k,
            head_offset,
            visible_seq,
            state,
            head_offset,
            head_dim,
            scale,
            &mut scores,
        );
        softmax(&mut scores);
        attention_values_avx2(
            &scores,
            v,
            visible_seq,
            state,
            head_offset,
            head_dim,
            &mut out[head_offset..head_offset + head_dim],
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn attention_decoder_incremental_cache_parallel(
    pool: &rayon::ThreadPool,
    mode: CpuKernelMode,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    visible_seq: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
) {
    use rayon::prelude::*;

    let state = n_head * head_dim;
    debug_assert_eq!(q.len(), state);
    debug_assert_eq!(k.len(), visible_seq * state);
    debug_assert_eq!(v.len(), visible_seq * state);
    debug_assert_eq!(out.len(), state);

    pool.install(|| {
        out.par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(head, out_head)| {
                let head_offset = head * head_dim;
                let mut scores = vec![0.0_f32; visible_seq];
                match mode {
                    CpuKernelMode::Scalar | CpuKernelMode::Optimized => {
                        for (ki, score) in scores.iter_mut().enumerate() {
                            let k_base = ki * state + head_offset;
                            let mut acc = 0.0_f32;
                            for d in 0..head_dim {
                                acc += q[head_offset + d] * k[k_base + d];
                            }
                            *score = acc * scale;
                        }
                        softmax(&mut scores);
                        out_head.fill(0.0);
                        for (ki, &p) in scores.iter().enumerate() {
                            let v_base = ki * state + head_offset;
                            for d in 0..head_dim {
                                out_head[d] += p * v[v_base + d];
                            }
                        }
                    }
                    CpuKernelMode::Avx2 => {
                        attention_scores_avx2(
                            q,
                            k,
                            head_offset,
                            visible_seq,
                            state,
                            head_offset,
                            head_dim,
                            scale,
                            &mut scores,
                        );
                        softmax(&mut scores);
                        attention_values_avx2(
                            &scores,
                            v,
                            visible_seq,
                            state,
                            head_offset,
                            head_dim,
                            out_head,
                        );
                    }
                }
            });
    });
}

#[allow(clippy::too_many_arguments)]
fn attention_scores_avx2(
    q: &[f32],
    k: &[f32],
    q_base: usize,
    rows: usize,
    state: usize,
    head_offset: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) {
    debug_assert!(q_base + head_dim <= q.len());
    debug_assert!(rows * state <= k.len());
    debug_assert!(head_offset + head_dim <= state);
    debug_assert!(scores.len() >= rows);

    for (row, score) in scores.iter_mut().take(rows).enumerate() {
        let k_base = row * state + head_offset;
        #[cfg(target_arch = "x86_64")]
        // SAFETY: AVX2 mode is accepted only after backend construction
        // verifies AVX2 and FMA support. Slice windows are validated by the
        // caller's tensor shape checks.
        let acc = unsafe {
            cpu_avx2::dot_f32_avx2(&q[q_base..q_base + head_dim], &k[k_base..k_base + head_dim])
        };
        #[cfg(not(target_arch = "x86_64"))]
        let acc = {
            let _ = (q, k, q_base, k_base);
            unreachable!("Avx2 mode rejected at construction on non-x86_64")
        };
        *score = acc * scale;
    }
}

fn attention_values_avx2(
    scores: &[f32],
    v: &[f32],
    rows: usize,
    state: usize,
    head_offset: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: AVX2 mode is accepted only after backend construction verifies
    // AVX2 and FMA support. Slice windows are validated by the caller's tensor
    // shape checks, and `out` is the exact output head slice.
    unsafe {
        cpu_avx2::attention_value_weighted_sum_avx2(
            scores,
            v,
            rows,
            state,
            head_offset,
            head_dim,
            out,
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    unreachable!("Avx2 mode rejected at construction on non-x86_64");
}

/// Scalar Whisper decoder causal self-attention (full-context).
///
/// Q, K, V: `[seq, state]` row-major where `state == n_head * head_dim`.
/// Row `qi` attends only keys `0..=qi` (causal mask). Writes `[seq, state]`
/// into `out`. This is the parity oracle for `attention_decoder_causal_d`.
///
/// Matches `attention_body_host` with `causal == true` and `q_seq == kv_seq`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_decoder_causal_scalar(
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
    debug_assert_eq!(q.len(), seq * state);
    debug_assert_eq!(k.len(), seq * state);
    debug_assert_eq!(v.len(), seq * state);
    debug_assert_eq!(out.len(), seq * state);

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
            softmax(&mut scores[..visible]);
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

/// Scalar single-token incremental decoder self-attention.
///
/// Q: `[state]`, past_k/past_v: `[past_seq, state]`, new_k/new_v: `[state]`.
/// Visible tokens = `past_seq + 1`. Writes `[state]` into `out`.
/// Matches `attention_incremental_body_host` from `whisper/primitives.rs`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_decoder_incremental_scalar(
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
    debug_assert_eq!(q.len(), state);
    debug_assert_eq!(past_k.len(), past_seq * state);
    debug_assert_eq!(past_v.len(), past_seq * state);
    debug_assert_eq!(new_k.len(), state);
    debug_assert_eq!(new_v.len(), state);
    debug_assert_eq!(out.len(), state);

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
        softmax(&mut scores);
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

/// Scalar incremental decoder self-attention over a fixed-capacity cache
/// prefix. This matches `attention_decoder_incremental_scalar` when `k`/`v`
/// are `past || new` and `visible_seq = past_seq + 1`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_decoder_incremental_cache_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    visible_seq: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
) {
    let state = n_head * head_dim;
    debug_assert_eq!(q.len(), state);
    debug_assert_eq!(k.len(), visible_seq * state);
    debug_assert_eq!(v.len(), visible_seq * state);
    debug_assert_eq!(out.len(), state);

    let mut scores = vec![0.0_f32; visible_seq];
    for head in 0..n_head {
        let q_base = head * head_dim;
        for (ki, score) in scores.iter_mut().enumerate() {
            let k_base = ki * state + head * head_dim;
            let mut acc = 0.0_f32;
            for d in 0..head_dim {
                acc += q[q_base + d] * k[k_base + d];
            }
            *score = acc * scale;
        }
        softmax(&mut scores);
        let out_base = head * head_dim;
        for d in 0..head_dim {
            let mut acc = 0.0_f32;
            for (ki, &p) in scores.iter().enumerate() {
                let value = v[ki * state + head * head_dim + d];
                acc += p * value;
            }
            out[out_base + d] = acc;
        }
    }
}

pub(crate) fn validate_copy_into_shapes(
    src: &DeviceTensor,
    dst: &DeviceTensor,
    dst_offset: usize,
) -> Result<()> {
    let end = dst_offset
        .checked_add(src.len())
        .ok_or_else(|| kernel_err("copy_into_d dst_offset + src.len overflowed usize"))?;
    if end > dst.len() {
        return Err(kernel_err(format!(
            "copy_into_d range {dst_offset}..{end} exceeds dst len {}",
            dst.len()
        )));
    }
    Ok(())
}

/// Shape validator for the `attention_encoder_d` device surface. Rejects
/// zero heads, mismatched `state = n_head * head_dim`, and wrong-sized
/// operands.
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_attention_encoder_shapes(
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    seq: usize,
    n_head: usize,
    head_dim: usize,
    out: &DeviceTensor,
) -> Result<()> {
    if n_head == 0 {
        return Err(kernel_err("attention_encoder_d n_head must be > 0"));
    }
    if head_dim == 0 {
        return Err(kernel_err("attention_encoder_d head_dim must be > 0"));
    }
    let state = n_head
        .checked_mul(head_dim)
        .ok_or_else(|| kernel_err("attention_encoder_d n_head*head_dim overflowed usize"))?;
    let expected = seq
        .checked_mul(state)
        .ok_or_else(|| kernel_err("attention_encoder_d seq*state overflowed usize"))?;
    for (label, len) in [
        ("q", q.len()),
        ("k", k.len()),
        ("v", v.len()),
        ("out", out.len()),
    ] {
        if len != expected {
            return Err(kernel_err(format!(
                "attention_encoder_d {label} len {len} != seq*state {expected}"
            )));
        }
    }
    Ok(())
}

/// Shape validator for `attention_decoder_causal_d`. Rejects zero heads/dim,
/// overflow in `state = n_head * head_dim`, and wrong-sized Q/K/V/out.
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_attention_decoder_causal_shapes(
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    seq: usize,
    n_head: usize,
    head_dim: usize,
    out: &DeviceTensor,
) -> Result<()> {
    if n_head == 0 {
        return Err(kernel_err("attention_decoder_causal_d n_head must be > 0"));
    }
    if head_dim == 0 {
        return Err(kernel_err(
            "attention_decoder_causal_d head_dim must be > 0",
        ));
    }
    let state = n_head
        .checked_mul(head_dim)
        .ok_or_else(|| kernel_err("attention_decoder_causal_d n_head*head_dim overflowed usize"))?;
    let expected = seq
        .checked_mul(state)
        .ok_or_else(|| kernel_err("attention_decoder_causal_d seq*state overflowed usize"))?;
    for (label, len) in [
        ("q", q.len()),
        ("k", k.len()),
        ("v", v.len()),
        ("out", out.len()),
    ] {
        if len != expected {
            return Err(kernel_err(format!(
                "attention_decoder_causal_d {label} len {len} != seq*state {expected}"
            )));
        }
    }
    Ok(())
}

/// Shape validator for `attention_decoder_incremental_d`. Rejects zero
/// heads/dim, wrong Q/new_k/new_v lengths (must be `state`), and wrong
/// past_k/past_v lengths (must be `past_seq * state`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_attention_decoder_incremental_shapes(
    q: &DeviceTensor,
    past_k: &DeviceTensor,
    past_v: &DeviceTensor,
    new_k: &DeviceTensor,
    new_v: &DeviceTensor,
    past_seq: usize,
    n_head: usize,
    head_dim: usize,
    out: &DeviceTensor,
) -> Result<()> {
    if n_head == 0 {
        return Err(kernel_err(
            "attention_decoder_incremental_d n_head must be > 0",
        ));
    }
    if head_dim == 0 {
        return Err(kernel_err(
            "attention_decoder_incremental_d head_dim must be > 0",
        ));
    }
    let state = n_head.checked_mul(head_dim).ok_or_else(|| {
        kernel_err("attention_decoder_incremental_d n_head*head_dim overflowed usize")
    })?;
    for (label, len) in [
        ("q", q.len()),
        ("new_k", new_k.len()),
        ("new_v", new_v.len()),
        ("out", out.len()),
    ] {
        if len != state {
            return Err(kernel_err(format!(
                "attention_decoder_incremental_d {label} len {len} != state {state}"
            )));
        }
    }
    let past_expected = past_seq.checked_mul(state).ok_or_else(|| {
        kernel_err("attention_decoder_incremental_d past_seq*state overflowed usize")
    })?;
    for (label, len) in [("past_k", past_k.len()), ("past_v", past_v.len())] {
        if len != past_expected {
            return Err(kernel_err(format!(
                "attention_decoder_incremental_d {label} len {len} != past_seq*state {past_expected}"
            )));
        }
    }
    Ok(())
}

/// Shape validator for fixed-capacity incremental decoder attention. Cache
/// tensors must have `cache_capacity * state` cells, `visible_seq` must be
/// non-zero and not exceed that capacity, and q/out must be one state row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_attention_decoder_incremental_cache_shapes(
    q: &DeviceTensor,
    key_cache: &DeviceTensor,
    value_cache: &DeviceTensor,
    visible_seq: usize,
    cache_capacity: usize,
    n_head: usize,
    head_dim: usize,
    out: &DeviceTensor,
) -> Result<()> {
    if visible_seq == 0 {
        return Err(kernel_err(
            "attention_decoder_incremental_cache_d visible_seq must be > 0",
        ));
    }
    if visible_seq > cache_capacity {
        return Err(kernel_err(format!(
            "attention_decoder_incremental_cache_d visible_seq {visible_seq} exceeds cache_capacity {cache_capacity}"
        )));
    }
    if n_head == 0 {
        return Err(kernel_err(
            "attention_decoder_incremental_cache_d n_head must be > 0",
        ));
    }
    if head_dim == 0 {
        return Err(kernel_err(
            "attention_decoder_incremental_cache_d head_dim must be > 0",
        ));
    }
    let state = n_head.checked_mul(head_dim).ok_or_else(|| {
        kernel_err("attention_decoder_incremental_cache_d n_head*head_dim overflowed usize")
    })?;
    for (label, len) in [("q", q.len()), ("out", out.len())] {
        if len != state {
            return Err(kernel_err(format!(
                "attention_decoder_incremental_cache_d {label} len {len} != state {state}"
            )));
        }
    }
    let cache_expected = cache_capacity.checked_mul(state).ok_or_else(|| {
        kernel_err("attention_decoder_incremental_cache_d cache_capacity*state overflowed usize")
    })?;
    for (label, len) in [
        ("key_cache", key_cache.len()),
        ("value_cache", value_cache.len()),
    ] {
        if len != cache_expected {
            return Err(kernel_err(format!(
                "attention_decoder_incremental_cache_d {label} len {len} != cache_capacity*state {cache_expected}"
            )));
        }
    }
    Ok(())
}

/// Shape validator for appending one K/V row to a fixed-capacity incremental
/// decoder cache and attending over the resulting visible prefix.
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_attention_decoder_incremental_cache_append_shapes(
    q: &DeviceTensor,
    key_cache: &DeviceTensor,
    value_cache: &DeviceTensor,
    new_k: &DeviceTensor,
    new_v: &DeviceTensor,
    past_seq: usize,
    cache_capacity: usize,
    n_head: usize,
    head_dim: usize,
    out: &DeviceTensor,
) -> Result<()> {
    if past_seq >= cache_capacity {
        return Err(kernel_err(format!(
            "attention_decoder_incremental_cache_append_d past_seq {past_seq} cannot append into cache_capacity {cache_capacity}"
        )));
    }
    validate_attention_decoder_incremental_cache_shapes(
        q,
        key_cache,
        value_cache,
        past_seq + 1,
        cache_capacity,
        n_head,
        head_dim,
        out,
    )?;
    let state = n_head.checked_mul(head_dim).ok_or_else(|| {
        kernel_err("attention_decoder_incremental_cache_append_d n_head*head_dim overflowed usize")
    })?;
    for (label, len) in [("new_k", new_k.len()), ("new_v", new_v.len())] {
        if len != state {
            return Err(kernel_err(format!(
                "attention_decoder_incremental_cache_append_d {label} len {len} != state {state}"
            )));
        }
    }
    Ok(())
}

/// Scalar Whisper decoder cross-attention (encoder-decoder attention).
///
/// Q: `[q_seq, state]` from decoder hidden state.
/// K, V: `[kv_seq, state]` from encoder output (precomputed, static per sequence).
/// No causal mask: each query row attends all `kv_seq` encoder positions.
/// Writes `[q_seq, state]` into `out`.
///
/// This is the parity oracle for `attention_decoder_cross_d`. Matches
/// `attention_body_host` with `causal == false`, `q_seq` decoder rows, and
/// `kv_seq` encoder rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_decoder_cross_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_seq: usize,
    kv_seq: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
) {
    let state = n_head * head_dim;
    debug_assert_eq!(q.len(), q_seq * state);
    debug_assert_eq!(k.len(), kv_seq * state);
    debug_assert_eq!(v.len(), kv_seq * state);
    debug_assert_eq!(out.len(), q_seq * state);

    let mut scores = vec![0.0_f32; kv_seq];
    for qi in 0..q_seq {
        for head in 0..n_head {
            let q_base = qi * state + head * head_dim;
            // Pass 1: scaled dot products across all encoder positions.
            for (ki, score) in scores.iter_mut().enumerate() {
                let k_base = ki * state + head * head_dim;
                let mut acc = 0.0_f32;
                for d in 0..head_dim {
                    acc += q[q_base + d] * k[k_base + d];
                }
                *score = acc * scale;
            }
            // Pass 2: numerically stable softmax over all kv_seq scores.
            softmax(&mut scores);
            // Pass 3: probability-weighted accumulation of V.
            let out_base = qi * state + head * head_dim;
            for d in 0..head_dim {
                out[out_base + d] = 0.0;
            }
            for (ki, &p) in scores.iter().enumerate() {
                let v_base = ki * state + head * head_dim;
                for d in 0..head_dim {
                    out[out_base + d] += p * v[v_base + d];
                }
            }
        }
    }
}

/// Shape validator for `attention_decoder_cross_d`. Rejects zero heads/dim,
/// overflow in `state = n_head * head_dim`, wrong Q shape (must be
/// `q_seq * state`), wrong K/V shape (must be `kv_seq * state`), and
/// wrong output shape (must be `q_seq * state`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_attention_decoder_cross_shapes(
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    q_seq: usize,
    kv_seq: usize,
    n_head: usize,
    head_dim: usize,
    out: &DeviceTensor,
) -> Result<()> {
    if n_head == 0 {
        return Err(kernel_err("attention_decoder_cross_d n_head must be > 0"));
    }
    if head_dim == 0 {
        return Err(kernel_err("attention_decoder_cross_d head_dim must be > 0"));
    }
    let state = n_head
        .checked_mul(head_dim)
        .ok_or_else(|| kernel_err("attention_decoder_cross_d n_head*head_dim overflowed usize"))?;
    let q_expected = q_seq
        .checked_mul(state)
        .ok_or_else(|| kernel_err("attention_decoder_cross_d q_seq*state overflowed usize"))?;
    let kv_expected = kv_seq
        .checked_mul(state)
        .ok_or_else(|| kernel_err("attention_decoder_cross_d kv_seq*state overflowed usize"))?;
    for (label, len, expected) in [("q", q.len(), q_expected), ("out", out.len(), q_expected)] {
        if len != expected {
            return Err(kernel_err(format!(
                "attention_decoder_cross_d {label} len {len} != q_seq*state {expected}"
            )));
        }
    }
    for (label, len) in [("k", k.len()), ("v", v.len())] {
        if len != kv_expected {
            return Err(kernel_err(format!(
                "attention_decoder_cross_d {label} len {len} != kv_seq*state {kv_expected}"
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_layer_norm_shapes(
    x: &DeviceTensor,
    rows: usize,
    hidden: usize,
    weight: &DeviceTensor,
    bias: &DeviceTensor,
    out: &DeviceTensor,
) -> Result<()> {
    let expected = rows
        .checked_mul(hidden)
        .ok_or_else(|| kernel_err("layer_norm_d rows*hidden overflowed usize"))?;
    if x.len() != expected {
        return Err(kernel_err(format!(
            "layer_norm_d x len {} != rows*hidden {}",
            x.len(),
            expected
        )));
    }
    if out.len() != expected {
        return Err(kernel_err(format!(
            "layer_norm_d out len {} != rows*hidden {}",
            out.len(),
            expected
        )));
    }
    if weight.len() != hidden {
        return Err(kernel_err(format!(
            "layer_norm_d weight len {} != hidden {hidden}",
            weight.len()
        )));
    }
    if bias.len() != hidden {
        return Err(kernel_err(format!(
            "layer_norm_d bias len {} != hidden {hidden}",
            bias.len()
        )));
    }
    Ok(())
}

pub(crate) fn validate_add_positional_embedding_shapes(
    x: &DeviceTensor,
    rows: usize,
    cols: usize,
    pe: &DeviceTensor,
    pe_rows: usize,
    start_pos: usize,
) -> Result<()> {
    let x_expected = rows
        .checked_mul(cols)
        .ok_or_else(|| kernel_err("add_positional_embedding_d rows*cols overflowed usize"))?;
    let pe_expected = pe_rows
        .checked_mul(cols)
        .ok_or_else(|| kernel_err("add_positional_embedding_d pe_rows*cols overflowed usize"))?;
    if x.len() != x_expected {
        return Err(kernel_err(format!(
            "add_positional_embedding_d x len {} != rows*cols {}",
            x.len(),
            x_expected
        )));
    }
    if pe.len() != pe_expected {
        return Err(kernel_err(format!(
            "add_positional_embedding_d pe len {} != pe_rows*cols {}",
            pe.len(),
            pe_expected
        )));
    }
    let end = start_pos
        .checked_add(rows)
        .ok_or_else(|| kernel_err("add_positional_embedding_d start_pos+rows overflowed usize"))?;
    if end > pe_rows {
        return Err(kernel_err(format!(
            "add_positional_embedding_d start_pos {start_pos} + rows {rows} exceeds pe_rows {pe_rows}"
        )));
    }
    Ok(())
}

pub type SharedKernelBackend = Arc<dyn KernelBackend>;

pub fn default_kernel_backend() -> SharedKernelBackend {
    Arc::new(CpuKernelBackend::default())
}

pub fn optimized_cpu_kernel_backend() -> SharedKernelBackend {
    Arc::new(CpuKernelBackend::optimized())
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

// ---------------------------------------------------------------------------
// CPU reference primitives (M1.7)
//
// Reference-only. Not vectorized. Used to make the rest of the inference path
// testable end-to-end on a laptop with no GPU, and as the parity oracle for
// future GPU kernels.
// ---------------------------------------------------------------------------

pub(crate) fn kernel_err(message: impl Into<String>) -> OcelotlError {
    OcelotlError::Kernel(KernelError {
        backend: "cpu".to_string(),
        message: message.into(),
    })
}

pub(crate) fn checked_len_product(kernel: &str, label: &str, dims: &[usize]) -> Result<usize> {
    dims.iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| {
            kernel_err(format!(
                "{kernel} {label} shape product overflows usize: {:?}",
                dims
            ))
        })
}

/// Element-wise addition: `out[i] = a[i] + b[i]`.
///
/// All three slices must have the same length. M1 is contiguous-only — there
/// is no stride argument.
///
/// # Errors
///
/// Returns `KernelError` (backend = `"cpu"`) when the input slices and the
/// output buffer do not all share the same length.
///
/// # Example
///
/// ```
/// use ocelotl_kernels::vec_add;
/// let a = [1.0_f32, 2.0, 3.0];
/// let b = [10.0_f32, 20.0, 30.0];
/// let mut out = [0.0_f32; 3];
/// vec_add(&a, &b, &mut out).unwrap();
/// assert_eq!(out, [11.0, 22.0, 33.0]);
/// ```
pub fn vec_add(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
    if a.len() != b.len() || a.len() != out.len() {
        return Err(kernel_err(format!(
            "vec_add length mismatch: a.len={}, b.len={}, out.len={}",
            a.len(),
            b.len(),
            out.len()
        )));
    }
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
    Ok(())
}

/// Inner product: `sum(a[i] * b[i])`.
///
/// Both slices must have the same length. M1 is contiguous-only — there is no
/// stride argument.
///
/// # Errors
///
/// Returns `KernelError` (backend = `"cpu"`) when the two input slices have
/// different lengths.
///
/// # Example
///
/// ```
/// use ocelotl_kernels::dot;
/// let a = [1.0_f32, 2.0, 3.0];
/// let b = [4.0_f32, 5.0, 6.0];
/// assert_eq!(dot(&a, &b).unwrap(), 32.0);
/// ```
pub fn dot(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(kernel_err(format!(
            "dot length mismatch: a.len={}, b.len={}",
            a.len(),
            b.len()
        )));
    }
    let mut acc = 0.0_f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    Ok(acc)
}

/// Numerically stable softmax, in place over a single slice.
///
/// Computes `x[i] = exp(x[i] - max(x)) / sum_j exp(x[j] - max(x))`.
/// Subtracting the max before exponentiating is the standard stability
/// technique: it leaves the result mathematically unchanged but bounds the
/// largest argument to `exp` at zero, preventing overflow for inputs whose
/// magnitude exceeds `~88` in `f32`. M1 is contiguous-only.
///
/// An empty slice is a no-op (softmax of nothing is nothing). A slice that is
/// all `-∞` or all `NaN` will produce `NaN` outputs — that is upstream's
/// responsibility, not the kernel's.
///
/// # Example
///
/// ```
/// use ocelotl_kernels::softmax;
/// let mut x = [1.0_f32, 2.0, 3.0];
/// softmax(&mut x);
/// let sum: f32 = x.iter().sum();
/// assert!((sum - 1.0).abs() < 4.0 * f32::EPSILON);
/// ```
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    let mut max = x[0];
    for &v in x.iter().skip(1) {
        if v > max {
            max = v;
        }
    }

    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }

    let inv_sum = 1.0_f32 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

/// Matrix multiplication: `out = a @ b`, all row-major contiguous.
///
/// Shapes:
/// - `a` is `m × k`, total length `m * k`.
/// - `b` is `k × n`, total length `k * n`.
/// - `out` is `m × n`, total length `m * n`.
///
/// This is a triple-loop reference implementation: `O(m * n * k)`. It is the
/// parity oracle for future GPU matmul kernels, not a fast kernel.
///
/// # Errors
///
/// Returns `KernelError` (backend = `"cpu"`) when:
/// - the inner dimensions of `a` and `b` disagree (`a_shape.1 != b_shape.0`),
/// - any input slice length does not match its declared shape,
/// - the output buffer length does not match `m * n`.
///
/// # Example
///
/// ```
/// use ocelotl_kernels::matmul;
/// // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
/// let a = [1.0_f32, 2.0, 3.0, 4.0];
/// let b = [5.0_f32, 6.0, 7.0, 8.0];
/// let mut out = [0.0_f32; 4];
/// matmul(&a, (2, 2), &b, (2, 2), &mut out).unwrap();
/// assert_eq!(out, [19.0, 22.0, 43.0, 50.0]);
/// ```
pub fn matmul(
    a: &[f32],
    a_shape: (usize, usize),
    b: &[f32],
    b_shape: (usize, usize),
    out: &mut [f32],
) -> Result<()> {
    let (m, k, n) = validate_matmul(a, a_shape, b, b_shape, out)?;
    matmul_compute(a, m, k, b, n, out);
    Ok(())
}

/// Scalar matmul body. Inputs are assumed pre-validated. Splits cleanly over
/// disjoint output-row chunks (M-axis), so the parallel dispatcher can call
/// this per chunk with its slice of `a` and `out` and the K-loop accumulation
/// order stays identical to the serial path (parity oracle for threaded runs).
fn matmul_compute(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, out: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}

fn matmul_optimized(
    a: &[f32],
    a_shape: (usize, usize),
    b: &[f32],
    b_shape: (usize, usize),
    out: &mut [f32],
) -> Result<()> {
    let (m, k, n) = validate_matmul(a, a_shape, b, b_shape, out)?;
    matmul_optimized_compute(a, m, k, b, n, out);
    Ok(())
}

/// Cache-friendlier matmul body (K outer, N inner, no transpose). Inputs are
/// assumed pre-validated. Same chunkability story as `matmul_compute`.
fn matmul_optimized_compute(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, out: &mut [f32]) {
    out.fill(0.0);
    for i in 0..m {
        let out_row = &mut out[i * n..(i + 1) * n];
        for p in 0..k {
            let a_ip = a[i * k + p];
            let b_row = &b[p * n..(p + 1) * n];
            for j in 0..n {
                out_row[j] += a_ip * b_row[j];
            }
        }
    }
}

fn validate_matmul(
    a: &[f32],
    a_shape: (usize, usize),
    b: &[f32],
    b_shape: (usize, usize),
    out: &[f32],
) -> Result<(usize, usize, usize)> {
    let (m, k_a) = a_shape;
    let (k_b, n) = b_shape;

    if k_a != k_b {
        return Err(kernel_err(format!(
            "matmul inner-dimension mismatch: a is {m}x{k_a}, b is {k_b}x{n}"
        )));
    }
    let a_expected = checked_len_product("matmul", "a", &[m, k_a])?;
    let b_expected = checked_len_product("matmul", "b", &[k_b, n])?;
    let out_expected = checked_len_product("matmul", "out", &[m, n])?;

    if a.len() != a_expected {
        return Err(kernel_err(format!(
            "matmul a slice length {} does not match shape {m}x{k_a}",
            a.len()
        )));
    }
    if b.len() != b_expected {
        return Err(kernel_err(format!(
            "matmul b slice length {} does not match shape {k_b}x{n}",
            b.len()
        )));
    }
    if out.len() != out_expected {
        return Err(kernel_err(format!(
            "matmul out slice length {} does not match shape {m}x{n}",
            out.len()
        )));
    }

    Ok((m, k_a, n))
}

#[allow(clippy::too_many_arguments)]
fn linear_out_by_in(
    x: &[f32],
    rows: usize,
    in_features: usize,
    weight_out_by_in: &[f32],
    out_features: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> Result<()> {
    validate_linear_out_by_in(
        x,
        rows,
        in_features,
        weight_out_by_in,
        out_features,
        bias,
        out,
    )?;
    linear_out_by_in_compute(
        x,
        rows,
        in_features,
        weight_out_by_in,
        out_features,
        bias,
        out,
    );
    Ok(())
}

/// Compute body for the scalar tiled `linear_out_by_in`. Inputs are assumed
/// pre-validated. Splits naturally over disjoint output-row chunks, so the
/// parallel dispatcher can call this per chunk with its slice of `x` and `out`
/// and the K-loop accumulation order stays identical to the serial path
/// (parity oracle for threaded runs).
fn linear_out_by_in_compute(
    x: &[f32],
    rows: usize,
    in_features: usize,
    weight_out_by_in: &[f32],
    out_features: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    let tiled_rows = rows - (rows % 4);
    let tiled_out = out_features - (out_features % 4);

    for row in (0..tiled_rows).step_by(4) {
        let x0 = &x[row * in_features..(row + 1) * in_features];
        let x1 = &x[(row + 1) * in_features..(row + 2) * in_features];
        let x2 = &x[(row + 2) * in_features..(row + 3) * in_features];
        let x3 = &x[(row + 3) * in_features..(row + 4) * in_features];

        for out_dim in (0..tiled_out).step_by(4) {
            // acc{output offset}{row offset}: four output dimensions by four activation rows.
            let mut acc00 = bias.map_or(0.0, |b| b[out_dim]);
            let mut acc01 = acc00;
            let mut acc02 = acc00;
            let mut acc03 = acc00;
            let mut acc10 = bias.map_or(0.0, |b| b[out_dim + 1]);
            let mut acc11 = acc10;
            let mut acc12 = acc10;
            let mut acc13 = acc10;
            let mut acc20 = bias.map_or(0.0, |b| b[out_dim + 2]);
            let mut acc21 = acc20;
            let mut acc22 = acc20;
            let mut acc23 = acc20;
            let mut acc30 = bias.map_or(0.0, |b| b[out_dim + 3]);
            let mut acc31 = acc30;
            let mut acc32 = acc30;
            let mut acc33 = acc30;
            let w0 = out_dim * in_features;
            let w1 = (out_dim + 1) * in_features;
            let w2 = (out_dim + 2) * in_features;
            let w3 = (out_dim + 3) * in_features;

            for in_dim in 0..in_features {
                let weight0 = weight_out_by_in[w0 + in_dim];
                let weight1 = weight_out_by_in[w1 + in_dim];
                let weight2 = weight_out_by_in[w2 + in_dim];
                let weight3 = weight_out_by_in[w3 + in_dim];
                let x0_value = x0[in_dim];
                let x1_value = x1[in_dim];
                let x2_value = x2[in_dim];
                let x3_value = x3[in_dim];

                acc00 += x0_value * weight0;
                acc10 += x0_value * weight1;
                acc20 += x0_value * weight2;
                acc30 += x0_value * weight3;

                acc01 += x1_value * weight0;
                acc11 += x1_value * weight1;
                acc21 += x1_value * weight2;
                acc31 += x1_value * weight3;

                acc02 += x2_value * weight0;
                acc12 += x2_value * weight1;
                acc22 += x2_value * weight2;
                acc32 += x2_value * weight3;

                acc03 += x3_value * weight0;
                acc13 += x3_value * weight1;
                acc23 += x3_value * weight2;
                acc33 += x3_value * weight3;
            }

            let out0 = row * out_features + out_dim;
            let out1 = (row + 1) * out_features + out_dim;
            let out2 = (row + 2) * out_features + out_dim;
            let out3 = (row + 3) * out_features + out_dim;

            out[out0] = acc00;
            out[out0 + 1] = acc10;
            out[out0 + 2] = acc20;
            out[out0 + 3] = acc30;
            out[out1] = acc01;
            out[out1 + 1] = acc11;
            out[out1 + 2] = acc21;
            out[out1 + 3] = acc31;
            out[out2] = acc02;
            out[out2 + 1] = acc12;
            out[out2 + 2] = acc22;
            out[out2 + 3] = acc32;
            out[out3] = acc03;
            out[out3 + 1] = acc13;
            out[out3 + 2] = acc23;
            out[out3 + 3] = acc33;
        }

        for tail_out in tiled_out..out_features {
            let mut acc0 = bias.map_or(0.0, |b| b[tail_out]);
            let mut acc1 = acc0;
            let mut acc2 = acc0;
            let mut acc3 = acc0;
            let weight_start = tail_out * in_features;
            for in_dim in 0..in_features {
                let weight = weight_out_by_in[weight_start + in_dim];
                acc0 += x0[in_dim] * weight;
                acc1 += x1[in_dim] * weight;
                acc2 += x2[in_dim] * weight;
                acc3 += x3[in_dim] * weight;
            }
            out[row * out_features + tail_out] = acc0;
            out[(row + 1) * out_features + tail_out] = acc1;
            out[(row + 2) * out_features + tail_out] = acc2;
            out[(row + 3) * out_features + tail_out] = acc3;
        }
    }

    for row in tiled_rows..rows {
        let x_row = &x[row * in_features..(row + 1) * in_features];
        let out_row = &mut out[row * out_features..(row + 1) * out_features];

        for out_dim in (0..tiled_out).step_by(4) {
            let mut acc0 = bias.map_or(0.0, |b| b[out_dim]);
            let mut acc1 = bias.map_or(0.0, |b| b[out_dim + 1]);
            let mut acc2 = bias.map_or(0.0, |b| b[out_dim + 2]);
            let mut acc3 = bias.map_or(0.0, |b| b[out_dim + 3]);
            let w0 = out_dim * in_features;
            let w1 = (out_dim + 1) * in_features;
            let w2 = (out_dim + 2) * in_features;
            let w3 = (out_dim + 3) * in_features;
            for in_dim in 0..in_features {
                let x_value = x_row[in_dim];
                acc0 += x_value * weight_out_by_in[w0 + in_dim];
                acc1 += x_value * weight_out_by_in[w1 + in_dim];
                acc2 += x_value * weight_out_by_in[w2 + in_dim];
                acc3 += x_value * weight_out_by_in[w3 + in_dim];
            }
            out_row[out_dim] = acc0;
            out_row[out_dim + 1] = acc1;
            out_row[out_dim + 2] = acc2;
            out_row[out_dim + 3] = acc3;
        }

        for out_dim in tiled_out..out_features {
            let mut acc = bias.map_or(0.0, |b| b[out_dim]);
            for in_dim in 0..in_features {
                acc += x_row[in_dim] * weight_out_by_in[out_dim * in_features + in_dim];
            }
            out_row[out_dim] = acc;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn linear_out_by_in_optimized(
    x: &[f32],
    rows: usize,
    in_features: usize,
    weight_out_by_in: &[f32],
    out_features: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> Result<()> {
    validate_linear_out_by_in(
        x,
        rows,
        in_features,
        weight_out_by_in,
        out_features,
        bias,
        out,
    )?;
    linear_out_by_in_optimized_compute(
        x,
        rows,
        in_features,
        weight_out_by_in,
        out_features,
        bias,
        out,
    );
    Ok(())
}

/// AVX2 + FMA implementation of `linear_out_by_in`. Validates the shape
/// contract once, then dispatches to the `unsafe` AVX2 compute body. The
/// host's AVX2 + FMA support must already be validated by
/// `validate_mode_supported` at backend construction.
#[cfg(target_arch = "x86_64")]
fn linear_out_by_in_avx2(
    x: &[f32],
    rows: usize,
    in_features: usize,
    weight_out_by_in: &[f32],
    out_features: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> Result<()> {
    validate_linear_out_by_in(
        x,
        rows,
        in_features,
        weight_out_by_in,
        out_features,
        bias,
        out,
    )?;
    // SAFETY: feature support was checked at backend construction; shape
    // contract was just validated.
    unsafe {
        cpu_avx2::linear_out_by_in_compute_avx2(
            x,
            rows,
            in_features,
            weight_out_by_in,
            out_features,
            bias,
            out,
        );
    }
    Ok(())
}

#[cfg(not(target_arch = "x86_64"))]
fn linear_out_by_in_avx2(
    _x: &[f32],
    _rows: usize,
    _in_features: usize,
    _weight_out_by_in: &[f32],
    _out_features: usize,
    _bias: Option<&[f32]>,
    _out: &mut [f32],
) -> Result<()> {
    // Unreachable: validate_mode_supported rejects Avx2 on non-x86_64 at
    // backend construction. Kept as a typed error so the dispatch arm
    // type-checks on all targets.
    Err(OcelotlError::Kernel(KernelError {
        backend: "cpu".to_string(),
        message: "CpuKernelMode::Avx2 is x86_64-only".to_string(),
    }))
}

/// Compute body for the optimized `linear_out_by_in`. Inputs are assumed
/// pre-validated.
fn linear_out_by_in_optimized_compute(
    x: &[f32],
    rows: usize,
    in_features: usize,
    weight_out_by_in: &[f32],
    out_features: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    for row in 0..rows {
        let out_row = &mut out[row * out_features..(row + 1) * out_features];
        match bias {
            Some(bias) => out_row.copy_from_slice(bias),
            None => out_row.fill(0.0),
        }
        let x_row = &x[row * in_features..(row + 1) * in_features];
        for in_dim in 0..in_features {
            let x_value = x_row[in_dim];
            for out_dim in 0..out_features {
                out_row[out_dim] += x_value * weight_out_by_in[out_dim * in_features + in_dim];
            }
        }
    }
}

/// Below this row count, single-threaded execution beats the rayon dispatch
/// overhead. Tuned for the Whisper encoder where M = audio_ctx (>=1500 for
/// all classic sizes). Single-token decoder projections use the output-axis
/// threshold below instead.
const PARALLEL_LINEAR_MIN_ROWS: usize = 32;

/// Below this output-feature count, single-row linear projections stay serial.
/// The high-value Whisper decoder case is the tied-embedding logits projection
/// (vocab-sized output); smaller 384-wide projections avoid rayon overhead.
const PARALLEL_LINEAR_MIN_OUTPUTS: usize = 1024;

/// Same rationale as `PARALLEL_LINEAR_MIN_ROWS` but for the generic `matmul`
/// kernel. Qwen prefill uses M = seq_len which can run into the hundreds for
/// realistic prompts; decode is M = 1 and stays serial.
const PARALLEL_MATMUL_MIN_ROWS: usize = 32;

/// Below this query count, single-threaded SDPA beats the rayon dispatch
/// overhead. Mirrors the Whisper attention threshold; chosen so that single-
/// token decode (seq_len = 1) stays serial.
const PARALLEL_SDPA_MIN_SEQ: usize = 32;

/// Parallel dispatcher for single-row `linear_out_by_in`. Partitions the
/// output-feature axis across the rayon pool. This is the hot Whisper decoder
/// append shape, especially the tied-embedding logits projection
/// (`rows == 1`, `out_features ~= vocab`). Each output cell keeps the same
/// K-loop accumulation order as the serial helper; only independent output
/// columns run on different threads.
#[allow(clippy::too_many_arguments)]
fn linear_out_by_in_output_parallel(
    pool: &rayon::ThreadPool,
    mode: CpuKernelMode,
    x: &[f32],
    in_features: usize,
    weight_out_by_in: &[f32],
    out_features: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> Result<()> {
    use rayon::prelude::*;

    validate_linear_out_by_in(x, 1, in_features, weight_out_by_in, out_features, bias, out)?;

    let threads = pool.current_num_threads().max(1);
    let tile = 4usize;
    let tiles_total = out_features.div_ceil(tile);
    let tiles_per_chunk = tiles_total.div_ceil(threads).max(1);
    let outputs_per_chunk = tiles_per_chunk * tile;

    pool.install(|| {
        out.par_chunks_mut(outputs_per_chunk)
            .enumerate()
            .for_each(|(idx, out_chunk)| {
                let out_start = idx * outputs_per_chunk;
                let chunk_out = out_chunk.len();
                let w_start = out_start * in_features;
                let w_end = w_start + chunk_out * in_features;
                let weight_chunk = &weight_out_by_in[w_start..w_end];
                let bias_chunk = bias.map(|b| &b[out_start..out_start + chunk_out]);

                match mode {
                    CpuKernelMode::Scalar => linear_out_by_in_compute(
                        x,
                        1,
                        in_features,
                        weight_chunk,
                        chunk_out,
                        bias_chunk,
                        out_chunk,
                    ),
                    CpuKernelMode::Optimized => linear_out_by_in_optimized_compute(
                        x,
                        1,
                        in_features,
                        weight_chunk,
                        chunk_out,
                        bias_chunk,
                        out_chunk,
                    ),
                    CpuKernelMode::Avx2 => {
                        // SAFETY: `with_mode_and_threads` validates AVX2+FMA
                        // support before constructing this backend. The full
                        // shape was validated above; chunk slices preserve the
                        // single-row `linear_out_by_in` contract.
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            cpu_avx2::linear_out_by_in_compute_avx2(
                                x,
                                1,
                                in_features,
                                weight_chunk,
                                chunk_out,
                                bias_chunk,
                                out_chunk,
                            );
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        unreachable!("Avx2 mode rejected at construction on non-x86_64");
                    }
                }
            });
    });

    Ok(())
}

/// Validate that the host CPU supports the requested mode. AVX2 needs both
/// the `avx2` and `fma` x86_64 features at runtime; the scalar/optimized
/// modes have no host requirements.
fn validate_mode_supported(mode: CpuKernelMode) -> Result<()> {
    match mode {
        CpuKernelMode::Scalar | CpuKernelMode::Optimized => Ok(()),
        CpuKernelMode::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                    Ok(())
                } else {
                    Err(OcelotlError::Kernel(KernelError {
                        backend: "cpu".to_string(),
                        message: "CpuKernelMode::Avx2 requires runtime avx2 + fma support; this host advertises neither or only one"
                            .to_string(),
                    }))
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                Err(OcelotlError::Kernel(KernelError {
                    backend: "cpu".to_string(),
                    message: "CpuKernelMode::Avx2 is x86_64-only; rebuild with a Scalar or Optimized mode on this target"
                        .to_string(),
                }))
            }
        }
    }
}

/// Parallel dispatcher for `linear_out_by_in`. Partitions the input/output
/// row range across the rayon pool, validates once, and calls the chosen
/// compute helper on each chunk. Each chunk writes a disjoint slice of `out`
/// and reads a disjoint slice of `x`, so the result is bit-identical to the
/// serial path (no cross-thread accumulation reorder).
#[allow(clippy::too_many_arguments)]
fn linear_out_by_in_parallel(
    pool: &rayon::ThreadPool,
    mode: CpuKernelMode,
    x: &[f32],
    rows: usize,
    in_features: usize,
    weight_out_by_in: &[f32],
    out_features: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> Result<()> {
    use rayon::prelude::*;

    validate_linear_out_by_in(
        x,
        rows,
        in_features,
        weight_out_by_in,
        out_features,
        bias,
        out,
    )?;

    let threads = pool.current_num_threads().max(1);
    // Align chunk size to a 4-row boundary so each chunk's scalar tile loop
    // hits its tiled fast path before falling back to the 1-row tail. The
    // last chunk may be shorter; that is fine because the compute helpers
    // accept any row count.
    let tile = 4usize;
    let tiles_total = rows.div_ceil(tile);
    let tiles_per_chunk = tiles_total.div_ceil(threads).max(1);
    let rows_per_chunk = tiles_per_chunk * tile;

    let chunk_out_len = rows_per_chunk * out_features;
    let chunk_x_len = rows_per_chunk * in_features;

    pool.install(|| {
        out.par_chunks_mut(chunk_out_len)
            .enumerate()
            .for_each(|(idx, out_chunk)| {
                let row_start = idx * rows_per_chunk;
                let chunk_rows = out_chunk.len() / out_features;
                let x_start = row_start * in_features;
                let x_chunk = &x[x_start..x_start + chunk_rows * in_features];
                debug_assert_eq!(out_chunk.len(), chunk_rows * out_features);
                debug_assert!(chunk_x_len >= chunk_rows * in_features);
                match mode {
                    CpuKernelMode::Scalar => linear_out_by_in_compute(
                        x_chunk,
                        chunk_rows,
                        in_features,
                        weight_out_by_in,
                        out_features,
                        bias,
                        out_chunk,
                    ),
                    CpuKernelMode::Optimized => linear_out_by_in_optimized_compute(
                        x_chunk,
                        chunk_rows,
                        in_features,
                        weight_out_by_in,
                        out_features,
                        bias,
                        out_chunk,
                    ),
                    CpuKernelMode::Avx2 => {
                        // SAFETY: the backend was constructed via
                        // `with_mode_and_threads` which calls
                        // `validate_mode_supported(Avx2)` and only succeeds
                        // when the host advertises avx2 + fma. Shape
                        // contract is upheld by the earlier
                        // `validate_linear_out_by_in` call on the full
                        // buffer; chunk slices preserve it.
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            cpu_avx2::linear_out_by_in_compute_avx2(
                                x_chunk,
                                chunk_rows,
                                in_features,
                                weight_out_by_in,
                                out_features,
                                bias,
                                out_chunk,
                            );
                        }
                        // On non-x86_64 hosts `validate_mode_supported`
                        // already rejected this mode at construction, so
                        // this arm is unreachable. Keep an explicit panic
                        // to avoid pulling in a no-op fallback.
                        #[cfg(not(target_arch = "x86_64"))]
                        unreachable!("Avx2 mode rejected at construction on non-x86_64");
                    }
                }
            });
    });

    Ok(())
}

/// Parallel dispatcher for `matmul`. Partitions the M (output-row) axis across
/// the rayon pool. Each chunk reads disjoint rows of `a` and writes disjoint
/// rows of `out`; `b` is shared read-only. The accumulation order within each
/// (i, j) cell is identical to the serial path, so the result is bit-identical
/// to running serially.
fn matmul_parallel(
    pool: &rayon::ThreadPool,
    mode: CpuKernelMode,
    a: &[f32],
    a_shape: (usize, usize),
    b: &[f32],
    b_shape: (usize, usize),
    out: &mut [f32],
) -> Result<()> {
    use rayon::prelude::*;

    let (m, k, n) = validate_matmul(a, a_shape, b, b_shape, out)?;

    let threads = pool.current_num_threads().max(1);
    let rows_per_chunk = m.div_ceil(threads).max(1);
    let chunk_out_len = rows_per_chunk * n;

    pool.install(|| {
        out.par_chunks_mut(chunk_out_len)
            .enumerate()
            .for_each(|(idx, out_chunk)| {
                let row_start = idx * rows_per_chunk;
                let chunk_rows = out_chunk.len() / n;
                let a_start = row_start * k;
                let a_chunk = &a[a_start..a_start + chunk_rows * k];
                debug_assert_eq!(out_chunk.len(), chunk_rows * n);
                match mode {
                    CpuKernelMode::Scalar => {
                        matmul_compute(a_chunk, chunk_rows, k, b, n, out_chunk);
                    }
                    // matmul Avx2 today falls back to optimized scalar (the
                    // AVX2 microkernel only covers `linear_out_by_in`'s
                    // [out, in] layout). Both modes therefore share the same
                    // optimized compute body.
                    CpuKernelMode::Optimized | CpuKernelMode::Avx2 => {
                        matmul_optimized_compute(a_chunk, chunk_rows, k, b, n, out_chunk);
                    }
                }
            });
    });

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_linear_out_by_in(
    x: &[f32],
    rows: usize,
    in_features: usize,
    weight_out_by_in: &[f32],
    out_features: usize,
    bias: Option<&[f32]>,
    out: &[f32],
) -> Result<()> {
    let x_expected = checked_len_product("linear_out_by_in", "x", &[rows, in_features])?;
    let weight_expected =
        checked_len_product("linear_out_by_in", "weight", &[out_features, in_features])?;
    let out_expected = checked_len_product("linear_out_by_in", "out", &[rows, out_features])?;

    if x.len() != x_expected {
        return Err(kernel_err(format!(
            "linear_out_by_in x.len()={} does not match rows*in_features={}*{}={}",
            x.len(),
            rows,
            in_features,
            x_expected
        )));
    }
    if weight_out_by_in.len() != weight_expected {
        return Err(kernel_err(format!(
            "linear_out_by_in weight.len()={} does not match out_features*in_features={}*{}={}",
            weight_out_by_in.len(),
            out_features,
            in_features,
            weight_expected
        )));
    }
    if let Some(bias) = bias {
        if bias.len() != out_features {
            return Err(kernel_err(format!(
                "linear_out_by_in bias.len()={} does not match out_features={out_features}",
                bias.len()
            )));
        }
    }
    if out.len() != out_expected {
        return Err(kernel_err(format!(
            "linear_out_by_in out.len()={} does not match rows*out_features={}*{}={}",
            out.len(),
            rows,
            out_features,
            out_expected
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests;
