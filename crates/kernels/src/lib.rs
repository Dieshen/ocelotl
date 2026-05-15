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

pub mod rope;
pub use rope::rope_apply_inplace;

use ocelotl_core::{Device, KernelError, OcelotlError, Result, UnsupportedError};

pub mod attention;
#[cfg(target_arch = "x86_64")]
mod cpu_avx2;
#[cfg(feature = "cubecl")]
pub mod cubecl_backend;
#[cfg(feature = "cubecl-wgpu")]
pub use cubecl_backend::{
    CUBECL_WGPU_BACKEND, WgpuDeviceBuffer, linear_out_by_in_wgpu, rope_apply_inplace_wgpu,
};
#[cfg(feature = "cubecl")]
pub use cubecl_backend::{CubeClKernelBackend, linear_out_by_in_cubecl, rope_apply_inplace_cubecl};
pub mod mlp;
pub mod rmsnorm;
pub mod tensor;
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

    fn rope_apply_inplace(
        &self,
        x: &mut [f32],
        head_dim: usize,
        position: usize,
        theta: f32,
    ) -> Result<()>;

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

#[derive(Debug, Clone)]
pub struct CpuKernelBackend {
    context: KernelContext,
    mode: CpuKernelMode,
    /// Optional thread pool. `None` means single-threaded execution, which is
    /// the parity oracle the default tests rely on. A pool is built when the
    /// caller asks for `threads >= 2` via `with_mode_and_threads`.
    pool: Option<Arc<rayon::ThreadPool>>,
}

impl Default for CpuKernelBackend {
    fn default() -> Self {
        Self::scalar()
    }
}

impl CpuKernelBackend {
    pub fn scalar() -> Self {
        Self::with_mode(CpuKernelMode::Scalar)
    }

    pub fn optimized() -> Self {
        Self::with_mode(CpuKernelMode::Optimized)
    }

    pub fn with_mode(mode: CpuKernelMode) -> Self {
        // Infallible variant used by tests and defaults. For modes that
        // require runtime feature detection (Avx2), prefer
        // `with_mode_checked`.
        Self {
            context: KernelContext {
                device: Device::Cpu,
            },
            mode,
            pool: None,
        }
    }

    /// Fallible counterpart to `with_mode` that validates host-CPU support
    /// for the requested mode. Currently only `Avx2` requires this check.
    pub fn with_mode_checked(mode: CpuKernelMode) -> Result<Self> {
        validate_mode_supported(mode)?;
        Ok(Self::with_mode(mode))
    }

    /// Construct a backend that runs hot kernels (currently `linear_out_by_in`
    /// and any caller that opts in via `cpu_thread_pool()`) across `threads`
    /// worker threads. `threads <= 1` is equivalent to `with_mode_checked`.
    /// The pool is built once and reused for the lifetime of this backend.
    pub fn with_mode_and_threads(mode: CpuKernelMode, threads: usize) -> Result<Self> {
        validate_mode_supported(mode)?;
        if threads <= 1 {
            return Ok(Self::with_mode(mode));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("ocelotl-cpu-{i}"))
            .build()
            .map_err(|err| {
                OcelotlError::Kernel(KernelError {
                    backend: "cpu".to_string(),
                    message: format!("failed to build rayon thread pool: {err}"),
                })
            })?;
        Ok(Self {
            context: KernelContext {
                device: Device::Cpu,
            },
            mode,
            pool: Some(Arc::new(pool)),
        })
    }

    pub fn mode(&self) -> CpuKernelMode {
        self.mode
    }

    /// Number of worker threads, or 1 if running serially.
    pub fn num_threads(&self) -> usize {
        self.pool.as_ref().map_or(1, |p| p.current_num_threads())
    }

    pub fn matmul(
        &self,
        a: &[f32],
        a_shape: (usize, usize),
        b: &[f32],
        b_shape: (usize, usize),
        out: &mut [f32],
    ) -> Result<()> {
        let m = a_shape.0;
        if let Some(pool) = &self.pool {
            if m >= PARALLEL_MATMUL_MIN_ROWS {
                return matmul_parallel(pool, self.mode, a, a_shape, b, b_shape, out);
            }
        }
        match self.mode {
            CpuKernelMode::Scalar => matmul(a, a_shape, b, b_shape, out),
            CpuKernelMode::Optimized => matmul_optimized(a, a_shape, b, b_shape, out),
            // matmul is used by the Qwen-shaped GEMM; AVX2 today only
            // accelerates `linear_out_by_in` (the [out, in] Whisper weight
            // layout). Other matmul callers fall back to the optimized
            // scalar path until AVX2 covers them.
            CpuKernelMode::Avx2 => matmul_optimized(a, a_shape, b, b_shape, out),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn linear_out_by_in(
        &self,
        x: &[f32],
        rows: usize,
        in_features: usize,
        weight_out_by_in: &[f32],
        out_features: usize,
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        if let Some(pool) = &self.pool {
            if rows >= PARALLEL_LINEAR_MIN_ROWS {
                return linear_out_by_in_parallel(
                    pool,
                    self.mode,
                    x,
                    rows,
                    in_features,
                    weight_out_by_in,
                    out_features,
                    bias,
                    out,
                );
            }
        }
        match self.mode {
            CpuKernelMode::Scalar => linear_out_by_in(
                x,
                rows,
                in_features,
                weight_out_by_in,
                out_features,
                bias,
                out,
            ),
            CpuKernelMode::Optimized => linear_out_by_in_optimized(
                x,
                rows,
                in_features,
                weight_out_by_in,
                out_features,
                bias,
                out,
            ),
            CpuKernelMode::Avx2 => linear_out_by_in_avx2(
                x,
                rows,
                in_features,
                weight_out_by_in,
                out_features,
                bias,
                out,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn scaled_dot_product_attention(
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
        if let Some(pool) = &self.pool {
            if seq_len >= PARALLEL_SDPA_MIN_SEQ {
                return attention::scaled_dot_product_attention_parallel(
                    pool,
                    self.mode,
                    q,
                    k,
                    v,
                    seq_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    out,
                );
            }
        }
        match self.mode {
            CpuKernelMode::Scalar => attention::scaled_dot_product_attention(
                q,
                k,
                v,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                out,
            ),
            CpuKernelMode::Optimized | CpuKernelMode::Avx2 => {
                // Same fallback rationale as matmul: this kernel is only
                // exercised by the Qwen path right now. AVX2 currently
                // accelerates the Whisper-shaped `linear_out_by_in`; the
                // Qwen-shaped attention falls back to optimized scalar.
                attention::scaled_dot_product_attention_optimized(
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

    fn cpu_thread_pool(&self) -> Option<&rayon::ThreadPool> {
        self.pool.as_deref()
    }

    fn matmul(
        &self,
        a: &[f32],
        a_shape: (usize, usize),
        b: &[f32],
        b_shape: (usize, usize),
        out: &mut [f32],
    ) -> Result<()> {
        CpuKernelBackend::matmul(self, a, a_shape, b, b_shape, out)
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
        CpuKernelBackend::linear_out_by_in(
            self,
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
        CpuKernelBackend::scaled_dot_product_attention(
            self,
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
        rope_apply_inplace(x, head_dim, position, theta)
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
        rmsnorm::rmsnorm(x, rows, hidden, weight, epsilon, out)
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
        mlp::mlp_gated_silu(
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
        vec_add(a, b, out)
    }

    /// CPU override: when every handle is host-resident (the common case
    /// for the CPU backend), borrow the underlying `Vec<f32>` slices directly
    /// and call the existing `linear_out_by_in` — no readback, no extra
    /// allocations. Falls back to the trait default for the rare device-on-
    /// CPU case (which produces a readback through the default impl).
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
        // All inputs are host-resident in practice on this backend; borrow
        // their slices directly. If any side is somehow device-resident we
        // delegate to the default impl which forces readback.
        let (x_borrow, w_borrow, out_borrow) = match (
            x.borrow_host_slice(),
            weight.borrow_host_slice(),
            out.borrow_host_slice_mut(),
        ) {
            (Ok(x_b), Ok(w_b), Ok(out_b)) => (x_b, w_b, out_b),
            _ => {
                let x_host = x.to_host_owned()?;
                let weight_host = weight.to_host_owned()?;
                let bias_host = bias.map(DeviceTensor::to_host_owned).transpose()?;
                let mut out_buf = vec![0.0_f32; rows * out_features];
                CpuKernelBackend::linear_out_by_in(
                    self,
                    &x_host,
                    rows,
                    in_features,
                    &weight_host,
                    out_features,
                    bias_host.as_deref(),
                    &mut out_buf,
                )?;
                return out.write_from_host_slice(&out_buf);
            }
        };
        let mut out_borrow = out_borrow;
        match bias {
            Some(b) => {
                let b_borrow = b.borrow_host_slice()?;
                CpuKernelBackend::linear_out_by_in(
                    self,
                    &x_borrow,
                    rows,
                    in_features,
                    &w_borrow,
                    out_features,
                    Some(&b_borrow),
                    &mut out_borrow,
                )
            }
            None => CpuKernelBackend::linear_out_by_in(
                self,
                &x_borrow,
                rows,
                in_features,
                &w_borrow,
                out_features,
                None,
                &mut out_borrow,
            ),
        }
    }

    /// CPU override: borrow both host slices and do an elementwise add.
    /// Falls through to the trait default if either side is device-resident
    /// (shouldn't happen on the CPU backend, but the default path stays
    /// correct).
    fn add_inplace_d(&self, lhs: &DeviceTensor, rhs: &DeviceTensor) -> Result<()> {
        let lhs_borrow = lhs.borrow_host_slice_mut();
        let rhs_borrow = rhs.borrow_host_slice();
        match (lhs_borrow, rhs_borrow) {
            (Ok(mut l), Ok(r)) => {
                if l.len() != r.len() {
                    return Err(kernel_err(format!(
                        "add_inplace_d length mismatch: lhs={} rhs={}",
                        l.len(),
                        r.len()
                    )));
                }
                for (lv, rv) in l.iter_mut().zip(r.iter()) {
                    *lv += *rv;
                }
                Ok(())
            }
            _ => {
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
        }
    }

    /// CPU override: borrow the host slice and apply the Whisper-exact
    /// GELU in place. Bit-identical to `gelu_inplace` in the models crate
    /// because both call the same `gelu_whisper_scalar` math.
    fn gelu_inplace_d(&self, x: &DeviceTensor) -> Result<()> {
        match x.borrow_host_slice_mut() {
            Ok(mut borrow) => {
                for v in borrow.iter_mut() {
                    *v = gelu_whisper_scalar(*v);
                }
                Ok(())
            }
            Err(_) => {
                let mut host = x.to_host_owned()?;
                for v in host.iter_mut() {
                    *v = gelu_whisper_scalar(*v);
                }
                x.write_from_host_slice(&host)
            }
        }
    }

    /// CPU override: borrow host slices and run the scalar Whisper-shape
    /// LayerNorm directly. Falls through to the trait default for the
    /// device-on-CPU case.
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
        let (x_b, w_b, b_b, out_b) = match (
            x.borrow_host_slice(),
            weight.borrow_host_slice(),
            bias.borrow_host_slice(),
            out.borrow_host_slice_mut(),
        ) {
            (Ok(x_b), Ok(w_b), Ok(b_b), Ok(out_b)) => (x_b, w_b, b_b, out_b),
            _ => {
                let x_host = x.to_host_owned()?;
                let w_host = weight.to_host_owned()?;
                let b_host = bias.to_host_owned()?;
                let mut out_buf = vec![0.0_f32; rows * hidden];
                layer_norm_whisper_scalar(
                    &x_host,
                    rows,
                    hidden,
                    &w_host,
                    &b_host,
                    eps,
                    &mut out_buf,
                );
                return out.write_from_host_slice(&out_buf);
            }
        };
        let mut out_b = out_b;
        layer_norm_whisper_scalar(&x_b, rows, hidden, &w_b, &b_b, eps, &mut out_b);
        Ok(())
    }

    /// CPU override: borrow host slices and run the scalar encoder
    /// attention directly. Falls through to the trait default if any
    /// operand is device-resident (shouldn't happen on the CPU backend,
    /// but the default path stays correct).
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
        let (q_b, k_b, v_b, out_b) = match (
            q.borrow_host_slice(),
            k.borrow_host_slice(),
            v.borrow_host_slice(),
            output.borrow_host_slice_mut(),
        ) {
            (Ok(q_b), Ok(k_b), Ok(v_b), Ok(out_b)) => (q_b, k_b, v_b, out_b),
            _ => {
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
                return output.write_from_host_slice(&out_buf);
            }
        };
        let mut out_b = out_b;
        attention_encoder_scalar(&q_b, &k_b, &v_b, seq, n_head, head_dim, scale, &mut out_b);
        Ok(())
    }

    /// CPU override: borrow host slices and accumulate the positional
    /// embedding into `x` in place. Falls through to the trait default
    /// for the device-on-CPU case.
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
        let (x_b, pe_b) = match (x.borrow_host_slice_mut(), pe.borrow_host_slice()) {
            (Ok(x_b), Ok(pe_b)) => (x_b, pe_b),
            _ => {
                let mut x_host = x.to_host_owned()?;
                let pe_host = pe.to_host_owned()?;
                for row in 0..rows {
                    let dst_start = row * cols;
                    let src_start = (start_pos + row) * cols;
                    for col in 0..cols {
                        x_host[dst_start + col] += pe_host[src_start + col];
                    }
                }
                return x.write_from_host_slice(&x_host);
            }
        };
        let mut x_b = x_b;
        for row in 0..rows {
            let dst_start = row * cols;
            let src_start = (start_pos + row) * cols;
            for col in 0..cols {
                x_b[dst_start + col] += pe_b[src_start + col];
            }
        }
        Ok(())
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
/// all classic sizes); decoder single-token decode has rows=1 and stays
/// serial regardless of pool configuration.
const PARALLEL_LINEAR_MIN_ROWS: usize = 32;

/// Same rationale as `PARALLEL_LINEAR_MIN_ROWS` but for the generic `matmul`
/// kernel. Qwen prefill uses M = seq_len which can run into the hundreds for
/// realistic prompts; decode is M = 1 and stays serial.
const PARALLEL_MATMUL_MIN_ROWS: usize = 32;

/// Below this query count, single-threaded SDPA beats the rayon dispatch
/// overhead. Mirrors the Whisper attention threshold; chosen so that single-
/// token decode (seq_len = 1) stays serial.
const PARALLEL_SDPA_MIN_SEQ: usize = 32;

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
mod tests {
    use super::*;

    // --- vec_add ---

    #[test]
    fn vec_add_handles_four_element_vectors() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [10.0_f32, 20.0, 30.0, 40.0];
        let mut out = [0.0_f32; 4];

        vec_add(&a, &b, &mut out).expect("equal-length vec_add must succeed");

        assert_eq!(out, [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn vec_add_rejects_mismatched_input_lengths() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [1.0_f32, 2.0];
        let mut out = [0.0_f32; 3];

        let err = vec_add(&a, &b, &mut out).expect_err("must reject mismatched input lengths");

        match err {
            OcelotlError::Kernel(KernelError { backend, message }) => {
                assert_eq!(backend, "cpu");
                assert!(
                    message.contains("vec_add"),
                    "expected error to mention kernel name, got {message:?}"
                );
            }
            other => panic!("expected KernelError, got {other:?}"),
        }
    }

    #[test]
    fn vec_add_rejects_mismatched_output_length() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [10.0_f32, 20.0, 30.0];
        let mut out = [0.0_f32; 4];

        let err = vec_add(&a, &b, &mut out).expect_err("must reject mismatched output length");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }

    // --- dot ---

    #[test]
    fn dot_computes_inner_product_of_four_element_vectors() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [10.0_f32, 20.0, 30.0, 40.0];

        // 1*10 + 2*20 + 3*30 + 4*40 = 10 + 40 + 90 + 160 = 300
        let result = dot(&a, &b).expect("equal-length dot must succeed");

        assert_eq!(result, 300.0);
    }

    #[test]
    fn dot_of_empty_slices_is_zero() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];

        assert_eq!(dot(&a, &b).unwrap(), 0.0);
    }

    #[test]
    fn dot_rejects_mismatched_lengths() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [1.0_f32, 2.0];

        let err = dot(&a, &b).expect_err("must reject mismatched lengths");

        match err {
            OcelotlError::Kernel(KernelError { backend, message }) => {
                assert_eq!(backend, "cpu");
                assert!(
                    message.contains("dot"),
                    "expected error to mention kernel name, got {message:?}"
                );
            }
            other => panic!("expected KernelError, got {other:?}"),
        }
    }

    // --- softmax ---

    #[test]
    fn softmax_produces_known_distribution_for_three_element_input() {
        // Hand-checked: softmax([1, 2, 3]) with max-subtraction stability:
        //   shifted = [-2, -1, 0]
        //   e^shifted ≈ [0.13533528, 0.36787944, 1.0]
        //   sum ≈ 1.50321472
        //   result ≈ [0.09003057, 0.24472847, 0.66524096]
        let mut x = [1.0_f32, 2.0, 3.0];
        softmax(&mut x);

        let expected = [0.09003057_f32, 0.24472847, 0.66524096];
        for (got, want) in x.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 4.0 * f32::EPSILON,
                "softmax mismatch: got {got}, want {want}"
            );
        }

        let sum: f32 = x.iter().sum();
        assert!(
            (sum - 1.0).abs() < 4.0 * f32::EPSILON,
            "softmax must sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn softmax_is_stable_for_large_inputs() {
        // Without max-subtraction, exp(1000) overflows to +inf and the result
        // is NaN. With max-subtraction, the largest exponent is 0 and the
        // result is well-defined.
        let mut x = [1000.0_f32, 1001.0, 1002.0];
        softmax(&mut x);

        let sum: f32 = x.iter().sum();
        assert!(
            (sum - 1.0).abs() < 4.0 * f32::EPSILON,
            "stable softmax must sum to 1.0 even for large inputs, got {sum}"
        );
        for v in x.iter() {
            assert!(v.is_finite(), "softmax output must be finite, got {v}");
        }
    }

    #[test]
    fn softmax_of_empty_slice_is_a_noop() {
        let mut x: [f32; 0] = [];
        softmax(&mut x);
        // No assertion needed — must not panic.
    }

    #[test]
    fn softmax_of_uniform_input_is_uniform_distribution() {
        let mut x = [5.0_f32; 4];
        softmax(&mut x);
        for v in x.iter() {
            assert!(
                (v - 0.25).abs() < 4.0 * f32::EPSILON,
                "uniform softmax must be 1/n, got {v}"
            );
        }
    }

    // --- matmul ---

    #[test]
    fn matmul_handles_two_by_two_times_two_by_two() {
        // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        // Hand check: row 0 of out = [1*5+2*7, 1*6+2*8] = [19, 22]
        //             row 1 of out = [3*5+4*7, 3*6+4*8] = [43, 50]
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [5.0_f32, 6.0, 7.0, 8.0];
        let mut out = [0.0_f32; 4];

        matmul(&a, (2, 2), &b, (2, 2), &mut out).expect("well-formed matmul must succeed");

        assert_eq!(out, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_handles_non_square_two_by_three_times_three_by_two() {
        // A = [[1, 2, 3], [4, 5, 6]]   (2x3)
        // B = [[7, 8], [9, 10], [11, 12]]   (3x2)
        // A@B row 0 = [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // A@B row 1 = [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out = [0.0_f32; 4];

        matmul(&a, (2, 3), &b, (3, 2), &mut out).expect("well-formed matmul must succeed");

        assert_eq!(out, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_rejects_inner_dimension_mismatch() {
        let a = [1.0_f32; 6]; // 2x3
        let b = [1.0_f32; 8]; // 4x2 — inner dims disagree
        let mut out = [0.0_f32; 4];

        let err =
            matmul(&a, (2, 3), &b, (4, 2), &mut out).expect_err("must reject inner-dim mismatch");

        match err {
            OcelotlError::Kernel(KernelError { backend, message }) => {
                assert_eq!(backend, "cpu");
                assert!(
                    message.contains("inner-dimension"),
                    "expected inner-dim message, got {message:?}"
                );
            }
            other => panic!("expected KernelError, got {other:?}"),
        }
    }

    #[test]
    fn matmul_rejects_wrong_a_slice_length() {
        let a = [1.0_f32; 5]; // claimed 2x3, actually 5
        let b = [1.0_f32; 6]; // 3x2
        let mut out = [0.0_f32; 4];

        let err = matmul(&a, (2, 3), &b, (3, 2), &mut out).expect_err("must reject wrong a length");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }

    #[test]
    fn matmul_rejects_wrong_output_length() {
        let a = [1.0_f32; 6]; // 2x3
        let b = [1.0_f32; 6]; // 3x2
        let mut out = [0.0_f32; 3]; // claimed 2x2 = 4

        let err =
            matmul(&a, (2, 3), &b, (3, 2), &mut out).expect_err("must reject wrong out length");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }

    #[test]
    fn matmul_rejects_shape_product_overflow() {
        let a = [];
        let b = [];
        let mut out = [];

        let err = matmul(&a, (usize::MAX, 2), &b, (2, 1), &mut out)
            .expect_err("overflowing shapes must be rejected");

        match err {
            OcelotlError::Kernel(KernelError { message, .. }) => {
                assert!(
                    message.contains("overflows"),
                    "expected overflow diagnostic, got {message:?}"
                );
            }
            other => panic!("expected KernelError, got {other:?}"),
        }
    }

    #[test]
    fn cpu_backend_defaults_to_scalar_mode() {
        let backend = CpuKernelBackend::default();

        assert_eq!(backend.mode(), CpuKernelMode::Scalar);
        assert_eq!(backend.name(), "cpu");
        assert_eq!(backend.context().device, Device::Cpu);
    }

    #[test]
    fn cpu_backend_rejects_gpu_requirement_with_typed_unsupported_error() {
        let backend = CpuKernelBackend::default();

        let err = require_gpu(&backend).expect_err("CPU backend must not satisfy GPU requirement");

        match err {
            OcelotlError::Unsupported(UnsupportedError {
                feature,
                requested,
                supported,
            }) => {
                assert_eq!(feature, "gpu_backend");
                assert_eq!(requested.as_deref(), Some("gpu"));
                assert_eq!(supported, vec!["cpu".to_string()]);
            }
            other => panic!("expected UnsupportedError, got {other:?}"),
        }
    }

    #[test]
    fn cpu_backend_can_select_optimized_mode() {
        let backend = CpuKernelBackend::optimized();

        assert_eq!(backend.mode(), CpuKernelMode::Optimized);
        assert_eq!(CpuKernelMode::Optimized.as_str(), "optimized");
    }

    #[test]
    fn optimized_matmul_matches_scalar_for_non_square_shape() {
        let a = [
            0.25_f32, -0.5, 1.0, //
            1.5, 0.75, -1.25,
        ];
        let b = [
            0.5_f32, -1.0, 0.25, 2.0, //
            -0.75, 0.5, 1.25, -0.5, //
            1.0, 1.5, -1.0, 0.75,
        ];
        let scalar = CpuKernelBackend::scalar();
        let optimized = CpuKernelBackend::optimized();
        let mut scalar_out = [0.0_f32; 8];
        let mut optimized_out = [0.0_f32; 8];

        scalar
            .matmul(&a, (2, 3), &b, (3, 4), &mut scalar_out)
            .unwrap();
        optimized
            .matmul(&a, (2, 3), &b, (3, 4), &mut optimized_out)
            .unwrap();

        for (got, want) in optimized_out.iter().zip(scalar_out.iter()) {
            assert!(
                (got - want).abs() <= 1.0e-6,
                "optimized matmul drifted: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn optimized_linear_out_by_in_matches_scalar_with_bias() {
        let x = [
            1.0_f32, -2.0, 0.5, //
            -0.25, 1.5, 2.0,
        ];
        // [out_features, in_features] layout.
        let w = [
            0.5_f32, -1.0, 0.25, //
            -0.75, 0.5, 1.25, //
            1.0, 1.5, -1.0, //
            0.0, -0.5, 0.75,
        ];
        let bias = [0.1_f32, -0.2, 0.3, -0.4];
        let scalar = CpuKernelBackend::scalar();
        let optimized = CpuKernelBackend::optimized();
        let mut scalar_out = [0.0_f32; 8];
        let mut optimized_out = [0.0_f32; 8];

        scalar
            .linear_out_by_in(&x, 2, 3, &w, 4, Some(&bias), &mut scalar_out)
            .unwrap();
        optimized
            .linear_out_by_in(&x, 2, 3, &w, 4, Some(&bias), &mut optimized_out)
            .unwrap();

        for (got, want) in optimized_out.iter().zip(scalar_out.iter()) {
            assert!(
                (got - want).abs() <= 1.0e-6,
                "optimized linear_out_by_in drifted: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn scalar_linear_out_by_in_handles_row_and_output_tile_tails() {
        let rows = 5;
        let in_features = 3;
        let out_features = 5;
        let x = [
            1.0_f32, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
            10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0,
        ];
        let w = [
            0.5_f32, 1.0, -0.5, //
            -1.0, 0.25, 0.75, //
            1.5, -0.25, 0.0, //
            0.0, -0.5, 2.0, //
            -0.75, 0.5, 1.25,
        ];
        let bias = [0.1_f32, -0.2, 0.3, -0.4, 0.5];
        let mut got = [0.0_f32; 25];
        let mut want = [0.0_f32; 25];

        CpuKernelBackend::scalar()
            .linear_out_by_in(
                &x,
                rows,
                in_features,
                &w,
                out_features,
                Some(&bias),
                &mut got,
            )
            .unwrap();

        for row in 0..rows {
            for out_dim in 0..out_features {
                let mut acc = bias[out_dim];
                for in_dim in 0..in_features {
                    acc += x[row * in_features + in_dim] * w[out_dim * in_features + in_dim];
                }
                want[row * out_features + out_dim] = acc;
            }
        }

        assert_eq!(got, want);
    }

    #[test]
    fn threaded_linear_out_by_in_matches_serial_bit_for_bit() {
        // Parity oracle for the rayon parallel dispatch. Each chunk writes
        // disjoint output rows; the K-loop accumulation order inside a chunk
        // is identical to the serial path, so the threaded output must be
        // bit-identical to the serial output. A run that drifts here would
        // indicate either a chunk boundary bug or that the compute helper
        // does not match the legacy path.
        let rows = 64usize; // > PARALLEL_LINEAR_MIN_ROWS so the pool dispatches
        let in_features = 17;
        let out_features = 13;
        let x: Vec<f32> = (0..(rows * in_features))
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let w: Vec<f32> = (0..(out_features * in_features))
            .map(|i| ((i as f32) * 0.019).cos())
            .collect();
        let b: Vec<f32> = (0..out_features).map(|i| (i as f32) * 0.05).collect();

        let mut serial = vec![0.0_f32; rows * out_features];
        let serial_backend = CpuKernelBackend::scalar();
        serial_backend
            .linear_out_by_in(
                &x,
                rows,
                in_features,
                &w,
                out_features,
                Some(&b),
                &mut serial,
            )
            .expect("serial linear must succeed");

        let mut threaded = vec![0.0_f32; rows * out_features];
        let threaded_backend = CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Scalar, 4)
            .expect("4-thread backend must build");
        threaded_backend
            .linear_out_by_in(
                &x,
                rows,
                in_features,
                &w,
                out_features,
                Some(&b),
                &mut threaded,
            )
            .expect("threaded linear must succeed");

        assert_eq!(
            serial, threaded,
            "threaded linear_out_by_in must produce bit-identical output to serial"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_linear_out_by_in_matches_scalar_within_tolerance() {
        // Parity oracle for the AVX2 + FMA path. The output is not bit-
        // identical to Scalar because FMA fuses one multiply and one add
        // into a single rounded operation; the scalar path does two
        // roundings. The relative error must stay tight on Whisper-sized
        // matmuls.
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            // Host doesn't support AVX2+FMA; skip rather than fail. The
            // backend constructor would return a typed error in this case.
            return;
        }
        let rows = 64usize;
        let in_features = 200; // not a multiple of 8, exercise K-tail
        let out_features = 33; // not a multiple of 4, exercise out-tail
        let x: Vec<f32> = (0..(rows * in_features))
            .map(|i| ((i as f32) * 0.011).sin())
            .collect();
        let w: Vec<f32> = (0..(out_features * in_features))
            .map(|i| ((i as f32) * 0.017).cos())
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
            .expect("scalar must succeed");

        let mut avx2 = vec![0.0_f32; rows * out_features];
        CpuKernelBackend::with_mode_checked(CpuKernelMode::Avx2)
            .expect("AVX2 backend must build on host that advertises avx2+fma")
            .linear_out_by_in(&x, rows, in_features, &w, out_features, Some(&b), &mut avx2)
            .expect("AVX2 must succeed");

        // Tolerance: 1e-4 relative or absolute. FMA fuses a multiply and
        // add into one rounding (vs scalar's two roundings) and the SIMD
        // path also accumulates into 8 partial-sum lanes that are reduced
        // after the K-loop, so order-of-addition differs. The drift is
        // bounded and tiny on Whisper-sized matmuls; Whisper exact-token
        // parity (much coarser, only argmax order matters) is still
        // preserved end-to-end and is pinned by the bench hook's
        // `matches_expected` field.
        for (idx, (a, s)) in avx2.iter().zip(scalar.iter()).enumerate() {
            let abs = (a - s).abs();
            let rel = if s.abs() > 1e-6 { abs / s.abs() } else { abs };
            assert!(
                abs <= 1e-4 || rel <= 1e-4,
                "AVX2 output drifted at idx {idx}: scalar={s} avx2={a} abs={abs} rel={rel}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_linear_out_by_in_threaded_matches_serial_avx2_within_tolerance() {
        // Compose AVX2 + threads. Each chunk runs the AVX2 compute body on
        // disjoint output rows; the result should match the single-thread
        // AVX2 output bit-for-bit because the per-row K-loop accumulation
        // order is identical between serial and parallel AVX2.
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let rows = 96usize;
        let in_features = 256;
        let out_features = 64;
        let x: Vec<f32> = (0..(rows * in_features))
            .map(|i| ((i as f32) * 0.007).sin())
            .collect();
        let w: Vec<f32> = (0..(out_features * in_features))
            .map(|i| ((i as f32) * 0.013).cos())
            .collect();

        let mut serial = vec![0.0_f32; rows * out_features];
        CpuKernelBackend::with_mode_checked(CpuKernelMode::Avx2)
            .unwrap()
            .linear_out_by_in(&x, rows, in_features, &w, out_features, None, &mut serial)
            .unwrap();

        let mut threaded = vec![0.0_f32; rows * out_features];
        CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Avx2, 4)
            .unwrap()
            .linear_out_by_in(&x, rows, in_features, &w, out_features, None, &mut threaded)
            .unwrap();

        assert_eq!(
            serial, threaded,
            "AVX2 + threads must match AVX2 serial bit-for-bit"
        );
    }

    #[test]
    fn threaded_linear_out_by_in_falls_back_to_serial_for_small_inputs() {
        // Rows below PARALLEL_LINEAR_MIN_ROWS must skip the pool dispatch.
        // We can't observe that directly, but we can confirm the small path
        // still produces the same result as the serial backend.
        let rows = 3usize;
        let in_features = 5;
        let out_features = 4;
        let x: Vec<f32> = (0..(rows * in_features))
            .map(|i| (i as f32) * 0.1)
            .collect();
        let w: Vec<f32> = (0..(out_features * in_features))
            .map(|i| (i as f32) * -0.07)
            .collect();

        let mut serial = vec![0.0_f32; rows * out_features];
        CpuKernelBackend::scalar()
            .linear_out_by_in(&x, rows, in_features, &w, out_features, None, &mut serial)
            .unwrap();

        let mut threaded = vec![0.0_f32; rows * out_features];
        CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Scalar, 4)
            .unwrap()
            .linear_out_by_in(&x, rows, in_features, &w, out_features, None, &mut threaded)
            .unwrap();

        assert_eq!(serial, threaded);
    }

    #[test]
    fn optimized_attention_matches_scalar_backend() {
        let q = [
            1.0_f32, 0.0, //
            0.0, 1.0, //
            0.5, 0.5,
        ];
        let k = q;
        let v = [
            1.0_f32, 2.0, //
            3.0, 4.0, //
            100.0, -50.0,
        ];
        let scalar = CpuKernelBackend::scalar();
        let optimized = CpuKernelBackend::optimized();
        let mut scalar_out = [0.0_f32; 6];
        let mut optimized_out = [0.0_f32; 6];

        scalar
            .scaled_dot_product_attention(&q, &k, &v, 3, 1, 1, 2, &mut scalar_out)
            .unwrap();
        optimized
            .scaled_dot_product_attention(&q, &k, &v, 3, 1, 1, 2, &mut optimized_out)
            .unwrap();

        for (got, want) in optimized_out.iter().zip(scalar_out.iter()) {
            assert!(
                (got - want).abs() <= 1.0e-6,
                "optimized attention drifted: got {got}, want {want}"
            );
        }
    }

    // --- parallel matmul parity ---

    #[test]
    fn threaded_matmul_scalar_matches_serial_bit_for_bit() {
        // Disjoint output-row chunks + identical accumulation order =
        // bit-identical to the serial scalar path.
        let m = 64usize;
        let k = 19usize;
        let n = 17usize;
        let a: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.013).sin()).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.019).cos()).collect();

        let mut serial = vec![0.0_f32; m * n];
        CpuKernelBackend::scalar()
            .matmul(&a, (m, k), &b, (k, n), &mut serial)
            .expect("serial scalar matmul must succeed");

        let mut threaded = vec![0.0_f32; m * n];
        CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Scalar, 4)
            .expect("4-thread scalar backend must build")
            .matmul(&a, (m, k), &b, (k, n), &mut threaded)
            .expect("threaded matmul must succeed");

        assert_eq!(
            serial, threaded,
            "threaded matmul must be bit-identical to serial scalar"
        );
    }

    #[test]
    fn threaded_matmul_optimized_matches_serial_bit_for_bit() {
        let m = 96usize;
        let k = 33usize;
        let n = 25usize;
        let a: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.011).sin()).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.017).cos()).collect();

        let mut serial = vec![0.0_f32; m * n];
        CpuKernelBackend::optimized()
            .matmul(&a, (m, k), &b, (k, n), &mut serial)
            .expect("serial optimized matmul must succeed");

        let mut threaded = vec![0.0_f32; m * n];
        CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Optimized, 4)
            .expect("4-thread optimized backend must build")
            .matmul(&a, (m, k), &b, (k, n), &mut threaded)
            .expect("threaded matmul must succeed");

        assert_eq!(
            serial, threaded,
            "threaded optimized matmul must be bit-identical to serial optimized"
        );
    }

    #[test]
    fn threaded_matmul_falls_back_to_serial_for_small_inputs() {
        // M below PARALLEL_MATMUL_MIN_ROWS must still produce the serial
        // result. We can't observe the dispatch path directly, but a small
        // shape must round-trip identically.
        let m = 5usize;
        let k = 7usize;
        let n = 3usize;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * -0.07).collect();

        let mut serial = vec![0.0_f32; m * n];
        CpuKernelBackend::scalar()
            .matmul(&a, (m, k), &b, (k, n), &mut serial)
            .unwrap();

        let mut threaded = vec![0.0_f32; m * n];
        CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Scalar, 4)
            .unwrap()
            .matmul(&a, (m, k), &b, (k, n), &mut threaded)
            .unwrap();

        assert_eq!(serial, threaded);
    }

    // --- parallel SDPA parity ---

    #[test]
    fn threaded_sdpa_scalar_matches_serial_bit_for_bit() {
        // Per-chunk scratch + per-query-position output cells = identical
        // accumulation order to the serial path. Use a seq_len above
        // PARALLEL_SDPA_MIN_SEQ so the pool actually dispatches.
        let seq_len = 48usize;
        let num_q_heads = 4usize;
        let num_kv_heads = 2usize;
        let head_dim = 6usize;
        let q: Vec<f32> = (0..(seq_len * num_q_heads * head_dim))
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let k: Vec<f32> = (0..(seq_len * num_kv_heads * head_dim))
            .map(|i| ((i as f32) * 0.019).cos())
            .collect();
        let v: Vec<f32> = (0..(seq_len * num_kv_heads * head_dim))
            .map(|i| ((i as f32) * 0.023).sin())
            .collect();

        let mut serial = vec![0.0_f32; seq_len * num_q_heads * head_dim];
        CpuKernelBackend::scalar()
            .scaled_dot_product_attention(
                &q,
                &k,
                &v,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                &mut serial,
            )
            .expect("serial SDPA must succeed");

        let mut threaded = vec![0.0_f32; seq_len * num_q_heads * head_dim];
        CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Scalar, 4)
            .expect("4-thread scalar backend must build")
            .scaled_dot_product_attention(
                &q,
                &k,
                &v,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                &mut threaded,
            )
            .expect("threaded SDPA must succeed");

        assert_eq!(
            serial, threaded,
            "threaded scalar SDPA must be bit-identical to serial scalar SDPA"
        );
    }

    #[test]
    fn threaded_sdpa_optimized_matches_serial_bit_for_bit() {
        let seq_len = 64usize;
        let num_q_heads = 6usize;
        let num_kv_heads = 2usize;
        let head_dim = 8usize;
        let q: Vec<f32> = (0..(seq_len * num_q_heads * head_dim))
            .map(|i| ((i as f32) * 0.011).sin())
            .collect();
        let k: Vec<f32> = (0..(seq_len * num_kv_heads * head_dim))
            .map(|i| ((i as f32) * 0.017).cos())
            .collect();
        let v: Vec<f32> = (0..(seq_len * num_kv_heads * head_dim))
            .map(|i| ((i as f32) * 0.021).sin())
            .collect();

        let mut serial = vec![0.0_f32; seq_len * num_q_heads * head_dim];
        CpuKernelBackend::optimized()
            .scaled_dot_product_attention(
                &q,
                &k,
                &v,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                &mut serial,
            )
            .expect("serial optimized SDPA must succeed");

        let mut threaded = vec![0.0_f32; seq_len * num_q_heads * head_dim];
        CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Optimized, 4)
            .expect("4-thread optimized backend must build")
            .scaled_dot_product_attention(
                &q,
                &k,
                &v,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                &mut threaded,
            )
            .expect("threaded SDPA must succeed");

        assert_eq!(
            serial, threaded,
            "threaded optimized SDPA must be bit-identical to serial optimized SDPA"
        );
    }

    #[test]
    fn threaded_sdpa_falls_back_to_serial_for_small_seq() {
        // seq_len below PARALLEL_SDPA_MIN_SEQ stays on the serial path.
        let seq_len = 3usize;
        let num_q_heads = 2usize;
        let num_kv_heads = 1usize;
        let head_dim = 2usize;
        let q = [
            1.0_f32, 0.0, 0.0, 1.0, 0.5, 0.5, 0.25, 0.75, -1.0, 1.0, 0.3, 0.7,
        ];
        let k = [1.0_f32, 0.0, 0.5, 0.5, -1.0, 1.0];
        let v = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let mut serial = vec![0.0_f32; seq_len * num_q_heads * head_dim];
        CpuKernelBackend::scalar()
            .scaled_dot_product_attention(
                &q,
                &k,
                &v,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                &mut serial,
            )
            .unwrap();

        let mut threaded = vec![0.0_f32; seq_len * num_q_heads * head_dim];
        CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Scalar, 4)
            .unwrap()
            .scaled_dot_product_attention(
                &q,
                &k,
                &v,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                &mut threaded,
            )
            .unwrap();

        assert_eq!(serial, threaded);
    }

    // --- GW.4 Stage 1A: linear_d parity ---

    #[test]
    fn cpu_linear_d_matches_linear_out_by_in_bit_for_bit() {
        // The CPU `linear_d` override borrows host slices directly and calls
        // the existing `linear_out_by_in`. The output must be bit-identical
        // to the slice-based path on the same inputs — `linear_d` is the
        // parity oracle the GPU implementation (GW.4-1B) will validate
        // against.
        let rows = 7usize;
        let in_features = 23usize;
        let out_features = 13usize;
        let x_vec: Vec<f32> = (0..rows * in_features)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let w_vec: Vec<f32> = (0..out_features * in_features)
            .map(|i| ((i as f32) * 0.019).cos())
            .collect();
        let b_vec: Vec<f32> = (0..out_features).map(|i| (i as f32) * 0.05).collect();

        let backend = CpuKernelBackend::scalar();

        // Slice path.
        let mut slice_out = vec![0.0_f32; rows * out_features];
        backend
            .linear_out_by_in(
                &x_vec,
                rows,
                in_features,
                &w_vec,
                out_features,
                Some(&b_vec),
                &mut slice_out,
            )
            .unwrap();

        // Handle path.
        let x_h = DeviceTensor::from_host(x_vec.clone());
        let w_h = DeviceTensor::from_host(w_vec.clone());
        let b_h = DeviceTensor::from_host(b_vec.clone());
        let out_h = DeviceTensor::host_zeros(rows * out_features);
        backend
            .linear_d(
                &x_h,
                rows,
                in_features,
                &w_h,
                out_features,
                Some(&b_h),
                &out_h,
            )
            .unwrap();
        let handle_out = out_h.to_host_owned().unwrap();

        assert_eq!(
            slice_out, handle_out,
            "linear_d must be bit-identical to linear_out_by_in on the CPU backend"
        );
    }

    #[test]
    fn cpu_linear_d_handles_no_bias() {
        let rows = 3usize;
        let in_features = 5usize;
        let out_features = 4usize;
        let x_vec: Vec<f32> = (0..rows * in_features).map(|i| (i as f32) * 0.1).collect();
        let w_vec: Vec<f32> = (0..out_features * in_features)
            .map(|i| (i as f32) * -0.07)
            .collect();

        let backend = CpuKernelBackend::scalar();

        let mut slice_out = vec![0.0_f32; rows * out_features];
        backend
            .linear_out_by_in(
                &x_vec,
                rows,
                in_features,
                &w_vec,
                out_features,
                None,
                &mut slice_out,
            )
            .unwrap();

        let x_h = DeviceTensor::from_host(x_vec);
        let w_h = DeviceTensor::from_host(w_vec);
        let out_h = DeviceTensor::host_zeros(rows * out_features);
        backend
            .linear_d(&x_h, rows, in_features, &w_h, out_features, None, &out_h)
            .unwrap();
        assert_eq!(slice_out, out_h.to_host_owned().unwrap());
    }

    // --- GW.4 Stage 2A: device-resident critical-chain primitives ---

    #[test]
    fn cpu_add_inplace_d_matches_scalar_add_bit_for_bit() {
        // Parity oracle: a hand-rolled lhs+rhs loop. CPU backend must
        // produce identical f32 bit patterns.
        let lhs_vec: Vec<f32> = (0..37).map(|i| (i as f32) * 0.013).collect();
        let rhs_vec: Vec<f32> = (0..37).map(|i| ((i as f32) * 0.019).sin()).collect();
        let mut expected = lhs_vec.clone();
        for (l, r) in expected.iter_mut().zip(rhs_vec.iter()) {
            *l += *r;
        }

        let backend = CpuKernelBackend::scalar();
        let lhs_h = DeviceTensor::from_host(lhs_vec.clone());
        let rhs_h = DeviceTensor::from_host(rhs_vec);
        backend.add_inplace_d(&lhs_h, &rhs_h).unwrap();
        assert_eq!(lhs_h.to_host_owned().unwrap(), expected);
    }

    #[test]
    fn cpu_add_inplace_d_rejects_mismatched_lengths() {
        let backend = CpuKernelBackend::scalar();
        let lhs_h = DeviceTensor::from_host(vec![1.0_f32, 2.0, 3.0]);
        let rhs_h = DeviceTensor::from_host(vec![1.0_f32, 2.0]);
        let err = backend
            .add_inplace_d(&lhs_h, &rhs_h)
            .expect_err("must reject");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }

    #[test]
    fn cpu_gelu_inplace_d_matches_whisper_primitive_bit_for_bit() {
        // Bit-exactness gate: this is the math both the kernels crate and
        // the Whisper primitive call. They share the same scalar function
        // (gelu_whisper_scalar), so the only way this can drift is if one
        // side changes the formula.
        let host: Vec<f32> = (-32..32).map(|i| (i as f32) * 0.25).collect();
        let mut expected = host.clone();
        for v in expected.iter_mut() {
            *v = gelu_whisper_scalar(*v);
        }

        let backend = CpuKernelBackend::scalar();
        let t = DeviceTensor::from_host(host);
        backend.gelu_inplace_d(&t).unwrap();
        assert_eq!(t.to_host_owned().unwrap(), expected);
    }

    #[test]
    fn cpu_layer_norm_d_matches_whisper_primitive_bit_for_bit() {
        // This test fails iff the kernels-crate scalar layer-norm has
        // drifted from the Whisper primitive. Both call layer_norm_whisper_scalar,
        // so a failure here is the canary.
        let rows = 17usize;
        let hidden = 23usize;
        let eps = 1e-5_f32;
        let x_vec: Vec<f32> = (0..rows * hidden)
            .map(|i| ((i as f32) * 0.011).sin())
            .collect();
        let weight: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let bias: Vec<f32> = (0..hidden).map(|i| (i as f32) * -0.005).collect();

        let mut expected = vec![0.0_f32; rows * hidden];
        layer_norm_whisper_scalar(&x_vec, rows, hidden, &weight, &bias, eps, &mut expected);

        let backend = CpuKernelBackend::scalar();
        let x_h = DeviceTensor::from_host(x_vec);
        let w_h = DeviceTensor::from_host(weight);
        let b_h = DeviceTensor::from_host(bias);
        let out_h = DeviceTensor::host_zeros(rows * hidden);
        backend
            .layer_norm_d(&x_h, rows, hidden, &w_h, &b_h, eps, &out_h)
            .unwrap();
        assert_eq!(out_h.to_host_owned().unwrap(), expected);
    }

    #[test]
    fn cpu_layer_norm_d_rejects_shape_mismatch() {
        let backend = CpuKernelBackend::scalar();
        let x_h = DeviceTensor::from_host(vec![0.0_f32; 12]);
        let w_h = DeviceTensor::from_host(vec![1.0_f32; 4]);
        let b_h = DeviceTensor::from_host(vec![0.0_f32; 4]);
        let out_h = DeviceTensor::host_zeros(11); // wrong length
        let err = backend
            .layer_norm_d(&x_h, 3, 4, &w_h, &b_h, 1e-5, &out_h)
            .expect_err("must reject");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }

    #[test]
    fn cpu_add_positional_embedding_d_matches_scalar_bit_for_bit() {
        let rows = 5usize;
        let cols = 11usize;
        let pe_rows = 12usize;
        let start_pos = 3usize;
        let x_vec: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.02).collect();
        let pe_vec: Vec<f32> = (0..pe_rows * cols).map(|i| (i as f32) * -0.013).collect();

        let mut expected = x_vec.clone();
        for row in 0..rows {
            let dst_start = row * cols;
            let src_start = (start_pos + row) * cols;
            for col in 0..cols {
                expected[dst_start + col] += pe_vec[src_start + col];
            }
        }

        let backend = CpuKernelBackend::scalar();
        let x_h = DeviceTensor::from_host(x_vec);
        let pe_h = DeviceTensor::from_host(pe_vec);
        backend
            .add_positional_embedding_d(&x_h, rows, cols, &pe_h, pe_rows, start_pos)
            .unwrap();
        assert_eq!(x_h.to_host_owned().unwrap(), expected);
    }

    #[test]
    fn cpu_add_positional_embedding_d_rejects_out_of_range_start_pos() {
        let backend = CpuKernelBackend::scalar();
        let x_h = DeviceTensor::from_host(vec![0.0_f32; 6]); // 2 rows × 3 cols
        let pe_h = DeviceTensor::from_host(vec![0.0_f32; 12]); // 4 rows × 3 cols
        // start_pos=3 + rows=2 = 5 > pe_rows=4
        let err = backend
            .add_positional_embedding_d(&x_h, 2, 3, &pe_h, 4, 3)
            .expect_err("must reject");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }

    #[test]
    fn upload_default_returns_host_resident_tensor() {
        let backend = CpuKernelBackend::scalar();
        let t = backend.upload(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(t.residency(), Residency::Host);
        assert_eq!(t.to_host_owned().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn alloc_default_returns_zero_filled_host_tensor() {
        let backend = CpuKernelBackend::scalar();
        let t = backend.alloc(4).unwrap();
        assert_eq!(t.residency(), Residency::Host);
        assert_eq!(t.to_host_owned().unwrap(), vec![0.0, 0.0, 0.0, 0.0]);
    }

    // --- attention_encoder_d ---

    /// Bit-for-bit parity gate between the CPU backend's
    /// `attention_encoder_d` override (which borrows host slices) and the
    /// scalar reference. The override and the reference call the same
    /// `attention_encoder_scalar` helper, so the two paths must agree
    /// exactly — no tolerance window.
    #[test]
    fn cpu_attention_encoder_d_matches_scalar_bit_for_bit() {
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

        let mut scalar_out = vec![0.0_f32; seq * state];
        attention_encoder_scalar(&q, &k, &v, seq, n_head, head_dim, scale, &mut scalar_out);

        let backend = CpuKernelBackend::scalar();
        let q_d = backend.upload(&q).expect("upload q");
        let k_d = backend.upload(&k).expect("upload k");
        let v_d = backend.upload(&v).expect("upload v");
        let out_d = backend.alloc(seq * state).expect("alloc out");
        backend
            .attention_encoder_d(&q_d, &k_d, &v_d, seq, n_head, head_dim, scale, &out_d)
            .expect("CPU attention_encoder_d must succeed");
        let got = out_d.to_host_owned().expect("readback");

        assert_eq!(
            got, scalar_out,
            "CPU override must match scalar bit-for-bit"
        );
    }

    // -----------------------------------------------------------------------
    // GW.4-5B CPU parity gates
    // -----------------------------------------------------------------------

    /// GW.4-5B: `attention_decoder_causal_d` on the CPU backend must match
    /// `attention_decoder_causal_scalar` bit-for-bit. Both take the same
    /// path so no tolerance window is needed.
    #[test]
    fn cpu_attention_decoder_causal_d_matches_scalar_bit_for_bit() {
        let seq = 5usize;
        let n_head = 2usize;
        let head_dim = 4usize;
        let state = n_head * head_dim;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..seq * state)
            .map(|i| ((i as f32) * 0.017).sin())
            .collect();
        let k: Vec<f32> = (0..seq * state)
            .map(|i| ((i as f32) * 0.023).cos())
            .collect();
        let v: Vec<f32> = (0..seq * state)
            .map(|i| ((i as f32) * 0.031).sin())
            .collect();

        let mut expected = vec![0.0_f32; seq * state];
        attention_decoder_causal_scalar(&q, &k, &v, seq, n_head, head_dim, scale, &mut expected);

        let backend = CpuKernelBackend::scalar();
        let q_d = backend.upload(&q).expect("upload q");
        let k_d = backend.upload(&k).expect("upload k");
        let v_d = backend.upload(&v).expect("upload v");
        let out_d = backend.alloc(seq * state).expect("alloc out");
        backend
            .attention_decoder_causal_d(&q_d, &k_d, &v_d, seq, n_head, head_dim, scale, &out_d)
            .expect("CPU attention_decoder_causal_d must succeed");
        let got = out_d.to_host_owned().expect("readback");
        assert_eq!(
            got, expected,
            "CPU causal decoder attention must be bit-identical to scalar"
        );
    }

    /// GW.4-5B: `attention_decoder_incremental_d` on the CPU backend must
    /// match `attention_decoder_incremental_scalar` bit-for-bit.
    #[test]
    fn cpu_attention_decoder_incremental_d_matches_scalar_bit_for_bit() {
        let past_seq = 3usize;
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

        let backend = CpuKernelBackend::scalar();
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
            .expect("CPU attention_decoder_incremental_d must succeed");
        let got = out_d.to_host_owned().expect("readback");
        assert_eq!(
            got, expected,
            "CPU incremental decoder attention must be bit-identical to scalar"
        );
    }

    /// GW.4-5B: `attention_decoder_causal_d` rejects wrong buffer lengths.
    #[test]
    fn validate_attention_decoder_causal_shapes_rejects_wrong_length() {
        let backend = CpuKernelBackend::scalar();
        // seq=3, n_head=2, head_dim=2 → expected len = 12. Give len=8 buffers.
        let q = backend.upload(&[0.0_f32; 8]).unwrap();
        let k = backend.upload(&[0.0_f32; 8]).unwrap();
        let v = backend.upload(&[0.0_f32; 8]).unwrap();
        let out = backend.alloc(8).unwrap();
        let err = backend
            .attention_decoder_causal_d(&q, &k, &v, 3, 2, 2, 0.5, &out)
            .expect_err("length mismatch must be rejected");
        match err {
            OcelotlError::Kernel(KernelError { message, .. }) => {
                assert!(
                    message.contains("attention_decoder_causal_d"),
                    "expected diagnostic, got {message}"
                );
            }
            other => panic!("expected KernelError, got {other:?}"),
        }
    }

    /// GW.4-5B: `attention_decoder_incremental_d` rejects wrong past_k length.
    #[test]
    fn validate_attention_decoder_incremental_shapes_rejects_wrong_past_length() {
        let backend = CpuKernelBackend::scalar();
        let n_head = 2usize;
        let head_dim = 4usize;
        let state = n_head * head_dim;
        let past_seq = 3usize;
        let q = backend.upload(&vec![0.0_f32; state]).unwrap();
        // Wrong: past_k is too short (2*state instead of 3*state).
        let past_k = backend.upload(&vec![0.0_f32; 2 * state]).unwrap();
        let past_v = backend.upload(&vec![0.0_f32; past_seq * state]).unwrap();
        let new_k = backend.upload(&vec![0.0_f32; state]).unwrap();
        let new_v = backend.upload(&vec![0.0_f32; state]).unwrap();
        let out = backend.alloc(state).unwrap();
        let err = backend
            .attention_decoder_incremental_d(
                &q, &past_k, &past_v, &new_k, &new_v, past_seq, n_head, head_dim, 0.5, &out,
            )
            .expect_err("length mismatch must be rejected");
        match err {
            OcelotlError::Kernel(KernelError { message, .. }) => {
                assert!(
                    message.contains("attention_decoder_incremental_d"),
                    "expected diagnostic, got {message}"
                );
            }
            other => panic!("expected KernelError, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // End GW.4-5B CPU tests
    // -----------------------------------------------------------------------

    /// Parity gate proving the scalar reference matches the
    /// `attention_body_host`-style host attention on the same inputs. The
    /// math here mirrors the Whisper primitive's loop body (head-major
    /// `[seq, n_head, head_dim]`, scaled-dot → softmax → P·V); this test
    /// guards against drift between the scalar reference and the host
    /// primitive without depending on the models crate.
    #[test]
    fn attention_encoder_scalar_matches_hand_computed_softmax_chain() {
        // seq=2, n_head=1, head_dim=2: small enough to hand-verify.
        let q = vec![1.0_f32, 0.0, 0.0, 1.0];
        let k = vec![1.0_f32, 0.0, 0.0, 1.0];
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let scale = 1.0_f32 / (2.0_f32).sqrt();

        let mut out = vec![0.0_f32; 4];
        attention_encoder_scalar(&q, &k, &v, 2, 1, 2, scale, &mut out);

        // Row 0: Q = [1, 0]. Scores [1*1+0*0, 1*0+0*1]*scale = [1/√2, 0].
        // Softmax: exp(1/√2-1/√2)=1, exp(0-1/√2)=exp(-1/√2). Normalize.
        let s0 = 1.0_f32 / std::f32::consts::SQRT_2;
        let e0 = (s0 - s0).exp();
        let e1 = (0.0 - s0).exp();
        let denom = e0 + e1;
        let p0 = e0 / denom;
        let p1 = e1 / denom;
        let expected_row0 = [p0 * 1.0 + p1 * 3.0, p0 * 2.0 + p1 * 4.0];
        // Row 1: Q = [0, 1]. Scores [0*1+1*0, 0*0+1*1]*scale = [0, 1/√2].
        let e0r1 = (0.0 - s0).exp();
        let e1r1 = (s0 - s0).exp();
        let denom1 = e0r1 + e1r1;
        let p0r1 = e0r1 / denom1;
        let p1r1 = e1r1 / denom1;
        let expected_row1 = [p0r1 * 1.0 + p1r1 * 3.0, p0r1 * 2.0 + p1r1 * 4.0];

        for (got, want) in out[..2].iter().zip(expected_row0.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "row 0 mismatch: got {got}, want {want}"
            );
        }
        for (got, want) in out[2..].iter().zip(expected_row1.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "row 1 mismatch: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn validate_attention_encoder_shapes_rejects_wrong_length() {
        let backend = CpuKernelBackend::scalar();
        let q = backend.upload(&[0.0_f32; 8]).unwrap();
        let k = backend.upload(&[0.0_f32; 8]).unwrap();
        let v = backend.upload(&[0.0_f32; 8]).unwrap();
        let out = backend.alloc(8).unwrap();
        // seq=4, n_head=1, head_dim=2 → expected 8. But pass seq=4, n_head=2
        // so expected = 16.
        let err = backend
            .attention_encoder_d(&q, &k, &v, 4, 2, 2, 0.5, &out)
            .expect_err("length mismatch must be rejected");
        match err {
            OcelotlError::Kernel(KernelError { message, .. }) => {
                assert!(
                    message.contains("attention_encoder_d"),
                    "expected diagnostic, got {message}"
                );
            }
            other => panic!("expected KernelError, got {other:?}"),
        }
    }

    // GW.4-5C: decoder cross-attention scalar oracle — no causal mask,
    // q_seq decoder rows attend all kv_seq encoder positions freely.
    #[test]
    fn attention_decoder_cross_scalar_matches_attention_body_host() {
        // Q from decoder: [q_seq=3, state], K/V from encoder: [kv_seq=5, state].
        // Verifies Q rows attend all 5 encoder positions (no causal restriction).
        let q_seq = 3usize;
        let kv_seq = 5usize;
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

        let mut got = vec![0.0_f32; q_seq * state];
        attention_decoder_cross_scalar(
            &q, &k, &v, q_seq, kv_seq, n_head, head_dim, scale, &mut got,
        );

        // Reference: attention_encoder_scalar with kv_seq != q_seq is not
        // directly applicable, so compute the reference inline using the same
        // algorithm attention_body_host uses with causal=false.
        let mut expected = vec![0.0_f32; q_seq * state];
        for qi in 0..q_seq {
            for head in 0..n_head {
                let q_base = qi * state + head * head_dim;
                let mut scores = vec![0.0_f32; kv_seq];
                for (ki, score) in scores.iter_mut().enumerate() {
                    let k_base = ki * state + head * head_dim;
                    let mut acc = 0.0_f32;
                    for d in 0..head_dim {
                        acc += q[q_base + d] * k[k_base + d];
                    }
                    *score = acc * scale;
                }
                softmax(&mut scores);
                let out_base = qi * state + head * head_dim;
                for d in 0..head_dim {
                    expected[out_base + d] = 0.0;
                }
                for (ki, &p) in scores.iter().enumerate() {
                    let v_base = ki * state + head * head_dim;
                    for d in 0..head_dim {
                        expected[out_base + d] += p * v[v_base + d];
                    }
                }
            }
        }

        assert_eq!(got.len(), expected.len());
        for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "cross scalar mismatch at idx {idx}: got={g} expected={e}"
            );
        }
    }

    // GW.4-5C: `attention_decoder_cross_d` default trait impl (readback +
    // scalar) must produce the same output as the scalar oracle.
    #[test]
    fn attention_decoder_cross_d_default_matches_scalar_oracle() {
        let q_seq = 3usize;
        let kv_seq = 5usize;
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
        attention_decoder_cross_scalar(
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

        let backend = CpuKernelBackend::scalar();
        let q_d = DeviceTensor::from_host(q.clone());
        let k_d = DeviceTensor::from_host(k.clone());
        let v_d = DeviceTensor::from_host(v.clone());
        let out_d = DeviceTensor::host_zeros(q_seq * state);
        backend
            .attention_decoder_cross_d(
                &q_d, &k_d, &v_d, q_seq, kv_seq, n_head, head_dim, scale, &out_d,
            )
            .expect("attention_decoder_cross_d must succeed");
        let got = out_d.to_host_owned().expect("readback");

        assert_eq!(got.len(), expected.len());
        for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "cross_d default mismatch at idx {idx}: got={g} expected={e}"
            );
        }
    }
}
