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

pub mod rope;
pub use rope::rope_apply_inplace;

use ocelotl_core::{Device, KernelError, OcelotlError, Result, UnsupportedError};

pub mod attention;
pub mod mlp;
pub mod rmsnorm;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CpuKernelMode {
    /// Original correctness-first CPU loops. This remains the default and the
    /// parity oracle for optimized CPU, GPU, and quantized kernels.
    #[default]
    Scalar,
    /// CPU loops with cache-friendlier accumulation order for hot matrix work.
    /// This path stays safe Rust and keeps the same slice/shape contract.
    Optimized,
}

impl CpuKernelMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Optimized => "optimized",
        }
    }
}

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
    mode: CpuKernelMode,
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
        Self {
            context: KernelContext {
                device: Device::Cpu,
            },
            mode,
        }
    }

    pub fn mode(&self) -> CpuKernelMode {
        self.mode
    }

    pub fn matmul(
        &self,
        a: &[f32],
        a_shape: (usize, usize),
        b: &[f32],
        b_shape: (usize, usize),
        out: &mut [f32],
    ) -> Result<()> {
        match self.mode {
            CpuKernelMode::Scalar => matmul(a, a_shape, b, b_shape, out),
            CpuKernelMode::Optimized => matmul_optimized(a, a_shape, b, b_shape, out),
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
            CpuKernelMode::Optimized => attention::scaled_dot_product_attention_optimized(
                q,
                k,
                v,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                out,
            ),
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

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    Ok(())
}

fn matmul_optimized(
    a: &[f32],
    a_shape: (usize, usize),
    b: &[f32],
    b_shape: (usize, usize),
    out: &mut [f32],
) -> Result<()> {
    let (m, k, n) = validate_matmul(a, a_shape, b, b_shape, out)?;

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
    Ok(())
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
    Ok(())
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
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_linear_out_by_in(
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
}
