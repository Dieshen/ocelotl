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

use ocelotl_core::{Device, KernelError, OcelotlError, Result, UnsupportedError};

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

// ---------------------------------------------------------------------------
// CPU reference primitives (M1.7)
//
// Reference-only. Not vectorized. Used to make the rest of the inference path
// testable end-to-end on a laptop with no GPU, and as the parity oracle for
// future GPU kernels.
// ---------------------------------------------------------------------------

fn kernel_err(message: impl Into<String>) -> OcelotlError {
    OcelotlError::Kernel(KernelError {
        backend: "cpu".to_string(),
        message: message.into(),
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
    let (m, k_a) = a_shape;
    let (k_b, n) = b_shape;

    if k_a != k_b {
        return Err(kernel_err(format!(
            "matmul inner-dimension mismatch: a is {m}x{k_a}, b is {k_b}x{n}"
        )));
    }
    if a.len() != m * k_a {
        return Err(kernel_err(format!(
            "matmul a slice length {} does not match shape {m}x{k_a}",
            a.len()
        )));
    }
    if b.len() != k_b * n {
        return Err(kernel_err(format!(
            "matmul b slice length {} does not match shape {k_b}x{n}",
            b.len()
        )));
    }
    if out.len() != m * n {
        return Err(kernel_err(format!(
            "matmul out slice length {} does not match shape {m}x{n}",
            out.len()
        )));
    }

    let k = k_a;
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
}
