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
}
