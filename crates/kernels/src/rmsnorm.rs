//! CPU reference RMSNorm.
//!
//! RMSNorm normalizes a vector by its root-mean-square magnitude and then
//! scales by a per-feature learned weight:
//!
//! ```text
//! rms     = sqrt(mean(x_i^2) + epsilon)
//! out_i   = (x_i / rms) * weight_i
//! ```
//!
//! Compared to LayerNorm there is no mean-subtraction and no learned bias —
//! that's the whole point of RMSNorm in modern decoder-only stacks (Qwen2.5,
//! LLaMA, etc.): one fewer reduction, one fewer parameter tensor.
//!
//! M3 layout & stride contract: contiguous row-major `&[f32]` only. A 2-D
//! `[rows, hidden]` input is normalized **per row** (last axis) — the
//! standard transformer convention and what the model forward path needs.
//! Strides are deferred to GPU work the same way M1.7 deferred them
//! (see crate-level docs).
//!
//! # M3.3 Phase 1 design notes
//!
//! - Out-of-place: `rmsnorm(x, rows, hidden, weight, epsilon, out)`.
//!   In-place would force the model forward path to clobber its residual
//!   stream copy; the model block can alias `out` to a scratch buffer and
//!   copy back if it wants in-place semantics.
//! - The per-row reduction reads `x` once and writes `out` once. The
//!   squared sum is accumulated in `f32`; precision-sensitive callers can
//!   lift to `f64` later behind a feature flag if parity tests demand it.
//! - Validation lives at the launch boundary, matching the M1.7 pattern.
//! - The "Done when: model forward path calls the kernel boundary" half of
//!   the M3.3 spec line defers to Phase 2 — it requires M3.1's metadata
//!   contract (in flight) and a model block (M3.5+).

use ocelotl_core::{KernelError, OcelotlError, Result};

fn kernel_err(message: impl Into<String>) -> OcelotlError {
    OcelotlError::Kernel(KernelError {
        backend: "cpu".to_string(),
        message: message.into(),
    })
}

/// Row-wise RMSNorm: `out[r, i] = (x[r, i] / sqrt(mean_i(x[r,i]^2) + eps)) * weight[i]`.
///
/// Shapes:
/// - `x`      is `rows × hidden`, total length `rows * hidden`.
/// - `weight` is `hidden`,         total length `hidden`.
/// - `out`    is `rows × hidden`, total length `rows * hidden`.
///
/// `rows` may be 1 (single-token / single-vector case). `hidden` must be
/// ≥ 1 — RMSNorm is undefined on an empty row (mean of zero values).
///
/// `epsilon` is added to the mean of squares **before** the square root for
/// numerical stability when the input row is near-zero. Typical values are
/// `1e-5` or `1e-6`; Qwen2.5 uses the value reported in its config.
///
/// # Errors
///
/// Returns `KernelError` (backend = `"cpu"`) when:
/// - `hidden` is zero,
/// - `weight.len()` does not equal `hidden`,
/// - `x.len()` is not equal to `rows * hidden`,
/// - `out.len()` does not equal `x.len()`,
/// - `epsilon` is non-finite or negative.
///
/// # Example
///
/// ```
/// use ocelotl_kernels::rmsnorm::rmsnorm;
/// let x = [1.0_f32, 2.0, 3.0];
/// let w = [1.0_f32, 1.0, 1.0];
/// let mut out = [0.0_f32; 3];
/// rmsnorm(&x, 1, 3, &w, 1e-6, &mut out).unwrap();
/// // mean(x^2) = 14/3, rms ≈ 2.1602469, out ≈ x / rms.
/// assert!((out[0] - 0.46291006).abs() < 1e-6);
/// ```
pub fn rmsnorm(
    x: &[f32],
    rows: usize,
    hidden: usize,
    weight: &[f32],
    epsilon: f32,
    out: &mut [f32],
) -> Result<()> {
    if hidden == 0 {
        return Err(kernel_err("rmsnorm hidden dimension must be non-zero"));
    }
    if weight.len() != hidden {
        return Err(kernel_err(format!(
            "rmsnorm weight length {} does not match hidden {hidden}",
            weight.len()
        )));
    }
    if x.len() != rows * hidden {
        return Err(kernel_err(format!(
            "rmsnorm x slice length {} does not match shape {rows}x{hidden}",
            x.len()
        )));
    }
    if out.len() != x.len() {
        return Err(kernel_err(format!(
            "rmsnorm out slice length {} does not match x length {}",
            out.len(),
            x.len()
        )));
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(kernel_err(format!(
            "rmsnorm epsilon must be finite and non-negative, got {epsilon}"
        )));
    }

    let hidden_f = hidden as f32;
    for r in 0..rows {
        let row_start = r * hidden;
        let row = &x[row_start..row_start + hidden];

        // Accumulate sum of squares for this row.
        let mut sum_sq = 0.0_f32;
        for &v in row.iter() {
            sum_sq += v * v;
        }
        let mean_sq = sum_sq / hidden_f;
        let rms = (mean_sq + epsilon).sqrt();
        let inv_rms = 1.0_f32 / rms;

        let out_row = &mut out[row_start..row_start + hidden];
        for i in 0..hidden {
            out_row[i] = row[i] * inv_rms * weight[i];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-checked baseline (M3.3 first failing test).
    ///
    /// Input row: `[1.0, 2.0, 3.0]`, weight `[1.0, 1.0, 1.0]`, epsilon `1e-6`.
    ///   sum_sq  = 1 + 4 + 9 = 14
    ///   mean_sq = 14 / 3   = 4.6666666...
    ///   rms     = sqrt(4.6666666... + 1e-6) ≈ 2.16024702...
    ///   inv_rms ≈ 0.46291006...
    ///   out[0]  = 1 * inv_rms ≈ 0.46291006
    ///   out[1]  = 2 * inv_rms ≈ 0.92582012
    ///   out[2]  = 3 * inv_rms ≈ 1.38873018
    #[test]
    fn rmsnorm_single_row_unit_weight_matches_hand_computation() {
        let x = [1.0_f32, 2.0, 3.0];
        let w = [1.0_f32, 1.0, 1.0];
        let mut out = [0.0_f32; 3];

        rmsnorm(&x, 1, 3, &w, 1e-6, &mut out).expect("well-formed rmsnorm must succeed");

        let expected = [0.46291006_f32, 0.92582012, 1.38873018];
        for (got, want) in out.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "rmsnorm mismatch: got {got}, want {want}"
            );
        }
    }
}
