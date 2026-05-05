//! Rotary Position Embedding (RoPE) — CPU reference.
//!
//! Half-rotation convention (HF / Llama / Qwen2.5): pair element `i` with
//! element `i + head_dim/2`, rotate by angle `pos * inv_freq[i]` where
//! `inv_freq[i] = theta^(-2i / head_dim)` for `i in 0..head_dim/2`.
//!
//! Phase-1 scope (M3.4): kernel function only. RoPE configuration is taken
//! as parameters; metadata-driven configuration and invalid-head-dim
//! rejection at the model boundary defer to Phase 2 (per
//! `docs/tasks/m3-single-model-forward.md` M3.4 "Done when" line + the
//! M3 Phase 1 scope split recorded in the assignments board).

use ocelotl_core::{KernelError, OcelotlError, Result};

fn kernel_err(message: impl Into<String>) -> OcelotlError {
    OcelotlError::Kernel(KernelError {
        backend: "cpu".to_string(),
        message: message.into(),
    })
}

/// Apply rotary position embeddings in place.
///
/// The slice `x` is treated as a contiguous row of `num_heads` head vectors,
/// each of length `head_dim`. The same rotation (defined by `position` and
/// `theta`) is applied independently to every head.
///
/// Half-rotation convention: for each head, element `i` (in `0..head_dim/2`)
/// is paired with element `i + head_dim/2`. Let
/// `inv_freq_i = theta.powf(-2.0 * (i as f32) / (head_dim as f32))`,
/// `angle = (position as f32) * inv_freq_i`, `c = angle.cos()`,
/// `s = angle.sin()`. Then:
///
/// ```text
/// out[i]            = x[i]            * c - x[i + head_dim/2] * s
/// out[i + head_dim/2] = x[i]          * s + x[i + head_dim/2] * c
/// ```
///
/// At `position = 0` this is the identity (`cos(0) = 1`, `sin(0) = 0`).
///
/// # Errors
///
/// Returns `KernelError` (backend = `"cpu"`) when:
/// - `head_dim` is zero,
/// - `head_dim` is odd (RoPE pairs elements; an odd dim has no valid pairing),
/// - `x.len()` is not a positive multiple of `head_dim`.
///
/// `position = 0` and any non-negative position are valid; very large
/// positions may lose precision because `f32::cos`/`f32::sin` argument
/// reduction degrades for large arguments — that is upstream's problem
/// (the model owns the max-position contract), not the kernel's.
///
/// # Example
///
/// ```
/// use ocelotl_kernels::rope_apply_inplace;
/// // head_dim=4, two heads laid out contiguously: [h0_0, h0_1, h0_2, h0_3,
/// //                                              h1_0, h1_1, h1_2, h1_3]
/// let mut x = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// // position 0 is the identity.
/// rope_apply_inplace(&mut x, 4, 0, 10_000.0).unwrap();
/// assert_eq!(x, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
/// ```
pub fn rope_apply_inplace(
    x: &mut [f32],
    head_dim: usize,
    position: usize,
    theta: f32,
) -> Result<()> {
    if head_dim == 0 {
        return Err(kernel_err(
            "rope_apply_inplace head_dim must be non-zero".to_string(),
        ));
    }
    if head_dim % 2 != 0 {
        return Err(kernel_err(format!(
            "rope_apply_inplace head_dim must be even, got {head_dim}"
        )));
    }
    if x.is_empty() || x.len() % head_dim != 0 {
        return Err(kernel_err(format!(
            "rope_apply_inplace x.len()={} must be a positive multiple of head_dim={}",
            x.len(),
            head_dim
        )));
    }

    let half = head_dim / 2;
    let pos_f = position as f32;
    let head_dim_f = head_dim as f32;

    // Walk each head row independently.
    for head_offset in (0..x.len()).step_by(head_dim) {
        for i in 0..half {
            // inv_freq[i] = theta^(-2i / head_dim)
            let exponent = -2.0_f32 * (i as f32) / head_dim_f;
            let inv_freq = theta.powf(exponent);
            let angle = pos_f * inv_freq;
            let c = angle.cos();
            let s = angle.sin();

            let lo = head_offset + i;
            let hi = head_offset + i + half;
            let x_lo = x[lo];
            let x_hi = x[hi];
            x[lo] = x_lo * c - x_hi * s;
            x[hi] = x_lo * s + x_hi * c;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Test 1: position 0 identity ---
    //
    // At pos=0, angle=0 for every i, so cos=1 and sin=0. The formula
    // reduces to x[i] = x[i]*1 - x[i+half]*0 = x[i] (and similarly for
    // the upper half). This is the simplest sanity check for the kernel
    // wiring and exercises the same arithmetic path as a non-zero
    // position; if this fails, the implementation is fundamentally
    // broken before any trig is involved.

    #[test]
    fn rope_at_position_zero_is_identity_for_single_head() {
        // Single head, head_dim=4.
        let mut x = [0.5_f32, -1.25, 3.0, 7.5];
        let original = x;

        rope_apply_inplace(&mut x, 4, 0, 10_000.0)
            .expect("position 0 with valid shape must succeed");

        assert_eq!(
            x, original,
            "RoPE at position 0 must be the identity transform"
        );
    }

    #[test]
    fn rope_at_position_zero_is_identity_for_multiple_heads() {
        // Two heads, head_dim=4. Same rotation applied per head; at pos=0
        // every head is identity.
        let mut x = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = x;

        rope_apply_inplace(&mut x, 4, 0, 10_000.0)
            .expect("position 0 with valid shape must succeed");

        assert_eq!(x, original, "position 0 must be identity across every head");
    }

    // --- Test 2: hand-computed non-zero position ---
    //
    // Setup: head_dim=4, theta=10000, position=1, single head with input
    // x = [1.0, 0.0, 0.0, 1.0]. half = 2.
    //
    // Pair indices: (0, 2) and (1, 3).
    //
    // Frequencies:
    //   i=0: inv_freq_0 = 10000^(-0/4) = 10000^0 = 1.0
    //        angle      = 1 * 1.0      = 1.0 rad
    //        cos(1.0)   ≈ 0.5403023058681398
    //        sin(1.0)   ≈ 0.8414709848078965
    //   i=1: inv_freq_1 = 10000^(-2/4) = 10000^(-0.5) = 1/100 = 0.01
    //        angle      = 1 * 0.01     = 0.01 rad
    //        cos(0.01)  ≈ 0.9999500004166653
    //        sin(0.01)  ≈ 0.009999833334166664
    //
    // Apply to x = [1.0, 0.0, 0.0, 1.0]:
    //   pair (0, 2): x[0]=1, x[2]=0
    //     new x[0] = 1*c0 - 0*s0 = c0 ≈ 0.5403023
    //     new x[2] = 1*s0 + 0*c0 = s0 ≈ 0.8414710
    //   pair (1, 3): x[1]=0, x[3]=1
    //     new x[1] = 0*c1 - 1*s1 = -s1 ≈ -0.0099998
    //     new x[3] = 0*s1 + 1*c1 = c1  ≈ 0.9999500
    //
    // Expected: [c0, -s1, s0, c1]
    //         ≈ [0.5403023, -0.009999833, 0.8414710, 0.9999500]
    //
    // Tolerance: 1e-6. All four expected values come from a 64-bit math
    // computation but the kernel runs in f32; 4*f32::EPSILON (~5e-7) is
    // tight enough to catch logic bugs (wrong pair, swapped sign,
    // missed inv_freq) but loose enough to absorb the 32-vs-64-bit
    // rounding gap on the trig calls.

    #[test]
    fn rope_at_position_one_with_unit_input_matches_hand_computation() {
        let head_dim = 4_usize;
        let theta = 10_000.0_f32;
        let position = 1_usize;

        let mut x = [1.0_f32, 0.0, 0.0, 1.0];

        rope_apply_inplace(&mut x, head_dim, position, theta)
            .expect("well-formed RoPE call must succeed");

        // Hand-computed values (see test comment for derivation).
        let c0 = 0.540_302_3_f32; // cos(1.0)
        let s0 = 0.841_471_0_f32; // sin(1.0)
        let c1 = 0.999_950_0_f32; // cos(0.01)
        let s1 = 0.009_999_833_5_f32; // sin(0.01)
        let expected = [c0, -s1, s0, c1];

        let tol = 1.0e-6_f32;
        for (idx, (got, want)) in x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < tol,
                "RoPE position-1 mismatch at index {idx}: got {got}, want {want}"
            );
        }
    }

    // --- Validation tests ---

    #[test]
    fn rope_rejects_zero_head_dim() {
        let mut x = [1.0_f32, 2.0, 3.0, 4.0];
        let err =
            rope_apply_inplace(&mut x, 0, 0, 10_000.0).expect_err("zero head_dim must be rejected");

        match err {
            OcelotlError::Kernel(KernelError { backend, message }) => {
                assert_eq!(backend, "cpu");
                assert!(
                    message.contains("head_dim"),
                    "expected message to mention head_dim, got {message:?}"
                );
            }
            other => panic!("expected KernelError, got {other:?}"),
        }
    }

    #[test]
    fn rope_rejects_odd_head_dim() {
        // head_dim=3 has no valid pairing under the half-rotation
        // convention. RoPE on an odd dim is not defined here; surface
        // the precondition explicitly.
        let mut x = [1.0_f32, 2.0, 3.0];
        let err =
            rope_apply_inplace(&mut x, 3, 0, 10_000.0).expect_err("odd head_dim must be rejected");

        match err {
            OcelotlError::Kernel(KernelError { backend, message }) => {
                assert_eq!(backend, "cpu");
                assert!(
                    message.contains("even"),
                    "expected message to mention even, got {message:?}"
                );
            }
            other => panic!("expected KernelError, got {other:?}"),
        }
    }

    #[test]
    fn rope_rejects_length_not_multiple_of_head_dim() {
        // 5 elements, head_dim=4 → not a multiple.
        let mut x = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let err = rope_apply_inplace(&mut x, 4, 0, 10_000.0)
            .expect_err("non-multiple length must be rejected");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }

    #[test]
    fn rope_rejects_empty_slice() {
        // An empty slice has no head rows to rotate. Reject explicitly so
        // the caller sees a typed error rather than a silent no-op.
        let mut x: [f32; 0] = [];
        let err =
            rope_apply_inplace(&mut x, 4, 0, 10_000.0).expect_err("empty slice must be rejected");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }
}
