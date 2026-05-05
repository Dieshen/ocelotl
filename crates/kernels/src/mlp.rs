//! CPU reference gated MLP (Qwen2.5 / LLaMA family).
//!
//! The block computes:
//!
//! ```text
//! out = down_proj(silu(gate_proj(x)) * up_proj(x))
//! ```
//!
//! where `silu(z) = z * sigmoid(z) = z / (1 + exp(-z))` and `*` is element-wise.
//!
//! M3 layout & stride contract: contiguous row-major `&[f32]` only, same as
//! the rest of the kernels (see crate-level docs). Out-of-place: the caller
//! provides scratch buffers for the two `[seq_len, intermediate]` projections
//! and the `[seq_len, hidden]` output. This avoids hidden allocations inside
//! the kernel — the model block decides the buffer story.
//!
//! # Weight layout
//!
//! HuggingFace stores the projection tensors as `[out_features, in_features]`
//! (row-major). Our `matmul` takes contiguous row-major `[m, k] @ [k, n]`
//! with no transpose, so this kernel takes the **already-transposed** view of
//! each weight, i.e.:
//!
//! - `gate_w` and `up_w` are passed as `[hidden, intermediate]`,
//! - `down_w` is passed as `[intermediate, hidden]`.
//!
//! The model layer owns the transpose (it materializes the slice once at
//! load time). Doing it here would force the kernel to allocate, which the
//! M1.7 / M3.3 / M3.4 boundary explicitly avoids.
//!
//! # M3.6 Phase 2 design notes
//!
//! - `silu` is its own free function so it can be unit-tested independently
//!   and reused if a different gating activation (GeGLU, etc.) shows up
//!   later. It runs in `f32` end-to-end; precision-sensitive callers can
//!   lift to `f64` later behind a feature flag if parity tests demand it,
//!   matching the M3.3 RMSNorm precision story.
//! - The three projections compose `crate::matmul` rather than re-rolling
//!   the triple loop. That keeps the parity-oracle behavior of `matmul` as
//!   the single source of truth and makes future GPU swaps mechanical
//!   (replace `matmul`, `mlp_gated_silu` is unchanged).
//! - Validation lives at the launch boundary, matching M1.7. Because we
//!   re-enter `matmul` three times we get its validation for free; we add
//!   only the cross-projection invariants (intermediate dimension agreement
//!   between gate/up, scratch buffer sizing, hidden agreement on output).
//! - The "Done when: activation, gate, up, and down projections are wired
//!   in the target model block" half of the spec defers to Phase 3 stitch.

use ocelotl_core::{KernelError, OcelotlError, Result};

use crate::{kernel_err, matmul};

/// Sigmoid Linear Unit: `silu(z) = z * sigmoid(z) = z / (1 + exp(-z))`.
///
/// Element-wise, in place. Empty input is a no-op.
///
/// `f32::exp` is finite for arguments down to roughly `-87.3` (below that
/// it underflows to `0.0`, which gives `silu(z) = z * 1.0 = z` — wrong by a
/// factor we never recover, but the "wrongness" is bounded by the input
/// magnitude that produced it). For the inverse extreme, `exp(-z)` for
/// very negative `z` overflows to `+inf`, producing `silu(z) -> 0.0 * z`
/// which is `0.0` — that matches the asymptote. Both cases are upstream's
/// problem only insofar as upstream feeds us pathological logits; for the
/// model forward path the magnitudes stay well within `f32` range.
///
/// # Example
///
/// ```
/// use ocelotl_kernels::mlp::silu_inplace;
/// let mut x = [0.0_f32, 1.0, -1.0];
/// silu_inplace(&mut x);
/// // silu(0) = 0, silu(1) ≈ 0.7310586, silu(-1) ≈ -0.26894143
/// assert!((x[0] - 0.0).abs() < 1e-6);
/// assert!((x[1] - 0.7310586).abs() < 1e-6);
/// assert!((x[2] - (-0.26894143)).abs() < 1e-6);
/// ```
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        // 1.0 / (1.0 + exp(-z)) is the standard stable form for sigmoid in
        // the typical range; for very negative z, exp(-z) overflows and the
        // sigmoid underflows to 0, giving silu = z * 0 = 0 — the correct
        // asymptote. No branchy fixup needed at f32 magnitudes the model
        // forward path produces.
        let sigmoid = 1.0_f32 / (1.0_f32 + (-*v).exp());
        *v *= sigmoid;
    }
}

/// Gated SiLU MLP: `out = down(silu(gate(x)) * up(x))`.
///
/// Shapes:
/// - `x`        is `seq_len × hidden`,        total length `seq_len * hidden`.
/// - `gate_w`   is `hidden  × intermediate`,  total length `hidden * intermediate`.
/// - `up_w`     is `hidden  × intermediate`,  total length `hidden * intermediate`.
/// - `down_w`   is `intermediate × hidden`,   total length `intermediate * hidden`.
/// - `gate_buf` is `seq_len × intermediate`,  scratch for `silu(gate(x))`.
/// - `up_buf`   is `seq_len × intermediate`,  scratch for `up(x)`.
/// - `out`      is `seq_len × hidden`,        total length `seq_len * hidden`.
///
/// All slices are contiguous row-major. The two scratch buffers (`gate_buf`
/// and `up_buf`) are owned by the caller so the model layer can pool /
/// reuse them across blocks.
///
/// `seq_len` may be 1 (decode case); `hidden` and `intermediate` must each
/// be ≥ 1 (a zero-width projection is undefined and almost certainly a
/// metadata bug).
///
/// # Errors
///
/// Returns `KernelError` (backend = `"cpu"`) when:
/// - any of `seq_len`, `hidden`, or `intermediate` is zero,
/// - any input/scratch/output slice length disagrees with its declared shape.
///
/// Inner-dimension and per-projection validation come from the underlying
/// `matmul` calls; they surface as `KernelError` with `matmul`'s own message.
///
/// # Example
///
/// ```
/// use ocelotl_kernels::mlp::mlp_gated_silu;
/// // Identity-ish toy: hidden=1, intermediate=1, seq_len=1, all weights = 1.
/// // gate(x) = up(x) = x; silu(x)*x = x*sigmoid(x)*x; down = same x.
/// let x = [2.0_f32];
/// let gate_w = [1.0_f32];
/// let up_w = [1.0_f32];
/// let down_w = [1.0_f32];
/// let mut gate_buf = [0.0_f32; 1];
/// let mut up_buf = [0.0_f32; 1];
/// let mut out = [0.0_f32; 1];
/// mlp_gated_silu(
///     &x, 1, 1, 1,
///     &gate_w, &up_w, &down_w,
///     &mut gate_buf, &mut up_buf, &mut out,
/// ).unwrap();
/// // silu(2) = 2 * sigmoid(2) ≈ 1.7615942; * up(2) = 2 -> ≈ 3.5231884; * 1 -> ≈ 3.5231884
/// assert!((out[0] - 3.5231884).abs() < 1e-5);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn mlp_gated_silu(
    x: &[f32],
    seq_len: usize,
    hidden: usize,
    intermediate: usize,
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    out: &mut [f32],
) -> Result<()> {
    if seq_len == 0 {
        return Err(kernel_err(
            "mlp_gated_silu seq_len must be non-zero".to_string(),
        ));
    }
    if hidden == 0 {
        return Err(kernel_err(
            "mlp_gated_silu hidden must be non-zero".to_string(),
        ));
    }
    if intermediate == 0 {
        return Err(kernel_err(
            "mlp_gated_silu intermediate must be non-zero".to_string(),
        ));
    }
    if x.len() != seq_len * hidden {
        return Err(kernel_err(format!(
            "mlp_gated_silu x slice length {} does not match shape {seq_len}x{hidden}",
            x.len()
        )));
    }
    if gate_w.len() != hidden * intermediate {
        return Err(kernel_err(format!(
            "mlp_gated_silu gate_w length {} does not match shape {hidden}x{intermediate}",
            gate_w.len()
        )));
    }
    if up_w.len() != hidden * intermediate {
        return Err(kernel_err(format!(
            "mlp_gated_silu up_w length {} does not match shape {hidden}x{intermediate}",
            up_w.len()
        )));
    }
    if down_w.len() != intermediate * hidden {
        return Err(kernel_err(format!(
            "mlp_gated_silu down_w length {} does not match shape {intermediate}x{hidden}",
            down_w.len()
        )));
    }
    if gate_buf.len() != seq_len * intermediate {
        return Err(kernel_err(format!(
            "mlp_gated_silu gate_buf length {} does not match shape {seq_len}x{intermediate}",
            gate_buf.len()
        )));
    }
    if up_buf.len() != seq_len * intermediate {
        return Err(kernel_err(format!(
            "mlp_gated_silu up_buf length {} does not match shape {seq_len}x{intermediate}",
            up_buf.len()
        )));
    }
    if out.len() != seq_len * hidden {
        return Err(kernel_err(format!(
            "mlp_gated_silu out length {} does not match shape {seq_len}x{hidden}",
            out.len()
        )));
    }

    // gate = x @ gate_w   ([seq, hidden] @ [hidden, intermediate])
    matmul(
        x,
        (seq_len, hidden),
        gate_w,
        (hidden, intermediate),
        gate_buf,
    )?;
    // up = x @ up_w
    matmul(x, (seq_len, hidden), up_w, (hidden, intermediate), up_buf)?;

    // silu in place on gate, then multiply by up element-wise (still in gate_buf).
    silu_inplace(gate_buf);
    for i in 0..gate_buf.len() {
        gate_buf[i] *= up_buf[i];
    }

    // out = gated @ down_w   ([seq, intermediate] @ [intermediate, hidden])
    matmul(
        gate_buf,
        (seq_len, intermediate),
        down_w,
        (intermediate, hidden),
        out,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- silu_inplace ---

    /// Hand-checked silu values at a few canonical inputs.
    ///   silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    ///   silu(1) = 1 * sigmoid(1) = 1 * 1/(1 + e^-1) ≈ 0.73105858
    ///   silu(-1) = -1 * sigmoid(-1) = -1 * 1/(1 + e^1) ≈ -0.26894143
    ///   silu(2) = 2 * sigmoid(2) = 2 * 1/(1 + e^-2) ≈ 1.76159416
    ///   silu(3) = 3 * sigmoid(3) = 3 * 1/(1 + e^-3) ≈ 2.85772238
    #[test]
    fn silu_inplace_matches_hand_checked_values() {
        let mut x = [0.0_f32, 1.0, -1.0, 2.0, 3.0];
        silu_inplace(&mut x);

        let expected = [0.0_f32, 0.7310586, -0.26894143, 1.7615942, 2.8577223];
        for (got, want) in x.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "silu mismatch: got {got}, want {want}"
            );
        }
    }

    /// silu(z) -> 0 as z -> -inf (sigmoid -> 0); silu(z) -> z as z -> +inf
    /// (sigmoid -> 1). At f32 magnitudes the forward path produces, both
    /// asymptotes are reached well before any overflow concern. This pins
    /// the `(-z).exp()` branch — using `z.exp()` directly would have
    /// overflowed for large positive `z` and produced NaN.
    #[test]
    fn silu_inplace_asymptotes_are_finite() {
        let mut x = [-30.0_f32, 30.0];
        silu_inplace(&mut x);

        // silu(-30) ≈ -30 * sigmoid(-30) ≈ -30 * 9.36e-14 ≈ ~-2.8e-12.
        assert!(x[0].is_finite(), "silu(-30) must be finite, got {}", x[0]);
        assert!(x[0].abs() < 1e-6, "silu(-30) must underflow toward 0");
        // silu(30) ≈ 30 * 1.0 ≈ 30.
        assert!(x[1].is_finite(), "silu(30) must be finite, got {}", x[1]);
        assert!((x[1] - 30.0).abs() < 1e-3, "silu(30) ≈ 30, got {}", x[1]);
    }

    #[test]
    fn silu_inplace_of_empty_slice_is_a_noop() {
        let mut x: [f32; 0] = [];
        silu_inplace(&mut x);
        // No assertion — must not panic.
    }

    // --- mlp_gated_silu ---

    /// Hand-checked gated MLP fixture (M3.6 Phase 2 first failing test).
    ///
    /// Shapes: `seq_len=1, hidden=2, intermediate=4`.
    ///
    /// Input row: `x = [1, 2]`.
    ///
    /// Weights (passed in `[in, out]` layout — already transposed from the
    /// HuggingFace `[out, in]` convention; the model layer owns that
    /// transpose):
    ///
    ///   gate_w (hidden=2 × intermediate=4) =
    ///     [[ 1,  0,  1, -1],
    ///      [ 0,  1,  1,  0]]    flat: [1, 0, 1, -1,  0, 1, 1, 0]
    ///
    ///   up_w (hidden=2 × intermediate=4) =
    ///     [[ 1,  1,  0,  2],
    ///      [ 1, -1,  1,  0]]    flat: [1, 1, 0, 2,  1, -1, 1, 0]
    ///
    ///   down_w (intermediate=4 × hidden=2) =
    ///     [[1, 0],
    ///      [0, 1],
    ///      [1, 0],
    ///      [0, 1]]               flat: [1, 0,  0, 1,  1, 0,  0, 1]
    ///
    /// Step 1 — gate(x) = x @ gate_w (1×2 @ 2×4):
    ///   col 0: 1*1 + 2*0  =  1
    ///   col 1: 1*0 + 2*1  =  2
    ///   col 2: 1*1 + 2*1  =  3
    ///   col 3: 1*-1 + 2*0 = -1
    ///   gate = [ 1, 2, 3, -1]
    ///
    /// Step 2 — up(x) = x @ up_w (1×2 @ 2×4):
    ///   col 0: 1*1 + 2*1  =  3
    ///   col 1: 1*1 + 2*-1 = -1
    ///   col 2: 1*0 + 2*1  =  2
    ///   col 3: 1*2 + 2*0  =  2
    ///   up = [ 3, -1, 2, 2]
    ///
    /// Step 3 — silu(gate) (per the silu_inplace test above):
    ///   silu( 1) ≈  0.7310586
    ///   silu( 2) ≈  1.7615942
    ///   silu( 3) ≈  2.8577223
    ///   silu(-1) ≈ -0.26894143
    ///
    /// Step 4 — gated = silu(gate) * up (element-wise):
    ///   [ 0.7310586 *  3,
    ///     1.7615942 * -1,
    ///     2.8577223 *  2,
    ///    -0.26894143 *  2 ]
    ///   ≈ [ 2.1931758, -1.7615942, 5.7154446, -0.53788286 ]
    ///
    /// Step 5 — out = gated @ down_w (1×4 @ 4×2):
    ///   col 0:  2.1931758*1 + (-1.7615942)*0 + 5.7154446*1 + (-0.53788286)*0
    ///         =  2.1931758 + 5.7154446
    ///         ≈  7.9086204
    ///   col 1:  2.1931758*0 + (-1.7615942)*1 + 5.7154446*0 + (-0.53788286)*1
    ///         = -1.7615942 + (-0.53788286)
    ///         ≈ -2.299477
    ///
    /// Expected: out ≈ [ 7.9086204, -2.299477 ].
    #[test]
    fn mlp_gated_silu_tiny_fixture_matches_hand_computation() {
        let x = [1.0_f32, 2.0];
        let gate_w = [1.0_f32, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0];
        let up_w = [1.0_f32, 1.0, 0.0, 2.0, 1.0, -1.0, 1.0, 0.0];
        let down_w = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];

        let mut gate_buf = [0.0_f32; 4];
        let mut up_buf = [0.0_f32; 4];
        let mut out = [0.0_f32; 2];

        mlp_gated_silu(
            &x,
            1, // seq_len
            2, // hidden
            4, // intermediate
            &gate_w,
            &up_w,
            &down_w,
            &mut gate_buf,
            &mut up_buf,
            &mut out,
        )
        .expect("well-formed mlp_gated_silu must succeed");

        let expected = [7.9086204_f32, -2.299477];
        for (got, want) in out.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-5,
                "mlp_gated_silu mismatch: got {got}, want {want}"
            );
        }
    }

    /// Multi-row inputs must be processed independently per row. The matmul
    /// already does this correctly, but we pin the contract here so a future
    /// "optimize the seq_len=1 path" change can't quietly break prefill.
    /// Row 0 reuses the hand-checked baseline; row 1 is `[0, 0]` and must
    /// produce zero output (silu(0) = 0, anything * 0 = 0).
    #[test]
    fn mlp_gated_silu_processes_rows_independently() {
        let x = [1.0_f32, 2.0, 0.0, 0.0];
        let gate_w = [1.0_f32, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0];
        let up_w = [1.0_f32, 1.0, 0.0, 2.0, 1.0, -1.0, 1.0, 0.0];
        let down_w = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];

        let mut gate_buf = [0.0_f32; 8];
        let mut up_buf = [0.0_f32; 8];
        let mut out = [0.0_f32; 4];

        mlp_gated_silu(
            &x,
            2, // seq_len
            2, // hidden
            4, // intermediate
            &gate_w,
            &up_w,
            &down_w,
            &mut gate_buf,
            &mut up_buf,
            &mut out,
        )
        .expect("well-formed mlp_gated_silu must succeed");

        // Row 0: same as the tiny-fixture test.
        assert!(
            (out[0] - 7.9086204).abs() < 1e-5,
            "row 0 col 0: got {}",
            out[0]
        );
        assert!(
            (out[1] - (-2.299477)).abs() < 1e-5,
            "row 0 col 1: got {}",
            out[1]
        );

        // Row 1: zero input -> zero gate, zero up, zero everything.
        assert_eq!(out[2], 0.0, "row 1 col 0 must be zero, got {}", out[2]);
        assert_eq!(out[3], 0.0, "row 1 col 1 must be zero, got {}", out[3]);
    }

    // --- validation errors ---

    #[test]
    fn mlp_gated_silu_rejects_zero_seq_len() {
        let err = mlp_gated_silu(
            &[],
            0,
            2,
            4,
            &[0.0; 8],
            &[0.0; 8],
            &[0.0; 8],
            &mut [0.0; 0],
            &mut [0.0; 0],
            &mut [0.0; 0],
        )
        .expect_err("must reject zero seq_len");
        match err {
            OcelotlError::Kernel(KernelError { backend, message }) => {
                assert_eq!(backend, "cpu");
                assert!(
                    message.contains("seq_len"),
                    "expected seq_len message, got {message:?}"
                );
            }
            other => panic!("expected KernelError, got {other:?}"),
        }
    }

    #[test]
    fn mlp_gated_silu_rejects_zero_hidden() {
        let err = mlp_gated_silu(
            &[],
            1,
            0,
            4,
            &[],
            &[],
            &[],
            &mut [0.0; 4],
            &mut [0.0; 4],
            &mut [],
        )
        .expect_err("must reject zero hidden");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }

    #[test]
    fn mlp_gated_silu_rejects_zero_intermediate() {
        let err = mlp_gated_silu(
            &[1.0, 2.0],
            1,
            2,
            0,
            &[],
            &[],
            &[],
            &mut [],
            &mut [],
            &mut [0.0; 2],
        )
        .expect_err("must reject zero intermediate");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }

    #[test]
    fn mlp_gated_silu_rejects_x_length_mismatch() {
        // claims 1x2 = 2 elements, has 3
        let err = mlp_gated_silu(
            &[1.0, 2.0, 3.0],
            1,
            2,
            4,
            &[0.0; 8],
            &[0.0; 8],
            &[0.0; 8],
            &mut [0.0; 4],
            &mut [0.0; 4],
            &mut [0.0; 2],
        )
        .expect_err("must reject x length mismatch");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }

    #[test]
    fn mlp_gated_silu_rejects_gate_w_length_mismatch() {
        // hidden*intermediate = 2*4 = 8, gate_w has 7
        let err = mlp_gated_silu(
            &[1.0, 2.0],
            1,
            2,
            4,
            &[0.0; 7],
            &[0.0; 8],
            &[0.0; 8],
            &mut [0.0; 4],
            &mut [0.0; 4],
            &mut [0.0; 2],
        )
        .expect_err("must reject gate_w length mismatch");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }

    #[test]
    fn mlp_gated_silu_rejects_down_w_length_mismatch() {
        // intermediate*hidden = 4*2 = 8, down_w has 9
        let err = mlp_gated_silu(
            &[1.0, 2.0],
            1,
            2,
            4,
            &[0.0; 8],
            &[0.0; 8],
            &[0.0; 9],
            &mut [0.0; 4],
            &mut [0.0; 4],
            &mut [0.0; 2],
        )
        .expect_err("must reject down_w length mismatch");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }

    #[test]
    fn mlp_gated_silu_rejects_scratch_buffer_length_mismatch() {
        // gate_buf must be seq_len*intermediate = 4; pass 3
        let err = mlp_gated_silu(
            &[1.0, 2.0],
            1,
            2,
            4,
            &[0.0; 8],
            &[0.0; 8],
            &[0.0; 8],
            &mut [0.0; 3],
            &mut [0.0; 4],
            &mut [0.0; 2],
        )
        .expect_err("must reject gate_buf length mismatch");
        assert!(matches!(err, OcelotlError::Kernel(_)));

        // up_buf must be seq_len*intermediate = 4; pass 5
        let err = mlp_gated_silu(
            &[1.0, 2.0],
            1,
            2,
            4,
            &[0.0; 8],
            &[0.0; 8],
            &[0.0; 8],
            &mut [0.0; 4],
            &mut [0.0; 5],
            &mut [0.0; 2],
        )
        .expect_err("must reject up_buf length mismatch");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }

    #[test]
    fn mlp_gated_silu_rejects_out_length_mismatch() {
        // out must be seq_len*hidden = 2; pass 3
        let err = mlp_gated_silu(
            &[1.0, 2.0],
            1,
            2,
            4,
            &[0.0; 8],
            &[0.0; 8],
            &[0.0; 8],
            &mut [0.0; 4],
            &mut [0.0; 4],
            &mut [0.0; 3],
        )
        .expect_err("must reject out length mismatch");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }
}
