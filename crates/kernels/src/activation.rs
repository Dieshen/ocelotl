//! Elementwise activations and small fused elementwise ops.
//!
//! Added for the Conformer/RNN-T family (Parakeet): the tree previously had only
//! `silu_inplace` and `gelu_tanh_inplace`, both fused inside the gated MLP, with
//! no standalone `relu`/`sigmoid`/`tanh` and no half-split GLU.

use crate::{Result, kernel_err};

/// `x = max(x, 0)`. Used by the `dw_striding` subsampler and the transducer joint.
pub fn relu_inplace(x: &mut [f32]) {
    for value in x.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
}

/// `x = 1 / (1 + exp(-x))`.
///
/// Guards the same overflow `silu_inplace` documents: for very negative `x`,
/// `exp(-x)` overflows to `+inf` and the quotient underflows to `0.0`, which is
/// the correct limit — but computing it via `exp(x)/(1+exp(x))` instead would
/// yield `inf/inf = NaN`. The branch below keeps both tails finite.
pub fn sigmoid_inplace(x: &mut [f32]) {
    for value in x.iter_mut() {
        *value = sigmoid(*value);
    }
}

#[inline]
pub fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// `x = tanh(x)`.
pub fn tanh_inplace(x: &mut [f32]) {
    for value in x.iter_mut() {
        *value = value.tanh();
    }
}

/// Gated Linear Unit over a half-split: `out[r, c] = a[r, c] * sigmoid(b[r, c])`
/// where each input row is `[a | b]` of width `2 * half`.
///
/// This is *not* expressible with the existing `mul_inplace` or the gated MLP
/// helpers: those take two separate tensors, while a Conformer convolution
/// module produces one tensor whose channel axis is split in half.
pub fn glu_halves(x: &[f32], rows: usize, half: usize, out: &mut [f32]) -> Result<()> {
    let width = half
        .checked_mul(2)
        .ok_or_else(|| kernel_err("glu_halves half*2 overflows usize"))?;
    if half == 0 {
        return Err(kernel_err("glu_halves half must be non-zero"));
    }
    if x.len() != rows * width {
        return Err(kernel_err(format!(
            "glu_halves x.len()={} does not match rows*2*half={}",
            x.len(),
            rows * width
        )));
    }
    if out.len() != rows * half {
        return Err(kernel_err(format!(
            "glu_halves out.len()={} does not match rows*half={}",
            out.len(),
            rows * half
        )));
    }
    for row in 0..rows {
        let src = row * width;
        let dst = row * half;
        for c in 0..half {
            out[dst + c] = x[src + c] * sigmoid(x[src + half + c]);
        }
    }
    Ok(())
}

/// `x += alpha * y`, the macaron half-step residual (`x = x + 0.5 * FFN(x)`).
///
/// The tree has `add_inplace` (alpha implicitly 1.0) but nothing scaled, and a
/// Conformer block needs the half-step twice per layer.
pub fn add_scaled_inplace(x: &mut [f32], y: &[f32], alpha: f32) -> Result<()> {
    if x.len() != y.len() {
        return Err(kernel_err(format!(
            "add_scaled_inplace length mismatch: x={} y={}",
            x.len(),
            y.len()
        )));
    }
    for (dst, src) in x.iter_mut().zip(y.iter()) {
        *dst += alpha * *src;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_clamps_negatives_and_preserves_positives_and_zero() {
        let mut x = [-2.5_f32, -0.0, 0.0, 0.5, 3.0];
        relu_inplace(&mut x);
        assert_eq!(x, [0.0, 0.0, 0.0, 0.5, 3.0]);
    }

    #[test]
    fn sigmoid_matches_hand_computed_values() {
        // sigmoid(0) = 0.5 exactly; sigmoid(1) = 1/(1+e^-1) = 0.7310585786;
        // sigmoid(-1) = 0.2689414214. Hand values, not a round-trip.
        assert_eq!(sigmoid(0.0), 0.5);
        assert!((sigmoid(1.0) - 0.731_058_6).abs() < 1e-6);
        assert!((sigmoid(-1.0) - 0.268_941_43).abs() < 1e-6);
        // Symmetry: sigmoid(-z) == 1 - sigmoid(z).
        for z in [0.3_f32, 2.0, 7.5] {
            assert!((sigmoid(-z) - (1.0 - sigmoid(z))).abs() < 1e-6);
        }
    }

    #[test]
    fn sigmoid_tails_are_finite_not_nan() {
        // The naive e^z/(1+e^z) form returns NaN here; this must not.
        assert_eq!(sigmoid(-200.0), 0.0);
        assert_eq!(sigmoid(200.0), 1.0);
        assert!(sigmoid(-200.0).is_finite() && sigmoid(200.0).is_finite());
    }

    #[test]
    fn tanh_matches_hand_computed_values() {
        // tanh(0) = 0 exactly; tanh(1) = 0.7615941559557649, whose nearest f32
        // is 0.7615942. The literal below is written to that f32 rather than to
        // full precision because the extra digits are not representable — but
        // the value they came from is recorded here, so the derivation stays
        // auditable. (A wrong literal in a sibling LSTM test cost real time this
        // port; the comment is what localized it.)
        let mut x = [0.0_f32, 1.0, -1.0];
        tanh_inplace(&mut x);
        assert_eq!(x[0], 0.0);
        assert!((x[1] - 0.761_594_2).abs() < 1e-6);
        assert!((x[2] + 0.761_594_2).abs() < 1e-6);
    }

    #[test]
    fn glu_halves_matches_hand_computed_gate() {
        // One row, half=2: a=[1,2], b=[0,1].
        // out = [1*sigmoid(0), 2*sigmoid(1)] = [0.5, 1.4621171].
        let x = [1.0_f32, 2.0, 0.0, 1.0];
        let mut out = [0.0_f32; 2];
        glu_halves(&x, 1, 2, &mut out).expect("glu");
        assert!((out[0] - 0.5).abs() < 1e-6, "got {}", out[0]);
        assert!((out[1] - 1.462_117_1).abs() < 1e-6, "got {}", out[1]);
    }

    #[test]
    fn glu_halves_gates_with_the_second_half_not_the_first() {
        // Discriminating: if the halves were swapped, out would be
        // [0*sigmoid(1), 1*sigmoid(2)] = [0, 0.8807971] instead.
        let x = [1.0_f32, 2.0, 0.0, 0.0];
        let mut out = [0.0_f32; 2];
        glu_halves(&x, 1, 2, &mut out).expect("glu");
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn glu_halves_rejects_shape_mismatch() {
        let mut out = [0.0_f32; 2];
        assert!(glu_halves(&[1.0, 2.0, 3.0], 1, 2, &mut out).is_err());
    }

    #[test]
    fn add_scaled_matches_hand_computed_half_step() {
        let mut x = [1.0_f32, 2.0, 3.0];
        add_scaled_inplace(&mut x, &[10.0, 20.0, 30.0], 0.5).expect("add_scaled");
        assert_eq!(x, [6.0, 12.0, 18.0]);
    }

    #[test]
    fn add_scaled_rejects_length_mismatch() {
        let mut x = [1.0_f32, 2.0];
        assert!(add_scaled_inplace(&mut x, &[1.0], 0.5).is_err());
    }
}
