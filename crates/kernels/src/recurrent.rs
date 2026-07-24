//! Recurrent cells.
//!
//! The tree had **no** recurrent code of any kind before this — no LSTM, no GRU,
//! no gated recurrence anywhere. Parakeet's TDT prediction network is a 2-layer
//! LSTM (hidden 640), so it is the first.

use crate::{Result, activation::sigmoid, kernel_err, linear_out_by_in};

/// One LSTM timestep, PyTorch gate convention.
///
/// Weights use PyTorch's `weight_ih_l*` / `weight_hh_l*` layout — `[4*hidden]`
/// rows over `input_size` / `hidden_size` columns respectively — which is the raw
/// `[out][in]` layout [`linear_out_by_in`] consumes directly, so no transpose.
///
/// **Gate order is `i, f, g, o`** (input, forget, cell, output), stacked along
/// the `4*hidden` axis in that order. This is not a free choice: PyTorch, ONNX
/// (`iofc`), and various papers disagree, and getting it wrong produces a network
/// that still runs and still emits plausible tokens. The order is asserted by a
/// discriminating test below.
///
/// ```text
/// gates = W_ih·x + b_ih + W_hh·h
/// i = σ(gates[0·H..1·H])   f = σ(gates[1·H..2·H])
/// g = tanh(gates[2·H..3·H]) o = σ(gates[3·H..4·H])
/// c' = f⊙c + i⊙g
/// h' = o⊙tanh(c')
/// ```
///
/// `bias_hh` is separate from `bias_ih` rather than pre-summed: PyTorch stores
/// both and their sum is what enters the gates, but keeping them apart lets a
/// loader map the checkpoint tensors one-for-one without a fold step that would
/// have to be undone to compare against the reference.
#[allow(clippy::too_many_arguments)]
pub fn lstm_step(
    x: &[f32],
    h: &[f32],
    c: &[f32],
    weight_ih: &[f32],
    weight_hh: &[f32],
    bias_ih: Option<&[f32]>,
    bias_hh: Option<&[f32]>,
    hidden: usize,
    h_out: &mut [f32],
    c_out: &mut [f32],
) -> Result<()> {
    if hidden == 0 {
        return Err(kernel_err("lstm_step hidden must be non-zero"));
    }
    let input_size = x.len();
    let gates_len = 4 * hidden;
    if h.len() != hidden || c.len() != hidden {
        return Err(kernel_err(format!(
            "lstm_step state length mismatch: h={} c={} expected {hidden}",
            h.len(),
            c.len()
        )));
    }
    if h_out.len() != hidden || c_out.len() != hidden {
        return Err(kernel_err(format!(
            "lstm_step output length mismatch: h_out={} c_out={} expected {hidden}",
            h_out.len(),
            c_out.len()
        )));
    }
    if weight_ih.len() != gates_len * input_size {
        return Err(kernel_err(format!(
            "lstm_step weight_ih.len()={} does not match 4*hidden*input_size={}",
            weight_ih.len(),
            gates_len * input_size
        )));
    }
    if weight_hh.len() != gates_len * hidden {
        return Err(kernel_err(format!(
            "lstm_step weight_hh.len()={} does not match 4*hidden*hidden={}",
            weight_hh.len(),
            gates_len * hidden
        )));
    }
    for (name, b) in [("bias_ih", bias_ih), ("bias_hh", bias_hh)] {
        if let Some(b) = b {
            if b.len() != gates_len {
                return Err(kernel_err(format!(
                    "lstm_step {name}.len()={} does not match 4*hidden={gates_len}",
                    b.len()
                )));
            }
        }
    }

    let mut gates = vec![0.0_f32; gates_len];
    linear_out_by_in(x, 1, input_size, weight_ih, gates_len, bias_ih, &mut gates)?;
    let mut from_h = vec![0.0_f32; gates_len];
    linear_out_by_in(h, 1, hidden, weight_hh, gates_len, bias_hh, &mut from_h)?;
    for (g, add) in gates.iter_mut().zip(from_h.iter()) {
        *g += *add;
    }

    for idx in 0..hidden {
        let i = sigmoid(gates[idx]);
        let f = sigmoid(gates[hidden + idx]);
        let g = gates[2 * hidden + idx].tanh();
        let o = sigmoid(gates[3 * hidden + idx]);
        let next_c = f * c[idx] + i * g;
        c_out[idx] = next_c;
        h_out[idx] = o * next_c.tanh();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// hidden=1, input=1, all weights zero so only the biases drive the gates.
    /// Lets every gate be set independently and checked by hand.
    #[allow(clippy::too_many_arguments)]
    fn step_with_bias(bias: [f32; 4], h0: f32, c0: f32) -> (f32, f32) {
        let mut h_out = [0.0_f32; 1];
        let mut c_out = [0.0_f32; 1];
        lstm_step(
            &[0.0],
            &[h0],
            &[c0],
            &[0.0; 4],
            &[0.0; 4],
            Some(&bias),
            None,
            1,
            &mut h_out,
            &mut c_out,
        )
        .expect("lstm_step");
        (h_out[0], c_out[0])
    }

    #[test]
    fn lstm_forget_gate_is_the_second_quarter() {
        // f = sigmoid(bias[1]). Drive f -> 0 with a large negative bias and
        // i -> 0 too, so c' = f*c0 + 0 must collapse to ~0 regardless of c0.
        let (_, c) = step_with_bias([-30.0, -30.0, 0.0, 0.0], 0.0, 5.0);
        assert!(c.abs() < 1e-6, "forget gate did not clear the cell: c={c}");
        // Now f -> 1 (large positive) with i -> 0: c' must PRESERVE c0 = 5.
        let (_, c) = step_with_bias([-30.0, 30.0, 0.0, 0.0], 0.0, 5.0);
        assert!((c - 5.0).abs() < 1e-5, "forget gate did not retain: c={c}");
    }

    #[test]
    fn lstm_input_and_cell_gates_are_the_first_and_third_quarters() {
        // f -> 0, i -> 1, g = tanh(bias[2]) = tanh(1) = 0.76159416.
        // c' = 0*c0 + 1*0.76159416
        let (_, c) = step_with_bias([30.0, -30.0, 1.0, 0.0], 0.0, 99.0);
        assert!(
            (c - 0.761_594_16).abs() < 1e-5,
            "i/g gates wrong: c={c} (expected tanh(1))"
        );
    }

    #[test]
    fn lstm_output_gate_is_the_fourth_quarter() {
        // Set c' = tanh(1) as above, then o = sigmoid(bias[3]).
        // o -> 0 => h' -> 0 even though c' is non-zero: this is exactly the
        // check that distinguishes the output gate from the others.
        let (h, c) = step_with_bias([30.0, -30.0, 1.0, -30.0], 0.0, 0.0);
        assert!(c.abs() > 0.7, "precondition: cell should be non-zero");
        assert!(h.abs() < 1e-6, "output gate did not close: h={h}");
        // o -> 1 => h' = tanh(c').
        let (h, c) = step_with_bias([30.0, -30.0, 1.0, 30.0], 0.0, 0.0);
        assert!((h - c.tanh()).abs() < 1e-5, "h={h} c={c}");
    }

    #[test]
    fn lstm_step_matches_a_fully_hand_computed_step() {
        // hidden=1, input=1, x=1, h=0, c=0.
        // W_ih = [i,f,g,o] = [0, 0, 1, 0]; all biases zero; W_hh zero.
        // gates = [0, 0, 1, 0]
        //   i = sigmoid(0) = 0.5
        //   f = sigmoid(0) = 0.5
        //   g = tanh(1)    = 0.76159416
        //   o = sigmoid(0) = 0.5
        // c' = 0.5*0 + 0.5*0.76159416 = 0.380797078
        // h' = 0.5 * tanh(0.380797078) = 0.5 * 0.363399484 = 0.181699742
        let mut h_out = [0.0_f32; 1];
        let mut c_out = [0.0_f32; 1];
        lstm_step(
            &[1.0],
            &[0.0],
            &[0.0],
            &[0.0, 0.0, 1.0, 0.0],
            &[0.0; 4],
            None,
            None,
            1,
            &mut h_out,
            &mut c_out,
        )
        .expect("lstm_step");
        assert!((c_out[0] - 0.380_797_08).abs() < 1e-6, "c={}", c_out[0]);
        assert!((h_out[0] - 0.181_699_74).abs() < 1e-6, "h={}", h_out[0]);
    }

    #[test]
    fn lstm_step_uses_the_recurrent_weights() {
        // Same as above but driven through h instead of x: W_hh g-row = 1, h = 1.
        // Must give the identical result — proving W_hh is actually applied.
        let mut h_out = [0.0_f32; 1];
        let mut c_out = [0.0_f32; 1];
        lstm_step(
            &[0.0],
            &[1.0],
            &[0.0],
            &[0.0; 4],
            &[0.0, 0.0, 1.0, 0.0],
            None,
            None,
            1,
            &mut h_out,
            &mut c_out,
        )
        .expect("lstm_step");
        assert!((c_out[0] - 0.380_797_08).abs() < 1e-6, "c={}", c_out[0]);
    }

    #[test]
    fn lstm_step_rejects_shape_mismatches() {
        let mut h_out = [0.0_f32; 1];
        let mut c_out = [0.0_f32; 1];
        // weight_hh sized for hidden=2 while hidden=1.
        assert!(
            lstm_step(
                &[0.0],
                &[0.0],
                &[0.0],
                &[0.0; 4],
                &[0.0; 8],
                None,
                None,
                1,
                &mut h_out,
                &mut c_out
            )
            .is_err()
        );
    }
}
