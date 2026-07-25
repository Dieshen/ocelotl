//! Parakeet TDT prediction network, joint network, and greedy decode.
//!
//! # Why TDT is not RNN-T
//!
//! A classic RNN-T joint emits a distribution over `vocab + blank` and the time
//! index advances by exactly one frame per blank. **Token-and-Duration
//! Transducer** widens the joint to `vocab + blank + |durations|` and makes the
//! time advance *predicted*: each step also argmaxes a duration and jumps that
//! many frames. That is where the speed comes from — and where the fragility
//! does, because a single flipped duration argmax shifts the time index by up to
//! 4 frames (320 ms at 8x subsampling) and desynchronizes everything after it.
//! There is no gradual degradation: the transcript is either right or visibly
//! wrong.
//!
//! Measured on the pinned `parity_jfk` fixture, the smallest top-2 duration
//! margin over the whole 46-step decode is **0.1824**, and driving the same
//! decode from ocelotl's frontend instead of the reference shifts those logits
//! by at most 1.4e-4 — about 1275x of headroom. That is why the parity gate here
//! is token-exact rather than a WER delta.
//!
//! # Loop semantics
//!
//! Three details decide whether the loop terminates and whether it agrees with
//! the reference. All three are easy to get wrong in a way that still produces
//! fluent-looking output:
//!
//! 1. **The prediction-network state commits only on a non-blank.** A blank
//!    leaves the LSTM state untouched. (`gpu-cli/parakeet-rs` commits on blank —
//!    a real, diffed bug in one of the more inviting references.)
//! 2. **A predicted duration of 0 does not advance time.** Progress then depends
//!    on the blank/`max_symbols` fallback below; without it the decode hangs.
//! 3. **`max_symbols` forces `t += 1`.** With `d == 0` and a stream of non-blank
//!    emissions, nothing else breaks the inner loop.
//!
//! The reference is `onnx_asr`'s transducer decoding, which ships with the same
//! ONNX exports used as this port's oracle, so it pins loop semantics and
//! tensors together.

use ocelotl_core::{OcelotlError, Result, RuntimeError};
use ocelotl_kernels::KernelBackend;
use ocelotl_kernels::activation::relu_inplace;
use ocelotl_kernels::recurrent::lstm_step;

/// Prediction-network hidden width (`pred_hidden`).
pub const PRED_HIDDEN: usize = 640;
/// Prediction-network LSTM depth (`pred_rnn_layers`).
pub const PRED_LAYERS: usize = 2;
/// Joint-network hidden width (`joint_hidden`).
pub const JOINT_HIDDEN: usize = 640;
/// Encoder width feeding the joint.
pub const ENC_HIDDEN: usize = 1024;
/// Token classes **including** blank: 8192 SPE pieces + `<blk>`.
pub const VOCAB_SIZE: usize = 8193;
/// Blank id. Also the SOS token: `decoder.embedding` row 8192 is PyTorch's
/// `padding_idx` and is exactly zero, so feeding blank reproduces NeMo's
/// "no previous label -> zero vector" start without a special case.
pub const BLANK: u32 = 8192;
/// Duration classes.
pub const NUM_DURATIONS: usize = 5;
/// Joint output width: `VOCAB_SIZE + NUM_DURATIONS`.
pub const JOINT_WIDTH: usize = VOCAB_SIZE + NUM_DURATIONS;
/// Frames to advance per duration class (`model_defaults.tdt_durations`).
pub const DURATIONS: [usize; NUM_DURATIONS] = [0, 1, 2, 3, 4];
/// Emissions allowed at one time index before time is forced forward.
pub const MAX_TOKENS_PER_STEP: usize = 10;

fn rt<S: Into<String>>(m: S) -> OcelotlError {
    OcelotlError::Runtime(RuntimeError { message: m.into() })
}

/// One LSTM layer's PyTorch tensors, in checkpoint layout (`[4*hidden][in]`).
#[derive(Debug, Clone, Default)]
pub struct LstmLayerWeights {
    pub weight_ih: Vec<f32>,
    pub weight_hh: Vec<f32>,
    pub bias_ih: Vec<f32>,
    pub bias_hh: Vec<f32>,
}

/// Prediction network: token embedding followed by a stacked LSTM.
#[derive(Debug, Clone, Default)]
pub struct PredNetWeights {
    /// `[VOCAB_SIZE][PRED_HIDDEN]`; row `BLANK` is the zero SOS vector.
    pub embedding: Vec<f32>,
    pub layers: Vec<LstmLayerWeights>,
}

/// Joint network: two projections summed, ReLU, then the wide head.
#[derive(Debug, Clone, Default)]
pub struct JointWeights {
    /// `[JOINT_HIDDEN][ENC_HIDDEN]` + bias.
    pub enc_proj_w: Vec<f32>,
    pub enc_proj_b: Vec<f32>,
    /// `[JOINT_HIDDEN][PRED_HIDDEN]` + bias.
    pub pred_proj_w: Vec<f32>,
    pub pred_proj_b: Vec<f32>,
    /// `[JOINT_WIDTH][JOINT_HIDDEN]` + bias.
    pub head_w: Vec<f32>,
    pub head_b: Vec<f32>,
}

/// Prediction-network recurrent state, `[PRED_LAYERS][PRED_HIDDEN]` per tensor.
#[derive(Debug, Clone)]
pub struct PredNetState {
    pub h: Vec<f32>,
    pub c: Vec<f32>,
}

impl PredNetState {
    /// Zero state — NeMo starts every utterance here.
    pub fn zeros(layers: usize, hidden: usize) -> Self {
        Self {
            h: vec![0.0; layers * hidden],
            c: vec![0.0; layers * hidden],
        }
    }
}

impl Default for PredNetState {
    fn default() -> Self {
        Self::zeros(PRED_LAYERS, PRED_HIDDEN)
    }
}

/// Run one prediction-network step for `token`, returning its output and the
/// **candidate** next state.
///
/// The state is returned rather than applied in place because the caller must
/// discard it on a blank. Making that the caller's explicit choice is
/// deliberate: an in-place variant is exactly the shape that produces the
/// commit-on-blank bug this port set out to avoid.
pub fn prednet_step(
    token: u32,
    weights: &PredNetWeights,
    state: &PredNetState,
) -> Result<(Vec<f32>, PredNetState)> {
    let hidden = PRED_HIDDEN;
    let layers = weights.layers.len();
    if layers == 0 {
        return Err(rt("prednet has no LSTM layers"));
    }
    if weights.embedding.len() != VOCAB_SIZE * hidden {
        return Err(rt(format!(
            "prednet embedding has {} values, expected {}x{hidden}",
            weights.embedding.len(),
            VOCAB_SIZE
        )));
    }
    if token as usize >= VOCAB_SIZE {
        return Err(rt(format!(
            "prednet token {token} is outside the vocabulary of {VOCAB_SIZE}"
        )));
    }
    if state.h.len() != layers * hidden || state.c.len() != layers * hidden {
        return Err(rt(format!(
            "prednet state is h={} c={}, expected {}",
            state.h.len(),
            state.c.len(),
            layers * hidden
        )));
    }

    let start = token as usize * hidden;
    let mut x = weights.embedding[start..start + hidden].to_vec();
    let mut next = PredNetState::zeros(layers, hidden);

    for (i, layer) in weights.layers.iter().enumerate() {
        let lo = i * hidden;
        let hi = lo + hidden;
        // `h` and `c` are separate allocations, so these two mutable borrows are
        // disjoint; the input state is a different value again.
        let PredNetState {
            h: next_h,
            c: next_c,
        } = &mut next;
        lstm_step(
            &x,
            &state.h[lo..hi],
            &state.c[lo..hi],
            &layer.weight_ih,
            &layer.weight_hh,
            Some(&layer.bias_ih),
            Some(&layer.bias_hh),
            hidden,
            &mut next_h[lo..hi],
            &mut next_c[lo..hi],
        )?;
        x.copy_from_slice(&next.h[lo..hi]);
    }
    Ok((x, next))
}

/// Project every encoder frame through the joint's encoder branch, once.
///
/// The joint is evaluated at least once per decode step, but the encoder side of
/// it depends only on the frame — so hoisting it out of the loop turns a
/// `[640,1024]` matmul per step into one `[T,1024]x[1024,640]` GEMM, which the
/// AVX2 microkernel handles far better anyway. Returns `[frames][JOINT_HIDDEN]`.
pub fn project_encoder(
    encoder: &[f32],
    frames: usize,
    weights: &JointWeights,
    kernels: &dyn KernelBackend,
) -> Result<Vec<f32>> {
    if encoder.len() != frames * ENC_HIDDEN {
        return Err(rt(format!(
            "encoder output has {} values, expected {frames}x{ENC_HIDDEN}",
            encoder.len()
        )));
    }
    let mut out = vec![0.0_f32; frames * JOINT_HIDDEN];
    kernels.linear_out_by_in(
        encoder,
        frames,
        ENC_HIDDEN,
        &weights.enc_proj_w,
        JOINT_HIDDEN,
        Some(&weights.enc_proj_b),
        &mut out,
    )?;
    Ok(out)
}

/// Evaluate the joint for one (frame, prediction) pair.
///
/// `enc_projected` is one row of [`project_encoder`]. Writes `JOINT_WIDTH`
/// logits: `[0..VOCAB_SIZE)` are token scores (blank at [`BLANK`]) and
/// `[VOCAB_SIZE..)` are the duration scores.
pub fn joint_step(
    enc_projected: &[f32],
    pred_out: &[f32],
    weights: &JointWeights,
    kernels: &dyn KernelBackend,
    logits: &mut [f32],
) -> Result<()> {
    if enc_projected.len() != JOINT_HIDDEN {
        return Err(rt(format!(
            "joint encoder row has {} values, expected {JOINT_HIDDEN}",
            enc_projected.len()
        )));
    }
    if pred_out.len() != PRED_HIDDEN {
        return Err(rt(format!(
            "joint prediction row has {} values, expected {PRED_HIDDEN}",
            pred_out.len()
        )));
    }
    if logits.len() != JOINT_WIDTH {
        return Err(rt(format!(
            "joint output buffer has {} values, expected {JOINT_WIDTH}",
            logits.len()
        )));
    }

    let mut hidden = vec![0.0_f32; JOINT_HIDDEN];
    kernels.linear_out_by_in(
        pred_out,
        1,
        PRED_HIDDEN,
        &weights.pred_proj_w,
        JOINT_HIDDEN,
        Some(&weights.pred_proj_b),
        &mut hidden,
    )?;
    for (h, e) in hidden.iter_mut().zip(enc_projected.iter()) {
        *h += *e;
    }
    relu_inplace(&mut hidden);
    kernels.linear_out_by_in(
        &hidden,
        1,
        JOINT_HIDDEN,
        &weights.head_w,
        JOINT_WIDTH,
        Some(&weights.head_b),
        logits,
    )
}

/// One iteration of the greedy loop, recorded for parity and debugging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TdtStep {
    /// Encoder frame the joint was evaluated at.
    pub frame: usize,
    /// Token fed to the prediction network (previous emission, or [`BLANK`]).
    pub prev_token: u32,
    /// Token argmax over `[0..VOCAB_SIZE)`.
    pub token: u32,
    /// Duration argmax, as an index into [`DURATIONS`].
    pub duration_index: usize,
    /// Frames advanced: `DURATIONS[duration_index]`.
    pub duration: usize,
}

/// Result of a greedy TDT decode.
#[derive(Debug, Clone, Default)]
pub struct TdtDecode {
    /// Emitted token ids, blanks excluded.
    pub tokens: Vec<u32>,
    /// Encoder frame each token was emitted at — parallel to `tokens`.
    pub frames: Vec<usize>,
    /// Every loop iteration, including the blank ones.
    pub steps: Vec<TdtStep>,
}

/// Knobs that change what the decode does. Defaults mirror the checkpoint.
#[derive(Debug, Clone, Copy)]
pub struct TdtConfig {
    pub durations: [usize; NUM_DURATIONS],
    pub blank: u32,
    pub max_tokens_per_step: usize,
    /// Hard stop on total iterations. A duration table containing 0 makes
    /// non-termination reachable through a weight bug, and an inference loop
    /// that hangs is worse to diagnose than one that fails.
    pub max_steps: usize,
}

impl Default for TdtConfig {
    fn default() -> Self {
        Self {
            durations: DURATIONS,
            blank: BLANK,
            max_tokens_per_step: MAX_TOKENS_PER_STEP,
            max_steps: 100_000,
        }
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, v) in values.iter().enumerate() {
        if *v > best_v {
            best_v = *v;
            best = i;
        }
    }
    best
}

/// Greedy TDT decode over an encoder output of `[frames][ENC_HIDDEN]`.
///
/// See the module docs for the three loop rules that matter. In short:
/// commit the prediction state only on a non-blank; advance time by the
/// predicted duration; and when that duration is 0, advance one frame anyway if
/// the token was blank or `max_tokens_per_step` emissions have accumulated.
pub fn greedy_decode(
    encoder: &[f32],
    frames: usize,
    prednet: &PredNetWeights,
    joint: &JointWeights,
    config: TdtConfig,
    kernels: &dyn KernelBackend,
) -> Result<TdtDecode> {
    if config.max_tokens_per_step == 0 {
        return Err(rt("max_tokens_per_step must be non-zero"));
    }
    let enc_projected = project_encoder(encoder, frames, joint, kernels)?;

    let mut state = PredNetState::zeros(prednet.layers.len(), PRED_HIDDEN);
    let mut out = TdtDecode::default();
    let mut logits = vec![0.0_f32; JOINT_WIDTH];

    let mut t = 0usize;
    let mut emitted = 0usize;
    while t < frames {
        if out.steps.len() >= config.max_steps {
            return Err(rt(format!(
                "TDT decode exceeded {} steps at frame {t} of {frames} — \
                 the duration head is predicting 0 without terminating",
                config.max_steps
            )));
        }
        let prev = out.tokens.last().copied().unwrap_or(config.blank);
        let (pred_out, next_state) = prednet_step(prev, prednet, &state)?;
        joint_step(
            &enc_projected[t * JOINT_HIDDEN..(t + 1) * JOINT_HIDDEN],
            &pred_out,
            joint,
            kernels,
            &mut logits,
        )?;

        let token = argmax(&logits[..VOCAB_SIZE]) as u32;
        let duration_index = argmax(&logits[VOCAB_SIZE..]);
        let duration = config.durations[duration_index];

        out.steps.push(TdtStep {
            frame: t,
            prev_token: prev,
            token,
            duration_index,
            duration,
        });

        if token != config.blank {
            // Commit the recurrent state ONLY here. A blank must leave the
            // prediction network exactly as it was.
            state = next_state;
            out.tokens.push(token);
            out.frames.push(t);
            emitted += 1;
        }

        if duration > 0 {
            t += duration;
            emitted = 0;
        } else if token == config.blank || emitted == config.max_tokens_per_step {
            // Duration 0 stalls time; blank or a full symbol budget is what
            // breaks the stall.
            t += 1;
            emitted = 0;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ocelotl_kernels::{CpuKernelBackend, CpuKernelMode};

    fn backend() -> CpuKernelBackend {
        CpuKernelBackend::with_mode(CpuKernelMode::Scalar).expect("cpu backend")
    }

    /// Build a prednet whose LSTM is all zeros, so its output is zero for any
    /// token and the joint sees only the encoder side. That makes the joint
    /// logits a pure function of the frame, which is what the loop tests below
    /// need to steer the decode deterministically.
    fn inert_prednet() -> PredNetWeights {
        PredNetWeights {
            embedding: vec![0.0; VOCAB_SIZE * PRED_HIDDEN],
            layers: (0..PRED_LAYERS)
                .map(|_| LstmLayerWeights {
                    weight_ih: vec![0.0; 4 * PRED_HIDDEN * PRED_HIDDEN],
                    weight_hh: vec![0.0; 4 * PRED_HIDDEN * PRED_HIDDEN],
                    bias_ih: vec![0.0; 4 * PRED_HIDDEN],
                    bias_hh: vec![0.0; 4 * PRED_HIDDEN],
                })
                .collect(),
        }
    }

    /// A joint that ignores its inputs and emits `head_b` verbatim: with both
    /// projections zeroed the ReLU output is zero, so the head contributes only
    /// its bias. Set `head_b` to place the token and duration argmaxes wherever
    /// a test needs them.
    fn scripted_joint(logits: Vec<f32>) -> JointWeights {
        assert_eq!(logits.len(), JOINT_WIDTH);
        JointWeights {
            enc_proj_w: vec![0.0; JOINT_HIDDEN * ENC_HIDDEN],
            enc_proj_b: vec![0.0; JOINT_HIDDEN],
            pred_proj_w: vec![0.0; JOINT_HIDDEN * PRED_HIDDEN],
            pred_proj_b: vec![0.0; JOINT_HIDDEN],
            head_w: vec![0.0; JOINT_WIDTH * JOINT_HIDDEN],
            head_b: logits,
        }
    }

    fn logits_with(token: u32, duration_index: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; JOINT_WIDTH];
        v[token as usize] = 1.0;
        v[VOCAB_SIZE + duration_index] = 1.0;
        v
    }

    #[test]
    fn blank_with_duration_two_advances_two_frames_and_emits_nothing() {
        let joint = scripted_joint(logits_with(BLANK, 2));
        let got = greedy_decode(
            &vec![0.0; 8 * ENC_HIDDEN],
            8,
            &inert_prednet(),
            &joint,
            TdtConfig::default(),
            &backend(),
        )
        .expect("decode");
        assert!(got.tokens.is_empty(), "blank must not emit");
        // 8 frames, +2 each: frames 0, 2, 4, 6 then t = 8 ends the loop.
        assert_eq!(got.steps.len(), 4);
        assert_eq!(
            got.steps.iter().map(|s| s.frame).collect::<Vec<_>>(),
            vec![0, 2, 4, 6]
        );
    }

    #[test]
    fn blank_with_duration_zero_still_advances_one_frame() {
        // The stall-breaker. Without the `token == blank` arm this hangs, and a
        // hang is the failure mode this test exists to make impossible.
        let joint = scripted_joint(logits_with(BLANK, 0));
        let got = greedy_decode(
            &vec![0.0; 3 * ENC_HIDDEN],
            3,
            &inert_prednet(),
            &joint,
            TdtConfig::default(),
            &backend(),
        )
        .expect("decode");
        assert!(got.tokens.is_empty());
        assert_eq!(
            got.steps.iter().map(|s| s.frame).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn duration_zero_non_blank_emits_until_max_tokens_then_advances() {
        // Duration 0 with a non-blank token is the case where only
        // `max_tokens_per_step` makes progress. Expect exactly
        // max_tokens_per_step emissions per frame, all stamped with that frame.
        let joint = scripted_joint(logits_with(7, 0));
        let config = TdtConfig {
            max_tokens_per_step: 3,
            ..TdtConfig::default()
        };
        let got = greedy_decode(
            &vec![0.0; 2 * ENC_HIDDEN],
            2,
            &inert_prednet(),
            &joint,
            config,
            &backend(),
        )
        .expect("decode");
        assert_eq!(got.tokens, vec![7; 6], "3 emissions on each of 2 frames");
        assert_eq!(got.frames, vec![0, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn a_duration_zero_loop_that_cannot_progress_errors_instead_of_hanging() {
        // max_tokens_per_step is unreachable within max_steps, so the loop can
        // never advance. It must fail loudly rather than spin.
        let joint = scripted_joint(logits_with(7, 0));
        let config = TdtConfig {
            max_tokens_per_step: usize::MAX,
            max_steps: 64,
            ..TdtConfig::default()
        };
        let err = greedy_decode(
            &vec![0.0; ENC_HIDDEN],
            1,
            &inert_prednet(),
            &joint,
            config,
            &backend(),
        )
        .expect_err("must not hang");
        assert!(
            format!("{err}").contains("exceeded"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn the_blank_embedding_row_is_the_sos_zero_vector() {
        // NeMo starts with "no previous label", implemented as embedding row
        // BLANK being padding_idx and therefore exactly zero. This asserts the
        // convention the loop relies on, so a checkpoint that violates it fails
        // here rather than silently shifting every prediction.
        let mut prednet = inert_prednet();
        prednet.embedding[BLANK as usize * PRED_HIDDEN] = 0.0;
        let state = PredNetState::zeros(PRED_LAYERS, PRED_HIDDEN);
        let (out, _) = prednet_step(BLANK, &prednet, &state).expect("step");
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn a_token_outside_the_vocabulary_is_rejected() {
        let prednet = inert_prednet();
        let state = PredNetState::zeros(PRED_LAYERS, PRED_HIDDEN);
        assert!(prednet_step(VOCAB_SIZE as u32, &prednet, &state).is_err());
    }
}
