//! Parakeet TDT decoder parity against the reference ONNX decoder/joint graph.
//!
//! The gate here is **token-exact**, not a tolerance. That is a deliberate
//! choice made on evidence rather than optimism: the Stage-0 probe measured the
//! smallest top-2 duration-argmax margin over the whole decode at **0.1824**,
//! while swapping ocelotl's frontend in for the reference one moves those logits
//! by at most 1.4e-4 — roughly **1275x** of headroom. With that much room, a
//! numeric gate would be strictly weaker than an exact one: any drift large
//! enough to matter flips an argmax and shows up as a different token.
//!
//! Frame indices are compared too, and they are the more sensitive assertion.
//! Tokens can coincide while the time alignment has silently drifted, and a
//! desynchronized time index is the failure mode TDT is uniquely prone to.
//!
//! When the exact gate does fail, the per-step joint-logit dump localizes it:
//! the reference graph was instrumented at seven sublayer boundaries, so the
//! test can name the first diverging step and the first diverging stage instead
//! of leaving "prednet or joint?" open.
//!
//! Opt-in, matching the rest of the Parakeet suite. Requires:
//!   `OCELOTL_PARAKEET_WEIGHTS`   path to `model.safetensors` (nvidia v3)
//!   `OCELOTL_PARAKEET_REF_DIR`   dir holding `enc/` and `decode/`
//!   `OCELOTL_PARAKEET_TOKENIZER` path to `tokenizer.json` (optional; enables
//!                                the detokenization check)

use std::path::PathBuf;

use ocelotl_core::TokenId;
use ocelotl_kernels::{CpuKernelBackend, CpuKernelMode, transpose_2d};
use ocelotl_loader::load_safetensors_tensors_f32;
use ocelotl_models::parakeet::decoder::{
    ENC_HIDDEN, JOINT_HIDDEN, JOINT_WIDTH, JointWeights, LstmLayerWeights, PRED_HIDDEN,
    PRED_LAYERS, PredNetState, PredNetWeights, TdtConfig, VOCAB_SIZE, greedy_decode, joint_step,
    prednet_step, project_encoder,
};
use ocelotl_models::parakeet::model::ParakeetModel;
use ocelotl_tokenizer::{JsonTokenizer, Tokenizer};

/// Encoder frames for the pinned `parity_jfk` fixture (138 = subsample(1101)).
const REF_FRAMES: usize = 138;

/// Numeric tolerance for the *diagnostic* tensor comparisons only.
///
/// This does not gate correctness — the token-exact assertions do. It exists so
/// a failure reports "the joint logits already differ by 3e-1 at step 7" rather
/// than only "the tokens differ", and so a drift that has not yet flipped an
/// argmax is still visible. Calibrated against the measured decode: reference
/// joint logits span roughly +/-30, and the observed ocelotl-vs-reference delta
/// sits near 1e-4, so 1e-2 flags real movement while tolerating the f32 floor.
const DIAGNOSTIC_ABS: f32 = 1e-2;

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key).ok().map(PathBuf::from)
}

/// Prefer the AVX2 microkernel, fall back to the portable path.
///
/// Worth spelling out, because the naming invites the wrong choice:
/// `CpuKernelMode::Optimized` is *safe-Rust scalar with a cache-friendlier
/// accumulation order* — the AVX2 + FMA microkernel is the separate `Avx2`
/// variant. Picking `Optimized` for the encoder costs **105x** (measured:
/// 5.2 s vs 552 s for one 11-second utterance), so the wrong choice reads as a
/// hang rather than as a slow test. `Avx2` construction fails on a non-AVX2
/// host rather than silently degrading, hence the explicit fallback.
fn backend() -> CpuKernelBackend {
    CpuKernelBackend::with_mode(CpuKernelMode::Avx2)
        .or_else(|_| CpuKernelBackend::with_mode(CpuKernelMode::Optimized))
        .expect("cpu backend")
}

fn read_f32(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_i32(path: &PathBuf) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn load_prednet(path: &PathBuf) -> PredNetWeights {
    let mut names = vec!["decoder.embedding.weight".to_string()];
    for l in 0..PRED_LAYERS {
        for t in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"] {
            names.push(format!("decoder.lstm.{t}_l{l}"));
        }
    }
    let loaded = load_safetensors_tensors_f32(path, &names).expect("load prednet weights");
    let mut it = loaded.into_iter().map(|t| t.values);
    let embedding = it.next().expect("embedding");
    let layers = (0..PRED_LAYERS)
        .map(|_| LstmLayerWeights {
            weight_ih: it.next().expect("weight_ih"),
            weight_hh: it.next().expect("weight_hh"),
            bias_ih: it.next().expect("bias_ih"),
            bias_hh: it.next().expect("bias_hh"),
        })
        .collect();
    PredNetWeights { embedding, layers }
}

fn load_joint(path: &PathBuf) -> JointWeights {
    let names = [
        "encoder_projector.weight",
        "encoder_projector.bias",
        "decoder.decoder_projector.weight",
        "decoder.decoder_projector.bias",
        "joint.head.weight",
        "joint.head.bias",
    ];
    let loaded = load_safetensors_tensors_f32(path, &names).expect("load joint weights");
    let mut it = loaded.into_iter().map(|t| t.values);
    let mut n = || it.next().expect("tensor");
    JointWeights {
        enc_proj_w: n(),
        enc_proj_b: n(),
        pred_proj_w: n(),
        pred_proj_b: n(),
        head_w: n(),
        head_b: n(),
    }
}

/// The reference encoder dump is `[d_model][frames]` (channel-major, as the
/// exported graph emits it); the decoder wants one contiguous frame at a time.
fn reference_encoder_time_major(ref_dir: &PathBuf) -> Vec<f32> {
    let channel_major = read_f32(&ref_dir.join("enc").join("outputs.f32"));
    assert_eq!(
        channel_major.len(),
        ENC_HIDDEN * REF_FRAMES,
        "reference encoder output is not {ENC_HIDDEN}x{REF_FRAMES}"
    );
    transpose_2d(&channel_major, ENC_HIDDEN, REF_FRAMES)
}

fn fixtures() -> Option<(PathBuf, PathBuf)> {
    match (
        env_path("OCELOTL_PARAKEET_WEIGHTS"),
        env_path("OCELOTL_PARAKEET_REF_DIR"),
    ) {
        (Some(w), Some(r)) => Some((w, r)),
        _ => {
            eprintln!("skipping: set OCELOTL_PARAKEET_WEIGHTS and OCELOTL_PARAKEET_REF_DIR");
            None
        }
    }
}

/// Step the prediction network and joint exactly as the reference did, along the
/// reference's own token/frame path, and diff at every sublayer boundary.
///
/// Run before the end-to-end decode on purpose: it isolates the *math* from the
/// *loop*. If this passes and the decode below fails, the bug is in the loop
/// (state commit, duration advance, symbol budget), not in the tensors — and
/// vice versa. Without this split, a wrong transcript implicates both at once.
#[test]
#[ignore = "requires OCELOTL_PARAKEET_WEIGHTS + OCELOTL_PARAKEET_REF_DIR"]
fn parakeet_prednet_and_joint_match_reference_step_by_step() {
    let Some((weights_path, ref_dir)) = fixtures() else {
        return;
    };
    let decode = ref_dir.join("decode");
    let step_frames = read_i32(&decode.join("step_frames.i32"));
    let step_tokens = read_i32(&decode.join("step_tokens.i32"));
    let steps = step_frames.len();
    let ref_pred = read_f32(&decode.join("pred_out.f32"));
    let ref_joint = read_f32(&decode.join("joint_logits.f32"));
    let ref_h = read_f32(&decode.join("states_h.f32"));
    assert_eq!(ref_pred.len(), steps * PRED_HIDDEN);
    assert_eq!(ref_joint.len(), steps * JOINT_WIDTH);

    let prednet = load_prednet(&weights_path);
    let joint = load_joint(&weights_path);
    let backend = backend();
    let encoder = reference_encoder_time_major(&ref_dir);
    let enc_projected = project_encoder(&encoder, REF_FRAMES, &joint, &backend).expect("project");

    let mut state = PredNetState::zeros(PRED_LAYERS, PRED_HIDDEN);
    let mut logits = vec![0.0_f32; JOINT_WIDTH];
    let mut worst_pred = 0.0_f32;
    let mut worst_h = 0.0_f32;
    let mut worst_joint = 0.0_f32;
    let mut worst_step = 0usize;
    let mut prev = ocelotl_models::parakeet::decoder::BLANK;

    for s in 0..steps {
        let (pred_out, next_state) = prednet_step(prev, &prednet, &state).expect("prednet");
        let t = step_frames[s] as usize;
        joint_step(
            &enc_projected[t * JOINT_HIDDEN..(t + 1) * JOINT_HIDDEN],
            &pred_out,
            &joint,
            &backend,
            &mut logits,
        )
        .expect("joint");

        let dp = pred_out
            .iter()
            .zip(ref_pred[s * PRED_HIDDEN..(s + 1) * PRED_HIDDEN].iter())
            .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
        let dh = next_state
            .h
            .iter()
            .zip(ref_h[s * PRED_LAYERS * PRED_HIDDEN..(s + 1) * PRED_LAYERS * PRED_HIDDEN].iter())
            .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
        let dj = logits
            .iter()
            .zip(ref_joint[s * JOINT_WIDTH..(s + 1) * JOINT_WIDTH].iter())
            .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
        if dj > worst_joint {
            worst_joint = dj;
            worst_step = s;
        }
        worst_pred = worst_pred.max(dp);
        worst_h = worst_h.max(dh);

        // Replay the reference's own decision, so one divergence cannot cascade
        // into a different path and hide every later comparison.
        let token = step_tokens[s] as u32;
        if token != ocelotl_models::parakeet::decoder::BLANK {
            state = next_state;
            prev = token;
        }
    }

    eprintln!(
        "PARAKEET_DECODER_STAGES steps={steps} pred={worst_pred:.3e} \
         lstm_h={worst_h:.3e} joint={worst_joint:.3e} worst_joint_step={worst_step}"
    );
    assert!(
        worst_pred <= DIAGNOSTIC_ABS,
        "prediction network diverges by {worst_pred:.3e}"
    );
    assert!(
        worst_h <= DIAGNOSTIC_ABS,
        "LSTM hidden state diverges by {worst_h:.3e}"
    );
    assert!(
        worst_joint <= DIAGNOSTIC_ABS,
        "joint logits diverge by {worst_joint:.3e} at step {worst_step}"
    );
}

/// The Stage-3 gate: the full greedy decode must reproduce the reference's
/// tokens, its per-token frame indices, and its per-step durations exactly.
#[test]
#[ignore = "requires OCELOTL_PARAKEET_WEIGHTS + OCELOTL_PARAKEET_REF_DIR"]
fn parakeet_greedy_decode_is_token_and_frame_exact() {
    let Some((weights_path, ref_dir)) = fixtures() else {
        return;
    };
    let decode = ref_dir.join("decode");
    let want_tokens: Vec<u32> = read_i32(&decode.join("tokens.i32"))
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let want_frames: Vec<usize> = read_i32(&decode.join("frames.i32"))
        .into_iter()
        .map(|v| v as usize)
        .collect();
    let want_durations: Vec<usize> = read_i32(&decode.join("durations.i32"))
        .into_iter()
        .map(|v| v as usize)
        .collect();

    let prednet = load_prednet(&weights_path);
    let joint = load_joint(&weights_path);
    let backend = backend();
    let encoder = reference_encoder_time_major(&ref_dir);

    let got = greedy_decode(
        &encoder,
        REF_FRAMES,
        &prednet,
        &joint,
        TdtConfig::default(),
        &backend,
    )
    .expect("greedy decode");

    let got_durations: Vec<usize> = got.steps.iter().map(|s| s.duration).collect();
    eprintln!(
        "PARAKEET_DECODE steps={} tokens={} frames={:?}",
        got.steps.len(),
        got.tokens.len(),
        &got.frames[..got.frames.len().min(8)]
    );

    // Report the FIRST divergence rather than a bare inequality: with a decode
    // this long, "sequences differ" is not a usable failure message.
    if let Some(i) = got_durations
        .iter()
        .zip(want_durations.iter())
        .position(|(a, b)| a != b)
    {
        let s = &got.steps[i];
        panic!(
            "duration diverges first at step {i} (frame {}): got {} want {} \
             (token got {} ). A duration flip desynchronizes every later step, \
             so this is the only step worth looking at.",
            s.frame, got_durations[i], want_durations[i], s.token
        );
    }
    assert_eq!(
        got_durations.len(),
        want_durations.len(),
        "step count differs: {} vs {}",
        got_durations.len(),
        want_durations.len()
    );

    if let Some(i) = got
        .tokens
        .iter()
        .zip(want_tokens.iter())
        .position(|(a, b)| a != b)
    {
        panic!(
            "token diverges first at index {i}: got {} want {} (frame {})",
            got.tokens[i], want_tokens[i], got.frames[i]
        );
    }
    assert_eq!(got.tokens, want_tokens, "token sequence length differs");

    // Frames last: they are the assertion tokens can pass while alignment has
    // silently drifted.
    assert_eq!(
        got.frames, want_frames,
        "frame indices differ despite identical tokens — the time axis has drifted"
    );
}

/// Detokenization closes the loop: token ids only matter if they render to the
/// right string. Separate from the decode test because a mismatch here is a
/// tokenizer-configuration problem (Metaspace handling), not a model problem.
#[test]
#[ignore = "requires OCELOTL_PARAKEET_* fixtures and OCELOTL_PARAKEET_TOKENIZER"]
fn parakeet_detokenizes_to_the_reference_transcript() {
    let Some((_weights, ref_dir)) = fixtures() else {
        return;
    };
    let Some(tokenizer_path) = env_path("OCELOTL_PARAKEET_TOKENIZER") else {
        eprintln!("skipping: set OCELOTL_PARAKEET_TOKENIZER");
        return;
    };
    let decode = ref_dir.join("decode");
    let tokens: Vec<TokenId> = read_i32(&decode.join("tokens.i32"))
        .into_iter()
        .map(|v| TokenId(v as u32))
        .collect();
    let want = std::fs::read_to_string(decode.join("text.txt")).expect("reference text");

    let tokenizer = JsonTokenizer::from_json_path(&tokenizer_path).expect("tokenizer.json");
    let got = tokenizer.decode(&tokens).expect("decode tokens");
    eprintln!("PARAKEET_TEXT {got:?}");
    assert_eq!(
        got.trim(),
        want.trim(),
        "detokenized transcript differs from the reference"
    );
    // Guard the gate itself: an empty or whitespace transcript would compare
    // equal to an empty reference and prove nothing.
    assert!(
        want.trim().len() > 20,
        "reference transcript is too short to be a meaningful check"
    );
}

/// The whole port, end to end: raw audio through ocelotl's own frontend,
/// subsampler, encoder and decoder, gated token-exact against the reference.
///
/// Every other test in the Parakeet suite starts from a reference tensor so a
/// stage's error is isolated. This one does the opposite on purpose — it is the
/// only test where all four stages' errors compound, and compounding is exactly
/// the thing the staged gates cannot see. It is also the only test that would
/// catch a stage being correct in isolation but wired up wrong.
#[test]
#[ignore = "requires OCELOTL_PARAKEET_WEIGHTS + OCELOTL_PARAKEET_REF_DIR"]
fn parakeet_transcribes_audio_token_exactly_end_to_end() {
    let Some((weights_path, ref_dir)) = fixtures() else {
        return;
    };
    let decode = ref_dir.join("decode");
    let want_tokens: Vec<u32> = read_i32(&decode.join("tokens.i32"))
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let want_frames: Vec<usize> = read_i32(&decode.join("frames.i32"))
        .into_iter()
        .map(|v| v as usize)
        .collect();

    let audio = read_f32(&ref_dir.join("parity_jfk_audio.f32"));
    let backend = backend();
    let model = ParakeetModel::from_safetensors(&weights_path).expect("load model");

    let (hidden, frames) = model.encode_audio(&audio, &backend).expect("encode");
    assert_eq!(
        frames, REF_FRAMES,
        "encoder frame count drifted — an off-by-one on the time axis survives \
         a passing value diff and desynchronizes the transducer"
    );

    // Report how far ocelotl's own encoder output sits from the reference. This
    // is the number that decides whether the token-exact gate is sustainable:
    // it must stay far below the 0.1824 duration-argmax margin.
    let reference = reference_encoder_time_major(&ref_dir);
    let worst = hidden
        .iter()
        .zip(reference.iter())
        .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("PARAKEET_E2E encoder_max_abs={worst:.3e} frames={frames}");

    let got = model.decode_audio(&audio, &backend).expect("decode");
    assert_eq!(
        got.tokens, want_tokens,
        "end-to-end tokens differ from the reference decode"
    );
    assert_eq!(
        got.frames, want_frames,
        "end-to-end frame indices differ despite matching tokens"
    );

    if let Some(tokenizer_path) = env_path("OCELOTL_PARAKEET_TOKENIZER") {
        let tokenizer = JsonTokenizer::from_json_path(&tokenizer_path).expect("tokenizer.json");
        let ids: Vec<TokenId> = got.tokens.iter().map(|t| TokenId(*t)).collect();
        let text = tokenizer.decode(&ids).expect("detokenize");
        let want = std::fs::read_to_string(decode.join("text.txt")).expect("reference text");
        eprintln!("PARAKEET_E2E_TEXT {text:?}");
        assert_eq!(text.trim(), want.trim());
    }
}

/// Sanity check that the token ids really are outside the duration band. A
/// decode that accidentally argmaxed over all 8198 columns would still produce
/// plausible text most of the time; this makes that impossible to miss.
#[test]
#[ignore = "requires OCELOTL_PARAKEET_REF_DIR"]
fn reference_tokens_never_fall_in_the_duration_band() {
    let Some(ref_dir) = env_path("OCELOTL_PARAKEET_REF_DIR") else {
        eprintln!("skipping");
        return;
    };
    let tokens = read_i32(&ref_dir.join("decode").join("tokens.i32"));
    assert!(!tokens.is_empty());
    for t in tokens {
        assert!(
            (t as usize) < VOCAB_SIZE,
            "token {t} lies in the duration band [{VOCAB_SIZE}, {JOINT_WIDTH})"
        );
    }
}
