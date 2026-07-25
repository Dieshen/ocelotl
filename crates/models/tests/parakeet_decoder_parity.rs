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

use std::path::{Path, PathBuf};

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

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_i32(path: &Path) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn load_prednet(path: &Path) -> PredNetWeights {
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

fn load_joint(path: &Path) -> JointWeights {
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
fn reference_encoder_time_major(ref_dir: &Path) -> Vec<f32> {
    let channel_major = read_f32(&ref_dir.join("enc").join("outputs.f32"));
    assert_eq!(
        channel_major.len(),
        ENC_HIDDEN * REF_FRAMES,
        "reference encoder output is not {ENC_HIDDEN}x{REF_FRAMES}"
    );
    transpose_2d(&channel_major, ENC_HIDDEN, REF_FRAMES)
}

/// Levenshtein distance over token ids, normalized by the reference length.
///
/// The ASR-standard measure, applied at the token level. Position-wise equality
/// is not a substitute: one insertion near the start of a 400-token sequence
/// misaligns everything after it and reports a catastrophic-looking number for
/// two transcripts that differ by a single word.
fn token_error_rate(reference: &[u32], hypothesis: &[u32]) -> f64 {
    if reference.is_empty() {
        return if hypothesis.is_empty() { 0.0 } else { 1.0 };
    }
    // Single-row DP; only the previous row is ever needed.
    let mut prev: Vec<usize> = (0..=hypothesis.len()).collect();
    let mut curr = vec![0usize; hypothesis.len() + 1];
    for (i, r) in reference.iter().enumerate() {
        curr[0] = i + 1;
        for (j, h) in hypothesis.iter().enumerate() {
            let cost = usize::from(r != h);
            curr[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(curr[j] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[hypothesis.len()] as f64 / reference.len() as f64
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

/// Chunked transcription must stay close to the unchunked decode.
///
/// The chunk size is deliberately **absurd**: 48 frames of body with 25 frames
/// of context, i.e. ~3.8 s chunks over an 11 s clip, forcing three windows and
/// two interior seams. Nobody would run it this way. That is the point — the
/// production default (375/62) puts this whole fixture in a single window,
/// where the chunking code does nothing and the test would prove nothing.
///
/// A test that exercises the boundary only at realistic settings is a test that
/// cannot fail, so this one manufactures the boundaries.
#[test]
#[ignore = "requires OCELOTL_PARAKEET_WEIGHTS + OCELOTL_PARAKEET_REF_DIR"]
fn parakeet_chunked_transcription_stays_close_to_the_unchunked_decode() {
    let Some((weights_path, ref_dir)) = fixtures() else {
        return;
    };
    let want_tokens: Vec<u32> = read_i32(&ref_dir.join("decode").join("tokens.i32"))
        .into_iter()
        .map(|v| v as u32)
        .collect();

    let audio = read_f32(&ref_dir.join("parity_jfk_audio.f32"));
    let backend = backend();
    let model = ParakeetModel::from_safetensors(&weights_path).expect("load model");

    const CHUNK: usize = 48;
    const CONTEXT: usize = 25;

    let (single, single_frames) = model.encode_audio(&audio, &backend).expect("single window");
    let (stitched, stitched_frames) = model
        .encode_audio_chunked(&audio, CHUNK, CONTEXT, &backend)
        .expect("chunked");

    // Frame count first. A stitching bug that drops or duplicates a frame shifts
    // the time axis for everything after it, and that is invisible in a value
    // diff computed over mismatched lengths.
    assert_eq!(
        stitched_frames, single_frames,
        "chunked encode produced {stitched_frames} frames, single window {single_frames}"
    );

    // How far the seams actually move the encoder output. This is the number
    // that says whether the context width is adequate: it must stay far below
    // the 0.1824 duration-argmax margin, or tokens will flip at boundaries.
    let worst = stitched
        .iter()
        .zip(single.iter())
        .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
    let scale = single.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    eprintln!(
        "PARAKEET_CHUNK chunk={CHUNK} context={CONTEXT} frames={stitched_frames} \
         max_abs={worst:.3e} scale={scale:.3}"
    );

    let got = model
        .decode_audio_chunked(&audio, CHUNK, CONTEXT, &backend)
        .expect("chunked decode");

    // Bounded degradation, NOT token equality.
    //
    // An earlier version of this test asserted equality and passed, which was
    // luck rather than a property: chunking a full-attention encoder truncates
    // every frame's receptive field, so it is lossy by construction (see
    // `parakeet::chunk`). Demanding exactness here contradicts the module's own
    // measured finding, and it duly broke the moment attention's accumulation
    // order changed — an f32-level shift, not a regression.
    //
    // 3.8 s chunks with 2 s of context is an abusive setting chosen to force
    // three windows out of an 11 s clip; the production defaults put this whole
    // fixture in one window. The gate is set well inside what a real stitching
    // bug produces: a dropped or duplicated frame desynchronizes the transducer
    // and drives TER toward 1.0, two orders above this line. The frame-count
    // assertion above is the exact one.
    let ter = token_error_rate(&want_tokens, &got.tokens);
    eprintln!("PARAKEET_CHUNK token_error_rate={ter:.4}");
    assert!(
        ter <= 0.15,
        "chunked decode has a {:.1}% token error rate against the reference \
         (max encoder delta {worst:.3e} across {} seams)",
        ter * 100.0,
        (single_frames.div_ceil(CHUNK)).saturating_sub(1)
    );

    // Guard the gate: if the plan collapsed to one window this test degenerates
    // into re-running the unchunked path and proves nothing about stitching.
    let windows =
        ocelotl_models::parakeet::chunk::plan_chunks(single_frames, CHUNK, CONTEXT).expect("plan");
    assert!(
        windows.len() >= 3,
        "expected at least 3 windows to exercise interior seams, got {}",
        windows.len()
    );
}

/// How much does chunking actually perturb the encoder, and where?
///
/// This exists because the test above passes while the stitched encoder output
/// differs from the unchunked one by ~50% of its own scale. Identical tokens
/// under that much movement is a result that needs explaining, not accepting:
/// either the decode is robust to encoder perturbation in a way worth knowing,
/// or the default context width is being chosen by luck.
///
/// Sweeps the context width and reports, for each, the worst deviation and
/// whether it is concentrated at the seams. Diagnostic — it asserts only that
/// more context helps monotonically, which is the property that would break if
/// the trimming logic were wrong (a bug there does not care how much context it
/// is given).
#[test]
#[ignore = "diagnostic: run alone, sweeps encoder passes"]
fn chunk_context_width_versus_seam_error() {
    let Some((weights_path, ref_dir)) = fixtures() else {
        return;
    };
    let audio = read_f32(&ref_dir.join("parity_jfk_audio.f32"));
    let backend = backend();
    let model = ParakeetModel::from_safetensors(&weights_path).expect("load model");
    let (single, frames) = model.encode_audio(&audio, &backend).expect("single");
    let d = 1024usize;

    const CHUNK: usize = 48;
    eprintln!("PARAKEET_CHUNK_SWEEP chunk={CHUNK} frames={frames} (seams at 48, 96)");
    eprintln!(
        "{:>7} {:>11} {:>11} {:>11} {:>8}",
        "context", "max_abs", "seam_max", "interior_max", "tokens"
    );

    let mut previous = f32::INFINITY;
    for context in [0usize, 12, 25, 50, 100] {
        let (stitched, n) = model
            .encode_audio_chunked(&audio, CHUNK, context, &backend)
            .expect("chunked");
        assert_eq!(n, frames);

        // Per-frame worst deviation, split by distance from a seam.
        let mut max_abs = 0.0_f32;
        let mut seam_max = 0.0_f32;
        let mut interior_max = 0.0_f32;
        for f in 0..frames {
            let worst = (0..d)
                .map(|k| (stitched[f * d + k] - single[f * d + k]).abs())
                .fold(0.0_f32, f32::max);
            max_abs = max_abs.max(worst);
            // Within 4 frames of a body boundary counts as "at a seam".
            let near_seam = (f % CHUNK) < 4 || (f % CHUNK) >= CHUNK - 4;
            if near_seam {
                seam_max = seam_max.max(worst);
            } else {
                interior_max = interior_max.max(worst);
            }
        }
        let decoded = model
            .decode_hidden(&stitched, frames, &backend)
            .expect("decode");
        eprintln!(
            "{context:>7} {max_abs:>11.3e} {seam_max:>11.3e} {interior_max:>11.3e} {:>8}",
            decoded.tokens.len()
        );

        // More context must not make it worse. If trimming were wrong — off-by-one
        // keep range, context not actually discarded — the error would be
        // insensitive to this knob, and that is what this catches.
        assert!(
            max_abs <= previous * 1.05,
            "context {context} gave {max_abs:.3e}, worse than the previous \
             width's {previous:.3e}; more context must not hurt"
        );
        previous = max_abs;
    }
}

/// Chunking at realistic settings on genuinely long audio.
///
/// The short-fixture tests cannot answer the question that matters, because on
/// 138 frames any context wide enough to be realistic simply swallows the whole
/// clip and the comparison degenerates. This runs the 121 s fixture (1513
/// frames) at the production defaults and reports what chunking actually costs.
///
/// Deliberately reports rather than asserts exactness: with full attention,
/// chunking changes the function being computed for *every* frame, so demanding
/// token equality would be demanding something the method cannot promise. What
/// it does assert is that the degradation is small and bounded.
#[test]
#[ignore = "slow: encodes 121 s twice. Run alone."]
fn chunking_long_audio_at_default_settings() {
    use ocelotl_models::parakeet::chunk::{DEFAULT_CHUNK_FRAMES, DEFAULT_CONTEXT_FRAMES};
    let Some((weights_path, ref_dir)) = fixtures() else {
        return;
    };
    let audio = read_f32(&ref_dir.join("parity_long_quiet_audio.f32"));
    // Threaded: the unchunked 1513-frame reference encode is the expensive half
    // of this test, and there is no reason to measure it serially now that
    // row-parallel dispatch exists and is bit-exact.
    let backend = CpuKernelBackend::with_mode_and_threads(
        CpuKernelMode::Avx2,
        std::thread::available_parallelism().map_or(1, |n| n.get()),
    )
    .unwrap_or_else(|_| backend());
    let model = ParakeetModel::from_safetensors(&weights_path).expect("load model");
    eprintln!("PARAKEET_LONG audio={:.1}s", audio.len() as f32 / 16_000.0);

    let t0 = std::time::Instant::now();
    let (single, frames) = model
        .encode_audio_chunked(&audio, usize::MAX, 0, &backend)
        .expect("unchunked");
    let single_secs = t0.elapsed().as_secs_f64();
    let reference = model
        .decode_hidden(&single, frames, &backend)
        .expect("decode");

    let t1 = std::time::Instant::now();
    let got = model
        .decode_audio_chunked(
            &audio,
            DEFAULT_CHUNK_FRAMES,
            DEFAULT_CONTEXT_FRAMES,
            &backend,
        )
        .expect("chunked");
    let chunked_secs = t1.elapsed().as_secs_f64();

    // Token error rate by EDIT DISTANCE, not position-wise equality.
    //
    // Position-wise comparison is a broken proxy here and it flatters nothing:
    // the two decodes have different lengths (a handful of insertions), and a
    // single insertion misaligns every token after it, so it reported 84%
    // agreement for sequences that are in fact nearly identical. Edit distance
    // is the quantity that actually corresponds to "how much worse is the
    // transcript".
    let ter = token_error_rate(&reference.tokens, &got.tokens);
    eprintln!(
        "PARAKEET_LONG frames={frames} unchunked={single_secs:.1}s ({} tokens, \
         RTF {:.2}) chunked={chunked_secs:.1}s ({} tokens, RTF {:.2}) \
         speedup={:.2}x token_error_rate={:.4}",
        reference.tokens.len(),
        single_secs / 121.0,
        got.tokens.len(),
        chunked_secs / 121.0,
        single_secs / chunked_secs,
        ter
    );

    assert!(
        !reference.tokens.is_empty(),
        "the reference decode is empty, so any error rate would be vacuous"
    );
    assert!(
        ter <= 0.05,
        "chunked decode has a {:.2}% token error rate against the unchunked one",
        ter * 100.0
    );
    // NOT asserting that chunking is faster here — at 121 s it is not, and that
    // is the correct outcome.
    //
    // Chunking was built when the encoder's quadratic term was 97% of runtime at
    // this length. That term turned out to be 97% *unvectorized kernel*: once
    // attention became a threaded GEMM the same encode went 493.7 s -> 11.7 s and
    // the quadratic share fell to ~3%. Chunking's ~33% context re-encoding
    // overhead now dominates whatever it saves, so it runs ~12% slower AND costs
    // 4.2% token error. Refitting the threaded encoder puts break-even near
    // 30 000 frames — about **40 minutes** of audio.
    //
    // The lesson this bakes in: chunking is a workaround, and a workaround
    // benchmarked against a broken baseline looks like a win. Asserting a
    // speedup here would lock that illusion into the test suite.
    let ratio = single_secs / chunked_secs;
    eprintln!(
        "PARAKEET_LONG chunking is {ratio:.2}x the unchunked speed at this length \
         (expected < 1.0 below ~40 min of audio)"
    );
}

/// Chunking with a window larger than the audio must be bit-identical to not
/// chunking at all — the same code path, not merely a similar answer.
#[test]
#[ignore = "requires OCELOTL_PARAKEET_WEIGHTS + OCELOTL_PARAKEET_REF_DIR"]
fn a_single_oversized_chunk_is_bit_identical_to_the_unchunked_encode() {
    let Some((weights_path, ref_dir)) = fixtures() else {
        return;
    };
    let audio = read_f32(&ref_dir.join("parity_jfk_audio.f32"));
    let backend = backend();
    let model = ParakeetModel::from_safetensors(&weights_path).expect("load model");

    let (single, n) = model.encode_audio(&audio, &backend).expect("single");
    let (chunked, m) = model
        .encode_audio_chunked(&audio, 100_000, 62, &backend)
        .expect("chunked");
    assert_eq!(n, m);
    assert_eq!(
        single, chunked,
        "an oversized single chunk must be bit-identical, not merely close"
    );
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
