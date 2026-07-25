//! Parakeet CPU benchmark: per-stage timings and real-time factor.
//!
//! Reports the number that matters for ASR — **RTF**, wall-clock seconds per
//! second of audio — broken down by stage, so a regression names its own cause.
//!
//! # Reading these numbers honestly
//!
//! Three traps this harness is built to avoid, all of which have bitten this
//! codebase before:
//!
//! - **Run it alone.** Timed runs must not share the machine with a build, a
//!   test suite, or another benchmark. The harness refuses to *enforce* that,
//!   because it cannot see the rest of the system — so it is `#[ignore]`d and
//!   the runbook says to run it on an idle box.
//! - **Warm up, then take the minimum.** The first iteration pays page faults on
//!   2.4 GB of freshly-loaded weights. The minimum over repeats is the
//!   least-noisy estimator of the underlying cost; the mean mostly measures
//!   whatever else the machine was doing.
//! - **Do not report the load time as inference.** Reading and parsing the
//!   safetensors archive dominates a single short utterance and has nothing to
//!   do with throughput, so it is measured and reported separately.
//!
//! The backend choice is itself a benchmark result. `CpuKernelMode::Optimized`
//! is safe-Rust scalar; `Avx2` is the microkernel. The measured gap on the
//! 24-block encoder is **105x**, so a run that silently picked the wrong one
//! would not look like a regression — it would look like a hang. This harness
//! prints which it used.
//!
//! Opt-in:
//!   `OCELOTL_PARAKEET_WEIGHTS`   path to `model.safetensors`
//!   `OCELOTL_PARAKEET_REF_DIR`   dir holding the `parity_*_audio.f32` fixtures
//!   `OCELOTL_PARAKEET_BENCH_REPEATS`  iterations per fixture (default 3)

use std::path::PathBuf;
use std::time::Instant;

use ocelotl_kernels::{CpuKernelBackend, CpuKernelMode, KernelBackend};
use ocelotl_models::parakeet::audio::{PARAKEET_SAMPLE_RATE_HZ, parakeet_log_mel};
use ocelotl_models::parakeet::decoder::{TdtConfig, greedy_decode};
use ocelotl_models::parakeet::encoder::{EncoderShape, encode};
use ocelotl_models::parakeet::model::{MAX_AUDIO_SECONDS, ParakeetModel};
use ocelotl_models::parakeet::subsample::subsample;

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key).ok().map(PathBuf::from)
}

fn read_f32(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn pick_backend() -> (CpuKernelBackend, &'static str) {
    match CpuKernelBackend::with_mode(CpuKernelMode::Avx2) {
        Ok(b) => (b, "avx2"),
        Err(_) => (
            CpuKernelBackend::with_mode(CpuKernelMode::Optimized).expect("cpu backend"),
            "optimized(scalar)",
        ),
    }
}

/// Backend with a rayon pool of `threads` workers.
///
/// `linear_out_by_in` dispatches to the pool on its own once one exists (above
/// 32 rows, which the encoder always clears at 138), so this is the whole of
/// "turn on multi-threading" for the Parakeet path — the machinery was already
/// in the kernels crate and simply was not being constructed.
fn threaded_backend(threads: usize) -> CpuKernelBackend {
    CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Avx2, threads)
        .or_else(|_| CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Optimized, threads))
        .expect("threaded cpu backend")
}

/// Minimum of `repeats` timings of `f`, in seconds, after one warm-up call.
fn best_of<T>(repeats: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut out = f(); // warm-up: page-faults the weights, primes caches
    let mut best = f64::INFINITY;
    for _ in 0..repeats {
        let start = Instant::now();
        out = f();
        best = best.min(start.elapsed().as_secs_f64());
    }
    (best, out)
}

#[test]
#[ignore = "benchmark: run alone on an idle machine, with OCELOTL_PARAKEET_* set"]
fn parakeet_cpu_benchmark_reports_per_stage_rtf() {
    let (Some(weights_path), Some(ref_dir)) = (
        env_path("OCELOTL_PARAKEET_WEIGHTS"),
        env_path("OCELOTL_PARAKEET_REF_DIR"),
    ) else {
        eprintln!("skipping: set OCELOTL_PARAKEET_WEIGHTS and OCELOTL_PARAKEET_REF_DIR");
        return;
    };
    let repeats: usize = std::env::var("OCELOTL_PARAKEET_BENCH_REPEATS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);

    // Threads default to the host's parallelism; set to 1 to measure serial.
    let threads: usize = std::env::var("OCELOTL_PARAKEET_BENCH_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    let (probe, backend_name) = pick_backend();
    drop(probe);
    let backend = threaded_backend(threads);
    let load_start = Instant::now();
    let model = ParakeetModel::from_safetensors(&weights_path).expect("load model");
    let load_secs = load_start.elapsed().as_secs_f64();

    eprintln!(
        "PARAKEET_BENCH backend={backend_name} threads={threads} repeats={repeats} \
         load_s={load_secs:.2}"
    );
    eprintln!(
        "{:<18} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7}",
        "fixture", "audio_s", "mel_s", "sub_s", "enc_s", "dec_s", "total_s", "RTF"
    );

    // Long fixture last: it is the one that shows the encoder's quadratic
    // attention, and it is the one most likely to be skipped.
    for name in ["parity_short", "parity_jfk", "parity_long_quiet"] {
        let audio = read_f32(&ref_dir.join(format!("{name}_audio.f32")));
        let audio_secs = audio.len() as f64 / PARAKEET_SAMPLE_RATE_HZ as f64;
        if audio.len() > MAX_AUDIO_SECONDS * PARAKEET_SAMPLE_RATE_HZ as usize {
            eprintln!(
                "{name:<18} {audio_secs:>7.1}  skipped: over the {MAX_AUDIO_SECONDS}s \
                 full-attention limit (needs chunking, not implemented)"
            );
            continue;
        }

        let (mel_s, features) = best_of(repeats, || parakeet_log_mel(&audio).expect("mel"));
        let shape = EncoderShape::default();
        let (sub_s, subsampled) = best_of(repeats, || {
            subsample(
                &features,
                &model.weights().subsample,
                shape.d_model,
                &backend,
            )
            .expect("subsample")
        });
        let frames = subsampled.frames;
        let (enc_s, hidden) = best_of(repeats, || {
            let mut h = subsampled.values.clone();
            encode(&mut h, frames, shape, &model.weights().blocks, &backend).expect("encode");
            h
        });
        let (dec_s, decoded) = best_of(repeats, || {
            greedy_decode(
                &hidden,
                frames,
                &model.weights().prednet,
                &model.weights().joint,
                TdtConfig::default(),
                &backend,
            )
            .expect("decode")
        });

        // The encoder timing above clones the subsampler output each iteration,
        // which is a real allocation but not part of inference; it is small
        // relative to 24 blocks and is called out rather than hidden.
        let total = mel_s + sub_s + enc_s + dec_s;
        let rtf = total / audio_secs;
        eprintln!(
            "{name:<18} {audio_secs:>7.2} {mel_s:>8.3} {sub_s:>8.3} {enc_s:>8.3} \
             {dec_s:>8.3} {total:>8.3} {rtf:>7.4}"
        );
        eprintln!(
            "{:<18} frames={frames} tokens={} steps={}",
            "",
            decoded.tokens.len(),
            decoded.steps.len()
        );
        assert!(
            rtf.is_finite() && rtf > 0.0,
            "{name}: nonsensical RTF {rtf} — the timer measured nothing"
        );
    }
    eprintln!(
        "PARAKEET_BENCH note: model load ({load_secs:.2}s) is excluded from RTF; \
         it is a one-time cost, not throughput."
    );
}

/// How the encoder scales across threads.
///
/// Reports speedup and parallel efficiency per thread count, and asserts the
/// outputs still agree — a thread count that changed the answer would be a
/// correctness bug wearing a performance result's clothing. Row-parallel
/// dispatch keeps each output row on one worker with an unchanged accumulation
/// order, so agreement here should be *exact*, not merely close; the assertion
/// is written to catch it if that ever stops being true.
#[test]
#[ignore = "benchmark: run alone on an idle machine"]
fn parakeet_encoder_thread_scaling() {
    let (Some(weights_path), Some(ref_dir)) = (
        env_path("OCELOTL_PARAKEET_WEIGHTS"),
        env_path("OCELOTL_PARAKEET_REF_DIR"),
    ) else {
        eprintln!("skipping");
        return;
    };
    let model = ParakeetModel::from_safetensors(&weights_path).expect("load model");
    let audio = read_f32(&ref_dir.join("parity_jfk_audio.f32"));
    let features = parakeet_log_mel(&audio).expect("mel");
    let shape = EncoderShape::default();
    let base = threaded_backend(1);
    let subsampled =
        subsample(&features, &model.weights().subsample, shape.d_model, &base).expect("subsample");
    let frames = subsampled.frames;

    let run = |kernels: &dyn KernelBackend| {
        let mut best = f64::INFINITY;
        let mut out = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            let mut h = subsampled.values.clone();
            encode(&mut h, frames, shape, &model.weights().blocks, kernels).expect("encode");
            best = best.min(start.elapsed().as_secs_f64());
            out = h;
        }
        (best, out)
    };

    let (serial_s, serial_out) = run(&threaded_backend(1));
    eprintln!(
        "{:>8} {:>9} {:>9} {:>12}",
        "threads", "encode_s", "speedup", "efficiency"
    );
    eprintln!("{:>8} {serial_s:>9.3} {:>9} {:>12}", 1, "1.00x", "100%");

    for threads in [2usize, 4, 8, 12] {
        let (secs, out) = run(&threaded_backend(threads));
        let speedup = serial_s / secs;
        eprintln!(
            "{threads:>8} {secs:>9.3} {:>8.2}x {:>11.0}%",
            speedup,
            speedup / threads as f64 * 100.0
        );
        let worst = serial_out
            .iter()
            .zip(out.iter())
            .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
        assert_eq!(
            worst, 0.0,
            "{threads} threads changed the encoder output by {worst:.3e}; \
             row-parallel dispatch is supposed to preserve accumulation order \
             exactly, so any drift means work is being split somewhere it should \
             not be"
        );
    }
}

/// Compare the AVX2 microkernel against the portable path on one encoder pass.
///
/// Kept as its own test because it answers a question the aggregate cannot: how
/// much of the CPU story is the microkernel. The answer sets expectations for
/// any host without AVX2, and it is the reason the parity tests select `Avx2`
/// explicitly rather than trusting a mode whose name merely sounds fast.
#[test]
#[ignore = "benchmark: run alone on an idle machine"]
fn parakeet_encoder_avx2_versus_portable_backend() {
    let (Some(weights_path), Some(ref_dir)) = (
        env_path("OCELOTL_PARAKEET_WEIGHTS"),
        env_path("OCELOTL_PARAKEET_REF_DIR"),
    ) else {
        eprintln!("skipping");
        return;
    };
    let Ok(avx2) = CpuKernelBackend::with_mode(CpuKernelMode::Avx2) else {
        eprintln!("skipping: host has no AVX2");
        return;
    };
    let portable = CpuKernelBackend::with_mode(CpuKernelMode::Optimized).expect("portable backend");

    let model = ParakeetModel::from_safetensors(&weights_path).expect("load model");
    let audio = read_f32(&ref_dir.join("parity_jfk_audio.f32"));
    let features = parakeet_log_mel(&audio).expect("mel");
    let shape = EncoderShape::default();
    let subsampled =
        subsample(&features, &model.weights().subsample, shape.d_model, &avx2).expect("subsample");
    let frames = subsampled.frames;

    let run = |kernels: &dyn KernelBackend| {
        let start = Instant::now();
        let mut h = subsampled.values.clone();
        encode(&mut h, frames, shape, &model.weights().blocks, kernels).expect("encode");
        (start.elapsed().as_secs_f64(), h)
    };
    let (avx2_s, avx2_out) = run(&avx2);
    let (portable_s, portable_out) = run(&portable);

    // The speedup only means something if both paths computed the same thing.
    let worst = avx2_out
        .iter()
        .zip(portable_out.iter())
        .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!(
        "PARAKEET_BACKENDS avx2={avx2_s:.3}s portable={portable_s:.3}s \
         speedup={:.1}x max_abs_diff={worst:.3e}",
        portable_s / avx2_s
    );
    assert!(
        worst <= 1e-3,
        "AVX2 and portable encoders disagree by {worst:.3e}; the speedup is \
         meaningless if they are not computing the same function"
    );
}
