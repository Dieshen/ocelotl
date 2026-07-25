//! Parakeet frontend parity against the exported NeMo reference graph.
//!
//! Oracle: `nemo128.onnx` from `istupakov/parakeet-tdt-0.6b-v3-onnx` (a stock
//! `ASRModel.export()` of the official checkpoint), run once to dump reference
//! tensors. This test replays those tensors — it does not need ONNX, Python, or
//! the 2.5 GB model at run time.
//!
//! Opt-in, following the same convention as the Whisper local-artifact proofs:
//! reference dumps are not committed. Point `OCELOTL_PARAKEET_REF_DIR` at a
//! directory holding, per fixture:
//!   `<name>_audio.f32`  raw little-endian f32 PCM (the decoded fixture wav)
//!   `<name>_mel.f32`    raw little-endian f32, mel-major `[mel_bins * frames]`
//!   `manifest.json`     `{"<name>": {"samples":N,"mel_bins":M,"frames":T}, ...}`
//!
//! Three fixtures, not one — per the fixture-per-failure-mode rule. `jfk` alone
//! cannot catch a per-feature-normalization bug, because that failure only shows
//! across clips of differing length and level:
//!   `parity_jfk`         ~11 s, ordinary speech level.
//!   `parity_short`       0.75 s — the frame-count off-by-one and the unbiased
//!                        `T-1` denominator at small `T`.
//!   `parity_long_quiet`  121 s at ~-30 dBFS — per-feature statistics and the
//!                        `T` arithmetic at scale.
//!
//! The frame count is asserted as an **integer equality**, never a tolerance: a
//! time-axis off-by-one survives a passing value diff and only surfaces later as
//! a desynchronized transducer.

use std::path::{Path, PathBuf};

use ocelotl_models::parakeet::audio::{frame_count, parakeet_log_mel};

/// Value tolerance, calibrated against the measured precision floor rather than
/// picked round — and justified by its measured downstream effect, not asserted.
///
/// ocelotl computes a naive DFT; the reference uses an FFT. Three successive
/// precision tightenings (exact integer range-reduction of the DFT phase, f64
/// accumulators in the transform and mel projection, f64 accumulators in the
/// per-feature normalization) moved the residual 3.5e-2 -> 1.75e-4 -> 1.60e-4
/// -> 1.57e-4. The first bought 200x; the last two bought 2% between them, and
/// the worst-offending mel band moved (127 -> 0 -> 111) instead of staying put.
/// Diminishing returns plus a wandering worst case is the signature of a
/// representation floor, not a remaining bug.
///
/// What settles it is the downstream measurement, not the argument: feeding
/// ocelotl's mel and the reference mel through the SAME reference encoder graph
/// gives a max encoder-output difference of 1.0e-5 — 100x inside the 1e-3
/// encoder gate, and 15x SMALLER than the frontend delta itself, i.e. the error
/// attenuates rather than amplifies.
///
/// 2.5e-4 therefore sits above the floor while retaining enormous discriminating
/// power: the real bug this test caught (an unreduced f32 DFT phase) showed up at
/// 3.5e-2, more than two orders of magnitude above this line.
const MAX_ABS_DIFF: f32 = 2.5e-4;

struct Fixture {
    name: &'static str,
    audio: Vec<f32>,
    mel: Vec<f32>,
    mel_bins: usize,
    frames: usize,
}

fn ref_dir() -> Option<PathBuf> {
    std::env::var("OCELOTL_PARAKEET_REF_DIR")
        .ok()
        .map(PathBuf::from)
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "{} is not f32-aligned", path.display());
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn load(dir: &Path, name: &'static str) -> Fixture {
    let manifest = std::fs::read_to_string(dir.join("manifest.json")).expect("manifest.json");
    // Minimal extraction — avoids a serde_json dependency in this test.
    let entry = manifest
        .split(&format!("\"{name}\""))
        .nth(1)
        .unwrap_or_else(|| panic!("manifest has no entry for {name}"));
    let field = |key: &str| -> usize {
        entry
            .split(&format!("\"{key}\""))
            .nth(1)
            .and_then(|s| s.split([',', '}']).next())
            .and_then(|s| s.trim().trim_start_matches(':').trim().parse().ok())
            .unwrap_or_else(|| panic!("manifest {name}.{key} unreadable"))
    };
    Fixture {
        name,
        audio: read_f32(&dir.join(format!("{name}_audio.f32"))),
        mel: read_f32(&dir.join(format!("{name}_mel.f32"))),
        mel_bins: field("mel_bins"),
        frames: field("frames"),
    }
}

fn assert_frontend_parity(f: &Fixture) {
    // 1. Frame count: integer equality against the reference, and against the
    //    closed form. A tolerance here would hide the off-by-one entirely.
    assert_eq!(
        frame_count(f.audio.len()),
        f.frames,
        "{}: frame_count({}) disagrees with the reference",
        f.name,
        f.audio.len()
    );

    let got = parakeet_log_mel(&f.audio).expect("frontend must run");
    assert_eq!(got.frames, f.frames, "{}: frame count", f.name);
    assert_eq!(got.mel_bins, f.mel_bins, "{}: mel bins", f.name);
    assert_eq!(
        got.values.len(),
        f.mel.len(),
        "{}: feature element count",
        f.name
    );

    // 2. Values, with the worst offender located rather than just counted —
    //    a bad mel band and a bad time frame need different debugging.
    let mut worst = 0.0_f32;
    let mut worst_at = (0usize, 0usize);
    for (idx, (g, r)) in got.values.iter().zip(f.mel.iter()).enumerate() {
        let diff = (g - r).abs();
        if diff > worst {
            worst = diff;
            worst_at = (idx / f.frames, idx % f.frames);
        }
    }
    assert!(
        worst <= MAX_ABS_DIFF,
        "{}: max abs diff {worst:.3e} exceeds {MAX_ABS_DIFF:.0e} at mel {} frame {} (of {}x{})",
        f.name,
        worst_at.0,
        worst_at.1,
        f.mel_bins,
        f.frames
    );
    eprintln!(
        "PARAKEET_FRONTEND {} frames={} mel_bins={} max_abs_diff={worst:.3e}",
        f.name, f.frames, f.mel_bins
    );
}

#[test]
#[ignore = "requires OCELOTL_PARAKEET_REF_DIR with nemo128.onnx reference dumps"]
fn parakeet_frontend_matches_nemo_reference_on_all_fixtures() {
    let Some(dir) = ref_dir() else {
        eprintln!("skipping: OCELOTL_PARAKEET_REF_DIR unset");
        return;
    };
    for name in ["parity_jfk", "parity_short", "parity_long_quiet"] {
        assert_frontend_parity(&load(&dir, name));
    }
}

/// Debug aid: dump ocelotl's features next to the reference for per-band diffing.
#[test]
#[ignore = "debug dump; requires OCELOTL_PARAKEET_REF_DIR and OCELOTL_PARAKEET_DUMP"]
fn parakeet_frontend_dump() {
    let (Some(dir), Ok(out)) = (ref_dir(), std::env::var("OCELOTL_PARAKEET_DUMP")) else {
        return;
    };
    for name in ["parity_jfk", "parity_short"] {
        let f = load(&dir, name);
        let got = parakeet_log_mel(&f.audio).expect("frontend");
        let bytes: Vec<u8> = got.values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{out}/{name}_got.f32"), bytes).expect("write dump");
    }
}
