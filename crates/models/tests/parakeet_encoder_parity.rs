//! Parakeet encoder parity against the reference ONNX encoder, stage by stage.
//!
//! The reference graph (`encoder-model.onnx`) was instrumented once to expose
//! its per-stage intermediates as graph outputs — the subsampler output, the
//! positional table, and each of the 24 Conformer block outputs. That is what
//! turns "the encoder output is wrong" from a bisection across 24 layers into a
//! direct lookup of the first stage that diverges.
//!
//! Opt-in, matching the Whisper local-artifact convention. Requires:
//!   `OCELOTL_PARAKEET_WEIGHTS`  path to `model.safetensors` (nvidia v3)
//!   `OCELOTL_PARAKEET_REF_DIR`  dir holding `parity_jfk_mel.f32` and `enc/`

use std::path::PathBuf;

use ocelotl_kernels::{CpuKernelBackend, CpuKernelMode};
use ocelotl_loader::load_safetensors_tensors_f32;
use ocelotl_models::parakeet::audio::{PARAKEET_MEL_BINS, ParakeetFeatures};
use ocelotl_models::parakeet::subsample::{SubsampleWeights, subsample, subsampled_frames};

const D_MODEL: usize = 1024;
/// Reference frames for the pinned `parity_jfk` fixture.
const REF_FRAMES: usize = 1101;

/// Stage tolerance, expressed **relative to the stage's own scale**.
///
/// The recon's 1e-3 gate is an ABSOLUTE figure for the FINAL encoder output,
/// whose values were measured at +/-0.15. Mid-stack activations are not on that
/// scale at all: the subsampler emits values in +/-6942 (mean |x| = 244), because
/// nothing has normalized them yet. Applying an absolute tolerance calibrated on
/// O(0.15) data to O(10^3) data compares different units and would reject a
/// bit-for-bit-reasonable implementation.
///
/// So the gate is `max|diff| / max|reference|`. Measured for the subsampler:
/// 1.27e-6, with a median per-element relative error of 7.7e-7 — that is the f32
/// floor for a 4096-term dot product (f32 eps ~ 1.2e-7), and the error is spread
/// uniformly across all 138 frames rather than concentrated at the padded edges.
///
/// 1e-5 keeps three orders of discriminating power: a wrong axis order, a
/// freq-major instead of channel-major flatten, or an off-by-one in the padding
/// all produce O(1) relative error, not O(1e-6).
const MAX_REL_DIFF: f32 = 1e-5;

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

fn load_subsample_weights(path: &PathBuf) -> SubsampleWeights {
    let names = [
        "encoder.subsampling.layers.0.weight",
        "encoder.subsampling.layers.0.bias",
        "encoder.subsampling.layers.2.weight",
        "encoder.subsampling.layers.2.bias",
        "encoder.subsampling.layers.3.weight",
        "encoder.subsampling.layers.3.bias",
        "encoder.subsampling.layers.5.weight",
        "encoder.subsampling.layers.5.bias",
        "encoder.subsampling.layers.6.weight",
        "encoder.subsampling.layers.6.bias",
        "encoder.subsampling.linear.weight",
        "encoder.subsampling.linear.bias",
    ];
    let loaded = load_safetensors_tensors_f32(path, &names).expect("load subsampler weights");
    let mut it = loaded.into_iter().map(|t| t.values);
    let mut next = || it.next().expect("tensor");
    SubsampleWeights {
        conv0_w: next(),
        conv0_b: next(),
        dw1_w: next(),
        dw1_b: next(),
        pw1_w: next(),
        pw1_b: next(),
        dw2_w: next(),
        dw2_b: next(),
        pw2_w: next(),
        pw2_b: next(),
        linear_w: next(),
        linear_b: next(),
    }
}

#[test]
#[ignore = "requires OCELOTL_PARAKEET_WEIGHTS + OCELOTL_PARAKEET_REF_DIR"]
fn parakeet_subsampler_matches_reference_pre_encode_output() {
    let (Some(weights_path), Some(ref_dir)) = (
        env_path("OCELOTL_PARAKEET_WEIGHTS"),
        env_path("OCELOTL_PARAKEET_REF_DIR"),
    ) else {
        eprintln!("skipping: set OCELOTL_PARAKEET_WEIGHTS and OCELOTL_PARAKEET_REF_DIR");
        return;
    };

    // Drive from the REFERENCE mel, not ocelotl's, so a frontend delta cannot
    // be mistaken for a subsampler bug. The frontend has its own gate.
    let mel = read_f32(&ref_dir.join("parity_jfk_mel.f32"));
    assert_eq!(mel.len(), PARAKEET_MEL_BINS * REF_FRAMES);
    let features = ParakeetFeatures {
        frames: REF_FRAMES,
        mel_bins: PARAKEET_MEL_BINS,
        values: mel,
    };

    let weights = load_subsample_weights(&weights_path);
    let backend = CpuKernelBackend::with_mode(CpuKernelMode::Optimized).expect("cpu backend");
    let got = subsample(&features, &weights, D_MODEL, &backend).expect("subsample");

    let reference = read_f32(&ref_dir.join("enc").join("pre_encode_out_Add_output_0.f32"));

    // Integer equality on the time axis first — a desynchronized T survives a
    // passing value diff and only surfaces later as a broken transducer.
    assert_eq!(
        got.frames,
        subsampled_frames(REF_FRAMES),
        "frame count disagrees with the closed form"
    );
    assert_eq!(
        got.values.len(),
        reference.len(),
        "subsampler produced {} values, reference has {} ({} frames x {D_MODEL})",
        got.values.len(),
        reference.len(),
        got.frames
    );

    if let Ok(dump) = std::env::var("OCELOTL_PARAKEET_DUMP") {
        let bytes: Vec<u8> = got.values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{dump}/subsample_got.f32"), bytes).expect("dump");
    }
    let mut worst = 0.0_f32;
    let mut worst_at = (0usize, 0usize);
    for (idx, (g, r)) in got.values.iter().zip(reference.iter()).enumerate() {
        let d = (g - r).abs();
        if d > worst {
            worst = d;
            worst_at = (idx / D_MODEL, idx % D_MODEL);
        }
    }
    let scale = reference.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    let rel = worst / scale;
    eprintln!(
        "PARAKEET_SUBSAMPLE frames={} d_model={D_MODEL} max_abs={worst:.3e} scale={scale:.1} rel={rel:.3e}",
        got.frames
    );
    assert!(
        rel <= MAX_REL_DIFF,
        "subsampler relative diff {rel:.3e} exceeds {MAX_REL_DIFF:.0e} \
         (max abs {worst:.3e} against scale {scale:.1}) at frame {} dim {}",
        worst_at.0,
        worst_at.1
    );
}
