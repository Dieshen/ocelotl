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

/// Absolute gate for stages whose output is already normalized — chiefly the
/// final block, whose values are +/-0.15. This is the recon's encoder figure,
/// used in the units it was actually calibrated in.
const MAX_ABS_ENCODER: f32 = 1e-3;

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

fn load_block_weights(path: &PathBuf, i: usize) -> ocelotl_models::parakeet::encoder::BlockWeights {
    use ocelotl_models::parakeet::encoder::BlockWeights;
    let p = format!("encoder.layers.{i}");
    let names: Vec<String> = [
        "norm_feed_forward1.weight",
        "norm_feed_forward1.bias",
        "feed_forward1.linear1.weight",
        "feed_forward1.linear2.weight",
        "norm_self_att.weight",
        "norm_self_att.bias",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.relative_k_proj.weight",
        "self_attn.bias_u",
        "self_attn.bias_v",
        "norm_conv.weight",
        "norm_conv.bias",
        "conv.pointwise_conv1.weight",
        "conv.depthwise_conv.weight",
        "conv.norm.running_mean",
        "conv.norm.running_var",
        "conv.norm.weight",
        "conv.norm.bias",
        "conv.pointwise_conv2.weight",
        "norm_feed_forward2.weight",
        "norm_feed_forward2.bias",
        "feed_forward2.linear1.weight",
        "feed_forward2.linear2.weight",
        "norm_out.weight",
        "norm_out.bias",
    ]
    .iter()
    .map(|s| format!("{p}.{s}"))
    .collect();
    let loaded = load_safetensors_tensors_f32(path, &names).expect("load block weights");
    let mut it = loaded.into_iter().map(|t| t.values);
    let mut n = || it.next().expect("tensor");
    BlockWeights {
        norm_ff1_w: n(),
        norm_ff1_b: n(),
        ff1_l1: n(),
        ff1_l2: n(),
        norm_attn_w: n(),
        norm_attn_b: n(),
        q_proj: n(),
        k_proj: n(),
        v_proj: n(),
        o_proj: n(),
        pos_proj: n(),
        bias_u: n(),
        bias_v: n(),
        norm_conv_w: n(),
        norm_conv_b: n(),
        pw1: n(),
        dw: n(),
        bn_mean: n(),
        bn_var: n(),
        bn_w: n(),
        bn_b: n(),
        pw2: n(),
        norm_ff2_w: n(),
        norm_ff2_b: n(),
        ff2_l1: n(),
        ff2_l2: n(),
        norm_out_w: n(),
        norm_out_b: n(),
    }
}

/// Walk all 24 Conformer blocks, checking each against the reference block
/// output. Because the instrumented graph exposes every block boundary, a
/// divergence is located directly instead of bisected across the stack.
#[test]
#[ignore = "requires OCELOTL_PARAKEET_WEIGHTS + OCELOTL_PARAKEET_REF_DIR"]
fn parakeet_encoder_blocks_match_reference_stage_by_stage() {
    use ocelotl_models::parakeet::encoder::{EncoderShape, block_forward};
    let (Some(wp), Some(rd)) = (
        env_path("OCELOTL_PARAKEET_WEIGHTS"),
        env_path("OCELOTL_PARAKEET_REF_DIR"),
    ) else {
        eprintln!("skipping");
        return;
    };
    let enc = rd.join("enc");
    // Start from the REFERENCE subsampler output so block error is isolated.
    let mut x = read_f32(&enc.join("pre_encode_out_Add_output_0.f32"));
    let pos = read_f32(&enc.join("pos_enc_Slice_output_0.f32"));
    let rows = x.len() / D_MODEL;
    assert_eq!(rows, 138);
    assert_eq!(pos.len(), (2 * rows - 1) * D_MODEL);

    let shape = EncoderShape::default();
    let backend = CpuKernelBackend::with_mode(CpuKernelMode::Optimized).expect("backend");
    let mut worst_overall = 0.0_f32;
    for i in 0..24 {
        let w = load_block_weights(&wp, i);
        block_forward(&mut x, rows, &pos, shape, &w, &backend).expect("block");
        let r = read_f32(&enc.join(format!(
            "layers.{i}_norm_out_LayerNormalization_output_0.f32"
        )));
        assert_eq!(x.len(), r.len(), "block {i} shape");
        let worst = x
            .iter()
            .zip(r.iter())
            .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
        let scale = r.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        let rel = worst / scale;
        worst_overall = worst_overall.max(rel);
        eprintln!("PARAKEET_BLOCK {i:2} abs={worst:.3e} scale={scale:7.2} rel={rel:.3e}");
        // Two legitimate criteria, and a stage passes on EITHER. Mid-stack
        // activations run to O(100-500) and are judged relatively; the FINAL
        // block is normalized down to ~0.15, so dividing by that scale inflates
        // its relative figure even though its ABSOLUTE error (1.5e-5) is the
        // smallest in the whole stack and sits 66x inside the recon's 1e-3
        // encoder gate — a gate calibrated for exactly this +/-0.15 output.
        // A real structural bug fails BOTH by orders of magnitude: it would put
        // the absolute error at O(scale), i.e. ~1e-1 here, 100x over.
        assert!(
            rel <= MAX_REL_DIFF || worst <= MAX_ABS_ENCODER,
            "block {i}: relative {rel:.3e} over {MAX_REL_DIFF:.0e} AND absolute \
             {worst:.3e} over {MAX_ABS_ENCODER:.0e} (scale {scale:.2})"
        );
    }
    eprintln!("PARAKEET_ENCODER worst_rel_over_24_blocks={worst_overall:.3e}");
}
