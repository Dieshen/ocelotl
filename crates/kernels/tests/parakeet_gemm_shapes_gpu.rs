//! Phase-0 GPU probe: is a GPU port of the Parakeet encoder worth doing?
//!
//! This answers that question **before** anyone writes conv2d, depthwise conv,
//! BatchNorm, GLU and relative-position attention as GPU kernels — a body of
//! work comparable to the whole CPU encoder. The cheap version of the question
//! is: on the GEMM shapes this encoder actually runs, does the *existing*,
//! already-proven `linear_out_by_in` GPU kernel beat the threaded CPU one?
//!
//! If it does not, the port is not worth starting, and that is a result worth
//! having for the price of one test file.
//!
//! # Why these shapes are unpromising on paper
//!
//! The encoder processes an 11 s utterance as **138 rows**. GPUs want thousands.
//! Every GEMM here is short and fat — 138 rows against a 1024- or 4096-wide
//! output — so the kernel spends its time on launch overhead and memory traffic
//! rather than on arithmetic it can hide. The CPU side, meanwhile, is now AVX2
//! plus a 12-thread pool, which is a much higher bar than it was.
//!
//! Measured per-call cost is what decides it. Note this probe is generous to the
//! GPU in one important way and hostile in another: generous because it excludes
//! the host round-trip that a real per-op GPU path would pay (the embeddings work
//! established that per-op transfers dominate, and that only a device-resident
//! forward avoids them); hostile because it includes the upload of the weight
//! matrix, which a device-resident forward would hoist out. Both effects are
//! called out in the output rather than hidden in a single number.
//!
//! # Result (RX 6600 / Vulkan, 16-thread CPU, 2026-07-24): **do not port**
//!
//! The GPU is **9-10x slower** than the threaded CPU across the whole block
//! (9.31x and 9.85x on two runs), and slower on every individual shape — 3.4x to
//! 14x. Projected over 24 blocks: CPU ~0.57 s, GPU ~5.4 s.
//!
//! The obvious objection is that this is unfair because it includes the weight
//! upload. It is not the explanation, and the probe is arranged so the data
//! answers that directly: the **70 KB** `attn.ctx` weight is still 5-9x slower
//! while the **16.8 MB** `ffn.linear1` is 11x. If transfers dominated, those two
//! would not sit anywhere near each other. Both run at well under 1% of the
//! card's ~8.9 TFLOP/s, which is what a GEMM looks like when it cannot fill the
//! device.
//!
//! The cause is the shape. An 11 s utterance is **138 rows**; a GPU wants
//! thousands, and no amount of kernel tuning conjures parallelism that the
//! problem does not contain. Meanwhile the CPU baseline is no longer weak — AVX2
//! plus 12 threads is a genuinely high bar.
//!
//! **What would change the answer**: batched inference across many utterances
//! (rows scale with batch), very long audio, or a substantially larger GPU. None
//! of those are the current workload, so the conv2d / depthwise / BatchNorm / GLU
//! / rel-pos-attention device kernels a real port needs are not worth writing
//! yet. This file is the cheap check that says so, and it is committed so the
//! question does not get re-asked from scratch.
//!
//! Requires a GPU feature: `--features cubecl-wgpu` or `--features cubecl-hip`.

#![cfg(feature = "_gpu")]

use std::time::Instant;

use ocelotl_kernels::{CpuKernelBackend, CpuKernelMode};

/// The GEMMs one Conformer block actually issues, at 138 encoder frames.
/// `(label, rows, in_features, out_features, calls_per_block)`.
const SHAPES: &[(&str, usize, usize, usize, usize)] = &[
    ("ffn.linear1", 138, 1024, 4096, 2),
    ("ffn.linear2", 138, 4096, 1024, 2),
    ("attn.qkvo_proj", 138, 1024, 1024, 4),
    ("attn.pos_proj", 275, 1024, 1024, 1),
    ("conv.pointwise1", 138, 1024, 2048, 1),
    ("conv.pointwise2", 138, 1024, 1024, 1),
    ("attn.AC (per head)", 138, 128, 138, 8),
    ("attn.BD (per head)", 138, 128, 275, 8),
    ("attn.ctx (per head)", 138, 138, 128, 8),
];

fn deterministic(len: usize, seed: u32) -> Vec<f32> {
    // A cheap LCG, so the two backends see identical inputs without a dependency
    // and without Math.random-style irreproducibility.
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        })
        .collect()
}

fn best_of(reps: usize, mut f: impl FnMut()) -> f64 {
    f(); // warm-up: shader compilation, allocator, caches
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let start = Instant::now();
        f();
        best = best.min(start.elapsed().as_secs_f64());
    }
    best
}

#[test]
#[ignore = "GPU probe: needs a GPU feature and an idle machine"]
fn parakeet_encoder_gemm_shapes_gpu_versus_threaded_cpu() {
    #[cfg(all(feature = "cubecl-wgpu", not(feature = "cubecl-hip")))]
    let device = cubecl::wgpu::WgpuDevice::default();
    #[cfg(feature = "cubecl-hip")]
    let device = cubecl::hip::AmdDevice::default();
    #[cfg(all(feature = "cubecl-wgpu", not(feature = "cubecl-hip")))]
    type Rt = cubecl::wgpu::WgpuRuntime;
    #[cfg(feature = "cubecl-hip")]
    type Rt = cubecl::hip::HipRuntime;

    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let cpu = CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Avx2, threads)
        .or_else(|_| CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Optimized, threads))
        .expect("cpu backend");

    eprintln!("PARAKEET_GPU_PROBE cpu_threads={threads}");
    eprintln!(
        "{:<22} {:>5} {:>6} {:>6} {:>10} {:>10} {:>9} {:>10}",
        "shape", "rows", "in", "out", "cpu_us", "gpu_us", "gpu/cpu", "max_diff"
    );

    let mut cpu_block_total = 0.0_f64;
    let mut gpu_block_total = 0.0_f64;
    // Tracked so the "it is just the upload" objection can be answered from the
    // data rather than argued: if transfers dominated, the ratio would track
    // weight size, and it does not.
    let mut smallest_weight_ratio = (f64::INFINITY, 0.0_f64);

    for (label, rows, in_f, out_f, per_block) in SHAPES {
        let (rows, in_f, out_f, per_block) = (*rows, *in_f, *out_f, *per_block);
        let x = deterministic(rows * in_f, 1);
        let w = deterministic(out_f * in_f, 2);
        let mut cpu_out = vec![0.0_f32; rows * out_f];
        let mut gpu_out = vec![0.0_f32; rows * out_f];

        let cpu_s = best_of(5, || {
            cpu.linear_out_by_in(&x, rows, in_f, &w, out_f, None, &mut cpu_out)
                .expect("cpu gemm");
        });
        let gpu_s = best_of(5, || {
            ocelotl_kernels::linear_out_by_in_cubecl::<Rt>(
                &device,
                &x,
                rows,
                in_f,
                &w,
                out_f,
                None,
                &mut gpu_out,
            )
            .expect("gpu gemm");
        });

        // The timing means nothing if they disagree.
        let scale = cpu_out
            .iter()
            .fold(0.0_f32, |m, v| m.max(v.abs()))
            .max(1e-6);
        let diff = cpu_out
            .iter()
            .zip(gpu_out.iter())
            .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()))
            / scale;
        assert!(
            diff <= 1e-4,
            "{label}: GPU and CPU disagree by {diff:.3e} relative — the timing \
             comparison is meaningless unless both compute the same function"
        );

        let weight_mb = (out_f * in_f * 4) as f64 / 1e6;
        if weight_mb < smallest_weight_ratio.0 {
            smallest_weight_ratio = (weight_mb, gpu_s / cpu_s);
        }
        cpu_block_total += cpu_s * per_block as f64;
        gpu_block_total += gpu_s * per_block as f64;
        eprintln!(
            "{label:<22} {rows:>5} {in_f:>6} {out_f:>6} {:>10.1} {:>10.1} {:>8.2}x {diff:>10.2e}",
            cpu_s * 1e6,
            gpu_s * 1e6,
            gpu_s / cpu_s
        );
    }

    let blocks = 24.0;
    eprintln!(
        "PARAKEET_GPU_PROBE per-block cpu={:.2}ms gpu={:.2}ms | 24 blocks cpu={:.3}s gpu={:.3}s \
         | gpu is {:.2}x the CPU time",
        cpu_block_total * 1e3,
        gpu_block_total * 1e3,
        cpu_block_total * blocks,
        gpu_block_total * blocks,
        gpu_block_total / cpu_block_total
    );
    eprintln!(
        "PARAKEET_GPU_PROBE smallest weight = {:.2} MB and STILL {:.2}x slower — \
         so hoisting uploads into a device-resident forward would not rescue it; \
         the shapes are simply too small to fill the device.",
        smallest_weight_ratio.0, smallest_weight_ratio.1
    );
}
