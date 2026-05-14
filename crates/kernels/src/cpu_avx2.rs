//! AVX2 + FMA implementation of `linear_out_by_in_compute`.
//!
//! This module is the project's single `unsafe` boundary. The shape contract
//! is identical to the safe scalar `linear_out_by_in_compute`: contiguous
//! row-major `x: [rows, in_features]`, `weight_out_by_in: [out_features,
//! in_features]`, optional `bias: [out_features]`, and `out: [rows,
//! out_features]`. The Scalar mode remains the project parity oracle; this
//! file's output is validated against it within a pinned f32 tolerance on
//! every test run.
//!
//! # Tile shape
//!
//! The hot path is a 4-row × 4-output tile with 16 `__m256` accumulators
//! that fit comfortably in the 16 YMM registers available on AVX2. The K
//! direction is vectorized with 8-lane `_mm256_fmadd_ps`. After the SIMD
//! K-loop the 16 vector accumulators are horizontally reduced to scalar f32
//! and any K, output-dimension, or row tail is computed with the scalar
//! fallback. Row and output-dimension tails are intentionally simple — the
//! Whisper encoder shapes that motivate this path always have row and
//! out_features divisible by 4 (state ∈ {384, 512, 768, 1024, 1280} and
//! ffn ∈ {1536, 2048, 3072, 4096, 5120}; the conv-mapped audio_ctx is
//! 1500).
//!
//! # Numerical contract
//!
//! `_mm256_fmadd_ps` fuses one multiply and one add into a single rounded
//! operation; the scalar compute does two separate roundings. Because of
//! that the AVX2 output is **not** bit-identical to scalar, but the
//! per-output deviation on typical Whisper-sized matmuls stays under
//! 1e-5 in relative terms. The parity test in
//! `crates/kernels/src/lib.rs` pins that tolerance.
//!
//! # Safety
//!
//! All public entry points in this file are `unsafe` and require the caller
//! to have validated the host's AVX2 + FMA support (the
//! `CpuKernelBackend::with_mode_*` constructors do this via
//! `validate_mode_supported`). Pointer arithmetic stays inside slice bounds
//! that the upstream `validate_linear_out_by_in` already enforces, so
//! there are no out-of-bounds reads or writes.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// AVX2 + FMA tiled compute body matching `linear_out_by_in_compute`.
///
/// # Safety
///
/// - The host CPU must support AVX2 and FMA (guaranteed by
///   `validate_mode_supported(CpuKernelMode::Avx2)` at backend construction).
/// - The slice lengths must satisfy the same shape contract enforced by
///   `validate_linear_out_by_in`: `x.len() == rows * in_features`,
///   `weight_out_by_in.len() == out_features * in_features`, optional
///   `bias.len() == out_features`, `out.len() == rows * out_features`.
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn linear_out_by_in_compute_avx2(
    x: &[f32],
    rows: usize,
    in_features: usize,
    weight_out_by_in: &[f32],
    out_features: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    let tiled_rows = rows - (rows % 4);
    let tiled_out = out_features - (out_features % 4);
    let k_simd = in_features - (in_features % 8);

    let x_ptr = x.as_ptr();
    let w_ptr = weight_out_by_in.as_ptr();
    let out_ptr = out.as_mut_ptr();

    // 4-row × 4-output tile body.
    for row in (0..tiled_rows).step_by(4) {
        let x0_base = x_ptr.add(row * in_features);
        let x1_base = x_ptr.add((row + 1) * in_features);
        let x2_base = x_ptr.add((row + 2) * in_features);
        let x3_base = x_ptr.add((row + 3) * in_features);

        for out_dim in (0..tiled_out).step_by(4) {
            let w0_base = w_ptr.add(out_dim * in_features);
            let w1_base = w_ptr.add((out_dim + 1) * in_features);
            let w2_base = w_ptr.add((out_dim + 2) * in_features);
            let w3_base = w_ptr.add((out_dim + 3) * in_features);

            let mut acc00 = _mm256_setzero_ps();
            let mut acc01 = _mm256_setzero_ps();
            let mut acc02 = _mm256_setzero_ps();
            let mut acc03 = _mm256_setzero_ps();
            let mut acc10 = _mm256_setzero_ps();
            let mut acc11 = _mm256_setzero_ps();
            let mut acc12 = _mm256_setzero_ps();
            let mut acc13 = _mm256_setzero_ps();
            let mut acc20 = _mm256_setzero_ps();
            let mut acc21 = _mm256_setzero_ps();
            let mut acc22 = _mm256_setzero_ps();
            let mut acc23 = _mm256_setzero_ps();
            let mut acc30 = _mm256_setzero_ps();
            let mut acc31 = _mm256_setzero_ps();
            let mut acc32 = _mm256_setzero_ps();
            let mut acc33 = _mm256_setzero_ps();

            // SIMD K-loop: 8 lanes per FMA, 16 FMAs per K-tile of 8.
            let mut k = 0;
            while k < k_simd {
                let x0 = _mm256_loadu_ps(x0_base.add(k));
                let x1 = _mm256_loadu_ps(x1_base.add(k));
                let x2 = _mm256_loadu_ps(x2_base.add(k));
                let x3 = _mm256_loadu_ps(x3_base.add(k));

                let w0 = _mm256_loadu_ps(w0_base.add(k));
                let w1 = _mm256_loadu_ps(w1_base.add(k));
                let w2 = _mm256_loadu_ps(w2_base.add(k));
                let w3 = _mm256_loadu_ps(w3_base.add(k));

                acc00 = _mm256_fmadd_ps(x0, w0, acc00);
                acc01 = _mm256_fmadd_ps(x0, w1, acc01);
                acc02 = _mm256_fmadd_ps(x0, w2, acc02);
                acc03 = _mm256_fmadd_ps(x0, w3, acc03);

                acc10 = _mm256_fmadd_ps(x1, w0, acc10);
                acc11 = _mm256_fmadd_ps(x1, w1, acc11);
                acc12 = _mm256_fmadd_ps(x1, w2, acc12);
                acc13 = _mm256_fmadd_ps(x1, w3, acc13);

                acc20 = _mm256_fmadd_ps(x2, w0, acc20);
                acc21 = _mm256_fmadd_ps(x2, w1, acc21);
                acc22 = _mm256_fmadd_ps(x2, w2, acc22);
                acc23 = _mm256_fmadd_ps(x2, w3, acc23);

                acc30 = _mm256_fmadd_ps(x3, w0, acc30);
                acc31 = _mm256_fmadd_ps(x3, w1, acc31);
                acc32 = _mm256_fmadd_ps(x3, w2, acc32);
                acc33 = _mm256_fmadd_ps(x3, w3, acc33);

                k += 8;
            }

            // Horizontal-reduce each accumulator to scalar f32.
            let mut s00 = hsum_ps_avx(acc00);
            let mut s01 = hsum_ps_avx(acc01);
            let mut s02 = hsum_ps_avx(acc02);
            let mut s03 = hsum_ps_avx(acc03);
            let mut s10 = hsum_ps_avx(acc10);
            let mut s11 = hsum_ps_avx(acc11);
            let mut s12 = hsum_ps_avx(acc12);
            let mut s13 = hsum_ps_avx(acc13);
            let mut s20 = hsum_ps_avx(acc20);
            let mut s21 = hsum_ps_avx(acc21);
            let mut s22 = hsum_ps_avx(acc22);
            let mut s23 = hsum_ps_avx(acc23);
            let mut s30 = hsum_ps_avx(acc30);
            let mut s31 = hsum_ps_avx(acc31);
            let mut s32 = hsum_ps_avx(acc32);
            let mut s33 = hsum_ps_avx(acc33);

            // K-tail (scalar). Keeps the AVX2 path correct when in_features
            // is not a multiple of 8.
            for k_tail in k_simd..in_features {
                let x0v = *x0_base.add(k_tail);
                let x1v = *x1_base.add(k_tail);
                let x2v = *x2_base.add(k_tail);
                let x3v = *x3_base.add(k_tail);
                let w0v = *w0_base.add(k_tail);
                let w1v = *w1_base.add(k_tail);
                let w2v = *w2_base.add(k_tail);
                let w3v = *w3_base.add(k_tail);

                s00 += x0v * w0v;
                s01 += x0v * w1v;
                s02 += x0v * w2v;
                s03 += x0v * w3v;
                s10 += x1v * w0v;
                s11 += x1v * w1v;
                s12 += x1v * w2v;
                s13 += x1v * w3v;
                s20 += x2v * w0v;
                s21 += x2v * w1v;
                s22 += x2v * w2v;
                s23 += x2v * w3v;
                s30 += x3v * w0v;
                s31 += x3v * w1v;
                s32 += x3v * w2v;
                s33 += x3v * w3v;
            }

            let (b0, b1, b2, b3) = if let Some(b) = bias {
                (b[out_dim], b[out_dim + 1], b[out_dim + 2], b[out_dim + 3])
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };

            let o0 = row * out_features + out_dim;
            let o1 = (row + 1) * out_features + out_dim;
            let o2 = (row + 2) * out_features + out_dim;
            let o3 = (row + 3) * out_features + out_dim;

            *out_ptr.add(o0) = s00 + b0;
            *out_ptr.add(o0 + 1) = s01 + b1;
            *out_ptr.add(o0 + 2) = s02 + b2;
            *out_ptr.add(o0 + 3) = s03 + b3;
            *out_ptr.add(o1) = s10 + b0;
            *out_ptr.add(o1 + 1) = s11 + b1;
            *out_ptr.add(o1 + 2) = s12 + b2;
            *out_ptr.add(o1 + 3) = s13 + b3;
            *out_ptr.add(o2) = s20 + b0;
            *out_ptr.add(o2 + 1) = s21 + b1;
            *out_ptr.add(o2 + 2) = s22 + b2;
            *out_ptr.add(o2 + 3) = s23 + b3;
            *out_ptr.add(o3) = s30 + b0;
            *out_ptr.add(o3 + 1) = s31 + b1;
            *out_ptr.add(o3 + 2) = s32 + b2;
            *out_ptr.add(o3 + 3) = s33 + b3;
        }

        // out-dimension tail: 4 rows × 1 output column, scalar fallback.
        for tail_out in tiled_out..out_features {
            let w_base = w_ptr.add(tail_out * in_features);
            let bias_v = bias.map_or(0.0, |b| b[tail_out]);
            let mut sum0 = bias_v;
            let mut sum1 = bias_v;
            let mut sum2 = bias_v;
            let mut sum3 = bias_v;
            for k_dim in 0..in_features {
                let w = *w_base.add(k_dim);
                sum0 += *x0_base.add(k_dim) * w;
                sum1 += *x1_base.add(k_dim) * w;
                sum2 += *x2_base.add(k_dim) * w;
                sum3 += *x3_base.add(k_dim) * w;
            }
            *out_ptr.add(row * out_features + tail_out) = sum0;
            *out_ptr.add((row + 1) * out_features + tail_out) = sum1;
            *out_ptr.add((row + 2) * out_features + tail_out) = sum2;
            *out_ptr.add((row + 3) * out_features + tail_out) = sum3;
        }
    }

    // Row tail: 1 row at a time, scalar fallback.
    for row in tiled_rows..rows {
        let x_base = x_ptr.add(row * in_features);
        for out_dim in 0..out_features {
            let w_base = w_ptr.add(out_dim * in_features);
            let bias_v = bias.map_or(0.0, |b| b[out_dim]);
            let mut sum = bias_v;
            for k_dim in 0..in_features {
                sum += *x_base.add(k_dim) * *w_base.add(k_dim);
            }
            *out_ptr.add(row * out_features + out_dim) = sum;
        }
    }
}

/// Horizontal sum of 8 f32 lanes packed in a `__m256`.
///
/// Uses the standard "reduce to xmm, two passes of horizontal add" pattern
/// from the Intel intrinsics guide. Kept on its own so the inner kernel
/// reads as a sequence of 16 hsum calls rather than 16 inlined shuffle
/// chains.
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn hsum_ps_avx(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);
    let shuf1 = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf1);
    let shuf2 = _mm_movehl_ps(shuf1, sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);
    _mm_cvtss_f32(sum32)
}
