use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use ocelotl_kernels::{CpuKernelBackend, GgmlKQuantKind, GgmlKQuantMatrixRef, linear_q8_k_k_quant};

fn benchmark_linear_out_by_in(criterion: &mut Criterion) {
    const ROWS: usize = 32;
    const INPUTS: usize = 256;
    const OUTPUTS: usize = 256;

    let backend = CpuKernelBackend::scalar();
    let input = deterministic_values(ROWS * INPUTS, 97);
    let weights = deterministic_values(OUTPUTS * INPUTS, 89);
    let bias = deterministic_values(OUTPUTS, 31);
    let mut output = vec![0.0_f32; ROWS * OUTPUTS];

    criterion.bench_function("kernel/linear_out_by_in/scalar/32x256x256", |bencher| {
        bencher.iter(|| {
            backend
                .linear_out_by_in(
                    black_box(&input),
                    ROWS,
                    INPUTS,
                    black_box(&weights),
                    OUTPUTS,
                    Some(black_box(&bias)),
                    black_box(&mut output),
                )
                .expect("deterministic linear benchmark shape must remain valid");
            black_box(&output);
        });
    });
}

fn benchmark_q6_k_projection(criterion: &mut Criterion) {
    const ROWS: usize = 4;
    const INPUTS: usize = 256;
    const OUTPUTS: usize = 64;
    const Q6_K_BLOCK_BYTES: usize = 210;

    let input = deterministic_values(ROWS * INPUTS, 127);
    let raw_weights = vec![0_u8; OUTPUTS * Q6_K_BLOCK_BYTES];
    let matrix = GgmlKQuantMatrixRef {
        input_features: INPUTS,
        output_features: OUTPUTS,
        kind: GgmlKQuantKind::Q6K,
        data: &raw_weights,
    };
    let mut output = vec![0.0_f32; ROWS * OUTPUTS];

    criterion.bench_function("kernel/linear_q8_k_q6_k/4x256x64", |bencher| {
        bencher.iter(|| {
            linear_q8_k_k_quant(
                black_box(&input),
                ROWS,
                black_box(matrix),
                black_box(&mut output),
            )
            .expect("deterministic Q6_K benchmark shape must remain valid");
            black_box(&output);
        });
    });
}

fn deterministic_values(len: usize, modulus: usize) -> Vec<f32> {
    (0..len)
        .map(|index| ((index % modulus) as f32 - (modulus / 2) as f32) / modulus as f32)
        .collect()
}

criterion_group!(
    benches,
    benchmark_linear_out_by_in,
    benchmark_q6_k_projection
);
criterion_main!(benches);
