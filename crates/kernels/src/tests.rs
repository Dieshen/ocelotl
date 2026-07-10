use super::*;

fn assert_close_with_tolerance(got: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label} length mismatch: got {} expected {}",
        got.len(),
        expected.len()
    );
    for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs = (g - e).abs();
        let rel = if e.abs() > 1e-6 { abs / e.abs() } else { abs };
        assert!(
            abs <= tolerance || rel <= tolerance,
            "{label} drifted at idx {idx}: expected={e} got={g} abs={abs} rel={rel}"
        );
    }
}

// --- vec_add ---

#[test]
fn vec_add_handles_four_element_vectors() {
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [10.0_f32, 20.0, 30.0, 40.0];
    let mut out = [0.0_f32; 4];

    vec_add(&a, &b, &mut out).expect("equal-length vec_add must succeed");

    assert_eq!(out, [11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn vec_add_rejects_mismatched_input_lengths() {
    let a = [1.0_f32, 2.0, 3.0];
    let b = [1.0_f32, 2.0];
    let mut out = [0.0_f32; 3];

    let err = vec_add(&a, &b, &mut out).expect_err("must reject mismatched input lengths");

    match err {
        OcelotlError::Kernel(KernelError { backend, message }) => {
            assert_eq!(backend, "cpu");
            assert!(
                message.contains("vec_add"),
                "expected error to mention kernel name, got {message:?}"
            );
        }
        other => panic!("expected KernelError, got {other:?}"),
    }
}

#[test]
fn vec_add_rejects_mismatched_output_length() {
    let a = [1.0_f32, 2.0, 3.0];
    let b = [10.0_f32, 20.0, 30.0];
    let mut out = [0.0_f32; 4];

    let err = vec_add(&a, &b, &mut out).expect_err("must reject mismatched output length");
    assert!(matches!(err, OcelotlError::Kernel(_)));
}

// --- dot ---

#[test]
fn dot_computes_inner_product_of_four_element_vectors() {
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [10.0_f32, 20.0, 30.0, 40.0];

    // 1*10 + 2*20 + 3*30 + 4*40 = 10 + 40 + 90 + 160 = 300
    let result = dot(&a, &b).expect("equal-length dot must succeed");

    assert_eq!(result, 300.0);
}

#[test]
fn dot_of_empty_slices_is_zero() {
    let a: [f32; 0] = [];
    let b: [f32; 0] = [];

    assert_eq!(dot(&a, &b).unwrap(), 0.0);
}

#[test]
fn dot_rejects_mismatched_lengths() {
    let a = [1.0_f32, 2.0, 3.0];
    let b = [1.0_f32, 2.0];

    let err = dot(&a, &b).expect_err("must reject mismatched lengths");

    match err {
        OcelotlError::Kernel(KernelError { backend, message }) => {
            assert_eq!(backend, "cpu");
            assert!(
                message.contains("dot"),
                "expected error to mention kernel name, got {message:?}"
            );
        }
        other => panic!("expected KernelError, got {other:?}"),
    }
}

// --- softmax ---

#[test]
fn softmax_produces_known_distribution_for_three_element_input() {
    // Hand-checked: softmax([1, 2, 3]) with max-subtraction stability:
    //   shifted = [-2, -1, 0]
    //   e^shifted ≈ [0.13533528, 0.36787944, 1.0]
    //   sum ≈ 1.50321472
    //   result ≈ [0.09003057, 0.24472847, 0.66524096]
    let mut x = [1.0_f32, 2.0, 3.0];
    softmax(&mut x);

    let expected = [0.09003057_f32, 0.24472847, 0.66524096];
    for (got, want) in x.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 4.0 * f32::EPSILON,
            "softmax mismatch: got {got}, want {want}"
        );
    }

    let sum: f32 = x.iter().sum();
    assert!(
        (sum - 1.0).abs() < 4.0 * f32::EPSILON,
        "softmax must sum to 1.0, got {sum}"
    );
}

#[test]
fn softmax_is_stable_for_large_inputs() {
    // Without max-subtraction, exp(1000) overflows to +inf and the result
    // is NaN. With max-subtraction, the largest exponent is 0 and the
    // result is well-defined.
    let mut x = [1000.0_f32, 1001.0, 1002.0];
    softmax(&mut x);

    let sum: f32 = x.iter().sum();
    assert!(
        (sum - 1.0).abs() < 4.0 * f32::EPSILON,
        "stable softmax must sum to 1.0 even for large inputs, got {sum}"
    );
    for v in x.iter() {
        assert!(v.is_finite(), "softmax output must be finite, got {v}");
    }
}

#[test]
fn softmax_of_empty_slice_is_a_noop() {
    let mut x: [f32; 0] = [];
    softmax(&mut x);
    // No assertion needed — must not panic.
}

#[test]
fn softmax_of_uniform_input_is_uniform_distribution() {
    let mut x = [5.0_f32; 4];
    softmax(&mut x);
    for v in x.iter() {
        assert!(
            (v - 0.25).abs() < 4.0 * f32::EPSILON,
            "uniform softmax must be 1/n, got {v}"
        );
    }
}

// --- matmul ---

#[test]
fn matmul_handles_two_by_two_times_two_by_two() {
    // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
    // Hand check: row 0 of out = [1*5+2*7, 1*6+2*8] = [19, 22]
    //             row 1 of out = [3*5+4*7, 3*6+4*8] = [43, 50]
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [5.0_f32, 6.0, 7.0, 8.0];
    let mut out = [0.0_f32; 4];

    matmul(&a, (2, 2), &b, (2, 2), &mut out).expect("well-formed matmul must succeed");

    assert_eq!(out, [19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn matmul_handles_non_square_two_by_three_times_three_by_two() {
    // A = [[1, 2, 3], [4, 5, 6]]   (2x3)
    // B = [[7, 8], [9, 10], [11, 12]]   (3x2)
    // A@B row 0 = [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // A@B row 1 = [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut out = [0.0_f32; 4];

    matmul(&a, (2, 3), &b, (3, 2), &mut out).expect("well-formed matmul must succeed");

    assert_eq!(out, [58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn matmul_rejects_inner_dimension_mismatch() {
    let a = [1.0_f32; 6]; // 2x3
    let b = [1.0_f32; 8]; // 4x2 — inner dims disagree
    let mut out = [0.0_f32; 4];

    let err = matmul(&a, (2, 3), &b, (4, 2), &mut out).expect_err("must reject inner-dim mismatch");

    match err {
        OcelotlError::Kernel(KernelError { backend, message }) => {
            assert_eq!(backend, "cpu");
            assert!(
                message.contains("inner-dimension"),
                "expected inner-dim message, got {message:?}"
            );
        }
        other => panic!("expected KernelError, got {other:?}"),
    }
}

#[test]
fn matmul_rejects_wrong_a_slice_length() {
    let a = [1.0_f32; 5]; // claimed 2x3, actually 5
    let b = [1.0_f32; 6]; // 3x2
    let mut out = [0.0_f32; 4];

    let err = matmul(&a, (2, 3), &b, (3, 2), &mut out).expect_err("must reject wrong a length");
    assert!(matches!(err, OcelotlError::Kernel(_)));
}

#[test]
fn matmul_rejects_wrong_output_length() {
    let a = [1.0_f32; 6]; // 2x3
    let b = [1.0_f32; 6]; // 3x2
    let mut out = [0.0_f32; 3]; // claimed 2x2 = 4

    let err = matmul(&a, (2, 3), &b, (3, 2), &mut out).expect_err("must reject wrong out length");
    assert!(matches!(err, OcelotlError::Kernel(_)));
}

#[test]
fn matmul_rejects_shape_product_overflow() {
    let a = [];
    let b = [];
    let mut out = [];

    let err = matmul(&a, (usize::MAX, 2), &b, (2, 1), &mut out)
        .expect_err("overflowing shapes must be rejected");

    match err {
        OcelotlError::Kernel(KernelError { message, .. }) => {
            assert!(
                message.contains("overflows"),
                "expected overflow diagnostic, got {message:?}"
            );
        }
        other => panic!("expected KernelError, got {other:?}"),
    }
}

#[test]
fn cpu_backend_defaults_to_scalar_mode() {
    let backend = CpuKernelBackend::default();

    assert_eq!(backend.mode(), CpuKernelMode::Scalar);
    assert_eq!(backend.name(), "cpu");
    assert_eq!(backend.context().device, Device::Cpu);
}

#[test]
fn cpu_backend_rejects_gpu_requirement_with_typed_unsupported_error() {
    let backend = CpuKernelBackend::default();

    let err = require_gpu(&backend).expect_err("CPU backend must not satisfy GPU requirement");

    match err {
        OcelotlError::Unsupported(UnsupportedError {
            feature,
            requested,
            supported,
        }) => {
            assert_eq!(feature, "gpu_backend");
            assert_eq!(requested.as_deref(), Some("gpu"));
            assert_eq!(supported, vec!["cpu".to_string()]);
        }
        other => panic!("expected UnsupportedError, got {other:?}"),
    }
}

#[test]
fn cpu_backend_can_select_optimized_mode() {
    let backend = CpuKernelBackend::optimized();

    assert_eq!(backend.mode(), CpuKernelMode::Optimized);
    assert_eq!(CpuKernelMode::Optimized.as_str(), "optimized");
}

#[test]
fn public_cpu_mode_constructor_validates_before_storing_mode() {
    let scalar = CpuKernelBackend::with_mode(CpuKernelMode::Scalar)
        .expect("scalar mode must always be supported");
    assert_eq!(scalar.mode(), CpuKernelMode::Scalar);

    let optimized = CpuKernelBackend::with_mode(CpuKernelMode::Optimized)
        .expect("optimized mode must always be supported");
    assert_eq!(optimized.mode(), CpuKernelMode::Optimized);

    let avx2 = CpuKernelBackend::with_mode(CpuKernelMode::Avx2);
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        assert_eq!(
            avx2.expect("advertised AVX2 + FMA must be accepted").mode(),
            CpuKernelMode::Avx2
        );
    } else {
        assert!(
            avx2.is_err(),
            "safe construction must reject AVX2 without both AVX2 and FMA"
        );
    }

    #[cfg(not(target_arch = "x86_64"))]
    assert!(
        avx2.is_err(),
        "safe construction must reject AVX2 outside x86_64"
    );
}

#[test]
fn optimized_matmul_matches_scalar_for_non_square_shape() {
    let a = [
        0.25_f32, -0.5, 1.0, //
        1.5, 0.75, -1.25,
    ];
    let b = [
        0.5_f32, -1.0, 0.25, 2.0, //
        -0.75, 0.5, 1.25, -0.5, //
        1.0, 1.5, -1.0, 0.75,
    ];
    let scalar = CpuKernelBackend::scalar();
    let optimized = CpuKernelBackend::optimized();
    let mut scalar_out = [0.0_f32; 8];
    let mut optimized_out = [0.0_f32; 8];

    scalar
        .matmul(&a, (2, 3), &b, (3, 4), &mut scalar_out)
        .unwrap();
    optimized
        .matmul(&a, (2, 3), &b, (3, 4), &mut optimized_out)
        .unwrap();

    for (got, want) in optimized_out.iter().zip(scalar_out.iter()) {
        assert!(
            (got - want).abs() <= 1.0e-6,
            "optimized matmul drifted: got {got}, want {want}"
        );
    }
}

#[test]
fn optimized_linear_out_by_in_matches_scalar_with_bias() {
    let x = [
        1.0_f32, -2.0, 0.5, //
        -0.25, 1.5, 2.0,
    ];
    // [out_features, in_features] layout.
    let w = [
        0.5_f32, -1.0, 0.25, //
        -0.75, 0.5, 1.25, //
        1.0, 1.5, -1.0, //
        0.0, -0.5, 0.75,
    ];
    let bias = [0.1_f32, -0.2, 0.3, -0.4];
    let scalar = CpuKernelBackend::scalar();
    let optimized = CpuKernelBackend::optimized();
    let mut scalar_out = [0.0_f32; 8];
    let mut optimized_out = [0.0_f32; 8];

    scalar
        .linear_out_by_in(&x, 2, 3, &w, 4, Some(&bias), &mut scalar_out)
        .unwrap();
    optimized
        .linear_out_by_in(&x, 2, 3, &w, 4, Some(&bias), &mut optimized_out)
        .unwrap();

    for (got, want) in optimized_out.iter().zip(scalar_out.iter()) {
        assert!(
            (got - want).abs() <= 1.0e-6,
            "optimized linear_out_by_in drifted: got {got}, want {want}"
        );
    }
}

#[test]
fn scalar_linear_out_by_in_handles_row_and_output_tile_tails() {
    let rows = 5;
    let in_features = 3;
    let out_features = 5;
    let x = [
        1.0_f32, 2.0, 3.0, //
        4.0, 5.0, 6.0, //
        7.0, 8.0, 9.0, //
        10.0, 11.0, 12.0, //
        13.0, 14.0, 15.0,
    ];
    let w = [
        0.5_f32, 1.0, -0.5, //
        -1.0, 0.25, 0.75, //
        1.5, -0.25, 0.0, //
        0.0, -0.5, 2.0, //
        -0.75, 0.5, 1.25,
    ];
    let bias = [0.1_f32, -0.2, 0.3, -0.4, 0.5];
    let mut got = [0.0_f32; 25];
    let mut want = [0.0_f32; 25];

    CpuKernelBackend::scalar()
        .linear_out_by_in(
            &x,
            rows,
            in_features,
            &w,
            out_features,
            Some(&bias),
            &mut got,
        )
        .unwrap();

    for row in 0..rows {
        for out_dim in 0..out_features {
            let mut acc = bias[out_dim];
            for in_dim in 0..in_features {
                acc += x[row * in_features + in_dim] * w[out_dim * in_features + in_dim];
            }
            want[row * out_features + out_dim] = acc;
        }
    }

    assert_eq!(got, want);
}

#[test]
fn threaded_linear_out_by_in_matches_serial_bit_for_bit() {
    // Parity oracle for the rayon parallel dispatch. Each chunk writes
    // disjoint output rows; the K-loop accumulation order inside a chunk
    // is identical to the serial path, so the threaded output must be
    // bit-identical to the serial output. A run that drifts here would
    // indicate either a chunk boundary bug or that the compute helper
    // does not match the legacy path.
    let rows = 64usize; // > PARALLEL_LINEAR_MIN_ROWS so the pool dispatches
    let in_features = 17;
    let out_features = 13;
    let x: Vec<f32> = (0..(rows * in_features))
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    let w: Vec<f32> = (0..(out_features * in_features))
        .map(|i| ((i as f32) * 0.019).cos())
        .collect();
    let b: Vec<f32> = (0..out_features).map(|i| (i as f32) * 0.05).collect();

    let mut serial = vec![0.0_f32; rows * out_features];
    let serial_backend = CpuKernelBackend::scalar();
    serial_backend
        .linear_out_by_in(
            &x,
            rows,
            in_features,
            &w,
            out_features,
            Some(&b),
            &mut serial,
        )
        .expect("serial linear must succeed");

    let mut threaded = vec![0.0_f32; rows * out_features];
    let threaded_backend = CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Scalar, 4)
        .expect("4-thread backend must build");
    threaded_backend
        .linear_out_by_in(
            &x,
            rows,
            in_features,
            &w,
            out_features,
            Some(&b),
            &mut threaded,
        )
        .expect("threaded linear must succeed");

    assert_eq!(
        serial, threaded,
        "threaded linear_out_by_in must produce bit-identical output to serial"
    );
}

#[test]
fn threaded_linear_out_by_in_single_row_splits_output_axis_bit_for_bit() {
    let rows = 1usize;
    let in_features = 17;
    let out_features = PARALLEL_LINEAR_MIN_OUTPUTS + 7;
    let x: Vec<f32> = (0..(rows * in_features))
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    let w: Vec<f32> = (0..(out_features * in_features))
        .map(|i| ((i as f32) * 0.019).cos())
        .collect();
    let b: Vec<f32> = (0..out_features).map(|i| (i as f32) * 0.002).collect();

    let mut serial = vec![0.0_f32; rows * out_features];
    CpuKernelBackend::scalar()
        .linear_out_by_in(
            &x,
            rows,
            in_features,
            &w,
            out_features,
            Some(&b),
            &mut serial,
        )
        .expect("serial linear must succeed");

    let mut threaded = vec![0.0_f32; rows * out_features];
    CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Scalar, 4)
        .expect("4-thread backend must build")
        .linear_out_by_in(
            &x,
            rows,
            in_features,
            &w,
            out_features,
            Some(&b),
            &mut threaded,
        )
        .expect("threaded output-axis linear must succeed");

    assert_eq!(
        serial, threaded,
        "threaded output-axis linear_out_by_in must match serial bit-for-bit"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_linear_out_by_in_matches_scalar_within_tolerance() {
    // Parity oracle for the AVX2 + FMA path. The output is not bit-
    // identical to Scalar because FMA fuses one multiply and one add
    // into a single rounded operation; the scalar path does two
    // roundings. The relative error must stay tight on Whisper-sized
    // matmuls.
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        // Host doesn't support AVX2+FMA; skip rather than fail. The
        // backend constructor would return a typed error in this case.
        return;
    }
    let rows = 64usize;
    let in_features = 200; // not a multiple of 8, exercise K-tail
    let out_features = 33; // not a multiple of 4, exercise out-tail
    let x: Vec<f32> = (0..(rows * in_features))
        .map(|i| ((i as f32) * 0.011).sin())
        .collect();
    let w: Vec<f32> = (0..(out_features * in_features))
        .map(|i| ((i as f32) * 0.017).cos())
        .collect();
    let b: Vec<f32> = (0..out_features).map(|i| (i as f32) * 0.05).collect();

    let mut scalar = vec![0.0_f32; rows * out_features];
    CpuKernelBackend::scalar()
        .linear_out_by_in(
            &x,
            rows,
            in_features,
            &w,
            out_features,
            Some(&b),
            &mut scalar,
        )
        .expect("scalar must succeed");

    let mut avx2 = vec![0.0_f32; rows * out_features];
    CpuKernelBackend::with_mode_checked(CpuKernelMode::Avx2)
        .expect("AVX2 backend must build on host that advertises avx2+fma")
        .linear_out_by_in(&x, rows, in_features, &w, out_features, Some(&b), &mut avx2)
        .expect("AVX2 must succeed");

    // Tolerance: 1e-4 relative or absolute. FMA fuses a multiply and
    // add into one rounding (vs scalar's two roundings) and the SIMD
    // path also accumulates into 8 partial-sum lanes that are reduced
    // after the K-loop, so order-of-addition differs. The drift is
    // bounded and tiny on Whisper-sized matmuls; Whisper exact-token
    // parity (much coarser, only argmax order matters) is still
    // preserved end-to-end and is pinned by the bench hook's
    // `matches_expected` field.
    for (idx, (a, s)) in avx2.iter().zip(scalar.iter()).enumerate() {
        let abs = (a - s).abs();
        let rel = if s.abs() > 1e-6 { abs / s.abs() } else { abs };
        assert!(
            abs <= 1e-4 || rel <= 1e-4,
            "AVX2 output drifted at idx {idx}: scalar={s} avx2={a} abs={abs} rel={rel}"
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_linear_out_by_in_single_row_matches_scalar_within_tolerance() {
    // Whisper decoder append uses rows=1 for each projection. This pins
    // the AVX2 row-tail SIMD path instead of only exercising the 4-row
    // tile body above.
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        return;
    }
    let rows = 1usize;
    let in_features = 384;
    let out_features = 257; // exercises the out-dimension tail too
    let x: Vec<f32> = (0..(rows * in_features))
        .map(|i| ((i as f32) * 0.009).sin())
        .collect();
    let w: Vec<f32> = (0..(out_features * in_features))
        .map(|i| ((i as f32) * 0.014).cos())
        .collect();

    let mut scalar = vec![0.0_f32; rows * out_features];
    CpuKernelBackend::scalar()
        .linear_out_by_in(&x, rows, in_features, &w, out_features, None, &mut scalar)
        .expect("scalar must succeed");

    let mut avx2 = vec![0.0_f32; rows * out_features];
    CpuKernelBackend::with_mode_checked(CpuKernelMode::Avx2)
        .expect("AVX2 backend must build on host that advertises avx2+fma")
        .linear_out_by_in(&x, rows, in_features, &w, out_features, None, &mut avx2)
        .expect("AVX2 must succeed");

    for (idx, (a, s)) in avx2.iter().zip(scalar.iter()).enumerate() {
        let abs = (a - s).abs();
        let rel = if s.abs() > 1e-6 { abs / s.abs() } else { abs };
        assert!(
            abs <= 1e-4 || rel <= 1e-4,
            "AVX2 single-row output drifted at idx {idx}: scalar={s} avx2={a} abs={abs} rel={rel}"
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_dot_f32_matches_scalar_within_tolerance() {
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        return;
    }
    let a: Vec<f32> = (0..130).map(|i| ((i as f32) * 0.011).sin()).collect();
    let b: Vec<f32> = (0..130).map(|i| ((i as f32) * 0.017).cos()).collect();
    let scalar = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    // SAFETY: the host feature check above verified AVX2+FMA support.
    let avx2 = unsafe { cpu_avx2::dot_f32_avx2(&a, &b) };
    let abs = (avx2 - scalar).abs();
    let rel = if scalar.abs() > 1e-6 {
        abs / scalar.abs()
    } else {
        abs
    };
    assert!(
        abs <= 1e-4 || rel <= 1e-4,
        "AVX2 dot drifted: scalar={scalar} avx2={avx2} abs={abs} rel={rel}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_attention_value_weighted_sum_matches_scalar_within_tolerance() {
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        return;
    }
    let seq = 17usize;
    let state = 24usize;
    let head_offset = 8usize;
    let head_dim = 10usize;
    let probs: Vec<f32> = (0..seq).map(|i| 0.001 + (i as f32) * 0.003).collect();
    let v: Vec<f32> = (0..seq * state)
        .map(|i| ((i as f32) * 0.017).cos())
        .collect();

    let mut scalar = vec![0.0_f32; head_dim];
    for d in 0..head_dim {
        for (ki, &p) in probs.iter().enumerate() {
            scalar[d] += p * v[ki * state + head_offset + d];
        }
    }

    let mut avx2 = vec![0.0_f32; head_dim];
    // SAFETY: the host feature check above verified AVX2+FMA support, and
    // the synthetic slices satisfy the helper's documented shape contract.
    unsafe {
        cpu_avx2::attention_value_weighted_sum_avx2(
            &probs,
            &v,
            seq,
            state,
            head_offset,
            head_dim,
            &mut avx2,
        );
    }

    for (idx, (a, s)) in avx2.iter().zip(scalar.iter()).enumerate() {
        let abs = (a - s).abs();
        let rel = if s.abs() > 1e-6 { abs / s.abs() } else { abs };
        assert!(
            abs <= 1e-4 || rel <= 1e-4,
            "AVX2 weighted sum drifted at idx {idx}: scalar={s} avx2={a} abs={abs} rel={rel}"
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn threaded_avx2_linear_out_by_in_single_row_matches_serial_avx2_bit_for_bit() {
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        return;
    }
    let rows = 1usize;
    let in_features = 384;
    let out_features = PARALLEL_LINEAR_MIN_OUTPUTS + 7;
    let x: Vec<f32> = (0..(rows * in_features))
        .map(|i| ((i as f32) * 0.009).sin())
        .collect();
    let w: Vec<f32> = (0..(out_features * in_features))
        .map(|i| ((i as f32) * 0.014).cos())
        .collect();

    let mut serial = vec![0.0_f32; rows * out_features];
    CpuKernelBackend::with_mode_checked(CpuKernelMode::Avx2)
        .expect("AVX2 backend must build on host that advertises avx2+fma")
        .linear_out_by_in(&x, rows, in_features, &w, out_features, None, &mut serial)
        .expect("serial AVX2 must succeed");

    let mut threaded = vec![0.0_f32; rows * out_features];
    CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Avx2, 4)
        .expect("threaded AVX2 backend must build")
        .linear_out_by_in(&x, rows, in_features, &w, out_features, None, &mut threaded)
        .expect("threaded output-axis AVX2 must succeed");

    assert_eq!(
        serial, threaded,
        "threaded output-axis AVX2 must match serial AVX2 bit-for-bit"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_linear_out_by_in_threaded_matches_serial_avx2_within_tolerance() {
    // Compose AVX2 + threads. Each chunk runs the AVX2 compute body on
    // disjoint output rows; the result should match the single-thread
    // AVX2 output bit-for-bit because the per-row K-loop accumulation
    // order is identical between serial and parallel AVX2.
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        return;
    }
    let rows = 96usize;
    let in_features = 256;
    let out_features = 64;
    let x: Vec<f32> = (0..(rows * in_features))
        .map(|i| ((i as f32) * 0.007).sin())
        .collect();
    let w: Vec<f32> = (0..(out_features * in_features))
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();

    let mut serial = vec![0.0_f32; rows * out_features];
    CpuKernelBackend::with_mode_checked(CpuKernelMode::Avx2)
        .unwrap()
        .linear_out_by_in(&x, rows, in_features, &w, out_features, None, &mut serial)
        .unwrap();

    let mut threaded = vec![0.0_f32; rows * out_features];
    CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Avx2, 4)
        .unwrap()
        .linear_out_by_in(&x, rows, in_features, &w, out_features, None, &mut threaded)
        .unwrap();

    assert_eq!(
        serial, threaded,
        "AVX2 + threads must match AVX2 serial bit-for-bit"
    );
}

#[test]
fn threaded_linear_out_by_in_falls_back_to_serial_for_small_inputs() {
    // Rows below PARALLEL_LINEAR_MIN_ROWS must skip the pool dispatch.
    // We can't observe that directly, but we can confirm the small path
    // still produces the same result as the serial backend.
    let rows = 3usize;
    let in_features = 5;
    let out_features = 4;
    let x: Vec<f32> = (0..(rows * in_features))
        .map(|i| (i as f32) * 0.1)
        .collect();
    let w: Vec<f32> = (0..(out_features * in_features))
        .map(|i| (i as f32) * -0.07)
        .collect();

    let mut serial = vec![0.0_f32; rows * out_features];
    CpuKernelBackend::scalar()
        .linear_out_by_in(&x, rows, in_features, &w, out_features, None, &mut serial)
        .unwrap();

    let mut threaded = vec![0.0_f32; rows * out_features];
    CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Scalar, 4)
        .unwrap()
        .linear_out_by_in(&x, rows, in_features, &w, out_features, None, &mut threaded)
        .unwrap();

    assert_eq!(serial, threaded);
}

#[test]
fn optimized_attention_matches_scalar_backend() {
    let q = [
        1.0_f32, 0.0, //
        0.0, 1.0, //
        0.5, 0.5,
    ];
    let k = q;
    let v = [
        1.0_f32, 2.0, //
        3.0, 4.0, //
        100.0, -50.0,
    ];
    let scalar = CpuKernelBackend::scalar();
    let optimized = CpuKernelBackend::optimized();
    let mut scalar_out = [0.0_f32; 6];
    let mut optimized_out = [0.0_f32; 6];

    scalar
        .scaled_dot_product_attention(&q, &k, &v, 3, 1, 1, 2, &mut scalar_out)
        .unwrap();
    optimized
        .scaled_dot_product_attention(&q, &k, &v, 3, 1, 1, 2, &mut optimized_out)
        .unwrap();

    for (got, want) in optimized_out.iter().zip(scalar_out.iter()) {
        assert!(
            (got - want).abs() <= 1.0e-6,
            "optimized attention drifted: got {got}, want {want}"
        );
    }
}

// --- parallel matmul parity ---

#[test]
fn threaded_matmul_scalar_matches_serial_bit_for_bit() {
    // Disjoint output-row chunks + identical accumulation order =
    // bit-identical to the serial scalar path.
    let m = 64usize;
    let k = 19usize;
    let n = 17usize;
    let a: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.013).sin()).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.019).cos()).collect();

    let mut serial = vec![0.0_f32; m * n];
    CpuKernelBackend::scalar()
        .matmul(&a, (m, k), &b, (k, n), &mut serial)
        .expect("serial scalar matmul must succeed");

    let mut threaded = vec![0.0_f32; m * n];
    CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Scalar, 4)
        .expect("4-thread scalar backend must build")
        .matmul(&a, (m, k), &b, (k, n), &mut threaded)
        .expect("threaded matmul must succeed");

    assert_eq!(
        serial, threaded,
        "threaded matmul must be bit-identical to serial scalar"
    );
}

#[test]
fn threaded_matmul_optimized_matches_serial_bit_for_bit() {
    let m = 96usize;
    let k = 33usize;
    let n = 25usize;
    let a: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.011).sin()).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.017).cos()).collect();

    let mut serial = vec![0.0_f32; m * n];
    CpuKernelBackend::optimized()
        .matmul(&a, (m, k), &b, (k, n), &mut serial)
        .expect("serial optimized matmul must succeed");

    let mut threaded = vec![0.0_f32; m * n];
    CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Optimized, 4)
        .expect("4-thread optimized backend must build")
        .matmul(&a, (m, k), &b, (k, n), &mut threaded)
        .expect("threaded matmul must succeed");

    assert_eq!(
        serial, threaded,
        "threaded optimized matmul must be bit-identical to serial optimized"
    );
}

#[test]
fn threaded_matmul_falls_back_to_serial_for_small_inputs() {
    // M below PARALLEL_MATMUL_MIN_ROWS must still produce the serial
    // result. We can't observe the dispatch path directly, but a small
    // shape must round-trip identically.
    let m = 5usize;
    let k = 7usize;
    let n = 3usize;
    let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * -0.07).collect();

    let mut serial = vec![0.0_f32; m * n];
    CpuKernelBackend::scalar()
        .matmul(&a, (m, k), &b, (k, n), &mut serial)
        .unwrap();

    let mut threaded = vec![0.0_f32; m * n];
    CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Scalar, 4)
        .unwrap()
        .matmul(&a, (m, k), &b, (k, n), &mut threaded)
        .unwrap();

    assert_eq!(serial, threaded);
}

// --- parallel SDPA parity ---

#[test]
fn threaded_sdpa_scalar_matches_serial_bit_for_bit() {
    // Per-chunk scratch + per-query-position output cells = identical
    // accumulation order to the serial path. Use a seq_len above
    // PARALLEL_SDPA_MIN_SEQ so the pool actually dispatches.
    let seq_len = 48usize;
    let num_q_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 6usize;
    let q: Vec<f32> = (0..(seq_len * num_q_heads * head_dim))
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    let k: Vec<f32> = (0..(seq_len * num_kv_heads * head_dim))
        .map(|i| ((i as f32) * 0.019).cos())
        .collect();
    let v: Vec<f32> = (0..(seq_len * num_kv_heads * head_dim))
        .map(|i| ((i as f32) * 0.023).sin())
        .collect();

    let mut serial = vec![0.0_f32; seq_len * num_q_heads * head_dim];
    CpuKernelBackend::scalar()
        .scaled_dot_product_attention(
            &q,
            &k,
            &v,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            &mut serial,
        )
        .expect("serial SDPA must succeed");

    let mut threaded = vec![0.0_f32; seq_len * num_q_heads * head_dim];
    CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Scalar, 4)
        .expect("4-thread scalar backend must build")
        .scaled_dot_product_attention(
            &q,
            &k,
            &v,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            &mut threaded,
        )
        .expect("threaded SDPA must succeed");

    assert_eq!(
        serial, threaded,
        "threaded scalar SDPA must be bit-identical to serial scalar SDPA"
    );
}

#[test]
fn threaded_sdpa_optimized_matches_serial_bit_for_bit() {
    let seq_len = 64usize;
    let num_q_heads = 6usize;
    let num_kv_heads = 2usize;
    let head_dim = 8usize;
    let q: Vec<f32> = (0..(seq_len * num_q_heads * head_dim))
        .map(|i| ((i as f32) * 0.011).sin())
        .collect();
    let k: Vec<f32> = (0..(seq_len * num_kv_heads * head_dim))
        .map(|i| ((i as f32) * 0.017).cos())
        .collect();
    let v: Vec<f32> = (0..(seq_len * num_kv_heads * head_dim))
        .map(|i| ((i as f32) * 0.021).sin())
        .collect();

    let mut serial = vec![0.0_f32; seq_len * num_q_heads * head_dim];
    CpuKernelBackend::optimized()
        .scaled_dot_product_attention(
            &q,
            &k,
            &v,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            &mut serial,
        )
        .expect("serial optimized SDPA must succeed");

    let mut threaded = vec![0.0_f32; seq_len * num_q_heads * head_dim];
    CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Optimized, 4)
        .expect("4-thread optimized backend must build")
        .scaled_dot_product_attention(
            &q,
            &k,
            &v,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            &mut threaded,
        )
        .expect("threaded SDPA must succeed");

    assert_eq!(
        serial, threaded,
        "threaded optimized SDPA must be bit-identical to serial optimized SDPA"
    );
}

#[test]
fn threaded_sdpa_falls_back_to_serial_for_small_seq() {
    // seq_len below PARALLEL_SDPA_MIN_SEQ stays on the serial path.
    let seq_len = 3usize;
    let num_q_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 2usize;
    let q = [
        1.0_f32, 0.0, 0.0, 1.0, 0.5, 0.5, 0.25, 0.75, -1.0, 1.0, 0.3, 0.7,
    ];
    let k = [1.0_f32, 0.0, 0.5, 0.5, -1.0, 1.0];
    let v = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    let mut serial = vec![0.0_f32; seq_len * num_q_heads * head_dim];
    CpuKernelBackend::scalar()
        .scaled_dot_product_attention(
            &q,
            &k,
            &v,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            &mut serial,
        )
        .unwrap();

    let mut threaded = vec![0.0_f32; seq_len * num_q_heads * head_dim];
    CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Scalar, 4)
        .unwrap()
        .scaled_dot_product_attention(
            &q,
            &k,
            &v,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            &mut threaded,
        )
        .unwrap();

    assert_eq!(serial, threaded);
}

// --- GW.4 Stage 1A: linear_d parity ---

#[test]
fn cpu_linear_d_matches_linear_out_by_in_bit_for_bit() {
    // The CPU `linear_d` override borrows host slices directly and calls
    // the existing `linear_out_by_in`. The output must be bit-identical
    // to the slice-based path on the same inputs — `linear_d` is the
    // parity oracle the GPU implementation (GW.4-1B) will validate
    // against.
    let rows = 7usize;
    let in_features = 23usize;
    let out_features = 13usize;
    let x_vec: Vec<f32> = (0..rows * in_features)
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    let w_vec: Vec<f32> = (0..out_features * in_features)
        .map(|i| ((i as f32) * 0.019).cos())
        .collect();
    let b_vec: Vec<f32> = (0..out_features).map(|i| (i as f32) * 0.05).collect();

    let backend = CpuKernelBackend::scalar();

    // Slice path.
    let mut slice_out = vec![0.0_f32; rows * out_features];
    backend
        .linear_out_by_in(
            &x_vec,
            rows,
            in_features,
            &w_vec,
            out_features,
            Some(&b_vec),
            &mut slice_out,
        )
        .unwrap();

    // Handle path.
    let x_h = DeviceTensor::from_host(x_vec.clone());
    let w_h = DeviceTensor::from_host(w_vec.clone());
    let b_h = DeviceTensor::from_host(b_vec.clone());
    let out_h = DeviceTensor::host_zeros(rows * out_features);
    backend
        .linear_d(
            &x_h,
            rows,
            in_features,
            &w_h,
            out_features,
            Some(&b_h),
            &out_h,
        )
        .unwrap();
    let handle_out = out_h.to_host_owned().unwrap();

    assert_eq!(
        slice_out, handle_out,
        "linear_d must be bit-identical to linear_out_by_in on the CPU backend"
    );
}

#[test]
fn cpu_linear_d_handles_no_bias() {
    let rows = 3usize;
    let in_features = 5usize;
    let out_features = 4usize;
    let x_vec: Vec<f32> = (0..rows * in_features).map(|i| (i as f32) * 0.1).collect();
    let w_vec: Vec<f32> = (0..out_features * in_features)
        .map(|i| (i as f32) * -0.07)
        .collect();

    let backend = CpuKernelBackend::scalar();

    let mut slice_out = vec![0.0_f32; rows * out_features];
    backend
        .linear_out_by_in(
            &x_vec,
            rows,
            in_features,
            &w_vec,
            out_features,
            None,
            &mut slice_out,
        )
        .unwrap();

    let x_h = DeviceTensor::from_host(x_vec);
    let w_h = DeviceTensor::from_host(w_vec);
    let out_h = DeviceTensor::host_zeros(rows * out_features);
    backend
        .linear_d(&x_h, rows, in_features, &w_h, out_features, None, &out_h)
        .unwrap();
    assert_eq!(slice_out, out_h.to_host_owned().unwrap());
}

// --- GW.4 Stage 2A: device-resident critical-chain primitives ---

#[test]
fn cpu_add_inplace_d_matches_scalar_add_bit_for_bit() {
    // Parity oracle: a hand-rolled lhs+rhs loop. CPU backend must
    // produce identical f32 bit patterns.
    let lhs_vec: Vec<f32> = (0..37).map(|i| (i as f32) * 0.013).collect();
    let rhs_vec: Vec<f32> = (0..37).map(|i| ((i as f32) * 0.019).sin()).collect();
    let mut expected = lhs_vec.clone();
    for (l, r) in expected.iter_mut().zip(rhs_vec.iter()) {
        *l += *r;
    }

    let backend = CpuKernelBackend::scalar();
    let lhs_h = DeviceTensor::from_host(lhs_vec.clone());
    let rhs_h = DeviceTensor::from_host(rhs_vec);
    backend.add_inplace_d(&lhs_h, &rhs_h).unwrap();
    assert_eq!(lhs_h.to_host_owned().unwrap(), expected);
}

#[test]
fn cpu_add_inplace_d_rejects_mismatched_lengths() {
    let backend = CpuKernelBackend::scalar();
    let lhs_h = DeviceTensor::from_host(vec![1.0_f32, 2.0, 3.0]);
    let rhs_h = DeviceTensor::from_host(vec![1.0_f32, 2.0]);
    let err = backend
        .add_inplace_d(&lhs_h, &rhs_h)
        .expect_err("must reject");
    assert!(matches!(err, OcelotlError::Kernel(_)));
}

#[test]
fn cpu_gelu_inplace_d_matches_whisper_primitive_bit_for_bit() {
    // Bit-exactness gate: this is the math both the kernels crate and
    // the Whisper primitive call. They share the same scalar function
    // (gelu_whisper_scalar), so the only way this can drift is if one
    // side changes the formula.
    let host: Vec<f32> = (-32..32).map(|i| (i as f32) * 0.25).collect();
    let mut expected = host.clone();
    for v in expected.iter_mut() {
        *v = gelu_whisper_scalar(*v);
    }

    let backend = CpuKernelBackend::scalar();
    let t = DeviceTensor::from_host(host);
    backend.gelu_inplace_d(&t).unwrap();
    assert_eq!(t.to_host_owned().unwrap(), expected);
}

#[test]
fn cpu_layer_norm_d_matches_whisper_primitive_bit_for_bit() {
    // This test fails iff the kernels-crate scalar layer-norm has
    // drifted from the Whisper primitive. Both call layer_norm_whisper_scalar,
    // so a failure here is the canary.
    let rows = 17usize;
    let hidden = 23usize;
    let eps = 1e-5_f32;
    let x_vec: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.011).sin())
        .collect();
    let weight: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let bias: Vec<f32> = (0..hidden).map(|i| (i as f32) * -0.005).collect();

    let mut expected = vec![0.0_f32; rows * hidden];
    layer_norm_whisper_scalar(&x_vec, rows, hidden, &weight, &bias, eps, &mut expected);

    let backend = CpuKernelBackend::scalar();
    let x_h = DeviceTensor::from_host(x_vec);
    let w_h = DeviceTensor::from_host(weight);
    let b_h = DeviceTensor::from_host(bias);
    let out_h = DeviceTensor::host_zeros(rows * hidden);
    backend
        .layer_norm_d(&x_h, rows, hidden, &w_h, &b_h, eps, &out_h)
        .unwrap();
    assert_eq!(out_h.to_host_owned().unwrap(), expected);
}

#[test]
fn cpu_layer_norm_d_rejects_shape_mismatch() {
    let backend = CpuKernelBackend::scalar();
    let x_h = DeviceTensor::from_host(vec![0.0_f32; 12]);
    let w_h = DeviceTensor::from_host(vec![1.0_f32; 4]);
    let b_h = DeviceTensor::from_host(vec![0.0_f32; 4]);
    let out_h = DeviceTensor::host_zeros(11); // wrong length
    let err = backend
        .layer_norm_d(&x_h, 3, 4, &w_h, &b_h, 1e-5, &out_h)
        .expect_err("must reject");
    assert!(matches!(err, OcelotlError::Kernel(_)));
}

#[test]
fn cpu_add_positional_embedding_d_matches_scalar_bit_for_bit() {
    let rows = 5usize;
    let cols = 11usize;
    let pe_rows = 12usize;
    let start_pos = 3usize;
    let x_vec: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.02).collect();
    let pe_vec: Vec<f32> = (0..pe_rows * cols).map(|i| (i as f32) * -0.013).collect();

    let mut expected = x_vec.clone();
    for row in 0..rows {
        let dst_start = row * cols;
        let src_start = (start_pos + row) * cols;
        for col in 0..cols {
            expected[dst_start + col] += pe_vec[src_start + col];
        }
    }

    let backend = CpuKernelBackend::scalar();
    let x_h = DeviceTensor::from_host(x_vec);
    let pe_h = DeviceTensor::from_host(pe_vec);
    backend
        .add_positional_embedding_d(&x_h, rows, cols, &pe_h, pe_rows, start_pos)
        .unwrap();
    assert_eq!(x_h.to_host_owned().unwrap(), expected);
}

#[test]
fn cpu_add_positional_embedding_d_rejects_out_of_range_start_pos() {
    let backend = CpuKernelBackend::scalar();
    let x_h = DeviceTensor::from_host(vec![0.0_f32; 6]); // 2 rows × 3 cols
    let pe_h = DeviceTensor::from_host(vec![0.0_f32; 12]); // 4 rows × 3 cols
    // start_pos=3 + rows=2 = 5 > pe_rows=4
    let err = backend
        .add_positional_embedding_d(&x_h, 2, 3, &pe_h, 4, 3)
        .expect_err("must reject");
    assert!(matches!(err, OcelotlError::Kernel(_)));
}

#[test]
fn upload_default_returns_host_resident_tensor() {
    let backend = CpuKernelBackend::scalar();
    let t = backend.upload(&[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(t.residency(), Residency::Host);
    assert_eq!(t.to_host_owned().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn alloc_default_returns_zero_filled_host_tensor() {
    let backend = CpuKernelBackend::scalar();
    let t = backend.alloc(4).unwrap();
    assert_eq!(t.residency(), Residency::Host);
    assert_eq!(t.to_host_owned().unwrap(), vec![0.0, 0.0, 0.0, 0.0]);
}

// --- attention_encoder_d ---

/// Bit-for-bit parity gate between the CPU backend's
/// `attention_encoder_d` override (which borrows host slices) and the
/// scalar reference. The override and the reference call the same
/// `attention_encoder_scalar` helper, so the two paths must agree
/// exactly — no tolerance window.
#[test]
fn cpu_attention_encoder_d_matches_scalar_bit_for_bit() {
    let seq = 8usize;
    let n_head = 2usize;
    let head_dim = 4usize;
    let state = n_head * head_dim;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..seq * state)
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    let k: Vec<f32> = (0..seq * state)
        .map(|i| ((i as f32) * 0.019).cos())
        .collect();
    let v: Vec<f32> = (0..seq * state)
        .map(|i| ((i as f32) * 0.023).sin())
        .collect();

    let mut scalar_out = vec![0.0_f32; seq * state];
    attention_encoder_scalar(&q, &k, &v, seq, n_head, head_dim, scale, &mut scalar_out);

    let backend = CpuKernelBackend::scalar();
    let q_d = backend.upload(&q).expect("upload q");
    let k_d = backend.upload(&k).expect("upload k");
    let v_d = backend.upload(&v).expect("upload v");
    let out_d = backend.alloc(seq * state).expect("alloc out");
    backend
        .attention_encoder_d(&q_d, &k_d, &v_d, seq, n_head, head_dim, scale, &out_d)
        .expect("CPU attention_encoder_d must succeed");
    let got = out_d.to_host_owned().expect("readback");

    assert_eq!(
        got, scalar_out,
        "CPU override must match scalar bit-for-bit"
    );
}

#[test]
fn cpu_attention_encoder_d_threaded_matches_scalar_bit_for_bit() {
    let seq = PARALLEL_SDPA_MIN_SEQ;
    let n_head = 2usize;
    let head_dim = 4usize;
    let state = n_head * head_dim;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..seq * state)
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    let k: Vec<f32> = (0..seq * state)
        .map(|i| ((i as f32) * 0.019).cos())
        .collect();
    let v: Vec<f32> = (0..seq * state)
        .map(|i| ((i as f32) * 0.023).sin())
        .collect();

    let mut scalar_out = vec![0.0_f32; seq * state];
    attention_encoder_scalar(&q, &k, &v, seq, n_head, head_dim, scale, &mut scalar_out);

    let backend = CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Scalar, 4)
        .expect("threaded scalar backend must build");
    let q_d = backend.upload(&q).expect("upload q");
    let k_d = backend.upload(&k).expect("upload k");
    let v_d = backend.upload(&v).expect("upload v");
    let out_d = backend.alloc(seq * state).expect("alloc out");
    backend
        .attention_encoder_d(&q_d, &k_d, &v_d, seq, n_head, head_dim, scale, &out_d)
        .expect("threaded CPU attention_encoder_d must succeed");
    let got = out_d.to_host_owned().expect("readback");

    assert_eq!(
        got, scalar_out,
        "threaded CPU override must match scalar bit-for-bit"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn cpu_attention_encoder_d_threaded_avx2_matches_scalar_within_tolerance() {
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        return;
    }
    let seq = PARALLEL_SDPA_MIN_SEQ;
    let n_head = 2usize;
    let head_dim = 8usize;
    let state = n_head * head_dim;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..seq * state)
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    let k: Vec<f32> = (0..seq * state)
        .map(|i| ((i as f32) * 0.019).cos())
        .collect();
    let v: Vec<f32> = (0..seq * state)
        .map(|i| ((i as f32) * 0.023).sin())
        .collect();

    let mut scalar_out = vec![0.0_f32; seq * state];
    attention_encoder_scalar(&q, &k, &v, seq, n_head, head_dim, scale, &mut scalar_out);

    let backend = CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Avx2, 4)
        .expect("threaded AVX2 backend must build");
    let q_d = backend.upload(&q).expect("upload q");
    let k_d = backend.upload(&k).expect("upload k");
    let v_d = backend.upload(&v).expect("upload v");
    let out_d = backend.alloc(seq * state).expect("alloc out");
    backend
        .attention_encoder_d(&q_d, &k_d, &v_d, seq, n_head, head_dim, scale, &out_d)
        .expect("threaded AVX2 attention_encoder_d must succeed");
    let got = out_d.to_host_owned().expect("readback");

    for (idx, (g, s)) in got.iter().zip(scalar_out.iter()).enumerate() {
        let abs = (g - s).abs();
        let rel = if s.abs() > 1e-6 { abs / s.abs() } else { abs };
        assert!(
            abs <= 1e-4 || rel <= 1e-4,
            "threaded AVX2 encoder attention drifted at idx {idx}: scalar={s} avx2={g} abs={abs} rel={rel}"
        );
    }
}

// -----------------------------------------------------------------------
// GW.4-5B CPU parity gates
// -----------------------------------------------------------------------

/// GW.4-5B: `attention_decoder_causal_d` on the CPU backend must match
/// `attention_decoder_causal_scalar` bit-for-bit. Both take the same
/// path so no tolerance window is needed.
#[test]
fn cpu_attention_decoder_causal_d_matches_scalar_bit_for_bit() {
    let seq = 5usize;
    let n_head = 2usize;
    let head_dim = 4usize;
    let state = n_head * head_dim;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..seq * state)
        .map(|i| ((i as f32) * 0.017).sin())
        .collect();
    let k: Vec<f32> = (0..seq * state)
        .map(|i| ((i as f32) * 0.023).cos())
        .collect();
    let v: Vec<f32> = (0..seq * state)
        .map(|i| ((i as f32) * 0.031).sin())
        .collect();

    let mut expected = vec![0.0_f32; seq * state];
    attention_decoder_causal_scalar(&q, &k, &v, seq, n_head, head_dim, scale, &mut expected);

    let backend = CpuKernelBackend::scalar();
    let q_d = backend.upload(&q).expect("upload q");
    let k_d = backend.upload(&k).expect("upload k");
    let v_d = backend.upload(&v).expect("upload v");
    let out_d = backend.alloc(seq * state).expect("alloc out");
    backend
        .attention_decoder_causal_d(&q_d, &k_d, &v_d, seq, n_head, head_dim, scale, &out_d)
        .expect("CPU attention_decoder_causal_d must succeed");
    let got = out_d.to_host_owned().expect("readback");
    assert_eq!(
        got, expected,
        "CPU causal decoder attention must be bit-identical to scalar"
    );
}

/// GW.4-5B: `attention_decoder_incremental_d` on the CPU backend must
/// match `attention_decoder_incremental_scalar` bit-for-bit.
#[test]
fn cpu_attention_decoder_incremental_d_matches_scalar_bit_for_bit() {
    let past_seq = 3usize;
    let n_head = 2usize;
    let head_dim = 4usize;
    let state = n_head * head_dim;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..state).map(|i| ((i as f32) * 0.041).sin()).collect();
    let past_k: Vec<f32> = (0..past_seq * state)
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();
    let past_v: Vec<f32> = (0..past_seq * state)
        .map(|i| ((i as f32) * 0.019).sin())
        .collect();
    let new_k: Vec<f32> = (0..state).map(|i| ((i as f32) * 0.027).cos()).collect();
    let new_v: Vec<f32> = (0..state).map(|i| ((i as f32) * 0.033).sin()).collect();

    let mut expected = vec![0.0_f32; state];
    attention_decoder_incremental_scalar(
        &q,
        &past_k,
        &past_v,
        &new_k,
        &new_v,
        past_seq,
        n_head,
        head_dim,
        scale,
        &mut expected,
    );

    let backend = CpuKernelBackend::scalar();
    let q_d = backend.upload(&q).expect("upload q");
    let past_k_d = backend.upload(&past_k).expect("upload past_k");
    let past_v_d = backend.upload(&past_v).expect("upload past_v");
    let new_k_d = backend.upload(&new_k).expect("upload new_k");
    let new_v_d = backend.upload(&new_v).expect("upload new_v");
    let out_d = backend.alloc(state).expect("alloc out");
    backend
        .attention_decoder_incremental_d(
            &q_d, &past_k_d, &past_v_d, &new_k_d, &new_v_d, past_seq, n_head, head_dim, scale,
            &out_d,
        )
        .expect("CPU attention_decoder_incremental_d must succeed");
    let got = out_d.to_host_owned().expect("readback");
    assert_eq!(
        got, expected,
        "CPU incremental decoder attention must be bit-identical to scalar"
    );
}

/// PostGW.1: cache append must be expressible as a backend device copy.
/// The CPU default path bounces through host, but it exercises the public
/// contract that GPU backends override without changing semantics.
#[test]
fn cpu_copy_into_d_updates_destination_window() {
    let backend = CpuKernelBackend::scalar();
    let src = backend.upload(&[9.0_f32, 8.0]).expect("upload src");
    let dst = backend
        .upload(&[0.0_f32, 1.0, 2.0, 3.0, 4.0])
        .expect("upload dst");

    backend
        .copy_into_d(&src, &dst, 2)
        .expect("copy into existing tensor");

    let got = dst.to_host_owned().expect("readback dst");
    assert_eq!(got, [0.0, 1.0, 9.0, 8.0, 4.0]);
}

#[test]
fn cpu_copy_into_d_rejects_out_of_range_window() {
    let backend = CpuKernelBackend::scalar();
    let src = backend.upload(&[1.0_f32, 2.0, 3.0]).expect("upload src");
    let dst = backend.upload(&[0.0_f32, 0.0]).expect("upload dst");

    let err = backend
        .copy_into_d(&src, &dst, 1)
        .expect_err("copy must reject a window past destination end");

    match err {
        OcelotlError::Kernel(KernelError { message, .. }) => {
            assert!(
                message.contains("copy_into_d"),
                "expected copy diagnostic, got {message}"
            );
        }
        other => panic!("expected KernelError, got {other:?}"),
    }
}

/// PostGW.1: fixed-capacity cache attention must match the older
/// `past || new` incremental contract while ignoring unused cache tail.
#[test]
fn cpu_attention_decoder_incremental_cache_d_matches_incremental_scalar() {
    let past_seq = 3usize;
    let visible_seq = past_seq + 1;
    let cache_capacity = 6usize;
    let n_head = 2usize;
    let head_dim = 4usize;
    let state = n_head * head_dim;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..state).map(|i| ((i as f32) * 0.041).sin()).collect();
    let past_k: Vec<f32> = (0..past_seq * state)
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();
    let past_v: Vec<f32> = (0..past_seq * state)
        .map(|i| ((i as f32) * 0.019).sin())
        .collect();
    let new_k: Vec<f32> = (0..state).map(|i| ((i as f32) * 0.027).cos()).collect();
    let new_v: Vec<f32> = (0..state).map(|i| ((i as f32) * 0.033).sin()).collect();

    let mut expected = vec![0.0_f32; state];
    attention_decoder_incremental_scalar(
        &q,
        &past_k,
        &past_v,
        &new_k,
        &new_v,
        past_seq,
        n_head,
        head_dim,
        scale,
        &mut expected,
    );

    let mut key_cache = vec![-99.0_f32; cache_capacity * state];
    let mut value_cache = vec![99.0_f32; cache_capacity * state];
    key_cache[..past_seq * state].copy_from_slice(&past_k);
    value_cache[..past_seq * state].copy_from_slice(&past_v);
    key_cache[past_seq * state..visible_seq * state].copy_from_slice(&new_k);
    value_cache[past_seq * state..visible_seq * state].copy_from_slice(&new_v);

    let backend = CpuKernelBackend::scalar();
    let q_d = backend.upload(&q).expect("upload q");
    let key_cache_d = backend.upload(&key_cache).expect("upload key cache");
    let value_cache_d = backend.upload(&value_cache).expect("upload value cache");
    let out_d = backend.alloc(state).expect("alloc out");
    backend
        .attention_decoder_incremental_cache_d(
            &q_d,
            &key_cache_d,
            &value_cache_d,
            visible_seq,
            cache_capacity,
            n_head,
            head_dim,
            scale,
            &out_d,
        )
        .expect("CPU cache attention must succeed");
    let got = out_d.to_host_owned().expect("readback");
    assert_eq!(
        got, expected,
        "cache-prefix attention must match past-plus-new incremental scalar"
    );
}

#[test]
fn cpu_attention_decoder_incremental_cache_append_d_matches_incremental_scalar_and_updates_cache() {
    let past_seq = 3usize;
    let visible_seq = past_seq + 1;
    let cache_capacity = 6usize;
    let n_head = 2usize;
    let head_dim = 4usize;
    let state = n_head * head_dim;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..state).map(|i| ((i as f32) * 0.041).sin()).collect();
    let past_k: Vec<f32> = (0..past_seq * state)
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();
    let past_v: Vec<f32> = (0..past_seq * state)
        .map(|i| ((i as f32) * 0.019).sin())
        .collect();
    let new_k: Vec<f32> = (0..state).map(|i| ((i as f32) * 0.027).cos()).collect();
    let new_v: Vec<f32> = (0..state).map(|i| ((i as f32) * 0.033).sin()).collect();

    let mut expected = vec![0.0_f32; state];
    attention_decoder_incremental_scalar(
        &q,
        &past_k,
        &past_v,
        &new_k,
        &new_v,
        past_seq,
        n_head,
        head_dim,
        scale,
        &mut expected,
    );

    let mut key_cache = vec![-99.0_f32; cache_capacity * state];
    let mut value_cache = vec![99.0_f32; cache_capacity * state];
    key_cache[..past_seq * state].copy_from_slice(&past_k);
    value_cache[..past_seq * state].copy_from_slice(&past_v);

    let backend = CpuKernelBackend::scalar();
    let q_d = backend.upload(&q).expect("upload q");
    let key_cache_d = backend.upload(&key_cache).expect("upload key cache");
    let value_cache_d = backend.upload(&value_cache).expect("upload value cache");
    let new_k_d = backend.upload(&new_k).expect("upload new k");
    let new_v_d = backend.upload(&new_v).expect("upload new v");
    let out_d = backend.alloc(state).expect("alloc out");
    backend
        .attention_decoder_incremental_cache_append_d(
            &q_d,
            &key_cache_d,
            &value_cache_d,
            &new_k_d,
            &new_v_d,
            past_seq,
            cache_capacity,
            n_head,
            head_dim,
            scale,
            &out_d,
        )
        .expect("CPU cache append attention must succeed");

    let got = out_d.to_host_owned().expect("readback");
    assert_eq!(got, expected);
    let got_key_cache = key_cache_d.to_host_owned().expect("read key cache");
    let got_value_cache = value_cache_d.to_host_owned().expect("read value cache");
    assert_eq!(
        &got_key_cache[past_seq * state..visible_seq * state],
        &new_k[..]
    );
    assert_eq!(
        &got_value_cache[past_seq * state..visible_seq * state],
        &new_v[..]
    );
}

#[test]
fn cpu_attention_decoder_incremental_cache_d_threaded_matches_scalar_bit_for_bit() {
    let visible_seq = PARALLEL_SDPA_MIN_SEQ;
    let cache_capacity = visible_seq + 3;
    let n_head = 2usize;
    let head_dim = 4usize;
    let state = n_head * head_dim;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..state).map(|i| ((i as f32) * 0.041).sin()).collect();
    let mut key_cache = vec![-99.0_f32; cache_capacity * state];
    let mut value_cache = vec![99.0_f32; cache_capacity * state];
    for i in 0..visible_seq * state {
        key_cache[i] = ((i as f32) * 0.013).cos();
        value_cache[i] = ((i as f32) * 0.019).sin();
    }

    let mut expected = vec![0.0_f32; state];
    attention_decoder_incremental_cache_scalar(
        &q,
        &key_cache[..visible_seq * state],
        &value_cache[..visible_seq * state],
        visible_seq,
        n_head,
        head_dim,
        scale,
        &mut expected,
    );

    let backend = CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Scalar, 4)
        .expect("threaded scalar backend must build");
    let q_d = backend.upload(&q).expect("upload q");
    let key_cache_d = backend.upload(&key_cache).expect("upload key cache");
    let value_cache_d = backend.upload(&value_cache).expect("upload value cache");
    let out_d = backend.alloc(state).expect("alloc out");
    backend
        .attention_decoder_incremental_cache_d(
            &q_d,
            &key_cache_d,
            &value_cache_d,
            visible_seq,
            cache_capacity,
            n_head,
            head_dim,
            scale,
            &out_d,
        )
        .expect("threaded cache attention must succeed");
    let got = out_d.to_host_owned().expect("readback");
    assert_eq!(
        got, expected,
        "threaded cache attention must match scalar bit-for-bit"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn cpu_attention_decoder_incremental_cache_d_threaded_avx2_matches_scalar_within_tolerance() {
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        return;
    }
    let visible_seq = PARALLEL_SDPA_MIN_SEQ;
    let cache_capacity = visible_seq + 3;
    let n_head = 2usize;
    let head_dim = 8usize;
    let state = n_head * head_dim;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..state).map(|i| ((i as f32) * 0.041).sin()).collect();
    let mut key_cache = vec![-99.0_f32; cache_capacity * state];
    let mut value_cache = vec![99.0_f32; cache_capacity * state];
    for i in 0..visible_seq * state {
        key_cache[i] = ((i as f32) * 0.013).cos();
        value_cache[i] = ((i as f32) * 0.019).sin();
    }

    let mut expected = vec![0.0_f32; state];
    attention_decoder_incremental_cache_scalar(
        &q,
        &key_cache[..visible_seq * state],
        &value_cache[..visible_seq * state],
        visible_seq,
        n_head,
        head_dim,
        scale,
        &mut expected,
    );

    let backend = CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Avx2, 4)
        .expect("threaded AVX2 backend must build");
    let q_d = backend.upload(&q).expect("upload q");
    let key_cache_d = backend.upload(&key_cache).expect("upload key cache");
    let value_cache_d = backend.upload(&value_cache).expect("upload value cache");
    let out_d = backend.alloc(state).expect("alloc out");
    backend
        .attention_decoder_incremental_cache_d(
            &q_d,
            &key_cache_d,
            &value_cache_d,
            visible_seq,
            cache_capacity,
            n_head,
            head_dim,
            scale,
            &out_d,
        )
        .expect("threaded AVX2 cache attention must succeed");
    let got = out_d.to_host_owned().expect("readback");
    assert_close_with_tolerance(&got, &expected, 1.0e-4, "threaded AVX2 cache attention");
}

/// GW.4-5B: `attention_decoder_causal_d` rejects wrong buffer lengths.
#[test]
fn validate_attention_decoder_causal_shapes_rejects_wrong_length() {
    let backend = CpuKernelBackend::scalar();
    // seq=3, n_head=2, head_dim=2 → expected len = 12. Give len=8 buffers.
    let q = backend.upload(&[0.0_f32; 8]).unwrap();
    let k = backend.upload(&[0.0_f32; 8]).unwrap();
    let v = backend.upload(&[0.0_f32; 8]).unwrap();
    let out = backend.alloc(8).unwrap();
    let err = backend
        .attention_decoder_causal_d(&q, &k, &v, 3, 2, 2, 0.5, &out)
        .expect_err("length mismatch must be rejected");
    match err {
        OcelotlError::Kernel(KernelError { message, .. }) => {
            assert!(
                message.contains("attention_decoder_causal_d"),
                "expected diagnostic, got {message}"
            );
        }
        other => panic!("expected KernelError, got {other:?}"),
    }
}

/// GW.4-5B: `attention_decoder_incremental_d` rejects wrong past_k length.
#[test]
fn validate_attention_decoder_incremental_shapes_rejects_wrong_past_length() {
    let backend = CpuKernelBackend::scalar();
    let n_head = 2usize;
    let head_dim = 4usize;
    let state = n_head * head_dim;
    let past_seq = 3usize;
    let q = backend.upload(&vec![0.0_f32; state]).unwrap();
    // Wrong: past_k is too short (2*state instead of 3*state).
    let past_k = backend.upload(&vec![0.0_f32; 2 * state]).unwrap();
    let past_v = backend.upload(&vec![0.0_f32; past_seq * state]).unwrap();
    let new_k = backend.upload(&vec![0.0_f32; state]).unwrap();
    let new_v = backend.upload(&vec![0.0_f32; state]).unwrap();
    let out = backend.alloc(state).unwrap();
    let err = backend
        .attention_decoder_incremental_d(
            &q, &past_k, &past_v, &new_k, &new_v, past_seq, n_head, head_dim, 0.5, &out,
        )
        .expect_err("length mismatch must be rejected");
    match err {
        OcelotlError::Kernel(KernelError { message, .. }) => {
            assert!(
                message.contains("attention_decoder_incremental_d"),
                "expected diagnostic, got {message}"
            );
        }
        other => panic!("expected KernelError, got {other:?}"),
    }
}

// -----------------------------------------------------------------------
// End GW.4-5B CPU tests
// -----------------------------------------------------------------------

/// Parity gate proving the scalar reference matches the
/// `attention_body_host`-style host attention on the same inputs. The
/// math here mirrors the Whisper primitive's loop body (head-major
/// `[seq, n_head, head_dim]`, scaled-dot → softmax → P·V); this test
/// guards against drift between the scalar reference and the host
/// primitive without depending on the models crate.
#[test]
fn attention_encoder_scalar_matches_hand_computed_softmax_chain() {
    // seq=2, n_head=1, head_dim=2: small enough to hand-verify.
    let q = vec![1.0_f32, 0.0, 0.0, 1.0];
    let k = vec![1.0_f32, 0.0, 0.0, 1.0];
    let v = vec![1.0_f32, 2.0, 3.0, 4.0];
    let scale = 1.0_f32 / (2.0_f32).sqrt();

    let mut out = vec![0.0_f32; 4];
    attention_encoder_scalar(&q, &k, &v, 2, 1, 2, scale, &mut out);

    // Row 0: Q = [1, 0]. Scores [1*1+0*0, 1*0+0*1]*scale = [1/√2, 0].
    // Softmax: exp(1/√2-1/√2)=1, exp(0-1/√2)=exp(-1/√2). Normalize.
    let s0 = 1.0_f32 / std::f32::consts::SQRT_2;
    let e0 = (s0 - s0).exp();
    let e1 = (0.0 - s0).exp();
    let denom = e0 + e1;
    let p0 = e0 / denom;
    let p1 = e1 / denom;
    let expected_row0 = [p0 * 1.0 + p1 * 3.0, p0 * 2.0 + p1 * 4.0];
    // Row 1: Q = [0, 1]. Scores [0*1+1*0, 0*0+1*1]*scale = [0, 1/√2].
    let e0r1 = (0.0 - s0).exp();
    let e1r1 = (s0 - s0).exp();
    let denom1 = e0r1 + e1r1;
    let p0r1 = e0r1 / denom1;
    let p1r1 = e1r1 / denom1;
    let expected_row1 = [p0r1 * 1.0 + p1r1 * 3.0, p0r1 * 2.0 + p1r1 * 4.0];

    for (got, want) in out[..2].iter().zip(expected_row0.iter()) {
        assert!(
            (got - want).abs() < 1e-6,
            "row 0 mismatch: got {got}, want {want}"
        );
    }
    for (got, want) in out[2..].iter().zip(expected_row1.iter()) {
        assert!(
            (got - want).abs() < 1e-6,
            "row 1 mismatch: got {got}, want {want}"
        );
    }
}

#[test]
fn validate_attention_encoder_shapes_rejects_wrong_length() {
    let backend = CpuKernelBackend::scalar();
    let q = backend.upload(&[0.0_f32; 8]).unwrap();
    let k = backend.upload(&[0.0_f32; 8]).unwrap();
    let v = backend.upload(&[0.0_f32; 8]).unwrap();
    let out = backend.alloc(8).unwrap();
    // seq=4, n_head=1, head_dim=2 → expected 8. But pass seq=4, n_head=2
    // so expected = 16.
    let err = backend
        .attention_encoder_d(&q, &k, &v, 4, 2, 2, 0.5, &out)
        .expect_err("length mismatch must be rejected");
    match err {
        OcelotlError::Kernel(KernelError { message, .. }) => {
            assert!(
                message.contains("attention_encoder_d"),
                "expected diagnostic, got {message}"
            );
        }
        other => panic!("expected KernelError, got {other:?}"),
    }
}

// GW.4-5C: decoder cross-attention scalar oracle — no causal mask,
// q_seq decoder rows attend all kv_seq encoder positions freely.
#[test]
fn attention_decoder_cross_scalar_matches_attention_body_host() {
    // Q from decoder: [q_seq=3, state], K/V from encoder: [kv_seq=5, state].
    // Verifies Q rows attend all 5 encoder positions (no causal restriction).
    let q_seq = 3usize;
    let kv_seq = 5usize;
    let n_head = 2usize;
    let head_dim = 4usize;
    let state = n_head * head_dim;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..q_seq * state)
        .map(|i| ((i as f32) * 0.017).sin())
        .collect();
    let k: Vec<f32> = (0..kv_seq * state)
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();
    let v: Vec<f32> = (0..kv_seq * state)
        .map(|i| ((i as f32) * 0.023).sin())
        .collect();

    let mut got = vec![0.0_f32; q_seq * state];
    attention_decoder_cross_scalar(&q, &k, &v, q_seq, kv_seq, n_head, head_dim, scale, &mut got);

    // Reference: attention_encoder_scalar with kv_seq != q_seq is not
    // directly applicable, so compute the reference inline using the same
    // algorithm attention_body_host uses with causal=false.
    let mut expected = vec![0.0_f32; q_seq * state];
    for qi in 0..q_seq {
        for head in 0..n_head {
            let q_base = qi * state + head * head_dim;
            let mut scores = vec![0.0_f32; kv_seq];
            for (ki, score) in scores.iter_mut().enumerate() {
                let k_base = ki * state + head * head_dim;
                let mut acc = 0.0_f32;
                for d in 0..head_dim {
                    acc += q[q_base + d] * k[k_base + d];
                }
                *score = acc * scale;
            }
            softmax(&mut scores);
            let out_base = qi * state + head * head_dim;
            for d in 0..head_dim {
                expected[out_base + d] = 0.0;
            }
            for (ki, &p) in scores.iter().enumerate() {
                let v_base = ki * state + head * head_dim;
                for d in 0..head_dim {
                    expected[out_base + d] += p * v[v_base + d];
                }
            }
        }
    }

    assert_eq!(got.len(), expected.len());
    for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-6,
            "cross scalar mismatch at idx {idx}: got={g} expected={e}"
        );
    }
}

// GW.4-5C/PostGW.2: `attention_decoder_cross_d` on CPU must produce the
// same output as the scalar oracle. The CPU backend now borrows host slices
// directly instead of falling through the cloning trait default.
#[test]
fn attention_decoder_cross_d_cpu_matches_scalar_oracle() {
    let q_seq = 3usize;
    let kv_seq = 5usize;
    let n_head = 2usize;
    let head_dim = 4usize;
    let state = n_head * head_dim;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..q_seq * state)
        .map(|i| ((i as f32) * 0.017).sin())
        .collect();
    let k: Vec<f32> = (0..kv_seq * state)
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();
    let v: Vec<f32> = (0..kv_seq * state)
        .map(|i| ((i as f32) * 0.023).sin())
        .collect();

    let mut expected = vec![0.0_f32; q_seq * state];
    attention_decoder_cross_scalar(
        &q,
        &k,
        &v,
        q_seq,
        kv_seq,
        n_head,
        head_dim,
        scale,
        &mut expected,
    );

    let backend = CpuKernelBackend::scalar();
    let q_d = DeviceTensor::from_host(q.clone());
    let k_d = DeviceTensor::from_host(k.clone());
    let v_d = DeviceTensor::from_host(v.clone());
    let out_d = DeviceTensor::host_zeros(q_seq * state);
    backend
        .attention_decoder_cross_d(
            &q_d, &k_d, &v_d, q_seq, kv_seq, n_head, head_dim, scale, &out_d,
        )
        .expect("attention_decoder_cross_d must succeed");
    let got = out_d.to_host_owned().expect("readback");

    assert_eq!(got.len(), expected.len());
    for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-6,
            "cross_d default mismatch at idx {idx}: got={g} expected={e}"
        );
    }
}

#[test]
fn cpu_attention_decoder_cross_d_threaded_matches_scalar_bit_for_bit() {
    let q_seq = 3usize;
    let kv_seq = PARALLEL_SDPA_MIN_SEQ;
    let n_head = 2usize;
    let head_dim = 4usize;
    let state = n_head * head_dim;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..q_seq * state)
        .map(|i| ((i as f32) * 0.017).sin())
        .collect();
    let k: Vec<f32> = (0..kv_seq * state)
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();
    let v: Vec<f32> = (0..kv_seq * state)
        .map(|i| ((i as f32) * 0.023).sin())
        .collect();

    let mut expected = vec![0.0_f32; q_seq * state];
    attention_decoder_cross_scalar(
        &q,
        &k,
        &v,
        q_seq,
        kv_seq,
        n_head,
        head_dim,
        scale,
        &mut expected,
    );

    let backend = CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Scalar, 4)
        .expect("threaded scalar backend must build");
    let q_d = backend.upload(&q).expect("upload q");
    let k_d = backend.upload(&k).expect("upload k");
    let v_d = backend.upload(&v).expect("upload v");
    let out_d = backend.alloc(q_seq * state).expect("alloc out");
    backend
        .attention_decoder_cross_d(
            &q_d, &k_d, &v_d, q_seq, kv_seq, n_head, head_dim, scale, &out_d,
        )
        .expect("threaded cross attention must succeed");
    let got = out_d.to_host_owned().expect("readback");
    assert_eq!(
        got, expected,
        "threaded cross attention must match scalar bit-for-bit"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn cpu_attention_decoder_cross_d_threaded_avx2_matches_scalar_within_tolerance() {
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        return;
    }
    let q_seq = 3usize;
    let kv_seq = PARALLEL_SDPA_MIN_SEQ;
    let n_head = 2usize;
    let head_dim = 8usize;
    let state = n_head * head_dim;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..q_seq * state)
        .map(|i| ((i as f32) * 0.017).sin())
        .collect();
    let k: Vec<f32> = (0..kv_seq * state)
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();
    let v: Vec<f32> = (0..kv_seq * state)
        .map(|i| ((i as f32) * 0.023).sin())
        .collect();

    let mut expected = vec![0.0_f32; q_seq * state];
    attention_decoder_cross_scalar(
        &q,
        &k,
        &v,
        q_seq,
        kv_seq,
        n_head,
        head_dim,
        scale,
        &mut expected,
    );

    let backend = CpuKernelBackend::with_mode_and_threads(CpuKernelMode::Avx2, 4)
        .expect("threaded AVX2 backend must build");
    let q_d = backend.upload(&q).expect("upload q");
    let k_d = backend.upload(&k).expect("upload k");
    let v_d = backend.upload(&v).expect("upload v");
    let out_d = backend.alloc(q_seq * state).expect("alloc out");
    backend
        .attention_decoder_cross_d(
            &q_d, &k_d, &v_d, q_seq, kv_seq, n_head, head_dim, scale, &out_d,
        )
        .expect("threaded AVX2 cross attention must succeed");
    let got = out_d.to_host_owned().expect("readback");
    assert_close_with_tolerance(&got, &expected, 1.0e-4, "threaded AVX2 cross attention");
}
