//! M3.7 — pinned tiny-synthetic prefill logits.
//!
//! End-to-end determinism test for `Qwen2_5Model::prefill`. Builds a
//! tiny Qwen2.5-shaped model with deterministic synthetic weights and
//! asserts the final-token logits match a committed fixture within a
//! documented tolerance. This is the proof that the M3.7 prefill path
//! is reproducible across builds, hosts, and minor refactors that
//! shouldn't change numerical behavior.
//!
//! # Why two layers and tiny dims
//!
//! - Two layers exercises the residual-stream wiring: a one-layer model
//!   would not catch a bug where the post-attention residual was wired
//!   to the wrong buffer.
//! - hidden=16, intermediate=32, vocab=32 keeps the worst-case f32
//!   accumulation under control. With ~16 dot-product accumulations per
//!   matmul cell and ~32 in the attention reduction, the worst-case
//!   relative drift through the chain stays under 1e-5; we pin at
//!   1e-4 to leave slack for cross-platform `f32::sin/cos/exp`
//!   differences (the M3.5 attention test already noted ~2e-6 drift on
//!   a much shorter chain).
//!
//! # Tolerance budget (1e-4)
//!
//! The prefill chain depth, in order:
//! 1. embedding lookup (no math)
//! 2. RMSNorm: 1 sum-of-squares reduction over 16 (≈1 ULP) + sqrt + div + mul
//! 3. 4 matmuls at hidden->q/k/v + bias adds (k=16 each)
//! 4. RoPE: cos/sin per pair (≈4 ULP each)
//! 5. attention: dot of head_dim=4, softmax over seq, weighted accumulation
//! 6. o_proj matmul (k=16) + residual add
//! 7. RMSNorm again
//! 8. MLP: 3 matmuls (k=16, k=16, k=32) + silu (1 div, 1 exp ≈ 4 ULP)
//! 9. residual add
//!
//! Two layers of (3..9) plus a final RMSNorm + lm_head matmul (k=16).
//! Empirical worst-case observed in the project to date: ~2e-6 over the
//! M3.5 single-layer attention test. Multiplying by the deeper chain
//! and adding cross-platform trig drift gives an estimated ~5e-5 worst
//! case. 1e-4 leaves an order-of-magnitude headroom and still catches
//! a logic bug (those typically move logits by >>1e-3).
//!
//! # Regenerating the fixture
//!
//! When the prefill math intentionally changes (kernel swap, different
//! activation, etc.), regenerate by running:
//!
//! ```bash
//! cargo test -p ocelotl-models --test qwen2_5_tiny_synthetic_prefill -- \
//!     prefill_matches_fixture --nocapture --include-ignored
//! ```
//!
//! The `prefill_matches_fixture` test prints the actual logits when
//! `OCELOTL_PRINT_LOGITS=1` is set; copy them into the JSON fixture and
//! commit alongside the impl change.

use ocelotl_core::{DType, TokenId};
use ocelotl_models::qwen::{
    Qwen2_5Config, Qwen2_5LayerWeights, Qwen2_5Model, Qwen2_5Weights, transpose_2d,
};

/// Fixture path: kept in `fixtures/logits/` alongside the M1 smoke fixture
/// per the project's "fixtures live at the repo root, not inside crates"
/// convention.
const FIXTURE_PATH: &str = "../../fixtures/logits/qwen2_5_tiny_synthetic_prefill.json";

/// Tolerance for f32 prefill parity. See module docs for derivation.
const TOLERANCE: f32 = 1.0e-4;

/// The synthetic-weight generator MUST match the one used to capture the
/// fixture values. Changing this formula or the seed schema is a fixture-
/// regeneration event.
fn synth(seed: u32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (seed as f32 * 0.123) + (i as f32 * 0.0177);
            0.05 * x.sin()
        })
        .collect()
}

fn tiny_config() -> Qwen2_5Config {
    Qwen2_5Config {
        vocab_size: 32,
        num_hidden_layers: 2,
        hidden_size: 16,
        intermediate_size: 32,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 4,
        context_length: 128,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        dtype: DType::F32,
    }
}

fn tiny_weights(cfg: &Qwen2_5Config) -> Qwen2_5Weights {
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let q_out = cfg.num_attention_heads * cfg.head_dim;
    let kv_out = cfg.num_key_value_heads * cfg.head_dim;
    let i_size = cfg.intermediate_size;

    let embed = synth(1, v * h);
    let lm_head_w = transpose_2d(&embed, v, h);

    let layers = (0..cfg.num_hidden_layers)
        .map(|i| {
            let s = (i as u32) * 100 + 10;
            Qwen2_5LayerWeights {
                q_proj_w: synth(s, h * q_out),
                q_proj_b: synth(s + 1, q_out),
                k_proj_w: synth(s + 2, h * kv_out),
                k_proj_b: synth(s + 3, kv_out),
                v_proj_w: synth(s + 4, h * kv_out),
                v_proj_b: synth(s + 5, kv_out),
                o_proj_w: synth(s + 6, q_out * h),
                input_layernorm_w: vec![1.0; h],
                post_attention_layernorm_w: vec![1.0; h],
                gate_proj_w: synth(s + 7, h * i_size),
                up_proj_w: synth(s + 8, h * i_size),
                down_proj_w: synth(s + 9, i_size * h),
            }
        })
        .collect();

    Qwen2_5Weights {
        embed_tokens: embed,
        layers,
        final_norm_w: vec![1.0; h],
        lm_head_w,
        tie_word_embeddings: true,
    }
}

#[derive(Debug, serde::Deserialize)]
struct LogitsFixture {
    /// Identifier for the model shape used to compute the fixture.
    model_shape: String,
    /// Token ids of the prompt that produced the pinned logits.
    prompt_tokens: Vec<u32>,
    /// Pinned final-position logits over the vocabulary.
    expected_logits: Vec<f32>,
    /// The tolerance under which the test compares against `expected_logits`.
    /// Pinned in the fixture so a tolerance-only change (loosening or
    /// tightening) shows up in the diff.
    tolerance: f32,
    /// Free-form rationale for the fixture and tolerance choices.
    rationale: String,
}

#[test]
fn prefill_matches_pinned_fixture_within_tolerance() {
    // Load the fixture from disk. The path is relative to the test crate's
    // manifest dir (CARGO_MANIFEST_DIR), per cargo's documented behavior
    // for integration tests.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR must be set when running cargo test");
    let path = std::path::Path::new(&manifest_dir).join(FIXTURE_PATH);
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    let fixture: LogitsFixture = serde_json::from_str(&json)
        .unwrap_or_else(|e| panic!("parse fixture {}: {e}", path.display()));

    // Sanity: the fixture must be the one we expect. If the model_shape
    // string drifts, the test should fail loudly rather than silently
    // compare against the wrong artifact.
    assert_eq!(
        fixture.model_shape, "qwen2_5_tiny_synthetic_prefill",
        "fixture model_shape must match this test"
    );
    assert!(
        (fixture.tolerance - TOLERANCE).abs() < f32::EPSILON,
        "fixture tolerance ({}) must match the test's TOLERANCE ({}) — \
         a tolerance change is a deliberate event, not silent drift",
        fixture.tolerance,
        TOLERANCE,
    );

    let cfg = tiny_config();
    let weights = tiny_weights(&cfg);
    let model = Qwen2_5Model::new(cfg.clone(), weights).expect("tiny model must construct");

    let prompt: Vec<TokenId> = fixture.prompt_tokens.iter().copied().map(TokenId).collect();
    let logits = model.prefill(&prompt).expect("prefill must succeed");

    // Optional debug: when regenerating the fixture, set
    // OCELOTL_PRINT_LOGITS=1 and re-run with --nocapture to dump the
    // actual logits in JSON-array form.
    if std::env::var("OCELOTL_PRINT_LOGITS").as_deref() == Ok("1") {
        let formatted: Vec<String> = logits.iter().map(|v| format!("{v:.7}")).collect();
        eprintln!("logits = [{}]", formatted.join(", "));
    }

    assert_eq!(
        logits.len(),
        fixture.expected_logits.len(),
        "logits length must match fixture (vocab_size = {})",
        cfg.vocab_size,
    );

    for (i, (got, want)) in logits
        .iter()
        .zip(fixture.expected_logits.iter())
        .enumerate()
    {
        let diff = (got - want).abs();
        assert!(
            diff < TOLERANCE,
            "logit {i}: got {got}, want {want}, diff {diff} exceeds tolerance {TOLERANCE}\n\
             rationale: {}",
            fixture.rationale,
        );
    }
}

#[cfg(feature = "cubecl-wgpu")]
#[test]
#[ignore = "requires a CubeCL WGPU-capable local runtime"]
fn cubecl_wgpu_prefill_matches_cpu_reference_within_tolerance() {
    use std::sync::Arc;

    let cfg = tiny_config();
    let cpu =
        Qwen2_5Model::new(cfg.clone(), tiny_weights(&cfg)).expect("CPU tiny model must construct");
    let cubecl = Qwen2_5Model::with_kernel_backend(
        cfg.clone(),
        tiny_weights(&cfg),
        Arc::new(ocelotl_kernels::CubeClKernelBackend::new_gpu(0)),
    )
    .expect("CubeCL tiny model must construct");

    assert_eq!(cubecl.execution_backend().name(), "cubecl");
    let prompt = [TokenId(3), TokenId(7), TokenId(11)];
    let expected = cpu.prefill(&prompt).expect("CPU prefill must succeed");
    let actual = cubecl
        .prefill(&prompt)
        .expect("CubeCL prefill must succeed");

    assert_eq!(actual.len(), expected.len());
    for (idx, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff < TOLERANCE,
            "CubeCL-backed prefill logit {idx}: got {got}, want {want}, diff {diff} exceeds {TOLERANCE}"
        );
    }
}
