//! M3.8 -- pinned tiny-synthetic one-token decode.
//!
//! End-to-end determinism + pinning test for `runtime::decode_one_token`.
//! Builds the *same* tiny Qwen2.5-shaped synthetic model that the M3.7
//! prefill fixture is captured against, runs the prompt through the
//! public runtime decode path, and asserts:
//!
//! 1. Determinism: identical inputs yield bit-identical TokenIds.
//! 2. Pinning: a specific (prompt, model) pair produces `TokenId(16)` --
//!    the argmax of the M3.7-pinned final-token logits.
//! 3. Validation propagation: an empty prompt produces the same
//!    `InvalidRequest(tokens, ...)` `runtime::prefill` would.
//!
//! # Why share the M3.7 model + prompt
//!
//! The M3.7 integration test pins
//! `expected_logits = [..., 0.7908793 at index 16, ...]` for the prompt
//! `[3, 7, 11]` against a specific synthetic model. Decoding one token
//! with greedy argmax on those exact logits MUST select index 16
//! (lowest-index tie-break is irrelevant here -- 0.7908793 is a strict
//! maximum; the next-largest value is 0.7776330 at index 15, comfortably
//! below).
//! This pinning is therefore not an independently captured number: it's
//! a direct consequence of the M3.7 fixture + the M1.8 greedy_sample
//! contract. If either changes, this test fails -- which is exactly
//! the kind of guard the brief asks for.
//!
//! # Crate-scope discipline
//!
//! The synthetic config + weight builder is duplicated from
//! `crates/models/tests/qwen2_5_tiny_synthetic_prefill.rs` because Rust
//! integration tests cannot share helpers across crate boundaries
//! without an extra published library. Keeping the duplication local
//! is cheaper than introducing a `dev-utils` crate for two callers.
//! The `synth(seed, len)` formula and seed schema MUST stay in lockstep
//! with the M3.7 builder; the only way the two models disagree is if
//! one of these files drifts. A future refactor that extracts both
//! into a `crates/models/src/test_support.rs` (gated behind a
//! `test-support` feature) is worth doing once a third caller appears.

use ocelotl_core::{DType, OcelotlError, TokenId};
use ocelotl_models::{
    Qwen2_5Config, Qwen2_5LayerWeights, Qwen2_5Model, Qwen2_5Weights, transpose_2d,
};
use ocelotl_runtime::decode_one_token;

/// The TokenId M3.8 pins for the M3.7 prompt `[3, 7, 11]` on this tiny
/// synthetic Qwen2.5 model. Derived from the M3.7 fixture: the largest
/// entry of `expected_logits` (`0.7908793`) is at index 16. With the
/// M1.8 greedy_sample contract (strict-`>` argmax, lowest-id tie-break)
/// that index is the deterministic decode result. See module docs for
/// the chain of reasoning.
const EXPECTED_DECODED_TOKEN: TokenId = TokenId(16);

/// MUST match `synth` in `crates/models/tests/qwen2_5_tiny_synthetic_prefill.rs`.
/// Changing this formula or the seed schema is a fixture-regeneration
/// event in BOTH places.
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

#[test]
fn decode_one_token_is_deterministic_for_identical_inputs() {
    let cfg = tiny_config();
    let weights = tiny_weights(&cfg);
    let model = Qwen2_5Model::new(cfg, weights).expect("tiny model must construct");

    let prompt = [TokenId(3), TokenId(7), TokenId(11)];

    let a = decode_one_token(&model, &prompt).expect("decode must succeed");
    let b = decode_one_token(&model, &prompt).expect("decode must succeed (second call)");

    assert_eq!(
        a, b,
        "identical inputs must yield bit-identical decoded TokenIds"
    );
}

#[test]
fn decode_one_token_matches_pinned_argmax_of_m3_7_fixture() {
    // Pinning test. The M3.7 fixture pins
    // expected_logits[16] = 0.7908793 as the maximum entry of the
    // final-position logits for prompt [3, 7, 11] on this tiny model.
    // greedy_sample picks the strict argmax with lowest-id tie-break;
    // the maximum here is strict, so the pinned token id is 16.
    let cfg = tiny_config();
    let weights = tiny_weights(&cfg);
    let model = Qwen2_5Model::new(cfg, weights).expect("tiny model must construct");

    let prompt = [TokenId(3), TokenId(7), TokenId(11)];

    let token = decode_one_token(&model, &prompt).expect("decode must succeed");

    assert_eq!(
        token, EXPECTED_DECODED_TOKEN,
        "decode_one_token([3, 7, 11]) on the M3.7 tiny synthetic model \
         must select TokenId(16) -- the argmax of the M3.7-pinned logits. \
         If this fails, either prefill numerics drifted (check the M3.7 \
         fixture test first) or greedy_sample's tie-break contract changed."
    );
}

#[test]
fn decode_one_token_propagates_invalid_request_for_empty_prompt() {
    // The runtime's decode path goes through runtime::prefill, which
    // goes through Qwen2_5Model::prefill. The empty-prompt rejection
    // is at the model boundary; it must surface unchanged through
    // both layers of the public API. No swallowing, no remapping.
    let cfg = tiny_config();
    let weights = tiny_weights(&cfg);
    let model = Qwen2_5Model::new(cfg, weights).expect("tiny model must construct");

    let err = decode_one_token(&model, &[]).expect_err("empty prompt must be rejected");

    match err {
        OcelotlError::InvalidRequest(invalid) => {
            assert_eq!(invalid.field, "tokens");
            assert!(
                invalid.message.contains("at least one"),
                "expected the model-boundary 'at least one' message to \
                 propagate verbatim through runtime::decode_one_token, \
                 got {:?}",
                invalid.message,
            );
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}
