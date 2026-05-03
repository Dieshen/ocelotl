//! M1 climax: integration smoke test.
//!
//! Sends a fixed prompt token through `generate_one_token` and asserts the
//! sampled next token matches the value pinned at
//! `fixtures/logits/m1_smoke_expected.json`. This is what proves the M1
//! CPU reference path is wired end to end through the public API:
//! request validation, the synthetic forward (matmul kernel), and greedy
//! sampling.
//!
//! The test deliberately does not import any private module from any
//! crate. Everything it touches is the public surface server code will
//! call in M2.
//!
//! When the synthetic forward intentionally changes (different SCALE,
//! different formula, anything that moves logits), regenerate the value
//! below and update the fixture file in lockstep. Both the assertion and
//! the fixture must be updated together — they are one truth, expressed
//! twice on purpose so a stale fixture is detectable on review.

use ocelotl_core::{DType, GenerationOptions, ModelMetadata, TokenId};
use ocelotl_runtime::{GenerateRequest, generate_one_token};

/// The token the deterministic synthetic forward produces for the prompt
/// `[TokenId(7)]` against the qwen2_5_tiny_synthetic shape. Pinned by
/// running the test once with a deliberately wrong value and capturing the
/// actual one from the assertion failure. Lives in lockstep with
/// `fixtures/logits/m1_smoke_expected.json`.
const EXPECTED_NEXT: u32 = 5;

#[test]
fn m1_cpu_reference_smoke_produces_expected_token() {
    let model = ModelMetadata {
        architecture: "qwen2".to_string(),
        vocab_size: 32,
        num_hidden_layers: 2,
        hidden_size: 16,
        intermediate_size: 32,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 4,
        context_length: 128,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        dtype: DType::F32,
        tokenizer_model_hint: None,
    };

    let req = GenerateRequest {
        prompt_tokens: vec![TokenId(7)],
        // max_new_tokens is set explicitly rather than via Default::default()
        // because the default is 256, which would fail validation against
        // the 128-token context. M1 only generates one token regardless;
        // the request just needs a positive budget that fits.
        options: GenerationOptions {
            max_new_tokens: 1,
            temperature: None,
        },
    };

    let resp = generate_one_token(&req, &model).expect("M1 smoke path must succeed");

    assert_eq!(
        resp.tokens.len(),
        1,
        "generate_one_token must return exactly one token"
    );
    assert_eq!(
        resp.tokens[0],
        TokenId(EXPECTED_NEXT),
        "M1 smoke path must produce the pinned token; if you intentionally \
         changed the synthetic forward, also update \
         fixtures/logits/m1_smoke_expected.json"
    );
}
