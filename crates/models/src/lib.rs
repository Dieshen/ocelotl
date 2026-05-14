//! Model-family implementations.
//!
//! M1 ships a single deterministic synthetic decoder-only forward pass that
//! exercises the full CPU reference path (`ocelotl_kernels::matmul` plus
//! `ocelotl_runtime::greedy_sample`) without implementing real attention,
//! RoPE, or RMSNorm. Its only job is to produce reproducible logits given a
//! fixed prompt so the M1 integration test can prove the wiring works end
//! to end. The real Qwen2.5 implementation now lives under `qwen`; the
//! synthetic path remains as a small compatibility fixture for the original
//! M1 runtime smoke test.

pub mod gemma;
pub mod qwen;
pub mod whisper;
pub use gemma::{Gemma4Config, Gemma4Quantization};
pub use qwen::{
    Qwen2_5Config, Qwen2_5LayerWeights, Qwen2_5Model, Qwen2_5Weights, Qwen3_5Config,
    parse_qwen3_5_config_json, required_tensor_names, transpose_2d, validate_qwen2_5_tensors,
};

use ocelotl_core::{InvalidRequestError, ModelMetadata, OcelotlError, Result, TokenId};
use ocelotl_kernels::matmul;

/// A scale factor for the deterministic synthetic forward.
///
/// Chosen so that `cos(v * h * SCALE)` traverses several oscillations
/// across the (vocab × hidden) grid — without that, the cos terms stay
/// near 1 for low indices and the argmax is trivially `0` for every
/// prompt, which would make the M1 smoke test pass even if the runtime
/// were short-circuited. With this value, the argmax depends
/// non-trivially on the prompt token (e.g. prompt `[7]` argmaxes at `5`,
/// prompt `[8]` at `7`).
///
/// The exact value is not load-bearing for correctness, but **changing it
/// requires regenerating** the committed expected-token fixture at
/// `fixtures/logits/m1_smoke_expected.json` and updating the integration
/// test's pinned next-token value in lockstep.
const SYNTHETIC_SCALE: f32 = 0.1;

/// Deterministic synthetic forward pass over the last prompt token.
///
/// Returns logits over the vocabulary corresponding to the position
/// **after** the final prompt token. The output is fully reproducible
/// without committing weight files: both the (synthetic) embedding and the
/// (synthetic) output projection are derived from index trigonometry.
///
/// # Pipeline
///
/// 1. Embed the last prompt token into a `hidden_size`-dimensional vector:
///    `hidden[i] = sin((token_id + 1) * (i + 1) * SCALE)`.
/// 2. Build an output projection matrix `W_out` of shape
///    `(vocab_size, hidden_size)` with
///    `W_out[v, h] = cos((v + 1) * (h + 1) * SCALE)`.
/// 3. `logits = W_out @ hidden_vec` via `kernels::matmul` with shapes
///    `(vocab, hidden) @ (hidden, 1) → (vocab, 1)`.
///
/// # Why matmul, not a hand-written dot loop
///
/// The projection is conceptually a single linear op
/// `logits = W_out @ hidden_vec`. Per-row `kernels::dot` would compute the
/// same numbers at the same `O(vocab * hidden)` cost, but it would obscure
/// the fact that this is one matrix-vector multiply and would put the
/// outer loop in user code rather than the kernel. The pair (Matt + James)
/// landed on `matmul` because the kernel is the right level of abstraction
/// for the operation, the test surface is smaller (we exercise one
/// kernel-call shape rather than `vocab` calls), and there is no
/// hand-rolled accumulation that GPU kernels would later need to re-prove.
///
/// # Errors
///
/// Returns `InvalidRequest` when `prompt_tokens` is empty. The runtime's
/// `validate_request` is expected to have already rejected this upstream;
/// the model-boundary check is belt-and-braces. Propagates kernel errors
/// from `matmul` if the shapes derived from the metadata are inconsistent.
///
/// # Determinism
///
/// Output is bit-identical for identical inputs. **Changing
/// `SYNTHETIC_SCALE` or the embedding/projection formulae requires
/// regenerating `fixtures/logits/m1_smoke_expected.json` and updating the
/// integration test's pinned next-token value.**
pub fn tiny_synthetic_forward(
    model: &ModelMetadata,
    prompt_tokens: &[TokenId],
) -> Result<Vec<f32>> {
    if prompt_tokens.is_empty() {
        return Err(OcelotlError::InvalidRequest(InvalidRequestError {
            field: "prompt_tokens".to_string(),
            message: "tiny_synthetic_forward requires at least one prompt token".to_string(),
        }));
    }

    let vocab = model.vocab_size;
    let hidden = model.hidden_size;

    let last_token = prompt_tokens
        .last()
        .copied()
        .expect("non-empty checked above");

    // Step 1: synthetic embedding of the last prompt token.
    let token_f = (last_token.0 as f32) + 1.0;
    let mut hidden_vec = Vec::with_capacity(hidden);
    for i in 0..hidden {
        let arg = token_f * ((i as f32) + 1.0) * SYNTHETIC_SCALE;
        hidden_vec.push(arg.sin());
    }

    // Step 2: synthetic output projection W_out, shape (vocab, hidden).
    let mut w_out = Vec::with_capacity(vocab * hidden);
    for v in 0..vocab {
        let v_f = (v as f32) + 1.0;
        for h in 0..hidden {
            let arg = v_f * ((h as f32) + 1.0) * SYNTHETIC_SCALE;
            w_out.push(arg.cos());
        }
    }

    // Step 3: logits = W_out @ hidden_vec.
    // Shapes: (vocab, hidden) @ (hidden, 1) → (vocab, 1).
    let mut logits = vec![0.0_f32; vocab];
    matmul(
        &w_out,
        (vocab, hidden),
        &hidden_vec,
        (hidden, 1),
        &mut logits,
    )?;

    Ok(logits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ocelotl_core::DType;

    fn tiny_model() -> ModelMetadata {
        // Same shape as the qwen2_5_tiny_synthetic metadata fixture, kept
        // in step deliberately so this test and the integration smoke test
        // exercise the same model.
        ModelMetadata {
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
        }
    }

    #[test]
    fn tiny_synthetic_forward_returns_one_logit_per_vocab_entry() {
        let model = tiny_model();
        let prompt = [TokenId(7)];

        let logits =
            tiny_synthetic_forward(&model, &prompt).expect("non-empty prompt must succeed");

        assert_eq!(logits.len(), model.vocab_size);
        for v in &logits {
            assert!(v.is_finite(), "synthetic logits must be finite, got {v}");
        }
    }

    #[test]
    fn tiny_synthetic_forward_is_deterministic_for_identical_inputs() {
        let model = tiny_model();
        let prompt = [TokenId(7)];

        let a = tiny_synthetic_forward(&model, &prompt).unwrap();
        let b = tiny_synthetic_forward(&model, &prompt).unwrap();

        assert_eq!(a, b, "identical inputs must yield bit-identical logits");
    }

    #[test]
    fn tiny_synthetic_forward_rejects_empty_prompt_with_invalid_request() {
        let model = tiny_model();
        let prompt: [TokenId; 0] = [];

        let err = tiny_synthetic_forward(&model, &prompt)
            .expect_err("empty prompt must be rejected at the model boundary");

        match err {
            OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "prompt_tokens");
                assert!(
                    invalid.message.contains("tiny_synthetic_forward"),
                    "expected error to name the function, got {:?}",
                    invalid.message
                );
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn tiny_synthetic_forward_responds_to_different_last_tokens() {
        // The forward only sees the *last* token. Two prompts with different
        // last tokens must produce different logits — otherwise we have a
        // bug where the synthetic forward ignores its input.
        let model = tiny_model();

        let logits_a = tiny_synthetic_forward(&model, &[TokenId(7)]).unwrap();
        let logits_b = tiny_synthetic_forward(&model, &[TokenId(8)]).unwrap();

        assert_ne!(
            logits_a, logits_b,
            "different last-token must yield different logits"
        );
    }
}
