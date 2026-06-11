//! MF.7 -- pinned tiny-synthetic Gemma4 text prefill.
//!
//! This test is intentionally narrower than the selected real Gemma4 Q4_K_M
//! artifact. It pins the first supported execution subset: text-only, dense
//! F32 weights, no multimodal inputs, and synthetic shared-KV attention. The
//! real GGUF artifact remains rejected until multimodal execution and reference
//! parity are complete.

use ocelotl_core::{OcelotlError, TokenId};
use ocelotl_models::gemma::{
    Gemma4Config, Gemma4Quantization, Gemma4TextLayerWeights, Gemma4TextModel, Gemma4TextWeights,
};
use ocelotl_runtime::gemma::{decode_one_token, prefill};

const FIXTURE_PATH: &str = "../../fixtures/logits/gemma4_tiny_synthetic_text_prefill.json";
const TOLERANCE: f32 = 1.0e-4;
const SLIDING_WINDOW_DRIFT_TOLERANCE: f32 = 1.0e-8;

fn synth(seed: u32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (seed as f32 * 0.071) + (i as f32 * 0.013);
            0.04 * x.sin()
        })
        .collect()
}

fn transpose_2d(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), rows * cols);
    let mut dst = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
}

fn tiny_config() -> Gemma4Config {
    Gemma4Config {
        context_length: 64,
        block_count: 2,
        embedding_length: 16,
        embedding_length_per_layer_input: 16,
        feed_forward_length: 32,
        attention_head_count: 4,
        attention_head_count_kv: 2,
        attention_key_length: 4,
        attention_value_length: 4,
        attention_key_length_swa: 4,
        attention_value_length_swa: 4,
        rope_dimension_count: 4,
        rope_dimension_count_swa: 4,
        rope_freq_base: 10_000.0,
        rope_freq_base_swa: 10_000.0,
        rms_norm_eps: 1.0e-6,
        attention_sliding_window: None,
        attention_shared_kv_layers: None,
        attention_sliding_window_pattern: None,
        final_logit_softcap: None,
        tokenizer_model: Some("gemma4".to_string()),
        tokenizer_token_count: 32,
        quantization: Gemma4Quantization::Unquantized,
        has_quantized_tensors: false,
        tensor_count: 40,
        multimodal: false,
    }
}

fn tiny_weights(cfg: &Gemma4Config) -> Gemma4TextWeights {
    let h = cfg.embedding_length;
    let v = cfg.tokenizer_token_count;
    let q_out = cfg.attention_head_count * cfg.attention_key_length;
    let kv_out = cfg.attention_head_count_kv * cfg.attention_key_length;
    let f = cfg.feed_forward_length;
    let epl = cfg.embedding_length_per_layer_input;
    let per_layer_width = cfg.block_count * epl;

    let token_embd = synth(1, v * h);
    let lm_head_w = transpose_2d(&token_embd, v, h);
    let layers = (0..cfg.block_count)
        .map(|layer| {
            let s = 100 + (layer as u32) * 50;
            Gemma4TextLayerWeights {
                attn_norm_w: vec![1.0; h],
                attn_q_w: synth(s, h * q_out),
                attn_k_w: synth(s + 1, h * kv_out),
                attn_v_w: synth(s + 2, h * kv_out),
                attn_o_w: synth(s + 3, q_out * h),
                attn_q_norm_w: vec![1.0; cfg.attention_key_length],
                attn_k_norm_w: vec![1.0; cfg.attention_key_length],
                ffn_norm_w: vec![1.0; h],
                ffn_gate_w: synth(s + 4, h * f),
                ffn_up_w: synth(s + 5, h * f),
                ffn_down_w: synth(s + 6, f * h),
                attn_post_norm_w: vec![1.0; h],
                ffn_post_norm_w: vec![1.0; h],
                per_layer_inp_gate_w: vec![0.0; h * epl],
                per_layer_proj_w: vec![0.0; epl * h],
                per_layer_post_norm_w: vec![1.0; h],
                layer_output_scale: 1.0,
            }
        })
        .collect();

    Gemma4TextWeights {
        token_embd,
        per_layer_token_embd: vec![0.0; v * per_layer_width],
        per_layer_model_proj_w: vec![0.0; h * per_layer_width],
        per_layer_proj_norm_w: vec![1.0; epl],
        rope_freqs: vec![1.0; cfg.rope_dimension_count / 2],
        layers,
        output_norm_w: vec![1.0; h],
        lm_head_w,
        tie_word_embeddings: true,
    }
}

#[derive(Debug, serde::Deserialize)]
struct LogitsFixture {
    model_shape: String,
    prompt_tokens: Vec<u32>,
    expected_logits: Vec<f32>,
    expected_decode_token: u32,
    tolerance: f32,
    rationale: String,
}

fn load_fixture() -> LogitsFixture {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR must be set when running cargo test");
    let path = std::path::Path::new(&manifest_dir).join(FIXTURE_PATH);
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    let fixture: LogitsFixture = serde_json::from_str(&json)
        .unwrap_or_else(|e| panic!("parse fixture {}: {e}", path.display()));

    assert_eq!(fixture.model_shape, "gemma4_tiny_synthetic_text_prefill");
    assert!(
        (fixture.tolerance - TOLERANCE).abs() < f32::EPSILON,
        "fixture tolerance ({}) must match test tolerance ({TOLERANCE})",
        fixture.tolerance
    );
    fixture
}

fn tiny_model() -> Gemma4TextModel {
    let cfg = tiny_config();
    Gemma4TextModel::new(cfg.clone(), tiny_weights(&cfg))
        .expect("tiny Gemma4 text model must build")
}

#[test]
fn gemma4_prefill_matches_pinned_fixture_through_runtime_path() {
    let fixture = load_fixture();
    let model = tiny_model();
    let prompt: Vec<TokenId> = fixture.prompt_tokens.iter().copied().map(TokenId).collect();

    let logits = prefill(&model, &prompt).expect("runtime Gemma4 prefill must succeed");

    if std::env::var("OCELOTL_PRINT_LOGITS").as_deref() == Ok("1") {
        let formatted: Vec<String> = logits.iter().map(|v| format!("{v:.7}")).collect();
        eprintln!("logits = [{}]", formatted.join(", "));
    }

    assert_eq!(logits.len(), fixture.expected_logits.len());
    for (idx, (got, want)) in logits
        .iter()
        .zip(fixture.expected_logits.iter())
        .enumerate()
    {
        let diff = (got - want).abs();
        assert!(
            diff < TOLERANCE,
            "Gemma4 logit {idx}: got {got}, want {want}, diff {diff} exceeds {TOLERANCE}\n{}",
            fixture.rationale
        );
    }
}

#[test]
fn gemma4_prefill_applies_final_logit_softcap_through_runtime_path() {
    let fixture = load_fixture();
    let mut cfg = tiny_config();
    let cap = 0.25_f32;
    cfg.final_logit_softcap = Some(cap);
    let model = Gemma4TextModel::new(cfg.clone(), tiny_weights(&cfg))
        .expect("softcapped tiny Gemma4 text model must build");
    let prompt: Vec<TokenId> = fixture.prompt_tokens.iter().copied().map(TokenId).collect();

    let logits = prefill(&model, &prompt).expect("runtime Gemma4 prefill must succeed");

    assert_eq!(logits.len(), fixture.expected_logits.len());
    let mut changed = false;
    for (idx, (got, raw)) in logits
        .iter()
        .zip(fixture.expected_logits.iter())
        .enumerate()
    {
        let want = cap * (raw / cap).tanh();
        let diff = (got - want).abs();
        assert!(
            diff < TOLERANCE,
            "softcapped Gemma4 logit {idx}: got {got}, want {want}, diff {diff} exceeds {TOLERANCE}"
        );
        assert!(
            got.abs() <= cap + TOLERANCE,
            "softcapped Gemma4 logit {idx}: got {got}, cap {cap}"
        );
        changed |= (got - raw).abs() > TOLERANCE;
    }
    assert!(
        changed,
        "softcap test must prove at least one logit changed"
    );
}

#[test]
fn gemma4_prefill_applies_sliding_window_mask_through_runtime_path() {
    let fixture = load_fixture();
    let prompt: Vec<TokenId> = fixture.prompt_tokens.iter().copied().map(TokenId).collect();
    let full_model = tiny_model();

    let mut cfg = tiny_config();
    cfg.attention_sliding_window = Some(1);
    let windowed_model = Gemma4TextModel::new(cfg.clone(), tiny_weights(&cfg))
        .expect("sliding-window tiny Gemma4 text model must build");

    let full_logits = prefill(&full_model, &prompt).expect("full Gemma4 prefill must succeed");
    let windowed_logits =
        prefill(&windowed_model, &prompt).expect("windowed Gemma4 prefill must succeed");

    assert_eq!(windowed_logits.len(), fixture.expected_logits.len());
    assert!(windowed_logits.iter().all(|value| value.is_finite()));
    let max_diff = full_logits
        .iter()
        .zip(windowed_logits.iter())
        .map(|(full, windowed)| (full - windowed).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff > SLIDING_WINDOW_DRIFT_TOLERANCE,
        "sliding-window mask must produce a distinct output on the tripwire prompt; max diff {max_diff}"
    );
}

#[test]
fn gemma4_prefill_accepts_shared_kv_through_runtime_path() {
    let fixture = load_fixture();
    let prompt: Vec<TokenId> = fixture.prompt_tokens.iter().copied().map(TokenId).collect();
    let mut cfg = tiny_config();
    cfg.attention_shared_kv_layers = Some(1);
    cfg.attention_sliding_window_pattern = Some(vec![false, false]);
    let mut weights = tiny_weights(&cfg);
    weights.layers[1].attn_k_w.fill(f32::NAN);
    weights.layers[1].attn_v_w.fill(f32::NAN);
    weights.layers[1].attn_k_norm_w.fill(f32::NAN);
    let model = Gemma4TextModel::new(cfg.clone(), weights)
        .expect("shared-KV tiny Gemma4 text model must build");

    let logits = prefill(&model, &prompt).expect("shared-KV Gemma4 prefill must succeed");

    assert_eq!(logits.len(), fixture.expected_logits.len());
    assert!(
        logits.iter().all(|value| value.is_finite()),
        "shared-KV runtime path must ignore poisoned K/V tensors on the shared suffix layer"
    );
}

#[test]
fn gemma4_decode_one_token_matches_pinned_argmax_of_prefill_fixture() {
    let fixture = load_fixture();
    let model = tiny_model();
    let prompt: Vec<TokenId> = fixture.prompt_tokens.iter().copied().map(TokenId).collect();

    let max_index = fixture
        .expected_logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(idx, _)| idx as u32)
        .expect("fixture logits must not be empty");
    assert_eq!(
        fixture.expected_decode_token, max_index,
        "fixture expected_decode_token must be the argmax of expected_logits"
    );

    let first = decode_one_token(&model, &prompt).expect("Gemma4 decode must succeed");
    let second = decode_one_token(&model, &prompt).expect("Gemma4 decode must be repeatable");

    assert_eq!(first, TokenId(fixture.expected_decode_token));
    assert_eq!(first, second, "Gemma4 decode must be deterministic");
}

#[test]
fn gemma4_decode_one_token_propagates_invalid_request_for_empty_prompt() {
    let model = tiny_model();

    let err = decode_one_token(&model, &[]).expect_err("empty prompt must be rejected");

    match err {
        OcelotlError::InvalidRequest(invalid) => {
            assert_eq!(invalid.field, "tokens");
            assert!(
                invalid.message.contains("at least one"),
                "expected model-boundary empty prompt message, got {:?}",
                invalid.message
            );
        }
        other => panic!("expected InvalidRequest for empty prompt, got {other:?}"),
    }
}
