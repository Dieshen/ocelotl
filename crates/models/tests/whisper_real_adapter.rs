use ocelotl_core::{DType, TokenId};
use ocelotl_loader::{LoadedTensor, SupportedDtype};
use ocelotl_models::whisper::{WhisperConfig, WhisperModel, required_whisper_tensor_names};

#[test]
fn tiny_real_whisper_adapter_produces_pinned_next_token_logits() {
    let cfg = tiny_config();
    let weights = synthetic_weight_tensors(&cfg);
    let model = WhisperModel::new(cfg, weights).expect("synthetic real weights construct");
    let mel = vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, -0.5];

    let logits = model
        .forward_next_token_logits(&mel, 4, &[TokenId(1), TokenId(2)])
        .expect("tiny real Whisper adapter forward");

    assert_eq!(logits.len(), 4);
    assert!(logits.iter().all(|v| v.is_finite()));
    assert_close(&logits, &[0.5, 0.0, 0.25, -0.25], 1.0e-5);

    let token = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| TokenId(idx as u32))
        .expect("non-empty logits");
    assert_eq!(token, TokenId(0));
}

fn tiny_config() -> WhisperConfig {
    WhisperConfig {
        vocab_size: 4,
        mel_bins: 2,
        audio_context_length: 2,
        audio_state_size: 2,
        audio_attention_heads: 1,
        audio_layers: 1,
        audio_ffn_size: 2,
        text_context_length: 3,
        text_state_size: 2,
        text_attention_heads: 1,
        text_layers: 1,
        text_ffn_size: 2,
        dtype: DType::F32,
        tie_word_embeddings: true,
    }
}

fn synthetic_weight_tensors(cfg: &WhisperConfig) -> Vec<LoadedTensor> {
    required_whisper_tensor_names(cfg)
        .into_iter()
        .map(|name| {
            let shape = shape_for(&name, cfg);
            let len = shape.iter().product();
            let mut values = vec![0.0_f32; len];
            if name == "decoder.token_embedding.weight" {
                values = vec![
                    0.5, 0.0, // token 0
                    0.0, 0.0, // token 1
                    0.25, 0.0, // token 2
                    -0.25, 0.0, // token 3
                ];
            } else if name == "decoder.ln.bias" {
                values = vec![1.0, 0.0];
            }
            LoadedTensor {
                name,
                shape,
                dtype: SupportedDtype::F32,
                values,
            }
        })
        .collect()
}

fn shape_for(name: &str, cfg: &WhisperConfig) -> Vec<usize> {
    if name == "encoder.conv1.weight" {
        return vec![cfg.audio_state_size, cfg.mel_bins, 3];
    }
    if name == "encoder.conv1.bias" || name == "encoder.conv2.bias" {
        return vec![cfg.audio_state_size];
    }
    if name == "encoder.conv2.weight" {
        return vec![cfg.audio_state_size, cfg.audio_state_size, 3];
    }
    if name == "encoder.positional_embedding" {
        return vec![cfg.audio_context_length, cfg.audio_state_size];
    }
    if name == "encoder.ln_post.weight" || name == "encoder.ln_post.bias" {
        return vec![cfg.audio_state_size];
    }
    if name == "decoder.token_embedding.weight" || name == "decoder.proj_out.weight" {
        return vec![cfg.vocab_size, cfg.text_state_size];
    }
    if name == "decoder.positional_embedding" {
        return vec![cfg.text_context_length, cfg.text_state_size];
    }
    if name == "decoder.ln.weight" || name == "decoder.ln.bias" {
        return vec![cfg.text_state_size];
    }
    if let Some(rest) = name.strip_prefix("encoder.blocks.") {
        return block_shape(rest, cfg.audio_state_size, cfg.audio_ffn_size, None);
    }
    if let Some(rest) = name.strip_prefix("decoder.blocks.") {
        return block_shape(
            rest,
            cfg.text_state_size,
            cfg.text_ffn_size,
            Some(cfg.audio_state_size),
        );
    }
    panic!("unknown tensor name {name}");
}

fn block_shape(rest: &str, state: usize, ffn: usize, audio_state: Option<usize>) -> Vec<usize> {
    let dot = rest.find('.').expect("layer separator");
    let suffix = &rest[dot + 1..];
    if let Some(cross) = suffix.strip_prefix("cross_attn.") {
        let audio = audio_state.expect("cross attention audio state");
        return match cross {
            "query.weight" => vec![state, state],
            "query.bias" => vec![state],
            "key.weight" => vec![state, audio],
            "value.weight" => vec![state, audio],
            "value.bias" => vec![state],
            "out.weight" => vec![state, state],
            "out.bias" => vec![state],
            other => panic!("unknown cross attention suffix {other}"),
        };
    }
    match suffix {
        "attn.query.weight" => vec![state, state],
        "attn.query.bias" => vec![state],
        "attn.key.weight" => vec![state, state],
        "attn.value.weight" => vec![state, state],
        "attn.value.bias" => vec![state],
        "attn.out.weight" => vec![state, state],
        "attn.out.bias" => vec![state],
        "attn_ln.weight" | "attn_ln.bias" => vec![state],
        "cross_attn_ln.weight" | "cross_attn_ln.bias" => vec![state],
        "mlp.0.weight" => vec![ffn, state],
        "mlp.0.bias" => vec![ffn],
        "mlp.2.weight" => vec![state, ffn],
        "mlp.2.bias" => vec![state],
        "mlp_ln.weight" | "mlp_ln.bias" => vec![state],
        other => panic!("unknown block suffix {other}"),
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let delta = (actual - expected).abs();
        assert!(
            delta <= tolerance,
            "index {idx}: expected {expected}, got {actual}, delta {delta}"
        );
    }
}
