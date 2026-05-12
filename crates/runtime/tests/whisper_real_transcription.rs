use ocelotl_core::{DType, OcelotlError, TokenId};
use ocelotl_loader::{LoadedTensor, SupportedDtype};
use ocelotl_models::whisper::audio::{
    AudioMetadata, WHISPER_FFT_SIZE, WHISPER_MEL_BINS, log_mel_spectrogram,
};
use ocelotl_models::whisper::{WhisperConfig, WhisperModel, required_whisper_tensor_names};
use ocelotl_runtime::{
    WhisperDecodeRequest, WhisperTranscriptionRequest, decode_whisper_transcription,
    prepare_whisper_transcription, transcribe_whisper,
};
use ocelotl_tokenizer::{
    WhisperDecodeMask, WhisperStartupTokens, WhisperTimestampMode, WhisperTokenMaskDecision,
};

#[test]
fn real_whisper_transcription_reuses_encoded_audio_and_matches_legacy_loop() {
    let model = tiny_real_whisper_model();
    let request = whisper_request(2);

    let state = prepare_whisper_transcription(&model, &request).expect("prepare state");
    assert!(state.encoded_audio().frames() > 0);
    assert_eq!(
        state.encoded_audio().state_size(),
        model.config().audio_state_size
    );

    let from_state =
        decode_whisper_transcription(&model, &state, &request.decode).expect("decode from state");
    let composed = transcribe_whisper(&model, &request).expect("direct runtime transcription");
    let legacy = legacy_loop_recomputing_encoder(&model, &request);

    assert_eq!(from_state.tokens, legacy);
    assert_eq!(composed, from_state);
    assert_eq!(from_state.tokens.len(), request.decode.max_new_tokens);
    assert_eq!(from_state.logits.len(), model.config().vocab_size);
}

#[test]
fn real_whisper_transcription_rejects_zero_max_new_tokens_before_compute() {
    let model = tiny_real_whisper_model();
    let mut request = whisper_request(1);
    request.decode.max_new_tokens = 0;

    let err = transcribe_whisper(&model, &request)
        .expect_err("zero max_new_tokens must fail before preprocessing");

    match err {
        OcelotlError::InvalidRequest(invalid) => {
            assert_eq!(invalid.field, "max_new_tokens");
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn real_whisper_transcription_rejects_context_overflow_before_decode() {
    let model = tiny_real_whisper_model();
    let mut request = whisper_request(1);
    request.decode.decoder_prompt_tokens = vec![TokenId(0), TokenId(1), TokenId(2), TokenId(3)];
    request.decode.max_new_tokens = 2;

    let err = transcribe_whisper(&model, &request)
        .expect_err("context overflow must fail before preprocessing");

    match err {
        OcelotlError::InvalidRequest(invalid) => {
            assert_eq!(invalid.field, "decoder_context_length");
            assert!(invalid.message.contains("exceeds"));
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

fn legacy_loop_recomputing_encoder(
    model: &WhisperModel,
    request: &WhisperTranscriptionRequest,
) -> Vec<TokenId> {
    let mel = log_mel_spectrogram(&request.audio_samples, request.audio_metadata)
        .expect("log-mel should succeed");
    let mut context = request.decode.decoder_prompt_tokens.clone();
    let mut generated = Vec::new();

    for _ in 0..request.decode.max_new_tokens {
        let logits = model
            .forward_next_token_logits(&mel.values, mel.frames, &context)
            .expect("legacy forward");
        let next = masked_greedy_sample(&logits, request.decode.decode_mask);
        generated.push(next);
        context.push(next);
        if next == request.decode.stop_token {
            break;
        }
    }

    generated
}

fn masked_greedy_sample(logits: &[f32], mask: WhisperDecodeMask) -> TokenId {
    let mut best = None;
    for (idx, &logit) in logits.iter().enumerate() {
        let token = TokenId(u32::try_from(idx).expect("vocab index must fit in u32"));
        if mask.mask_token(token) == WhisperTokenMaskDecision::Suppress {
            continue;
        }
        if best.is_none_or(|(_, best_logit)| logit > best_logit) {
            best = Some((idx, logit));
        }
    }

    let (idx, _) = best.expect("Whisper decode mask suppressed every logit");
    TokenId(u32::try_from(idx).expect("vocab index must fit in u32"))
}

fn whisper_request(max_new_tokens: usize) -> WhisperTranscriptionRequest {
    let tokens = WhisperStartupTokens {
        end_of_text: TokenId(9_999),
        start_of_transcript: TokenId(9_998),
        language: TokenId(9_997),
        transcribe_task: TokenId(9_996),
        no_timestamps: TokenId(9_995),
        first_timestamp: TokenId(9_994),
    };
    WhisperTranscriptionRequest {
        audio_samples: tiny_waveform_fixture(),
        audio_metadata: AudioMetadata {
            sample_rate_hz: 16_000,
            channels: 1,
        },
        decode: WhisperDecodeRequest {
            decoder_prompt_tokens: vec![TokenId(0), TokenId(2)],
            max_new_tokens,
            decode_mask: WhisperDecodeMask::transcribe(tokens, WhisperTimestampMode::NoTimestamps),
            stop_token: tokens.end_of_text,
        },
    }
}

fn tiny_waveform_fixture() -> Vec<f32> {
    let mut audio = vec![0.0; WHISPER_FFT_SIZE];
    audio[0] = 1.0;
    audio[80] = -0.5;
    audio[160] = 0.25;
    audio[240] = -0.125;
    audio[320] = 0.0625;
    audio
}

fn tiny_real_whisper_model() -> WhisperModel {
    let cfg = tiny_config();
    WhisperModel::new(cfg.clone(), synthetic_weight_tensors(&cfg))
        .expect("synthetic real Whisper model must construct")
}

fn tiny_config() -> WhisperConfig {
    WhisperConfig {
        vocab_size: 4,
        mel_bins: WHISPER_MEL_BINS,
        audio_context_length: 2,
        audio_state_size: 2,
        audio_attention_heads: 1,
        audio_layers: 1,
        audio_ffn_size: 2,
        text_context_length: 5,
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
            let shape = shape_of(&name, cfg);
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

fn shape_of(name: &str, cfg: &WhisperConfig) -> Vec<usize> {
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
    panic!("unknown synthetic Whisper tensor name {name}");
}

fn block_shape(rest: &str, state: usize, ffn: usize, audio_state: Option<usize>) -> Vec<usize> {
    let dot = rest.find('.').expect("block tensor contains layer prefix");
    let suffix = &rest[dot + 1..];
    if let Some(cross) = suffix.strip_prefix("cross_attn.") {
        let audio = audio_state.expect("cross-attention only exists in decoder blocks");
        return match cross {
            "query.weight" => vec![state, state],
            "query.bias" => vec![state],
            "key.weight" => vec![state, audio],
            "value.weight" => vec![state, audio],
            "value.bias" => vec![state],
            "out.weight" => vec![state, state],
            "out.bias" => vec![state],
            _ => panic!("unknown cross-attention tensor suffix {cross}"),
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
        _ => panic!("unknown block tensor suffix {suffix}"),
    }
}
