//! Request lifecycle and generation runtime.

mod sampling;
mod whisper_streaming;

pub use sampling::greedy_sample;
pub use whisper_streaming::{
    ChunkedTranscriptionRequest, TranscriptionChunk, TranscriptionChunkingConfig,
    plan_transcription_chunks,
};

use ocelotl_core::{
    GenerationOptions, InvalidRequestError, ModelMetadata, OcelotlError, Result, RuntimeError,
    TokenId, UnsupportedError,
};
#[cfg(feature = "cubecl-wgpu")]
use ocelotl_kernels::CubeClKernelBackend;
use ocelotl_kernels::{CpuKernelBackend, KernelBackend};
use ocelotl_models::whisper::audio::{AudioMetadata, log_mel_spectrogram, validate_audio_metadata};
use ocelotl_models::whisper::{WhisperEncodedAudio, WhisperModel, WhisperTinyModel};
use ocelotl_models::{Qwen2_5Config, Qwen2_5Model, Qwen2_5Weights, tiny_synthetic_forward};
use ocelotl_tokenizer::{WhisperDecodeMask, WhisperTokenMaskDecision};
use serde::{Deserialize, Serialize};

// Re-export the response vocabulary type so callers can
// `use ocelotl_runtime::GenerateResponse;` without also pulling in
// `ocelotl_core` directly. The canonical definition still lives in core
// because the server crate will JSON-serialize it without depending on the
// runtime.
pub use ocelotl_core::GenerateResponse;

/// A generation request after tokenization. The runtime accepts token ids,
/// not raw strings; tokenization is the caller's responsibility (the
/// tokenizer crate owns that boundary).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt_tokens: Vec<TokenId>,
    pub options: GenerationOptions,
}

/// A Whisper transcription request after audio loading and tokenizer startup
/// policy have already run.
///
/// Runtime accepts raw mono samples and decoder prompt token IDs. Text decoding
/// is deliberately out of scope for W-ASR.6: the tokenizer crate owns Whisper
/// special-token policy and future token-to-text behavior.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionRequest {
    pub audio_samples: Vec<f32>,
    pub audio_metadata: AudioMetadata,
    pub decoder_prompt_tokens: Vec<TokenId>,
}

/// One synthetic Whisper decode step through the runtime API.
///
/// `tokens` is the greedy-selected next token. `logits` is returned alongside
/// it so early ASR tests can pin the model/runtime shape before text decoding
/// exists.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionResponse {
    pub tokens: Vec<TokenId>,
    pub logits: Vec<f32>,
}

/// Real Whisper transcription request for an autoregressive token loop.
///
/// The tokenizer layer still owns startup-token construction and timestamp
/// masking policy. Runtime receives the already-tokenized prompt, decode mask,
/// and stop token, then owns audio preprocessing, encoded-audio state, and the
/// decode lifecycle.
#[derive(Debug, Clone, PartialEq)]
pub struct WhisperTranscriptionRequest {
    pub audio_samples: Vec<f32>,
    pub audio_metadata: AudioMetadata,
    pub decode: WhisperDecodeRequest,
}

/// Real Whisper decoder controls after tokenization and policy selection.
#[derive(Debug, Clone, PartialEq)]
pub struct WhisperDecodeRequest {
    pub decoder_prompt_tokens: Vec<TokenId>,
    pub max_new_tokens: usize,
    pub decode_mask: WhisperDecodeMask,
    pub stop_token: TokenId,
}

/// Runtime-owned Whisper state that is invariant across token decode steps.
#[derive(Debug, Clone, PartialEq)]
pub struct WhisperTranscriptionState {
    encoded_audio: WhisperEncodedAudio,
}

impl WhisperTranscriptionState {
    pub fn encoded_audio(&self) -> &WhisperEncodedAudio {
        &self.encoded_audio
    }
}

/// Tokens produced by a real Whisper autoregressive transcription loop.
///
/// `tokens` contains only newly generated tokens, not the startup prompt. The
/// caller can concatenate `decoder_prompt_tokens + tokens` when it needs the
/// full model sequence for parity fixtures.
#[derive(Debug, Clone, PartialEq)]
pub struct WhisperTranscriptionResponse {
    pub tokens: Vec<TokenId>,
    pub logits: Vec<f32>,
}

/// Validate a generation request against the loaded model's metadata before
/// any compute is scheduled. Each check produces a typed error matching the
/// project's error taxonomy: sampling-mode requests we cannot fulfill yet
/// surface as `Unsupported`; shape and bound violations surface as
/// `InvalidRequest`.
///
/// Order of checks matters because some downstream checks would be
/// meaningless on an upstream violation (a context-overflow check on an
/// empty prompt, for example, doesn't carry the right diagnostic). The order
/// is therefore: sampling mode → token-budget bounds → prompt shape →
/// context fit. Each error category fires at exactly one gate.
pub fn validate_request(req: &GenerateRequest, model: &ModelMetadata) -> Result<()> {
    if req.options.temperature.is_some() {
        return Err(OcelotlError::Unsupported(UnsupportedError {
            feature: "sampling_mode".to_string(),
            requested: Some("temperature".to_string()),
            supported: vec!["greedy".to_string()],
        }));
    }

    if req.options.max_new_tokens == 0 {
        return Err(OcelotlError::InvalidRequest(InvalidRequestError {
            field: "max_new_tokens".to_string(),
            message: "must be greater than zero".to_string(),
        }));
    }

    if req.prompt_tokens.is_empty() {
        return Err(OcelotlError::InvalidRequest(InvalidRequestError {
            field: "prompt_tokens".to_string(),
            message: "must contain at least one token".to_string(),
        }));
    }

    let total = req
        .prompt_tokens
        .len()
        .checked_add(req.options.max_new_tokens)
        .ok_or_else(|| {
            OcelotlError::InvalidRequest(InvalidRequestError {
                field: "context_length".to_string(),
                message: format!(
                    "prompt_tokens ({}) + max_new_tokens ({}) overflows usize",
                    req.prompt_tokens.len(),
                    req.options.max_new_tokens,
                ),
            })
        })?;
    if total > model.context_length {
        return Err(OcelotlError::InvalidRequest(InvalidRequestError {
            field: "context_length".to_string(),
            message: format!(
                "prompt_tokens ({}) + max_new_tokens ({}) = {} exceeds model context_length ({})",
                req.prompt_tokens.len(),
                req.options.max_new_tokens,
                total,
                model.context_length,
            ),
        }));
    }

    Ok(())
}

/// Run the M1 CPU reference path end to end and return a single sampled
/// token.
///
/// This is the public entry point that wires together every component the
/// previous M1 milestones built:
///
/// 1. `validate_request` (M1.6) rejects empty prompts, zero
///    `max_new_tokens`, sampling-mode requests, and context overflow.
/// 2. `ocelotl_models::tiny_synthetic_forward` (M1.9) produces a
///    deterministic logits vector via `kernels::matmul` (M1.7).
/// 3. `greedy_sample` (M1.8) picks the argmax with lowest-token-id
///    tie-break.
///
/// # Why one token, not `max_new_tokens` tokens
///
/// M1 proves the *pipeline*. A loop that re-runs the synthetic forward
/// `max_new_tokens` times would just be a `for` loop wrapped around this
/// function and would not exercise any new component. Multi-step
/// generation needs a KV cache and a real decoder loop, which is M3 work.
/// Returning one token here keeps the contract honest: M1 is "the wires
/// are connected", not "the runtime can decode".
pub fn generate_one_token(
    req: &GenerateRequest,
    model: &ModelMetadata,
) -> Result<GenerateResponse> {
    validate_request(req, model)?;
    let logits = tiny_synthetic_forward(model, &req.prompt_tokens)?;
    let next_token = greedy_sample(&logits)?;
    Ok(GenerateResponse {
        tokens: vec![next_token],
    })
}

/// Run prefill on the given model and return final-position logits.
///
/// This is the public M3.7 entry point. It is a thin shim over
/// `Qwen2_5Model::prefill` that exists so callers (server code, tests,
/// future decoder loops) reach prefill through `ocelotl_runtime::prefill`
/// rather than reaching directly into the model crate. The shape of this
/// API is intentionally model-typed for now: M3 targets only Qwen2.5.
/// When a second model family lands, this will become a method on a
/// `CausalLanguageModel` trait. Until then, generic abstraction would be
/// premature (the M3 design doc's "keep generic abstractions minimal until
/// a second family is implemented" line).
///
/// # Errors
///
/// Propagates `OcelotlError` from the model's prefill verbatim:
/// - `InvalidRequest` for empty prompts, out-of-range token ids, or
///   prompt length exceeding `context_length`.
/// - `Kernel` for unreachable shape violations (would indicate the
///   `Qwen2_5Model::new` length checks have a bug).
pub fn prefill(model: &Qwen2_5Model, tokens: &[TokenId]) -> Result<Vec<f32>> {
    model.prefill(tokens)
}

/// Run one decode step: prefill the prompt, sample the next token greedily,
/// return that token id.
///
/// This is the public M3.8 entry point. It is the minimum composition of
/// two existing public APIs:
///
/// 1. `runtime::prefill` (M3.7) -- final-position logits for the prompt.
/// 2. `runtime::greedy_sample` (M1.8) -- argmax over the logits with a
///    lowest-token-id tie-break.
///
/// # Why call `runtime::prefill` rather than `Qwen2_5Model::prefill` directly
///
/// The M3.8 brief's "Done when" line is "decode does not bypass runtime or
/// model APIs used by prefill". Going through the runtime's own `prefill`
/// shim keeps a single hop between the public decode API and the model:
/// any future addition to the runtime's `prefill` (cache plumbing,
/// tracing, request-state hooks) is automatically inherited by decode.
/// Reaching into `Qwen2_5Model::prefill` directly here would create a
/// second path that those hooks would silently miss.
///
/// # State handling (no KV cache)
///
/// M3 is the correctness-first reference path. This function does not
/// reuse any prefill state across calls -- each call to `decode_one_token`
/// pays the full O(prompt_len) prefill cost. KV-cache reuse is M5/M6
/// work and will land as a separate API (likely `decode_with_cache`)
/// rather than complicating this signature now. The current shape
/// (`(&Qwen2_5Model, &[TokenId]) -> Result<TokenId>`) is intentionally
/// thin so a future cache-aware path can be added alongside without
/// breaking callers.
///
/// # Errors
///
/// Propagates errors from the composed APIs verbatim:
/// - `InvalidRequest` from `runtime::prefill` for empty prompts,
///   out-of-range token ids, or prompt length exceeding `context_length`.
/// - `Runtime` from `greedy_sample` if the logits vector is empty
///   (unreachable with a validly constructed model -- `vocab_size > 0`
///   is enforced at `Qwen2_5Config::try_from`).
pub fn decode_one_token(model: &Qwen2_5Model, tokens: &[TokenId]) -> Result<TokenId> {
    let logits = prefill(model, tokens)?;
    greedy_sample(&logits)
}

/// Run one synthetic Whisper transcription step through the runtime boundary.
///
/// W-ASR.6 keeps this intentionally narrow: runtime validates request-owned
/// audio shape, calls the Whisper log-mel reference path, reaches the
/// `WhisperTinyModel::forward` public model API, and greedily selects one next
/// token. Multi-token decode, timestamp policy, and token-to-text decoding are
/// future tokenizer/runtime work.
pub fn transcribe(
    model: &WhisperTinyModel,
    request: &TranscriptionRequest,
) -> Result<TranscriptionResponse> {
    validate_transcription_request(request)?;
    let mel = log_mel_spectrogram(&request.audio_samples, request.audio_metadata)?;
    let logits = model.forward(&mel, &request.decoder_prompt_tokens)?;
    let token = greedy_sample(&logits)?;

    Ok(TranscriptionResponse {
        tokens: vec![token],
        logits,
    })
}

/// Prepare real Whisper audio state once for a transcription request.
///
/// This is the W-ASR.21 public runtime seam: audio validation and log-mel
/// preprocessing happen once, `WhisperModel::encode_audio_features` produces
/// encoded audio once, and the returned state can be reused by every token
/// decode step for that audio window.
pub fn prepare_whisper_transcription(
    model: &WhisperModel,
    request: &WhisperTranscriptionRequest,
) -> Result<WhisperTranscriptionState> {
    validate_whisper_transcription_request(request)?;
    let mel = log_mel_spectrogram(&request.audio_samples, request.audio_metadata)?;
    let encoded_audio = model.encode_audio_features(&mel.values, mel.frames)?;
    Ok(WhisperTranscriptionState { encoded_audio })
}

/// Decode real Whisper tokens from a prepared encoded-audio state.
///
/// This is the W-ASR.22 runtime path: callers can hold
/// `WhisperTranscriptionState` and avoid recomputing the encoder for each
/// generated token. W-ASR.27 also keeps a decoder state inside this loop so
/// decoder self-attention K/V grows one token at a time instead of recomputing
/// the full decoder prefix for every generated token.
pub fn decode_whisper_transcription(
    model: &WhisperModel,
    state: &WhisperTranscriptionState,
    request: &WhisperDecodeRequest,
) -> Result<WhisperTranscriptionResponse> {
    validate_whisper_decode_request(model, request)?;

    let mut decoder_state = model
        .prepare_decoder_state_from_audio(state.encoded_audio(), &request.decoder_prompt_tokens)?;
    let mut tokens = Vec::with_capacity(request.max_new_tokens);
    let mut logits = Vec::new();

    for _ in 0..request.max_new_tokens {
        logits = decoder_state.next_token_logits().to_vec();
        let next = masked_greedy_sample(&logits, request.decode_mask)?;
        tokens.push(next);
        if next == request.stop_token {
            break;
        }
        if tokens.len() < request.max_new_tokens {
            model.append_decoder_token_from_audio(
                state.encoded_audio(),
                &mut decoder_state,
                next,
            )?;
        }
    }

    Ok(WhisperTranscriptionResponse { tokens, logits })
}

/// Run real Whisper transcription through the runtime boundary.
///
/// This convenience wrapper composes `prepare_whisper_transcription` and
/// `decode_whisper_transcription`, so the public path gets encoded-audio reuse
/// even when the caller does not manage the state directly.
pub fn transcribe_whisper(
    model: &WhisperModel,
    request: &WhisperTranscriptionRequest,
) -> Result<WhisperTranscriptionResponse> {
    validate_whisper_decode_request(model, &request.decode)?;
    let state = prepare_whisper_transcription(model, request)?;
    decode_whisper_transcription(model, &state, &request.decode)
}

fn validate_transcription_request(request: &TranscriptionRequest) -> Result<()> {
    if request.audio_samples.is_empty() {
        return Err(OcelotlError::InvalidRequest(InvalidRequestError {
            field: "audio_samples".to_string(),
            message: "must contain at least one sample".to_string(),
        }));
    }

    validate_audio_metadata(request.audio_metadata)
}

fn validate_whisper_transcription_request(request: &WhisperTranscriptionRequest) -> Result<()> {
    if request.audio_samples.is_empty() {
        return Err(invalid_request(
            "audio_samples",
            "must contain at least one sample",
        ));
    }

    validate_audio_metadata(request.audio_metadata)
}

fn validate_whisper_decode_request(
    model: &WhisperModel,
    request: &WhisperDecodeRequest,
) -> Result<()> {
    if request.decoder_prompt_tokens.is_empty() {
        return Err(invalid_request(
            "decoder_prompt_tokens",
            "must contain at least one token",
        ));
    }
    if request.max_new_tokens == 0 {
        return Err(invalid_request(
            "max_new_tokens",
            "must be greater than zero",
        ));
    }

    let total = request
        .decoder_prompt_tokens
        .len()
        .checked_add(request.max_new_tokens)
        .ok_or_else(|| {
            invalid_request(
                "decoder_context_length",
                "decoder_prompt_tokens + max_new_tokens overflows usize",
            )
        })?;
    if total > model.config().text_context_length {
        return Err(invalid_request(
            "decoder_context_length",
            &format!(
                "decoder_prompt_tokens ({}) + max_new_tokens ({}) = {} exceeds text_context_length ({})",
                request.decoder_prompt_tokens.len(),
                request.max_new_tokens,
                total,
                model.config().text_context_length,
            ),
        ));
    }

    Ok(())
}

fn masked_greedy_sample(logits: &[f32], mask: WhisperDecodeMask) -> Result<TokenId> {
    let mut best = None;
    for (idx, &logit) in logits.iter().enumerate() {
        let token = TokenId(u32::try_from(idx).map_err(|_| {
            OcelotlError::Runtime(RuntimeError {
                message: format!("logit index {idx} does not fit in TokenId"),
            })
        })?);
        if mask.mask_token(token) == WhisperTokenMaskDecision::Suppress {
            continue;
        }
        if best.is_none_or(|(_, best_logit)| logit > best_logit) {
            best = Some((idx, logit));
        }
    }

    let (idx, _) = best.ok_or_else(|| {
        OcelotlError::Runtime(RuntimeError {
            message: "Whisper decode mask suppressed every logit".to_string(),
        })
    })?;
    Ok(TokenId(u32::try_from(idx).map_err(|_| {
        OcelotlError::Runtime(RuntimeError {
            message: format!("logit index {idx} does not fit in TokenId"),
        })
    })?))
}

fn invalid_request(field: &str, message: &str) -> OcelotlError {
    OcelotlError::InvalidRequest(InvalidRequestError {
        field: field.to_string(),
        message: message.to_string(),
    })
}

pub struct Runtime<B: KernelBackend = CpuKernelBackend> {
    backend: B,
}

impl Runtime<CpuKernelBackend> {
    pub fn cpu() -> Self {
        Self {
            backend: CpuKernelBackend::default(),
        }
    }

    pub fn optimized_cpu() -> Self {
        Self {
            backend: CpuKernelBackend::optimized(),
        }
    }

    pub fn qwen2_5_model(
        &self,
        config: Qwen2_5Config,
        weights: Qwen2_5Weights,
    ) -> Result<Qwen2_5Model> {
        Qwen2_5Model::with_cpu_kernel_backend(config, weights, self.backend.clone())
    }
}

#[cfg(feature = "cubecl-wgpu")]
impl Runtime<CubeClKernelBackend> {
    pub fn cubecl_wgpu(ordinal: usize) -> Self {
        Self {
            backend: CubeClKernelBackend::new_gpu(ordinal),
        }
    }

    pub fn qwen2_5_model(
        &self,
        config: Qwen2_5Config,
        weights: Qwen2_5Weights,
    ) -> Result<Qwen2_5Model> {
        Qwen2_5Model::with_cubecl_wgpu_backend(config, weights, self.backend.clone())
    }
}

impl<B: KernelBackend> Runtime<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Run a generation request through the M1 CPU reference path. For now
    /// this is a thin shim around `generate_one_token`; multi-token
    /// generation arrives with the KV cache in M3.
    pub fn generate(
        &self,
        request: GenerateRequest,
        model: &ModelMetadata,
    ) -> Result<GenerateResponse> {
        generate_one_token(&request, model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ocelotl_core::DType;

    /// Build a model metadata fixture with a controllable context length.
    /// Other fields are placeholders — validation only inspects context_length.
    fn make_model(context_length: usize) -> ModelMetadata {
        ModelMetadata {
            architecture: "qwen2".to_string(),
            vocab_size: 32,
            num_hidden_layers: 2,
            hidden_size: 16,
            intermediate_size: 32,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            context_length,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            dtype: DType::F32,
            tokenizer_model_hint: None,
        }
    }

    fn make_request(prompt_len: usize, max_new_tokens: usize) -> GenerateRequest {
        GenerateRequest {
            prompt_tokens: (0..prompt_len as u32).map(TokenId).collect(),
            options: GenerationOptions {
                max_new_tokens,
                temperature: None,
            },
        }
    }

    fn tiny_qwen_config_and_weights() -> (Qwen2_5Config, Qwen2_5Weights) {
        use ocelotl_models::{Qwen2_5LayerWeights, transpose_2d};

        let cfg = Qwen2_5Config {
            vocab_size: 8,
            num_hidden_layers: 1,
            hidden_size: 4,
            intermediate_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            context_length: 16,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            dtype: DType::F32,
        };
        let h = cfg.hidden_size;
        let v = cfg.vocab_size;
        let q_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        let i_size = cfg.intermediate_size;
        let embed: Vec<f32> = (0..v * h).map(|i| (i as f32) * 0.01).collect();
        let lm_head_w = transpose_2d(&embed, v, h);
        let weights = Qwen2_5Weights {
            embed_tokens: embed,
            layers: vec![Qwen2_5LayerWeights {
                q_proj_w: vec![0.01; h * q_out],
                q_proj_b: vec![0.0; q_out],
                k_proj_w: vec![0.01; h * kv_out],
                k_proj_b: vec![0.0; kv_out],
                v_proj_w: vec![0.01; h * kv_out],
                v_proj_b: vec![0.0; kv_out],
                o_proj_w: vec![0.01; q_out * h],
                input_layernorm_w: vec![1.0; h],
                post_attention_layernorm_w: vec![1.0; h],
                gate_proj_w: vec![0.01; h * i_size],
                up_proj_w: vec![0.01; h * i_size],
                down_proj_w: vec![0.01; i_size * h],
            }],
            final_norm_w: vec![1.0; h],
            lm_head_w,
            tie_word_embeddings: true,
        };

        (cfg, weights)
    }

    #[test]
    fn optimized_cpu_runtime_selects_optimized_kernel_backend() {
        let runtime = Runtime::optimized_cpu();

        assert_eq!(
            runtime.backend().mode(),
            ocelotl_kernels::CpuKernelMode::Optimized
        );
    }

    #[test]
    fn cpu_runtime_builds_qwen_model_with_selected_cpu_backend() {
        let runtime = Runtime::optimized_cpu();
        let (cfg, weights) = tiny_qwen_config_and_weights();

        let model = runtime
            .qwen2_5_model(cfg, weights)
            .expect("runtime should construct Qwen model");

        assert_eq!(
            model.kernel_backend().mode(),
            ocelotl_kernels::CpuKernelMode::Optimized
        );
        assert_eq!(
            model.execution_backend().context().device,
            ocelotl_core::Device::Cpu
        );
    }

    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn cubecl_wgpu_runtime_builds_qwen_model_with_gpu_execution_backend_without_launch() {
        let runtime = Runtime::cubecl_wgpu(0);
        let (cfg, weights) = tiny_qwen_config_and_weights();

        let model = runtime
            .qwen2_5_model(cfg, weights)
            .expect("runtime should construct CubeCL-backed Qwen model");

        assert_eq!(model.execution_backend().name(), "cubecl");
        assert_eq!(
            model.execution_backend().context().device,
            ocelotl_core::Device::Gpu { ordinal: 0 }
        );
    }

    #[test]
    fn validate_request_accepts_well_formed_request() {
        let model = make_model(128);
        let req = make_request(4, 8);

        validate_request(&req, &model).expect("a well-formed request must validate");
    }

    #[test]
    fn validate_request_rejects_temperature_with_unsupported_sampling_mode() {
        let model = make_model(128);
        let mut req = make_request(4, 8);
        req.options.temperature = Some(0.7);

        let err = validate_request(&req, &model)
            .expect_err("requests with a temperature must be rejected for now");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(unsupported.feature, "sampling_mode");
                assert_eq!(unsupported.requested.as_deref(), Some("temperature"));
                assert_eq!(unsupported.supported, vec!["greedy".to_string()]);
            }
            other => panic!("expected Unsupported(sampling_mode), got {other:?}"),
        }
    }

    #[test]
    fn validate_request_rejects_zero_max_new_tokens() {
        let model = make_model(128);
        let req = make_request(4, 0);

        let err = validate_request(&req, &model).expect_err("max_new_tokens == 0 must be rejected");

        match err {
            OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "max_new_tokens");
                assert_eq!(invalid.message, "must be greater than zero");
            }
            other => panic!("expected InvalidRequest(max_new_tokens), got {other:?}"),
        }
    }

    #[test]
    fn validate_request_rejects_empty_prompt() {
        let model = make_model(128);
        let req = make_request(0, 8);

        let err = validate_request(&req, &model).expect_err("an empty prompt must be rejected");

        match err {
            OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "prompt_tokens");
                assert_eq!(invalid.message, "must contain at least one token");
            }
            other => panic!("expected InvalidRequest(prompt_tokens), got {other:?}"),
        }
    }

    #[test]
    fn validate_request_rejects_context_overflow() {
        let model = make_model(16);
        // prompt 10 + max_new 8 = 18, model context = 16 → overflow by 2.
        let req = make_request(10, 8);

        let err = validate_request(&req, &model)
            .expect_err("prompt + max_new_tokens > context_length must be rejected");

        match err {
            OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "context_length");
                assert!(
                    invalid.message.contains("10"),
                    "expected prompt length 10 in message, got {:?}",
                    invalid.message
                );
                assert!(
                    invalid.message.contains('8'),
                    "expected max_new_tokens 8 in message, got {:?}",
                    invalid.message
                );
                assert!(
                    invalid.message.contains("16"),
                    "expected context_length 16 in message, got {:?}",
                    invalid.message
                );
            }
            other => panic!("expected InvalidRequest(context_length), got {other:?}"),
        }
    }

    #[test]
    fn validate_request_rejects_context_sum_usize_overflow() {
        let model = make_model(usize::MAX);
        let req = make_request(1, usize::MAX);

        let err = validate_request(&req, &model)
            .expect_err("prompt + max_new_tokens overflow must be rejected");

        match err {
            OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "context_length");
                assert!(
                    invalid.message.contains("overflows"),
                    "expected overflow diagnostic, got {:?}",
                    invalid.message
                );
            }
            other => panic!("expected InvalidRequest(context_length), got {other:?}"),
        }
    }

    #[test]
    fn validate_request_accepts_request_exactly_filling_context() {
        let model = make_model(16);
        // prompt 10 + max_new 6 = 16 → exactly the limit, must be accepted.
        let req = make_request(10, 6);

        validate_request(&req, &model)
            .expect("request that exactly fills context_length must be accepted");
    }

    #[test]
    fn validate_request_temperature_check_fires_before_other_violations() {
        // Multiple violations present (temperature + zero max_new + empty prompt).
        // The contract reports the most upstream one (sampling mode), not the
        // first shape error. This is what M1.8 depends on.
        let model = make_model(128);
        let req = GenerateRequest {
            prompt_tokens: vec![],
            options: GenerationOptions {
                max_new_tokens: 0,
                temperature: Some(0.5),
            },
        };

        let err = validate_request(&req, &model)
            .expect_err("a request with multiple violations must still error");

        match err {
            OcelotlError::Unsupported(u) => assert_eq!(u.feature, "sampling_mode"),
            other => panic!("expected sampling_mode rejection to win, got {other:?}"),
        }
    }

    // --- generate_one_token (M1.9 wiring) ---

    #[test]
    fn generate_one_token_returns_one_token_for_valid_request() {
        let model = make_model(128);
        let req = make_request(1, 8);

        let resp = generate_one_token(&req, &model).expect("valid request must produce a token");

        assert_eq!(resp.tokens.len(), 1);
        assert!(
            (resp.tokens[0].0 as usize) < model.vocab_size,
            "sampled token must be within vocabulary, got {:?}",
            resp.tokens[0]
        );
    }

    // --- prefill (M3.7 wiring) ---

    #[test]
    fn prefill_returns_logits_through_runtime_api_surface() {
        // M3.7 contract: the runtime exposes prefill on the public API
        // so callers do not bypass the runtime layer to reach the model.
        // We keep the model construction inline here because the runtime
        // crate doesn't own model fixtures; the goal of this test is to
        // pin the API shape, not to re-validate prefill numerics (that's
        // pinned in the models crate's tiny-synthetic integration test).
        use ocelotl_models::{
            Qwen2_5Config, Qwen2_5LayerWeights, Qwen2_5Model, Qwen2_5Weights, transpose_2d,
        };
        let cfg = Qwen2_5Config {
            vocab_size: 8,
            num_hidden_layers: 1,
            hidden_size: 4,
            intermediate_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            context_length: 16,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            dtype: DType::F32,
        };
        let h = cfg.hidden_size;
        let v = cfg.vocab_size;
        let q_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        let i_size = cfg.intermediate_size;
        let embed: Vec<f32> = (0..v * h).map(|i| (i as f32) * 0.01).collect();
        let lm_head_w = transpose_2d(&embed, v, h);
        let weights = Qwen2_5Weights {
            embed_tokens: embed,
            layers: vec![Qwen2_5LayerWeights {
                q_proj_w: vec![0.01; h * q_out],
                q_proj_b: vec![0.0; q_out],
                k_proj_w: vec![0.01; h * kv_out],
                k_proj_b: vec![0.0; kv_out],
                v_proj_w: vec![0.01; h * kv_out],
                v_proj_b: vec![0.0; kv_out],
                o_proj_w: vec![0.01; q_out * h],
                input_layernorm_w: vec![1.0; h],
                post_attention_layernorm_w: vec![1.0; h],
                gate_proj_w: vec![0.01; h * i_size],
                up_proj_w: vec![0.01; h * i_size],
                down_proj_w: vec![0.01; i_size * h],
            }],
            final_norm_w: vec![1.0; h],
            lm_head_w,
            tie_word_embeddings: true,
        };
        let model = Qwen2_5Model::new(cfg, weights).expect("tiny model must construct");

        let logits = prefill(&model, &[TokenId(1), TokenId(2)])
            .expect("prefill via runtime API must succeed");

        assert_eq!(logits.len(), 8);
        for v in &logits {
            assert!(v.is_finite(), "prefill logits must be finite");
        }
    }

    #[test]
    fn prefill_propagates_invalid_request_for_empty_prompt() {
        // A runtime-level public API must surface model-level validation
        // failures verbatim. An empty prompt is InvalidRequest at the
        // model boundary (Qwen2_5Model::prefill); the runtime wrapper
        // must not swallow or remap it.
        use ocelotl_models::{
            Qwen2_5Config, Qwen2_5LayerWeights, Qwen2_5Model, Qwen2_5Weights, transpose_2d,
        };
        let cfg = Qwen2_5Config {
            vocab_size: 4,
            num_hidden_layers: 1,
            hidden_size: 2,
            intermediate_size: 4,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            context_length: 8,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            dtype: DType::F32,
        };
        let h = cfg.hidden_size;
        let v = cfg.vocab_size;
        let q_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        let i_size = cfg.intermediate_size;
        let embed: Vec<f32> = vec![0.1; v * h];
        let lm_head_w = transpose_2d(&embed, v, h);
        let weights = Qwen2_5Weights {
            embed_tokens: embed,
            layers: vec![Qwen2_5LayerWeights {
                q_proj_w: vec![0.1; h * q_out],
                q_proj_b: vec![0.0; q_out],
                k_proj_w: vec![0.1; h * kv_out],
                k_proj_b: vec![0.0; kv_out],
                v_proj_w: vec![0.1; h * kv_out],
                v_proj_b: vec![0.0; kv_out],
                o_proj_w: vec![0.1; q_out * h],
                input_layernorm_w: vec![1.0; h],
                post_attention_layernorm_w: vec![1.0; h],
                gate_proj_w: vec![0.1; h * i_size],
                up_proj_w: vec![0.1; h * i_size],
                down_proj_w: vec![0.1; i_size * h],
            }],
            final_norm_w: vec![1.0; h],
            lm_head_w,
            tie_word_embeddings: true,
        };
        let model = Qwen2_5Model::new(cfg, weights).unwrap();

        let err = prefill(&model, &[]).expect_err("empty prompt must be rejected");

        match err {
            OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "tokens");
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    // --- decode_one_token (M3.8 wiring) ---

    /// Build the same tiny Qwen2.5 model used by `prefill_returns_logits_through_runtime_api_surface`.
    /// Kept inline rather than extracted to a helper to keep these tests
    /// self-contained: the runtime crate doesn't own model fixtures, and a
    /// helper would only have the two callers in this module.
    fn tiny_model_for_decode() -> ocelotl_models::Qwen2_5Model {
        use ocelotl_models::{
            Qwen2_5Config, Qwen2_5LayerWeights, Qwen2_5Model, Qwen2_5Weights, transpose_2d,
        };
        let cfg = Qwen2_5Config {
            vocab_size: 8,
            num_hidden_layers: 1,
            hidden_size: 4,
            intermediate_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            context_length: 16,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            dtype: DType::F32,
        };
        let h = cfg.hidden_size;
        let v = cfg.vocab_size;
        let q_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        let i_size = cfg.intermediate_size;
        let embed: Vec<f32> = (0..v * h).map(|i| (i as f32) * 0.01).collect();
        let lm_head_w = transpose_2d(&embed, v, h);
        let weights = Qwen2_5Weights {
            embed_tokens: embed,
            layers: vec![Qwen2_5LayerWeights {
                q_proj_w: vec![0.01; h * q_out],
                q_proj_b: vec![0.0; q_out],
                k_proj_w: vec![0.01; h * kv_out],
                k_proj_b: vec![0.0; kv_out],
                v_proj_w: vec![0.01; h * kv_out],
                v_proj_b: vec![0.0; kv_out],
                o_proj_w: vec![0.01; q_out * h],
                input_layernorm_w: vec![1.0; h],
                post_attention_layernorm_w: vec![1.0; h],
                gate_proj_w: vec![0.01; h * i_size],
                up_proj_w: vec![0.01; h * i_size],
                down_proj_w: vec![0.01; i_size * h],
            }],
            final_norm_w: vec![1.0; h],
            lm_head_w,
            tie_word_embeddings: true,
        };
        Qwen2_5Model::new(cfg, weights).expect("tiny model must construct")
    }

    #[test]
    fn decode_one_token_returns_a_token_id_for_valid_prompt() {
        // Smallest meaningful contract for M3.8: decode_one_token returns
        // a TokenId that is within the model's vocabulary. Pins the API
        // shape (Vec<f32> -> TokenId) without yet asserting numerics; the
        // pinning test below covers the specific value.
        let model = tiny_model_for_decode();

        let token = decode_one_token(&model, &[TokenId(1), TokenId(2)])
            .expect("valid prompt must produce a token");

        assert!(
            (token.0 as usize) < model.config().vocab_size,
            "decoded token must be within vocabulary, got {token:?}",
        );
    }

    #[test]
    fn generate_one_token_propagates_validation_errors() {
        // The wired path must surface validation failures verbatim — no
        // swallowing, no remapping. A temperature request must produce the
        // same Unsupported error validate_request would.
        let model = make_model(128);
        let mut req = make_request(1, 8);
        req.options.temperature = Some(0.7);

        let err = generate_one_token(&req, &model)
            .expect_err("validation failure must propagate through generate_one_token");

        match err {
            OcelotlError::Unsupported(u) => assert_eq!(u.feature, "sampling_mode"),
            other => panic!("expected sampling_mode rejection, got {other:?}"),
        }
    }
}
