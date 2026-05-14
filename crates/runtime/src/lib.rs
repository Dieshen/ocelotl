//! Request lifecycle and generation runtime.

mod kv_cache;
mod qwen;
mod sampling;
mod scheduler;
mod whisper;
mod whisper_streaming;

use std::sync::Arc;

pub use kv_cache::{ContiguousKvCache, PagedKvCache, PagedKvCacheAllocator, qwen2_5_kv_layout};
pub use qwen::{
    Qwen2_5ContiguousCacheState, Qwen2_5PagedCacheState, decode_one_token,
    decode_one_token_with_contiguous_cache, decode_one_token_with_paged_cache, prefill,
    prepare_qwen2_5_contiguous_cache, prepare_qwen2_5_paged_cache,
};
pub use sampling::greedy_sample;
pub use scheduler::{
    ContinuousBatchScheduler, GreedyDecodeModel, QwenGreedyModel, ScheduledGenerationRequest,
    ScheduledGenerationResponse, SchedulerConfig, SchedulerEvent, SchedulerRequestState,
    generate_qwen_batch,
};
pub use whisper::{
    TranscriptionRequest, TranscriptionResponse, WhisperDecodeRequest, WhisperTranscriptionRequest,
    WhisperTranscriptionResponse, WhisperTranscriptionState, decode_whisper_transcription,
    prepare_whisper_transcription, transcribe, transcribe_whisper,
};
pub use whisper_streaming::{
    ChunkedTranscriptionRequest, TranscriptionChunk, TranscriptionChunkingConfig,
    plan_transcription_chunks,
};

use ocelotl_core::{
    GenerationOptions, InvalidRequestError, ModelMetadata, OcelotlError, Result, TokenId,
    UnsupportedError,
};
#[cfg(feature = "cubecl-wgpu")]
use ocelotl_kernels::CubeClKernelBackend;
use ocelotl_kernels::{CpuKernelBackend, KernelBackend};
use ocelotl_models::{Qwen2_5Config, Qwen2_5Model, Qwen2_5Weights, tiny_synthetic_forward};
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
        Qwen2_5Model::with_kernel_backend(config, weights, Arc::new(self.backend.clone()))
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
        Qwen2_5Model::with_kernel_backend(config, weights, Arc::new(self.backend.clone()))
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
    use ocelotl_core::{DType, KvCacheStore};

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

        assert_eq!(model.kernel_backend().name(), "cpu");
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
    fn contiguous_cache_decode_matches_no_cache_decode_and_appends_token() {
        let model = tiny_model_for_decode();
        let prompt = [TokenId(1), TokenId(2)];
        let expected = decode_one_token(&model, &prompt).unwrap();

        let mut state =
            prepare_qwen2_5_contiguous_cache(&model, &prompt).expect("cached prefill must succeed");

        assert_eq!(state.cache().len_tokens(), prompt.len());
        assert_eq!(
            state.cache().key_at(0, 0).unwrap().len(),
            model.config().num_key_value_heads * model.config().head_dim
        );
        assert!(state.last_logits().iter().all(|v| v.is_finite()));

        let actual = decode_one_token_with_contiguous_cache(&model, &mut state)
            .expect("cached decode must succeed");

        assert_eq!(actual, expected);
        assert_eq!(state.cache().len_tokens(), prompt.len() + 1);
    }

    #[test]
    fn contiguous_cache_rejects_capacity_overflow_before_advancing_len() {
        let model = tiny_model_for_decode();
        let layout = qwen2_5_kv_layout(model.config(), 2).unwrap();
        let mut cache = ContiguousKvCache::new(layout).unwrap();

        let err = model
            .prefill_with_cache(&[TokenId(1), TokenId(2), TokenId(3)], &mut cache)
            .expect_err("prompt beyond cache capacity must be rejected");

        match err {
            OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "kv_cache.capacity");
            }
            other => panic!("expected InvalidRequest(kv_cache.capacity), got {other:?}"),
        }
        assert_eq!(cache.len_tokens(), 0);
    }

    #[test]
    fn contiguous_cache_release_clears_runtime_state_on_cancellation() {
        let model = tiny_model_for_decode();
        let mut state = prepare_qwen2_5_contiguous_cache(&model, &[TokenId(1), TokenId(2)])
            .expect("cached prefill must succeed");

        state.release();

        assert!(state.is_released());
        assert!(state.last_logits().is_empty());
    }

    #[test]
    fn paged_cache_decode_matches_no_cache_decode_across_page_boundary() {
        let model = tiny_model_for_decode();
        let prompt = [TokenId(1), TokenId(2), TokenId(3)];
        let expected = decode_one_token(&model, &prompt).unwrap();
        let mut allocator = PagedKvCacheAllocator::for_qwen2_5(model.config(), 4, 2, 3)
            .expect("allocator must construct");

        let mut state = prepare_qwen2_5_paged_cache(&model, &prompt, &mut allocator, 4)
            .expect("paged cached prefill must succeed");

        let cache = state.cache().expect("state must hold a cache");
        assert_eq!(cache.page_table(), &[0, 1]);
        assert_eq!(cache.physical_page_for_position(2).unwrap(), 1);
        assert_eq!(cache.len_tokens(), prompt.len());
        assert_eq!(allocator.free_page_count(), 1);

        let actual = decode_one_token_with_paged_cache(&model, &mut state)
            .expect("paged cached decode must succeed");

        assert_eq!(actual, expected);
        assert_eq!(
            state.cache().expect("cache remains live").len_tokens(),
            prompt.len() + 1
        );

        state.release_into(&mut allocator);
        assert!(state.is_released());
        assert_eq!(allocator.free_page_count(), 3);
    }

    #[test]
    fn paged_cache_releases_pages_when_prefill_fails_after_allocation() {
        let model = tiny_model_for_decode();
        let mut allocator = PagedKvCacheAllocator::for_qwen2_5(model.config(), 4, 2, 3)
            .expect("allocator must construct");

        let err = prepare_qwen2_5_paged_cache(&model, &[TokenId(99)], &mut allocator, 4)
            .expect_err("invalid token should fail prefill after allocation");

        match err {
            OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "tokens");
            }
            other => panic!("expected InvalidRequest(tokens), got {other:?}"),
        }
        assert_eq!(allocator.free_page_count(), 3);
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
