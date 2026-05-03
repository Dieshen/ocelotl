//! Request lifecycle and generation runtime.

mod sampling;

pub use sampling::greedy_sample;

use ocelotl_core::{
    GenerationOptions, InvalidRequestError, ModelMetadata, OcelotlError, Result, TokenId,
    UnsupportedError,
};
use ocelotl_kernels::{CpuKernelBackend, KernelBackend};
use serde::{Deserialize, Serialize};

/// A generation request after tokenization. The runtime accepts token ids,
/// not raw strings; tokenization is the caller's responsibility (the
/// tokenizer crate owns that boundary).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt_tokens: Vec<TokenId>,
    pub options: GenerationOptions,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub text: String,
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

    let total = req.prompt_tokens.len() + req.options.max_new_tokens;
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

pub struct Runtime<B: KernelBackend = CpuKernelBackend> {
    backend: B,
}

impl Runtime<CpuKernelBackend> {
    pub fn cpu() -> Self {
        Self {
            backend: CpuKernelBackend::default(),
        }
    }
}

impl<B: KernelBackend> Runtime<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn generate(&self, _request: GenerateRequest) -> Result<GenerateResponse> {
        Err(OcelotlError::Unsupported(UnsupportedError {
            feature: "generate".to_string(),
            requested: None,
            supported: vec![],
        }))
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
}
