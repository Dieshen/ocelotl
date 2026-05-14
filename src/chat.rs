//! App-facing local chat facade.
//!
//! This layer is intentionally thin. It composes the lower-level crates into a
//! product-shaped API while keeping artifact parsing, tokenization, model
//! family behavior, runtime decode, and kernels in their owning crates.

use std::path::Path;

use ocelotl_core::{
    GenerationOptions, InvalidModelError, InvalidRequestError, IoError, OcelotlError, Result,
    TokenId, TokenizerError, UnsupportedError,
};
use ocelotl_models::Qwen2_5Model;
use ocelotl_tokenizer::{ChatMessage, ChatTemplate, JsonTokenizer, Tokenizer};
use serde::Deserialize;

#[derive(Debug)]
pub struct ChatModel {
    model: Qwen2_5Model,
    tokenizer: JsonTokenizer,
    chat_template: ChatTemplate,
    messages: Vec<ChatMessage>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatResponse {
    pub text: String,
    pub tokens: Vec<TokenId>,
}

impl ChatModel {
    /// Load a local Qwen2.5-style chat artifact directory.
    ///
    /// Expected files:
    /// - `config.json`
    /// - `model.safetensors`
    /// - `tokenizer.json`
    /// - `tokenizer_config.json` with a `chat_template` string
    ///
    /// This never downloads artifacts. Use the explicit fetch CLI before this
    /// call when artifacts are not already present on disk.
    pub fn load_local<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref();
        let model = Qwen2_5Model::load_from_dir(dir)?;
        let tokenizer = JsonTokenizer::from_json_path(dir.join("tokenizer.json"))?;
        let chat_template = load_chat_template(dir.join("tokenizer_config.json"))?;
        Ok(Self::from_qwen2_5_parts(model, tokenizer, chat_template))
    }

    fn from_qwen2_5_parts(
        model: Qwen2_5Model,
        tokenizer: JsonTokenizer,
        chat_template: ChatTemplate,
    ) -> Self {
        Self {
            model,
            tokenizer,
            chat_template,
            messages: Vec::new(),
        }
    }

    pub fn add_message(
        &mut self,
        role: impl Into<String>,
        content: impl Into<String>,
    ) -> &mut Self {
        self.messages.push(ChatMessage {
            role: role.into(),
            content: content.into(),
        });
        self
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    pub fn generate_text(&self, options: GenerationOptions) -> Result<ChatResponse> {
        let tokens = self.generate_tokens(options)?;
        let text = self.tokenizer.decode(&tokens)?;
        Ok(ChatResponse { text, tokens })
    }

    fn generate_tokens(&self, options: GenerationOptions) -> Result<Vec<TokenId>> {
        validate_chat_options(&options)?;
        if self.messages.is_empty() {
            return Err(OcelotlError::InvalidRequest(InvalidRequestError {
                field: "messages".to_string(),
                message: "must contain at least one message before generation".to_string(),
            }));
        }

        let rendered = self.chat_template.apply(&self.messages, true)?;
        let mut prompt_tokens = self.tokenizer.encode(&rendered)?;
        let mut generated = Vec::with_capacity(options.max_new_tokens);

        for _ in 0..options.max_new_tokens {
            let next = ocelotl_runtime::decode_one_token(&self.model, &prompt_tokens)?;
            generated.push(next);
            prompt_tokens.push(next);
        }

        Ok(generated)
    }
}

#[derive(Debug, Deserialize)]
struct TokenizerConfigChatTemplate {
    chat_template: Option<String>,
}

fn load_chat_template(path: impl AsRef<Path>) -> Result<ChatTemplate> {
    let path = path.as_ref();
    let raw = std::fs::read_to_string(path).map_err(|source| {
        OcelotlError::Io(IoError {
            path: Some(path.to_path_buf()),
            source,
        })
    })?;
    let cfg: TokenizerConfigChatTemplate = serde_json::from_str(&raw).map_err(|source| {
        OcelotlError::Tokenizer(TokenizerError {
            message: format!(
                "failed to parse tokenizer_config.json from `{}`",
                path.display()
            ),
            source: Some(Box::new(source)),
        })
    })?;
    let source = cfg.chat_template.ok_or_else(|| {
        OcelotlError::InvalidModel(InvalidModelError {
            path: Some(path.to_path_buf()),
            field: Some("chat_template".to_string()),
            message: "tokenizer_config.json must contain a chat_template string".to_string(),
        })
    })?;
    ChatTemplate::from_jinja(&source)
}

fn validate_chat_options(options: &GenerationOptions) -> Result<()> {
    if options.temperature.is_some() {
        return Err(OcelotlError::Unsupported(UnsupportedError {
            feature: "sampling_mode".to_string(),
            requested: Some("temperature".to_string()),
            supported: vec!["greedy".to_string()],
        }));
    }
    if options.max_new_tokens == 0 {
        return Err(OcelotlError::InvalidRequest(InvalidRequestError {
            field: "max_new_tokens".to_string(),
            message: "must be greater than zero".to_string(),
        }));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ocelotl_core::DType;
    use ocelotl_models::{Qwen2_5Config, Qwen2_5LayerWeights, Qwen2_5Weights};

    use super::*;

    #[test]
    fn chat_model_add_message_and_generate_text_uses_runtime_path() {
        let model = tiny_chat_model();
        let tokenizer =
            JsonTokenizer::from_json_path(tokenizer_fixture_path("tiny_wordlevel.json")).unwrap();
        let chat_template = ChatTemplate::from_jinja(
            "{% for message in messages %}{{ message.content }}{% endfor %}",
        )
        .unwrap();
        let mut chat = ChatModel::from_qwen2_5_parts(model, tokenizer, chat_template);

        chat.add_message("user", "hello");
        let response = chat
            .generate_text(GenerationOptions {
                max_new_tokens: 1,
                temperature: None,
            })
            .expect("one-token chat generation should succeed");

        assert_eq!(chat.messages()[0].role, "user");
        assert_eq!(response.tokens, vec![TokenId(2)]);
        assert_eq!(response.text, "world");
    }

    #[test]
    fn chat_model_rejects_empty_messages_before_tokenization() {
        let model = tiny_chat_model();
        let tokenizer =
            JsonTokenizer::from_json_path(tokenizer_fixture_path("tiny_wordlevel.json")).unwrap();
        let chat_template = ChatTemplate::from_jinja("{{ messages | length }}").unwrap();
        let chat = ChatModel::from_qwen2_5_parts(model, tokenizer, chat_template);

        let err = chat
            .generate_text(GenerationOptions {
                max_new_tokens: 1,
                temperature: None,
            })
            .expect_err("empty chat history must fail before generation");

        match err {
            OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "messages");
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn chat_options_reject_temperature_until_sampling_exists() {
        let err = validate_chat_options(&GenerationOptions {
            max_new_tokens: 1,
            temperature: Some(0.7),
        })
        .expect_err("temperature sampling is not supported yet");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(unsupported.feature, "sampling_mode");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    fn tiny_chat_model() -> Qwen2_5Model {
        let cfg = Qwen2_5Config {
            vocab_size: 3,
            num_hidden_layers: 1,
            hidden_size: 4,
            intermediate_size: 4,
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

        let mut embed_tokens = vec![0.0_f32; v * h];
        embed_tokens[h] = 1.0;
        let mut lm_head_w = vec![0.0_f32; h * v];
        lm_head_w[2] = 1.0;

        let weights = Qwen2_5Weights {
            embed_tokens,
            layers: vec![Qwen2_5LayerWeights {
                q_proj_w: vec![0.0; h * q_out],
                q_proj_b: vec![0.0; q_out],
                k_proj_w: vec![0.0; h * kv_out],
                k_proj_b: vec![0.0; kv_out],
                v_proj_w: vec![0.0; h * kv_out],
                v_proj_b: vec![0.0; kv_out],
                o_proj_w: vec![0.0; q_out * h],
                input_layernorm_w: vec![1.0; h],
                post_attention_layernorm_w: vec![1.0; h],
                gate_proj_w: vec![0.0; h * i_size],
                up_proj_w: vec![0.0; h * i_size],
                down_proj_w: vec![0.0; i_size * h],
            }],
            final_norm_w: vec![1.0; h],
            lm_head_w,
            tie_word_embeddings: false,
        };

        Qwen2_5Model::new(cfg, weights).expect("tiny chat model must construct")
    }

    fn tokenizer_fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/tokenizer")
            .join(name)
    }
}
