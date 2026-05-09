//! Real Whisper configuration contract.
//!
//! `WhisperTinyConfig` remains the W-ASR.4 synthetic fixture shape. This module
//! owns the real Whisper model-family configuration that adapter work can
//! consume before any tensor values are loaded or model compute starts.

use ocelotl_core::{DType, InvalidModelError, OcelotlError, Result, UnsupportedError};
use serde::Deserialize;

const WHISPER_ARCHITECTURE: &str = "whisper";
const WHISPER_MIN_VOCAB_SIZE: usize = 2;
const WHISPER_MAX_AUDIO_CONTEXT: usize = 3_000;
const WHISPER_MAX_TEXT_CONTEXT: usize = 8_192;
const WHISPER_SUPPORTED_DTYPES: &[DType] = &[DType::F32, DType::F16, DType::BF16];

/// Validated real Whisper model-family configuration.
///
/// Field names use Ocelotl semantics rather than Hugging Face names so later
/// model code can avoid carrying config-format details into compute code.
#[derive(Debug, Clone, PartialEq)]
pub struct WhisperConfig {
    pub vocab_size: usize,
    pub mel_bins: usize,
    pub audio_context_length: usize,
    pub audio_state_size: usize,
    pub audio_attention_heads: usize,
    pub audio_layers: usize,
    pub audio_ffn_size: usize,
    pub text_context_length: usize,
    pub text_state_size: usize,
    pub text_attention_heads: usize,
    pub text_layers: usize,
    pub text_ffn_size: usize,
    pub dtype: DType,
    pub tie_word_embeddings: bool,
}

impl WhisperConfig {
    pub fn validate(self) -> Result<Self> {
        validate_positive("mel_bins", self.mel_bins)?;
        validate_positive("audio_context_length", self.audio_context_length)?;
        validate_positive("audio_state_size", self.audio_state_size)?;
        validate_positive("audio_attention_heads", self.audio_attention_heads)?;
        validate_positive("audio_layers", self.audio_layers)?;
        validate_positive("audio_ffn_size", self.audio_ffn_size)?;
        validate_positive("text_context_length", self.text_context_length)?;
        validate_positive("text_state_size", self.text_state_size)?;
        validate_positive("text_attention_heads", self.text_attention_heads)?;
        validate_positive("text_layers", self.text_layers)?;
        validate_positive("text_ffn_size", self.text_ffn_size)?;

        if self.vocab_size < WHISPER_MIN_VOCAB_SIZE {
            return Err(invalid(
                "vocab_size",
                &format!(
                    "must be >= {WHISPER_MIN_VOCAB_SIZE}; got {}",
                    self.vocab_size
                ),
            ));
        }
        if self.audio_context_length > WHISPER_MAX_AUDIO_CONTEXT {
            return Err(invalid(
                "audio_context_length",
                &format!(
                    "must be <= {WHISPER_MAX_AUDIO_CONTEXT}; got {}",
                    self.audio_context_length
                ),
            ));
        }
        if self.text_context_length > WHISPER_MAX_TEXT_CONTEXT {
            return Err(invalid(
                "text_context_length",
                &format!(
                    "must be <= {WHISPER_MAX_TEXT_CONTEXT}; got {}",
                    self.text_context_length
                ),
            ));
        }
        if self.audio_state_size % self.audio_attention_heads != 0 {
            return Err(invalid(
                "audio_state_size",
                &format!(
                    "must be divisible by audio_attention_heads ({}); got {}",
                    self.audio_attention_heads, self.audio_state_size
                ),
            ));
        }
        if self.text_state_size % self.text_attention_heads != 0 {
            return Err(invalid(
                "text_state_size",
                &format!(
                    "must be divisible by text_attention_heads ({}); got {}",
                    self.text_attention_heads, self.text_state_size
                ),
            ));
        }
        if !WHISPER_SUPPORTED_DTYPES.contains(&self.dtype) {
            return Err(OcelotlError::from(UnsupportedError {
                feature: "whisper.dtype".to_string(),
                requested: Some(format!("{:?}", self.dtype)),
                supported: WHISPER_SUPPORTED_DTYPES
                    .iter()
                    .map(|d| format!("{d:?}"))
                    .collect(),
            }));
        }

        checked_dim_product(
            "audio_context_length*audio_state_size",
            &[self.audio_context_length, self.audio_state_size],
        )?;
        checked_dim_product(
            "text_context_length*text_state_size",
            &[self.text_context_length, self.text_state_size],
        )?;

        Ok(self)
    }
}

/// Parse a Hugging Face-style or Ocelotl/OpenAI-style Whisper `config.json`.
///
/// The first real parity artifact is tiny.en, but the config contract is not
/// tiny-only. HF exposes names like `d_model` and `max_source_positions`;
/// OpenAI checkpoints describe the same shape with `n_audio_state`,
/// `n_text_ctx`, and related fields. This parser accepts both JSON shapes and
/// returns one validated Ocelotl-owned `WhisperConfig`.
pub fn parse_whisper_config_json(raw: &str) -> Result<WhisperConfig> {
    let parsed: RawWhisperConfig = serde_json::from_str(raw).map_err(|source| {
        OcelotlError::from(InvalidModelError {
            path: None,
            field: Some("config.json".to_string()),
            message: format!("failed to parse Whisper config.json: {source}"),
        })
    })?;

    match parsed {
        RawWhisperConfig::Hf(cfg) => cfg.into_config(),
        RawWhisperConfig::OpenAi(cfg) => cfg.into_config(),
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawWhisperConfig {
    Hf(HfWhisperConfig),
    OpenAi(OpenAiWhisperConfig),
}

#[derive(Debug, Deserialize)]
struct HfWhisperConfig {
    model_type: String,
    vocab_size: usize,
    num_mel_bins: usize,
    d_model: usize,
    encoder_layers: usize,
    encoder_attention_heads: usize,
    encoder_ffn_dim: usize,
    decoder_layers: usize,
    decoder_attention_heads: usize,
    decoder_ffn_dim: usize,
    max_source_positions: usize,
    max_target_positions: usize,
    #[serde(default)]
    torch_dtype: Option<String>,
    #[serde(default = "default_tie_word_embeddings")]
    tie_word_embeddings: bool,
}

#[derive(Debug, Deserialize)]
struct OpenAiWhisperConfig {
    #[serde(default = "default_whisper_architecture")]
    model_type: String,
    n_vocab: usize,
    n_mels: usize,
    n_audio_ctx: usize,
    n_audio_state: usize,
    n_audio_head: usize,
    n_audio_layer: usize,
    n_text_ctx: usize,
    n_text_state: usize,
    n_text_head: usize,
    n_text_layer: usize,
    #[serde(default)]
    torch_dtype: Option<String>,
    #[serde(default = "default_tie_word_embeddings")]
    tie_word_embeddings: bool,
}

impl HfWhisperConfig {
    fn into_config(self) -> Result<WhisperConfig> {
        validate_architecture(&self.model_type)?;
        Ok(WhisperConfig {
            vocab_size: self.vocab_size,
            mel_bins: self.num_mel_bins,
            audio_context_length: self.max_source_positions,
            audio_state_size: self.d_model,
            audio_attention_heads: self.encoder_attention_heads,
            audio_layers: self.encoder_layers,
            audio_ffn_size: self.encoder_ffn_dim,
            text_context_length: self.max_target_positions,
            text_state_size: self.d_model,
            text_attention_heads: self.decoder_attention_heads,
            text_layers: self.decoder_layers,
            text_ffn_size: self.decoder_ffn_dim,
            dtype: parse_dtype(self.torch_dtype.as_deref())?,
            tie_word_embeddings: self.tie_word_embeddings,
        })
        .and_then(WhisperConfig::validate)
    }
}

impl OpenAiWhisperConfig {
    fn into_config(self) -> Result<WhisperConfig> {
        validate_architecture(&self.model_type)?;
        Ok(WhisperConfig {
            vocab_size: self.n_vocab,
            mel_bins: self.n_mels,
            audio_context_length: self.n_audio_ctx,
            audio_state_size: self.n_audio_state,
            audio_attention_heads: self.n_audio_head,
            audio_layers: self.n_audio_layer,
            audio_ffn_size: checked_dim_product("n_audio_state*4", &[self.n_audio_state, 4])?,
            text_context_length: self.n_text_ctx,
            text_state_size: self.n_text_state,
            text_attention_heads: self.n_text_head,
            text_layers: self.n_text_layer,
            text_ffn_size: checked_dim_product("n_text_state*4", &[self.n_text_state, 4])?,
            dtype: parse_dtype(self.torch_dtype.as_deref())?,
            tie_word_embeddings: self.tie_word_embeddings,
        })
        .and_then(WhisperConfig::validate)
    }
}

fn validate_architecture(model_type: &str) -> Result<()> {
    if model_type == WHISPER_ARCHITECTURE {
        Ok(())
    } else {
        Err(OcelotlError::from(UnsupportedError {
            feature: "whisper.architecture".to_string(),
            requested: Some(model_type.to_string()),
            supported: vec![WHISPER_ARCHITECTURE.to_string()],
        }))
    }
}

fn parse_dtype(raw: Option<&str>) -> Result<DType> {
    match raw.unwrap_or("float32") {
        "float32" | "f32" => Ok(DType::F32),
        "float16" | "float16_reduced_precision" | "f16" => Ok(DType::F16),
        "bfloat16" | "bf16" => Ok(DType::BF16),
        other => Err(OcelotlError::from(UnsupportedError {
            feature: "whisper.dtype".to_string(),
            requested: Some(other.to_string()),
            supported: vec!["float32".into(), "float16".into(), "bfloat16".into()],
        })),
    }
}

fn validate_positive(field: &str, value: usize) -> Result<()> {
    if value == 0 {
        Err(invalid(field, "must be > 0"))
    } else {
        Ok(())
    }
}

fn checked_dim_product(label: &str, dims: &[usize]) -> Result<usize> {
    dims.iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| invalid(label, &format!("shape product overflows usize: {:?}", dims)))
}

fn invalid(field: &str, message: &str) -> OcelotlError {
    OcelotlError::from(InvalidModelError {
        path: None,
        field: Some(field.to_string()),
        message: message.to_string(),
    })
}

fn default_whisper_architecture() -> String {
    WHISPER_ARCHITECTURE.to_string()
}

fn default_tie_word_embeddings() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    struct WhisperSizeCase {
        name: &'static str,
        state: usize,
        heads: usize,
        layers: usize,
    }

    const KNOWN_OPENAI_SIZE_CASES: &[WhisperSizeCase] = &[
        WhisperSizeCase {
            name: "tiny",
            state: 384,
            heads: 6,
            layers: 4,
        },
        WhisperSizeCase {
            name: "base",
            state: 512,
            heads: 8,
            layers: 6,
        },
        WhisperSizeCase {
            name: "small",
            state: 768,
            heads: 12,
            layers: 12,
        },
        WhisperSizeCase {
            name: "medium",
            state: 1_024,
            heads: 16,
            layers: 24,
        },
        WhisperSizeCase {
            name: "large",
            state: 1_280,
            heads: 20,
            layers: 32,
        },
    ];

    impl WhisperSizeCase {
        fn ffn(self) -> usize {
            self.state * 4
        }

        fn openai_style_json(self) -> String {
            format!(
                r#"{{
                  "model_type": "whisper",
                  "n_vocab": 51865,
                  "n_mels": 80,
                  "n_audio_ctx": 1500,
                  "n_audio_state": {state},
                  "n_audio_head": {heads},
                  "n_audio_layer": {layers},
                  "n_text_ctx": 448,
                  "n_text_state": {state},
                  "n_text_head": {heads},
                  "n_text_layer": {layers},
                  "torch_dtype": "float16"
                }}"#,
                state = self.state,
                heads = self.heads,
                layers = self.layers,
            )
        }

        fn hf_style_json(self) -> String {
            format!(
                r#"{{
                  "model_type": "whisper",
                  "vocab_size": 51865,
                  "num_mel_bins": 80,
                  "d_model": {state},
                  "encoder_layers": {layers},
                  "encoder_attention_heads": {heads},
                  "encoder_ffn_dim": {ffn},
                  "decoder_layers": {layers},
                  "decoder_attention_heads": {heads},
                  "decoder_ffn_dim": {ffn},
                  "max_source_positions": 1500,
                  "max_target_positions": 448,
                  "torch_dtype": "bfloat16",
                  "tie_word_embeddings": true
                }}"#,
                state = self.state,
                heads = self.heads,
                layers = self.layers,
                ffn = self.ffn(),
            )
        }
    }

    fn assert_known_size_config(cfg: &WhisperConfig, case: WhisperSizeCase, dtype: DType) {
        assert_eq!(cfg.vocab_size, 51_865, "{}", case.name);
        assert_eq!(cfg.mel_bins, 80, "{}", case.name);
        assert_eq!(cfg.audio_context_length, 1_500, "{}", case.name);
        assert_eq!(cfg.audio_state_size, case.state, "{}", case.name);
        assert_eq!(cfg.audio_attention_heads, case.heads, "{}", case.name);
        assert_eq!(cfg.audio_layers, case.layers, "{}", case.name);
        assert_eq!(cfg.audio_ffn_size, case.ffn(), "{}", case.name);
        assert_eq!(cfg.text_context_length, 448, "{}", case.name);
        assert_eq!(cfg.text_state_size, case.state, "{}", case.name);
        assert_eq!(cfg.text_attention_heads, case.heads, "{}", case.name);
        assert_eq!(cfg.text_layers, case.layers, "{}", case.name);
        assert_eq!(cfg.text_ffn_size, case.ffn(), "{}", case.name);
        assert_eq!(cfg.dtype, dtype, "{}", case.name);
        assert!(cfg.tie_word_embeddings, "{}", case.name);
    }

    #[test]
    fn parses_known_openai_whisper_size_dimensions() {
        for &case in KNOWN_OPENAI_SIZE_CASES {
            let cfg = parse_whisper_config_json(&case.openai_style_json()).unwrap_or_else(|err| {
                panic!("{} OpenAI-style config must parse: {err:?}", case.name)
            });

            assert_known_size_config(&cfg, case, DType::F16);
        }
    }

    #[test]
    fn parses_non_tiny_hf_whisper_size_dimensions() {
        for &case in KNOWN_OPENAI_SIZE_CASES
            .iter()
            .filter(|case| case.name != "tiny")
        {
            let cfg = parse_whisper_config_json(&case.hf_style_json())
                .unwrap_or_else(|err| panic!("{} HF-style config must parse: {err:?}", case.name));

            assert_known_size_config(&cfg, case, DType::BF16);
        }
    }

    #[test]
    fn rejects_oversized_audio_context_before_compute() {
        let err = parse_whisper_config_json(
            r#"{
              "model_type": "whisper",
              "n_vocab": 51865,
              "n_mels": 80,
              "n_audio_ctx": 3001,
              "n_audio_state": 1280,
              "n_audio_head": 20,
              "n_audio_layer": 32,
              "n_text_ctx": 448,
              "n_text_state": 1280,
              "n_text_head": 20,
              "n_text_layer": 32
            }"#,
        )
        .expect_err("unsupported audio context growth must fail before compute");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("audio_context_length"));
            }
            other => panic!("expected InvalidModel, got {other:?}"),
        }
    }

    #[test]
    fn parses_hf_tiny_en_config_shape() {
        let cfg = parse_whisper_config_json(
            r#"{
              "model_type": "whisper",
              "vocab_size": 51864,
              "num_mel_bins": 80,
              "d_model": 384,
              "encoder_layers": 4,
              "encoder_attention_heads": 6,
              "encoder_ffn_dim": 1536,
              "decoder_layers": 4,
              "decoder_attention_heads": 6,
              "decoder_ffn_dim": 1536,
              "max_source_positions": 1500,
              "max_target_positions": 448,
              "torch_dtype": "float16",
              "tie_word_embeddings": true
            }"#,
        )
        .expect("HF-style tiny.en config must parse");

        assert_eq!(cfg.vocab_size, 51_864);
        assert_eq!(cfg.mel_bins, 80);
        assert_eq!(cfg.audio_context_length, 1_500);
        assert_eq!(cfg.audio_state_size, 384);
        assert_eq!(cfg.audio_attention_heads, 6);
        assert_eq!(cfg.audio_layers, 4);
        assert_eq!(cfg.audio_ffn_size, 1_536);
        assert_eq!(cfg.text_context_length, 448);
        assert_eq!(cfg.text_state_size, 384);
        assert_eq!(cfg.text_attention_heads, 6);
        assert_eq!(cfg.text_layers, 4);
        assert_eq!(cfg.text_ffn_size, 1_536);
        assert_eq!(cfg.dtype, DType::F16);
        assert!(cfg.tie_word_embeddings);
    }

    #[test]
    fn parses_openai_style_dims_shape() {
        let cfg = parse_whisper_config_json(
            r#"{
              "n_vocab": 51864,
              "n_mels": 80,
              "n_audio_ctx": 1500,
              "n_audio_state": 384,
              "n_audio_head": 6,
              "n_audio_layer": 4,
              "n_text_ctx": 448,
              "n_text_state": 384,
              "n_text_head": 6,
              "n_text_layer": 4
            }"#,
        )
        .expect("OpenAI-style dims config must parse");

        assert_eq!(cfg.audio_ffn_size, 1_536);
        assert_eq!(cfg.text_ffn_size, 1_536);
        assert_eq!(cfg.dtype, DType::F32);
        assert!(cfg.tie_word_embeddings);
    }

    #[test]
    fn rejects_non_whisper_architecture() {
        let err = parse_whisper_config_json(
            r#"{
              "model_type": "qwen2",
              "vocab_size": 51864,
              "num_mel_bins": 80,
              "d_model": 384,
              "encoder_layers": 4,
              "encoder_attention_heads": 6,
              "encoder_ffn_dim": 1536,
              "decoder_layers": 4,
              "decoder_attention_heads": 6,
              "decoder_ffn_dim": 1536,
              "max_source_positions": 1500,
              "max_target_positions": 448
            }"#,
        )
        .expect_err("non-Whisper model_type must be rejected");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(unsupported.feature, "whisper.architecture");
                assert_eq!(unsupported.requested.as_deref(), Some("qwen2"));
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn rejects_zero_dimension_before_compute() {
        let err = parse_whisper_config_json(
            r#"{
              "model_type": "whisper",
              "vocab_size": 51864,
              "num_mel_bins": 0,
              "d_model": 384,
              "encoder_layers": 4,
              "encoder_attention_heads": 6,
              "encoder_ffn_dim": 1536,
              "decoder_layers": 4,
              "decoder_attention_heads": 6,
              "decoder_ffn_dim": 1536,
              "max_source_positions": 1500,
              "max_target_positions": 448
            }"#,
        )
        .expect_err("zero mel bins must be rejected");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("mel_bins"));
            }
            other => panic!("expected InvalidModel, got {other:?}"),
        }
    }

    #[test]
    fn rejects_inconsistent_head_divisibility() {
        let err = parse_whisper_config_json(
            r#"{
              "model_type": "whisper",
              "vocab_size": 51864,
              "num_mel_bins": 80,
              "d_model": 385,
              "encoder_layers": 4,
              "encoder_attention_heads": 6,
              "encoder_ffn_dim": 1536,
              "decoder_layers": 4,
              "decoder_attention_heads": 5,
              "decoder_ffn_dim": 1536,
              "max_source_positions": 1500,
              "max_target_positions": 448
            }"#,
        )
        .expect_err("non-divisible head dims must be rejected");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("audio_state_size"));
            }
            other => panic!("expected InvalidModel, got {other:?}"),
        }
    }
}
