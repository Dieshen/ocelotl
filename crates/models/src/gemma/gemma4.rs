//! Gemma4 GGUF metadata contract.
//!
//! Gemma4 support starts as an inspect/reject path. `ocelotl-loader` owns the
//! GGUF parser; this module projects the loader-owned manifest into the
//! Gemma4-specific facts the model layer must preserve before any execution
//! path is allowed to run.

use ocelotl_core::{InvalidModelError, OcelotlError, Result, UnsupportedError};
use ocelotl_loader::{GgmlTensorType, GgufManifest, GgufMetadataType, GgufMetadataValue};

const GEMMA4_ARCHITECTURE: &str = "gemma4";
const GGUF_FILE_TYPE_Q4_K_M: u32 = 15;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gemma4Quantization {
    Q4KM,
    FileType(u32),
}

impl Gemma4Quantization {
    fn from_file_type(file_type: u32) -> Self {
        match file_type {
            GGUF_FILE_TYPE_Q4_K_M => Self::Q4KM,
            other => Self::FileType(other),
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::Q4KM => "q4_k_m".to_string(),
            Self::FileType(file_type) => format!("gguf_file_type_{file_type}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4Config {
    pub context_length: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub attention_head_count: usize,
    pub attention_head_count_kv: usize,
    pub rope_dimension_count: usize,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f32,
    pub attention_sliding_window: Option<usize>,
    pub attention_shared_kv_layers: Option<usize>,
    pub final_logit_softcap: Option<f32>,
    pub tokenizer_model: Option<String>,
    pub tokenizer_token_count: usize,
    pub quantization: Gemma4Quantization,
    pub has_quantized_tensors: bool,
    pub tensor_count: usize,
    pub multimodal: bool,
}

impl Gemma4Config {
    pub fn requires_dequant_policy(&self) -> bool {
        matches!(
            self.quantization,
            Gemma4Quantization::Q4KM | Gemma4Quantization::FileType(_)
        ) || self.has_quantized_tensors
    }

    pub fn ensure_supported_for_execution(&self) -> Result<()> {
        let mut requested = Vec::new();

        if self.multimodal {
            requested.push("multimodal".to_string());
        }
        if self.attention_sliding_window.is_some() {
            requested.push("sliding_window_attention".to_string());
        }
        if self.attention_shared_kv_layers.is_some() {
            requested.push("shared_kv_layers".to_string());
        }
        if self.final_logit_softcap.is_some() {
            requested.push("final_logit_softcap".to_string());
        }
        if self.requires_dequant_policy() {
            requested.push(format!("quantization={}", self.quantization.label()));
        }

        if requested.is_empty() {
            return Ok(());
        }

        Err(OcelotlError::from(UnsupportedError {
            feature: "gemma4.execution_features".to_string(),
            requested: Some(requested.join(",")),
            supported: vec!["none for Gemma4 execution yet".to_string()],
        }))
    }
}

impl TryFrom<&GgufManifest> for Gemma4Config {
    type Error = OcelotlError;

    fn try_from(manifest: &GgufManifest) -> Result<Self> {
        let architecture = metadata_string(manifest, "general.architecture")?;
        if architecture != GEMMA4_ARCHITECTURE {
            return Err(OcelotlError::from(UnsupportedError {
                feature: "gemma4.architecture".to_string(),
                requested: Some(architecture),
                supported: vec![GEMMA4_ARCHITECTURE.to_string()],
            }));
        }

        let context_length = required_usize(manifest, "gemma4.context_length")?;
        let block_count = required_usize(manifest, "gemma4.block_count")?;
        let embedding_length = required_usize(manifest, "gemma4.embedding_length")?;
        let feed_forward_length = required_usize(manifest, "gemma4.feed_forward_length")?;
        let attention_head_count = required_usize(manifest, "gemma4.attention.head_count")?;
        let attention_head_count_kv = required_usize(manifest, "gemma4.attention.head_count_kv")?;
        let rope_dimension_count = required_usize(manifest, "gemma4.rope.dimension_count")?;
        let rope_freq_base = required_f32(manifest, "gemma4.rope.freq_base")?;
        let rms_norm_eps = required_f32(manifest, "gemma4.attention.layer_norm_rms_epsilon")?;
        let quantization =
            Gemma4Quantization::from_file_type(required_u32(manifest, "general.file_type")?);

        validate_positive("gemma4.context_length", context_length)?;
        validate_positive("gemma4.block_count", block_count)?;
        validate_positive("gemma4.embedding_length", embedding_length)?;
        validate_positive("gemma4.feed_forward_length", feed_forward_length)?;
        validate_positive("gemma4.attention.head_count", attention_head_count)?;
        validate_positive("gemma4.attention.head_count_kv", attention_head_count_kv)?;
        validate_positive("gemma4.rope.dimension_count", rope_dimension_count)?;
        validate_finite_positive("gemma4.rope.freq_base", rope_freq_base)?;
        validate_finite_positive("gemma4.attention.layer_norm_rms_epsilon", rms_norm_eps)?;

        if attention_head_count % attention_head_count_kv != 0 {
            return Err(invalid(
                "gemma4.attention.head_count",
                &format!(
                    "must be divisible by gemma4.attention.head_count_kv ({attention_head_count_kv}); got {attention_head_count}"
                ),
            ));
        }

        let tokenizer_token_count = tokenizer_token_count(manifest)?;
        let attention_sliding_window = optional_usize(manifest, "gemma4.attention.sliding_window")?;
        let attention_shared_kv_layers =
            optional_usize(manifest, "gemma4.attention.shared_kv_layers")?;
        let final_logit_softcap = optional_f32(manifest, "gemma4.final_logit_softcapping")?;

        if let Some(value) = attention_sliding_window {
            validate_positive("gemma4.attention.sliding_window", value)?;
        }
        if let Some(value) = attention_shared_kv_layers {
            validate_positive("gemma4.attention.shared_kv_layers", value)?;
        }
        if let Some(value) = final_logit_softcap {
            validate_finite_positive("gemma4.final_logit_softcapping", value)?;
        }

        Ok(Self {
            context_length,
            block_count,
            embedding_length,
            feed_forward_length,
            attention_head_count,
            attention_head_count_kv,
            rope_dimension_count,
            rope_freq_base,
            rms_norm_eps,
            attention_sliding_window,
            attention_shared_kv_layers,
            final_logit_softcap,
            tokenizer_model: optional_string(manifest, "tokenizer.ggml.model"),
            tokenizer_token_count,
            quantization,
            has_quantized_tensors: manifest
                .tensors
                .iter()
                .any(|tensor| is_quantized_tensor_type(tensor.tensor_type)),
            tensor_count: manifest.tensors.len(),
            multimodal: true,
        })
    }
}

fn metadata_string(manifest: &GgufManifest, key: &str) -> Result<String> {
    match manifest.metadata_value(key) {
        Some(GgufMetadataValue::String(value)) => Ok(value.clone()),
        Some(other) => Err(invalid(
            key,
            &format!("must be a GGUF string metadata value, got {other:?}"),
        )),
        None => Err(missing(key)),
    }
}

fn optional_string(manifest: &GgufManifest, key: &str) -> Option<String> {
    match manifest.metadata_value(key) {
        Some(GgufMetadataValue::String(value)) => Some(value.clone()),
        _ => None,
    }
}

fn required_usize(manifest: &GgufManifest, key: &str) -> Result<usize> {
    match manifest.metadata_value(key) {
        Some(GgufMetadataValue::U32(value)) => Ok(*value as usize),
        Some(GgufMetadataValue::U64(value)) => (*value)
            .try_into()
            .map_err(|_| invalid(key, &format!("{value} does not fit in usize"))),
        Some(other) => Err(invalid(
            key,
            &format!("must be a GGUF unsigned integer metadata value, got {other:?}"),
        )),
        None => Err(missing(key)),
    }
}

fn optional_usize(manifest: &GgufManifest, key: &str) -> Result<Option<usize>> {
    if manifest.metadata_value(key).is_none() {
        return Ok(None);
    }
    required_usize(manifest, key).map(Some)
}

fn required_u32(manifest: &GgufManifest, key: &str) -> Result<u32> {
    match manifest.metadata_value(key) {
        Some(GgufMetadataValue::U32(value)) => Ok(*value),
        Some(GgufMetadataValue::U64(value)) => (*value)
            .try_into()
            .map_err(|_| invalid(key, &format!("{value} does not fit in u32"))),
        Some(other) => Err(invalid(
            key,
            &format!("must be a GGUF unsigned integer metadata value, got {other:?}"),
        )),
        None => Err(missing(key)),
    }
}

fn required_f32(manifest: &GgufManifest, key: &str) -> Result<f32> {
    match manifest.metadata_value(key) {
        Some(GgufMetadataValue::F32(value)) => Ok(*value),
        Some(GgufMetadataValue::F64(value)) => Ok(*value as f32),
        Some(GgufMetadataValue::U32(value)) => Ok(*value as f32),
        Some(GgufMetadataValue::U64(value)) => Ok(*value as f32),
        Some(other) => Err(invalid(
            key,
            &format!("must be a GGUF numeric metadata value, got {other:?}"),
        )),
        None => Err(missing(key)),
    }
}

fn optional_f32(manifest: &GgufManifest, key: &str) -> Result<Option<f32>> {
    if manifest.metadata_value(key).is_none() {
        return Ok(None);
    }
    required_f32(manifest, key).map(Some)
}

fn tokenizer_token_count(manifest: &GgufManifest) -> Result<usize> {
    match manifest.metadata_value("tokenizer.ggml.tokens") {
        Some(GgufMetadataValue::Array {
            element_type: GgufMetadataType::String,
            len,
        }) if *len > 0 => (*len).try_into().map_err(|_| {
            invalid(
                "tokenizer.ggml.tokens",
                &format!("token count {len} does not fit in usize"),
            )
        }),
        Some(GgufMetadataValue::Array { len: 0, .. }) => Err(invalid(
            "tokenizer.ggml.tokens",
            "must contain at least one tokenizer token",
        )),
        Some(other) => Err(invalid(
            "tokenizer.ggml.tokens",
            &format!("must be a GGUF string array metadata value, got {other:?}"),
        )),
        None => Err(missing("tokenizer.ggml.tokens")),
    }
}

fn validate_positive(field: &str, value: usize) -> Result<()> {
    if value == 0 {
        Err(invalid(field, "must be > 0"))
    } else {
        Ok(())
    }
}

fn validate_finite_positive(field: &str, value: f32) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        Err(invalid(
            field,
            &format!("must be finite and > 0; got {value}"),
        ))
    } else {
        Ok(())
    }
}

fn is_quantized_tensor_type(tensor_type: GgmlTensorType) -> bool {
    !matches!(
        tensor_type,
        GgmlTensorType::F32
            | GgmlTensorType::F16
            | GgmlTensorType::BF16
            | GgmlTensorType::I8
            | GgmlTensorType::I16
            | GgmlTensorType::I32
            | GgmlTensorType::I64
            | GgmlTensorType::F64
    )
}

fn missing(field: &str) -> OcelotlError {
    invalid(field, "missing required Gemma4 GGUF metadata field")
}

fn invalid(field: &str, message: &str) -> OcelotlError {
    OcelotlError::from(InvalidModelError {
        path: None,
        field: Some(field.to_string()),
        message: message.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ocelotl_loader::{GgufMetadataEntry, GgufTensorEntry, inspect_gguf};
    use serde::Deserialize;
    use std::path::PathBuf;

    #[derive(Debug, Deserialize)]
    struct Gemma4Fixture {
        gguf: Gemma4FixtureGguf,
    }

    #[derive(Debug, Deserialize)]
    struct Gemma4FixtureGguf {
        version: u32,
        metadata: Vec<FixtureMetadataEntry>,
        tensors: Vec<FixtureTensor>,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureMetadataEntry {
        key: String,
        value: FixtureMetadataValue,
    }

    #[derive(Debug, Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    enum FixtureMetadataValue {
        U32 { value: u32 },
        F32 { value: f32 },
        String { value: String },
        Array { element_type: String, len: u64 },
    }

    #[derive(Debug, Deserialize)]
    struct FixtureTensor {
        name: String,
        shape: Vec<usize>,
        tensor_type: String,
    }

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/metadata")
            .join(name)
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn local_gemma4_gguf_path() -> PathBuf {
        if let Ok(path) = std::env::var("OCELOTL_GEMMA4_GGUF_PATH") {
            return PathBuf::from(path);
        }
        repo_root()
            .join("local-artifacts")
            .join("gemma4_e4b_it_q4_k_m")
            .join("google_gemma-4-E4B-it-Q4_K_M.gguf")
    }

    fn fixture_manifest() -> GgufManifest {
        let raw = std::fs::read_to_string(fixture_path("gemma4_e4b_it_q4_k_m_gguf_metadata.json"))
            .expect("Gemma4 metadata fixture must be readable");
        let fixture: Gemma4Fixture =
            serde_json::from_str(&raw).expect("Gemma4 metadata fixture must parse");

        GgufManifest {
            version: fixture.gguf.version,
            metadata: fixture
                .gguf
                .metadata
                .into_iter()
                .map(|entry| GgufMetadataEntry {
                    key: entry.key,
                    value: match entry.value {
                        FixtureMetadataValue::U32 { value } => GgufMetadataValue::U32(value),
                        FixtureMetadataValue::F32 { value } => GgufMetadataValue::F32(value),
                        FixtureMetadataValue::String { value } => GgufMetadataValue::String(value),
                        FixtureMetadataValue::Array { element_type, len } => {
                            assert_eq!(element_type, "string");
                            GgufMetadataValue::Array {
                                element_type: GgufMetadataType::String,
                                len,
                            }
                        }
                    },
                })
                .collect(),
            tensors: fixture
                .gguf
                .tensors
                .into_iter()
                .map(|tensor| GgufTensorEntry {
                    name: tensor.name,
                    shape: tensor.shape,
                    tensor_type: match tensor.tensor_type.as_str() {
                        "q4_k" => GgmlTensorType::Q4K,
                        "f32" => GgmlTensorType::F32,
                        other => panic!("unexpected tensor type in fixture: {other}"),
                    },
                    offset: 0,
                    file_offset: 0,
                    byte_len: None,
                })
                .collect(),
            alignment: 32,
            data_start: 0,
            file_len: 0,
        }
    }

    #[test]
    fn try_from_gguf_manifest_accepts_gemma4_fixture_and_preserves_features() {
        let manifest = fixture_manifest();

        let cfg = Gemma4Config::try_from(&manifest).expect("Gemma4 GGUF fixture must convert");

        assert_eq!(cfg.context_length, 131_072);
        assert_eq!(cfg.block_count, 42);
        assert_eq!(cfg.embedding_length, 2_560);
        assert_eq!(cfg.feed_forward_length, 10_240);
        assert_eq!(cfg.attention_head_count, 8);
        assert_eq!(cfg.attention_head_count_kv, 2);
        assert_eq!(cfg.rope_dimension_count, 512);
        assert_eq!(cfg.attention_sliding_window, Some(512));
        assert_eq!(cfg.attention_shared_kv_layers, Some(18));
        assert_eq!(cfg.final_logit_softcap, Some(30.0));
        assert_eq!(cfg.tokenizer_model.as_deref(), Some("gemma4"));
        assert_eq!(cfg.tokenizer_token_count, 262_144);
        assert_eq!(cfg.quantization, Gemma4Quantization::Q4KM);
        assert!(cfg.has_quantized_tensors);
        assert_eq!(cfg.tensor_count, 2);
        assert!(cfg.multimodal);
    }

    #[test]
    fn ensure_supported_for_execution_rejects_gemma4_quantized_multimodal_features() {
        let cfg = Gemma4Config::try_from(&fixture_manifest()).unwrap();

        let err = cfg
            .ensure_supported_for_execution()
            .expect_err("Gemma4 execution must stay rejected until kernels land");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(unsupported.feature, "gemma4.execution_features");
                let requested = unsupported.requested.unwrap();
                assert!(requested.contains("multimodal"));
                assert!(requested.contains("sliding_window_attention"));
                assert!(requested.contains("shared_kv_layers"));
                assert!(requested.contains("final_logit_softcap"));
                assert!(requested.contains("quantization=q4_k_m"));
            }
            other => panic!("expected Unsupported for Gemma4 execution, got {other:?}"),
        }
    }

    #[test]
    fn try_from_gguf_manifest_rejects_missing_tokenizer_metadata() {
        let mut manifest = fixture_manifest();
        manifest
            .metadata
            .retain(|entry| entry.key != "tokenizer.ggml.tokens");

        let err = Gemma4Config::try_from(&manifest)
            .expect_err("Gemma4 without embedded tokenizer metadata must fail");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("tokenizer.ggml.tokens"));
            }
            other => panic!("expected InvalidModel for missing tokenizer metadata, got {other:?}"),
        }
    }

    #[test]
    fn try_from_gguf_manifest_rejects_non_gemma4_architecture() {
        let mut manifest = fixture_manifest();
        let architecture = manifest
            .metadata
            .iter_mut()
            .find(|entry| entry.key == "general.architecture")
            .unwrap();
        architecture.value = GgufMetadataValue::String("llama".to_string());

        let err = Gemma4Config::try_from(&manifest)
            .expect_err("foreign GGUF architecture must be rejected");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(unsupported.feature, "gemma4.architecture");
                assert_eq!(unsupported.requested.as_deref(), Some("llama"));
                assert_eq!(unsupported.supported, vec!["gemma4".to_string()]);
            }
            other => panic!("expected Unsupported for foreign architecture, got {other:?}"),
        }
    }

    #[test]
    fn try_from_gguf_manifest_rejects_invalid_sliding_window_before_compute() {
        let mut manifest = fixture_manifest();
        let sliding_window = manifest
            .metadata
            .iter_mut()
            .find(|entry| entry.key == "gemma4.attention.sliding_window")
            .unwrap();
        sliding_window.value = GgufMetadataValue::U32(0);

        let err = Gemma4Config::try_from(&manifest)
            .expect_err("zero sliding window must be rejected before compute");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(
                    invalid.field.as_deref(),
                    Some("gemma4.attention.sliding_window")
                );
            }
            other => panic!("expected InvalidModel for zero sliding window, got {other:?}"),
        }
    }

    #[test]
    #[ignore = "requires local-artifacts/gemma4_e4b_it_q4_k_m/google_gemma-4-E4B-it-Q4_K_M.gguf or OCELOTL_GEMMA4_GGUF_PATH"]
    fn local_gemma4_q4_k_m_gguf_header_converts_to_gemma4_config() {
        let path = local_gemma4_gguf_path();
        assert!(
            path.exists(),
            "missing Gemma4 GGUF artifact at {}; set OCELOTL_GEMMA4_GGUF_PATH or see docs/artifact-preparation.md",
            path.display()
        );

        let manifest = inspect_gguf(&path).expect("local Gemma4 GGUF header must inspect");
        let cfg = Gemma4Config::try_from(&manifest)
            .expect("local Gemma4 GGUF header must convert into Gemma4Config");

        assert_eq!(cfg.context_length, 131_072);
        assert_eq!(cfg.attention_sliding_window, Some(512));
        assert_eq!(cfg.attention_shared_kv_layers, Some(18));
        assert_eq!(cfg.final_logit_softcap, Some(30.0));
        assert_eq!(cfg.quantization, Gemma4Quantization::Q4KM);
        cfg.ensure_supported_for_execution()
            .expect_err("local Gemma4 Q4_K_M execution must be rejected before compute");
    }
}
