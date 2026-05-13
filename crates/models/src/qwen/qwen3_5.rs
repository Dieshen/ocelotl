//! Qwen3.5 metadata recognition and pre-compute rejection contract.
//!
//! Qwen3.5 is a separate family contract from the M3 Qwen2.5 dense decoder
//! path. The first Ocelotl slice preserves the relevant Hugging Face config
//! facts and rejects unsupported hybrid/MoE/multimodal/FP8 execution features
//! before model compute is reachable.

use ocelotl_core::{InvalidModelError, OcelotlError, Result, UnsupportedError};
use serde_json::Value;

const QWEN3_5_MODEL_TYPE: &str = "qwen3_5_moe";
const QWEN3_5_TEXT_MODEL_TYPE: &str = "qwen3_5_moe_text";

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3_5Config {
    pub model_type: String,
    pub text_model_type: String,
    pub architectures: Vec<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub context_length: usize,
    pub rope_theta: f64,
    pub dtype: String,
    pub layer_types: Vec<String>,
    pub has_hybrid_attention: bool,
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    pub has_moe: bool,
    pub quantization: Option<String>,
    pub multimodal: bool,
    pub image_token_id: Option<u32>,
    pub video_token_id: Option<u32>,
    pub vision_hidden_size: Option<usize>,
}

impl Qwen3_5Config {
    pub fn ensure_supported_for_execution(&self) -> Result<()> {
        let mut requested = Vec::new();

        if self.has_hybrid_attention {
            requested.push("hybrid_attention".to_string());
        }
        if self.has_moe {
            requested.push("moe".to_string());
        }
        if self.multimodal {
            requested.push("multimodal".to_string());
        }
        if let Some(quantization) = &self.quantization {
            requested.push(format!("quantization={quantization}"));
        }

        if requested.is_empty() {
            return Ok(());
        }

        Err(OcelotlError::from(UnsupportedError {
            feature: "qwen3_5.execution_features".to_string(),
            requested: Some(requested.join(",")),
            supported: vec!["none for Qwen3.5 execution yet".to_string()],
        }))
    }
}

pub fn parse_qwen3_5_config_json(raw: &str) -> Result<Qwen3_5Config> {
    let value: Value = serde_json::from_str(raw).map_err(|source| {
        OcelotlError::from(InvalidModelError {
            path: None,
            field: Some("config.json".to_string()),
            message: format!("failed to parse Qwen3.5 config.json: {source}"),
        })
    })?;

    let model_type = required_string(&value, "model_type")?;
    if model_type != QWEN3_5_MODEL_TYPE {
        return Err(OcelotlError::from(UnsupportedError {
            feature: "qwen3_5.architecture".to_string(),
            requested: Some(model_type),
            supported: vec![QWEN3_5_MODEL_TYPE.to_string()],
        }));
    }

    let text = required_object(&value, "text_config")?;
    let text_model_type = required_string(text, "model_type")?;
    if text_model_type != QWEN3_5_TEXT_MODEL_TYPE {
        return Err(OcelotlError::from(UnsupportedError {
            feature: "qwen3_5.text_architecture".to_string(),
            requested: Some(text_model_type),
            supported: vec![QWEN3_5_TEXT_MODEL_TYPE.to_string()],
        }));
    }

    let vocab_size = required_usize(text, "vocab_size")?;
    let hidden_size = required_usize(text, "hidden_size")?;
    let num_hidden_layers = required_usize(text, "num_hidden_layers")?;
    let num_attention_heads = required_usize(text, "num_attention_heads")?;
    let num_key_value_heads = required_usize(text, "num_key_value_heads")?;
    let head_dim = required_usize(text, "head_dim")?;
    let context_length = required_usize(text, "max_position_embeddings")?;
    let dtype = required_string(text, "dtype")?;
    let rope_parameters = required_object(text, "rope_parameters")?;
    let rope_theta = required_f64(rope_parameters, "rope_theta")?;
    let layer_types = required_string_array(text, "layer_types")?;

    validate_positive("text_config.vocab_size", vocab_size)?;
    validate_positive("text_config.hidden_size", hidden_size)?;
    validate_positive("text_config.num_hidden_layers", num_hidden_layers)?;
    validate_positive("text_config.num_attention_heads", num_attention_heads)?;
    validate_positive("text_config.num_key_value_heads", num_key_value_heads)?;
    validate_positive("text_config.head_dim", head_dim)?;
    validate_positive("text_config.max_position_embeddings", context_length)?;
    if !rope_theta.is_finite() || rope_theta <= 0.0 {
        return Err(invalid(
            "text_config.rope_parameters.rope_theta",
            &format!("must be finite and > 0; got {rope_theta}"),
        ));
    }
    if num_attention_heads % num_key_value_heads != 0 {
        return Err(invalid(
            "text_config.num_attention_heads",
            &format!(
                "must be divisible by text_config.num_key_value_heads ({num_key_value_heads}); got {num_attention_heads}"
            ),
        ));
    }
    if layer_types.len() != num_hidden_layers {
        return Err(invalid(
            "text_config.layer_types",
            &format!(
                "must contain one entry per hidden layer ({num_hidden_layers}); got {}",
                layer_types.len()
            ),
        ));
    }

    let has_hybrid_attention = has_hybrid_attention(&layer_types);
    let num_experts = optional_usize(text, "num_experts")?;
    let num_experts_per_tok = optional_usize(text, "num_experts_per_tok")?;
    let moe_intermediate_size = optional_usize(text, "moe_intermediate_size")?;
    let has_moe = num_experts.unwrap_or(0) > 0
        || num_experts_per_tok.unwrap_or(0) > 0
        || moe_intermediate_size.unwrap_or(0) > 0;
    let quantization = value
        .get("quantization_config")
        .and_then(|config| config.get("quant_method"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let vision = value.get("vision_config").and_then(Value::as_object);
    let multimodal = vision.is_some()
        || value.get("image_token_id").is_some()
        || value.get("video_token_id").is_some();

    Ok(Qwen3_5Config {
        model_type,
        text_model_type,
        architectures: optional_string_array(&value, "architectures")?,
        vocab_size,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        context_length,
        rope_theta,
        dtype,
        layer_types,
        has_hybrid_attention,
        num_experts,
        num_experts_per_tok,
        moe_intermediate_size,
        has_moe,
        quantization,
        multimodal,
        image_token_id: optional_u32(&value, "image_token_id")?,
        video_token_id: optional_u32(&value, "video_token_id")?,
        vision_hidden_size: vision
            .and_then(|_| optional_usize(&value["vision_config"], "hidden_size").transpose())
            .transpose()?,
    })
}

fn has_hybrid_attention(layer_types: &[String]) -> bool {
    layer_types
        .iter()
        .any(|layer_type| layer_type != "full_attention")
}

fn required_object<'a>(value: &'a Value, field: &str) -> Result<&'a Value> {
    match value.get(field) {
        Some(Value::Object(_)) => Ok(&value[field]),
        Some(other) => Err(invalid(
            field,
            &format!("must be a JSON object, got {other:?}"),
        )),
        None => Err(missing(field)),
    }
}

fn required_string(value: &Value, field: &str) -> Result<String> {
    match value.get(field).and_then(Value::as_str) {
        Some(value) => Ok(value.to_string()),
        None if value.get(field).is_some() => Err(invalid(field, "must be a JSON string")),
        None => Err(missing(field)),
    }
}

fn required_usize(value: &Value, field: &str) -> Result<usize> {
    match value.get(field).and_then(Value::as_u64) {
        Some(value) => value
            .try_into()
            .map_err(|_| invalid(field, &format!("{value} does not fit in usize"))),
        None if value.get(field).is_some() => {
            Err(invalid(field, "must be a non-negative JSON integer"))
        }
        None => Err(missing(field)),
    }
}

fn optional_usize(value: &Value, field: &str) -> Result<Option<usize>> {
    if value.get(field).is_none() {
        return Ok(None);
    }
    required_usize(value, field).map(Some)
}

fn optional_u32(value: &Value, field: &str) -> Result<Option<u32>> {
    match value.get(field).and_then(Value::as_u64) {
        Some(value) => value
            .try_into()
            .map(Some)
            .map_err(|_| invalid(field, &format!("{value} does not fit in u32"))),
        None if value.get(field).is_some() => {
            Err(invalid(field, "must be a non-negative JSON integer"))
        }
        None => Ok(None),
    }
}

fn required_f64(value: &Value, field: &str) -> Result<f64> {
    match value.get(field).and_then(Value::as_f64) {
        Some(value) => Ok(value),
        None if value.get(field).is_some() => Err(invalid(field, "must be a JSON number")),
        None => Err(missing(field)),
    }
}

fn required_string_array(value: &Value, field: &str) -> Result<Vec<String>> {
    match value.get(field).and_then(Value::as_array) {
        Some(values) => values
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| invalid(field, "array entries must be JSON strings"))
            })
            .collect(),
        None if value.get(field).is_some() => Err(invalid(field, "must be a JSON array")),
        None => Err(missing(field)),
    }
}

fn optional_string_array(value: &Value, field: &str) -> Result<Vec<String>> {
    if value.get(field).is_none() {
        return Ok(Vec::new());
    }
    required_string_array(value, field)
}

fn validate_positive(field: &str, value: usize) -> Result<()> {
    if value == 0 {
        Err(invalid(field, "must be > 0"))
    } else {
        Ok(())
    }
}

fn missing(field: &str) -> OcelotlError {
    invalid(field, "missing required Qwen3.5 config field")
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
    use crate::qwen::Qwen2_5Config;
    use ocelotl_core::{DType, ModelMetadata};
    use std::path::PathBuf;

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/metadata")
            .join(name)
    }

    fn qwen3_5_fixture() -> String {
        std::fs::read_to_string(fixture_path("qwen3_5_35b_a3b_fp8_config.json"))
            .expect("Qwen3.5 metadata fixture must be readable")
    }

    #[test]
    fn parse_qwen3_5_config_recognizes_family_separately_from_qwen2_5() {
        let cfg =
            parse_qwen3_5_config_json(&qwen3_5_fixture()).expect("Qwen3.5 fixture must parse");

        assert_eq!(cfg.model_type, "qwen3_5_moe");
        assert_eq!(cfg.text_model_type, "qwen3_5_moe_text");
        assert_eq!(
            cfg.architectures,
            vec!["Qwen3_5MoeForConditionalGeneration"]
        );
        assert_eq!(cfg.vocab_size, 248_320);
        assert_eq!(cfg.hidden_size, 2_048);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.context_length, 262_144);
        assert_eq!(cfg.rope_theta, 10_000_000.0);
        assert_eq!(cfg.dtype, "bfloat16");
        assert!(cfg.has_hybrid_attention);
        assert!(cfg.has_moe);
        assert!(cfg.multimodal);
        assert_eq!(cfg.quantization.as_deref(), Some("fp8"));
        assert_eq!(cfg.num_experts, Some(256));
        assert_eq!(cfg.num_experts_per_tok, Some(8));
        assert_eq!(cfg.moe_intermediate_size, Some(512));
        assert_eq!(cfg.image_token_id, Some(248_056));
        assert_eq!(cfg.video_token_id, Some(248_057));
        assert_eq!(cfg.vision_hidden_size, Some(1_152));
    }

    #[test]
    fn ensure_supported_for_execution_rejects_qwen3_5_hybrid_moe_multimodal_fp8() {
        let cfg = parse_qwen3_5_config_json(&qwen3_5_fixture()).unwrap();

        let err = cfg
            .ensure_supported_for_execution()
            .expect_err("Qwen3.5 execution must stay rejected until contracts land");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(unsupported.feature, "qwen3_5.execution_features");
                let requested = unsupported.requested.unwrap();
                assert!(requested.contains("hybrid_attention"));
                assert!(requested.contains("moe"));
                assert!(requested.contains("multimodal"));
                assert!(requested.contains("quantization=fp8"));
            }
            other => panic!("expected Unsupported for Qwen3.5 execution, got {other:?}"),
        }
    }

    #[test]
    fn parse_qwen3_5_rejects_qwen2_config_with_typed_unsupported() {
        let raw = std::fs::read_to_string(fixture_path("qwen2_5_0_5b_instruct_config.json"))
            .expect("Qwen2.5 fixture must be readable");

        let err = parse_qwen3_5_config_json(&raw)
            .expect_err("Qwen2.5 config must not be accepted as Qwen3.5");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(unsupported.feature, "qwen3_5.architecture");
                assert_eq!(unsupported.requested.as_deref(), Some("qwen2"));
                assert_eq!(unsupported.supported, vec!["qwen3_5_moe".to_string()]);
            }
            other => panic!("expected Unsupported for Qwen2.5 fixture, got {other:?}"),
        }
    }

    #[test]
    fn qwen2_5_config_rejects_qwen3_5_metadata() {
        let meta = ModelMetadata {
            architecture: "qwen3_5_moe".to_string(),
            vocab_size: 248_320,
            num_hidden_layers: 40,
            hidden_size: 2_048,
            intermediate_size: 512,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            head_dim: 256,
            context_length: 262_144,
            rope_theta: 10_000_000.0,
            rms_norm_eps: 1e-6,
            dtype: DType::BF16,
            tokenizer_model_hint: None,
        };

        let err = Qwen2_5Config::try_from(&meta)
            .expect_err("Qwen2.5 path must not accept Qwen3.5 metadata");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(unsupported.feature, "qwen2_5.architecture");
                assert_eq!(unsupported.requested.as_deref(), Some("qwen3_5_moe"));
            }
            other => panic!("expected Unsupported from Qwen2.5 architecture gate, got {other:?}"),
        }
    }

    #[test]
    fn parse_qwen3_5_rejects_layer_type_count_mismatch_before_compute() {
        let mut value: Value =
            serde_json::from_str(&qwen3_5_fixture()).expect("fixture JSON must parse");
        value["text_config"]["layer_types"] = serde_json::json!(["full_attention"]);
        let raw = serde_json::to_string(&value).unwrap();

        let err = parse_qwen3_5_config_json(&raw)
            .expect_err("layer type count mismatch must fail before compute");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("text_config.layer_types"));
            }
            other => panic!("expected InvalidModel for layer type count mismatch, got {other:?}"),
        }
    }
}
