//! Real Whisper tensor-name and shape validation.
//!
//! The first W-ASR.8 contract uses OpenAI Whisper state-dict names as the
//! Ocelotl-owned canonical names: `encoder.*` and `decoder.*` without a
//! Hugging Face `model.` prefix. HF and converted safetensors artifacts may use
//! different prefixes or q/k/v projection names; this module intentionally does
//! not accept aliases until the ignored local-artifact parity harness proves a
//! concrete converted-name set.

use std::path::Path;

use ocelotl_core::{DType, InvalidModelError, OcelotlError, Result};
use ocelotl_loader::{SafetensorsManifest, SupportedDtype, require_tensors};

use super::WhisperConfig;

const CONV_KERNEL_WIDTH: usize = 3;

/// Build the canonical, ordered list of real Whisper tensors required by a
/// config.
pub fn required_whisper_tensor_names(config: &WhisperConfig) -> Vec<String> {
    let mut names =
        Vec::with_capacity(5 + config.audio_layers * 15 + 2 + config.text_layers * 24 + 3);

    names.push("encoder.conv1.weight".to_string());
    names.push("encoder.conv1.bias".to_string());
    names.push("encoder.conv2.weight".to_string());
    names.push("encoder.conv2.bias".to_string());
    names.push("encoder.positional_embedding".to_string());

    for layer in 0..config.audio_layers {
        push_self_attention_block(&mut names, "encoder", layer);
    }

    names.push("decoder.token_embedding.weight".to_string());
    names.push("decoder.positional_embedding".to_string());

    for layer in 0..config.text_layers {
        push_self_attention_block(&mut names, "decoder", layer);
        names.push(format!("decoder.blocks.{layer}.cross_attn.query.weight"));
        names.push(format!("decoder.blocks.{layer}.cross_attn.query.bias"));
        names.push(format!("decoder.blocks.{layer}.cross_attn.key.weight"));
        names.push(format!("decoder.blocks.{layer}.cross_attn.value.weight"));
        names.push(format!("decoder.blocks.{layer}.cross_attn.value.bias"));
        names.push(format!("decoder.blocks.{layer}.cross_attn.out.weight"));
        names.push(format!("decoder.blocks.{layer}.cross_attn.out.bias"));
        names.push(format!("decoder.blocks.{layer}.cross_attn_ln.weight"));
        names.push(format!("decoder.blocks.{layer}.cross_attn_ln.bias"));
    }

    names.push("decoder.ln.weight".to_string());
    names.push("decoder.ln.bias".to_string());
    if !config.tie_word_embeddings {
        names.push("decoder.proj_out.weight".to_string());
    }

    names
}

/// Validate a safetensors manifest against the real Whisper tensor contract.
pub fn validate_whisper_tensors(
    manifest: &SafetensorsManifest,
    config: &WhisperConfig,
    path: Option<&Path>,
) -> Result<()> {
    config.clone().validate()?;

    let required = required_whisper_tensor_names(config);
    let required_refs: Vec<&str> = required.iter().map(String::as_str).collect();
    require_tensors(manifest, &required_refs, path)?;

    check_shape(
        manifest,
        "encoder.conv1.weight",
        &[config.audio_state_size, config.mel_bins, CONV_KERNEL_WIDTH],
        &config.dtype,
        path,
    )?;
    check_shape(
        manifest,
        "encoder.conv1.bias",
        &[config.audio_state_size],
        &config.dtype,
        path,
    )?;
    check_shape(
        manifest,
        "encoder.conv2.weight",
        &[
            config.audio_state_size,
            config.audio_state_size,
            CONV_KERNEL_WIDTH,
        ],
        &config.dtype,
        path,
    )?;
    check_shape(
        manifest,
        "encoder.conv2.bias",
        &[config.audio_state_size],
        &config.dtype,
        path,
    )?;
    check_shape(
        manifest,
        "encoder.positional_embedding",
        &[config.audio_context_length, config.audio_state_size],
        &config.dtype,
        path,
    )?;

    for layer in 0..config.audio_layers {
        check_self_attention_block(
            manifest,
            "encoder",
            layer,
            config.audio_state_size,
            config.audio_ffn_size,
            &config.dtype,
            path,
        )?;
    }

    check_shape(
        manifest,
        "decoder.token_embedding.weight",
        &[config.vocab_size, config.text_state_size],
        &config.dtype,
        path,
    )?;
    check_shape(
        manifest,
        "decoder.positional_embedding",
        &[config.text_context_length, config.text_state_size],
        &config.dtype,
        path,
    )?;

    for layer in 0..config.text_layers {
        check_self_attention_block(
            manifest,
            "decoder",
            layer,
            config.text_state_size,
            config.text_ffn_size,
            &config.dtype,
            path,
        )?;
        check_cross_attention_block(manifest, layer, config, path)?;
    }

    check_shape(
        manifest,
        "decoder.ln.weight",
        &[config.text_state_size],
        &config.dtype,
        path,
    )?;
    check_shape(
        manifest,
        "decoder.ln.bias",
        &[config.text_state_size],
        &config.dtype,
        path,
    )?;

    if !config.tie_word_embeddings {
        check_shape(
            manifest,
            "decoder.proj_out.weight",
            &[config.vocab_size, config.text_state_size],
            &config.dtype,
            path,
        )?;
    }

    Ok(())
}

fn push_self_attention_block(names: &mut Vec<String>, prefix: &str, layer: usize) {
    names.push(format!("{prefix}.blocks.{layer}.attn.query.weight"));
    names.push(format!("{prefix}.blocks.{layer}.attn.query.bias"));
    names.push(format!("{prefix}.blocks.{layer}.attn.key.weight"));
    names.push(format!("{prefix}.blocks.{layer}.attn.value.weight"));
    names.push(format!("{prefix}.blocks.{layer}.attn.value.bias"));
    names.push(format!("{prefix}.blocks.{layer}.attn.out.weight"));
    names.push(format!("{prefix}.blocks.{layer}.attn.out.bias"));
    names.push(format!("{prefix}.blocks.{layer}.attn_ln.weight"));
    names.push(format!("{prefix}.blocks.{layer}.attn_ln.bias"));
    names.push(format!("{prefix}.blocks.{layer}.mlp.0.weight"));
    names.push(format!("{prefix}.blocks.{layer}.mlp.0.bias"));
    names.push(format!("{prefix}.blocks.{layer}.mlp.2.weight"));
    names.push(format!("{prefix}.blocks.{layer}.mlp.2.bias"));
    names.push(format!("{prefix}.blocks.{layer}.mlp_ln.weight"));
    names.push(format!("{prefix}.blocks.{layer}.mlp_ln.bias"));
}

fn check_self_attention_block(
    manifest: &SafetensorsManifest,
    prefix: &str,
    layer: usize,
    state: usize,
    ffn: usize,
    dtype: &DType,
    path: Option<&Path>,
) -> Result<()> {
    check_shape(
        manifest,
        &format!("{prefix}.blocks.{layer}.attn.query.weight"),
        &[state, state],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("{prefix}.blocks.{layer}.attn.query.bias"),
        &[state],
        dtype,
        path,
    )?;
    // OpenAI Whisper keys are bias-free in both self-attention and
    // cross-attention. Add aliases only when a real converted artifact needs
    // an alternate name, not preemptively.
    check_shape(
        manifest,
        &format!("{prefix}.blocks.{layer}.attn.key.weight"),
        &[state, state],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("{prefix}.blocks.{layer}.attn.value.weight"),
        &[state, state],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("{prefix}.blocks.{layer}.attn.value.bias"),
        &[state],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("{prefix}.blocks.{layer}.attn.out.weight"),
        &[state, state],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("{prefix}.blocks.{layer}.attn.out.bias"),
        &[state],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("{prefix}.blocks.{layer}.attn_ln.weight"),
        &[state],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("{prefix}.blocks.{layer}.attn_ln.bias"),
        &[state],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("{prefix}.blocks.{layer}.mlp.0.weight"),
        &[ffn, state],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("{prefix}.blocks.{layer}.mlp.0.bias"),
        &[ffn],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("{prefix}.blocks.{layer}.mlp.2.weight"),
        &[state, ffn],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("{prefix}.blocks.{layer}.mlp.2.bias"),
        &[state],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("{prefix}.blocks.{layer}.mlp_ln.weight"),
        &[state],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("{prefix}.blocks.{layer}.mlp_ln.bias"),
        &[state],
        dtype,
        path,
    )?;

    Ok(())
}

fn check_cross_attention_block(
    manifest: &SafetensorsManifest,
    layer: usize,
    config: &WhisperConfig,
    path: Option<&Path>,
) -> Result<()> {
    let text = config.text_state_size;
    let audio = config.audio_state_size;
    let dtype = &config.dtype;

    check_shape(
        manifest,
        &format!("decoder.blocks.{layer}.cross_attn.query.weight"),
        &[text, text],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("decoder.blocks.{layer}.cross_attn.query.bias"),
        &[text],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("decoder.blocks.{layer}.cross_attn.key.weight"),
        &[text, audio],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("decoder.blocks.{layer}.cross_attn.value.weight"),
        &[text, audio],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("decoder.blocks.{layer}.cross_attn.value.bias"),
        &[text],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("decoder.blocks.{layer}.cross_attn.out.weight"),
        &[text, text],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("decoder.blocks.{layer}.cross_attn.out.bias"),
        &[text],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("decoder.blocks.{layer}.cross_attn_ln.weight"),
        &[text],
        dtype,
        path,
    )?;
    check_shape(
        manifest,
        &format!("decoder.blocks.{layer}.cross_attn_ln.bias"),
        &[text],
        dtype,
        path,
    )?;

    Ok(())
}

fn check_shape(
    manifest: &SafetensorsManifest,
    name: &str,
    expected: &[usize],
    expected_dtype: &DType,
    path: Option<&Path>,
) -> Result<()> {
    let entry = manifest
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| {
            OcelotlError::from(InvalidModelError {
                path: path.map(|p| p.to_path_buf()),
                field: Some(name.to_string()),
                message: format!("tensor `{name}` not found in safetensors header"),
            })
        })?;

    if entry.shape != expected {
        return Err(OcelotlError::from(InvalidModelError {
            path: path.map(|p| p.to_path_buf()),
            field: Some(name.to_string()),
            message: format!(
                "tensor `{name}` has shape {:?}, expected {:?}",
                entry.shape, expected,
            ),
        }));
    }
    if !dtype_matches(entry.dtype, expected_dtype) {
        return Err(OcelotlError::from(InvalidModelError {
            path: path.map(|p| p.to_path_buf()),
            field: Some(name.to_string()),
            message: format!(
                "tensor `{name}` has dtype {}, expected {:?}",
                supported_dtype_name(entry.dtype),
                expected_dtype,
            ),
        }));
    }

    Ok(())
}

fn dtype_matches(actual: SupportedDtype, expected: &DType) -> bool {
    matches!(
        (actual, expected),
        (SupportedDtype::F32, DType::F32)
            | (SupportedDtype::F16, DType::F16)
            | (SupportedDtype::BF16, DType::BF16)
    )
}

fn supported_dtype_name(dtype: SupportedDtype) -> &'static str {
    match dtype {
        SupportedDtype::F32 => "F32",
        SupportedDtype::F16 => "F16",
        SupportedDtype::BF16 => "BF16",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ocelotl_loader::TensorEntry;

    fn tiny_config() -> WhisperConfig {
        WhisperConfig {
            vocab_size: 64,
            mel_bins: 80,
            audio_context_length: 8,
            audio_state_size: 12,
            audio_attention_heads: 3,
            audio_layers: 2,
            audio_ffn_size: 48,
            text_context_length: 10,
            text_state_size: 12,
            text_attention_heads: 3,
            text_layers: 2,
            text_ffn_size: 48,
            dtype: DType::F32,
            tie_word_embeddings: true,
        }
    }

    fn manifest_with(
        names: &[String],
        shape_for: impl Fn(&str) -> Vec<usize>,
    ) -> SafetensorsManifest {
        let tensors = names
            .iter()
            .map(|name| TensorEntry {
                name: name.clone(),
                shape: shape_for(name),
                dtype: SupportedDtype::F32,
                byte_range: (0, 0),
            })
            .collect();
        SafetensorsManifest {
            tensors,
            data_len: 0,
        }
    }

    fn shape_of(name: &str, cfg: &WhisperConfig) -> Vec<usize> {
        if name == "encoder.conv1.weight" {
            return vec![cfg.audio_state_size, cfg.mel_bins, CONV_KERNEL_WIDTH];
        }
        if name == "encoder.conv1.bias" || name == "encoder.conv2.bias" {
            return vec![cfg.audio_state_size];
        }
        if name == "encoder.conv2.weight" {
            return vec![
                cfg.audio_state_size,
                cfg.audio_state_size,
                CONV_KERNEL_WIDTH,
            ];
        }
        if name == "encoder.positional_embedding" {
            return vec![cfg.audio_context_length, cfg.audio_state_size];
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

        let (prefix, state, ffn) = if name.starts_with("encoder.blocks.") {
            ("encoder", cfg.audio_state_size, cfg.audio_ffn_size)
        } else if name.starts_with("decoder.blocks.") {
            ("decoder", cfg.text_state_size, cfg.text_ffn_size)
        } else {
            panic!("unknown tensor name {name}");
        };

        let rest = name
            .strip_prefix(&format!("{prefix}.blocks."))
            .expect("block prefix");
        let dot = rest.find('.').expect("layer suffix dot");
        let suffix = &rest[dot + 1..];

        if let Some(cross_suffix) = suffix.strip_prefix("cross_attn.") {
            let audio = cfg.audio_state_size;
            return match cross_suffix {
                "query.weight" => vec![state, state],
                "query.bias" => vec![state],
                "key.weight" => vec![state, audio],
                "value.weight" => vec![state, audio],
                "value.bias" => vec![state],
                "out.weight" => vec![state, state],
                "out.bias" => vec![state],
                other => panic!("unknown cross-attn suffix {other}"),
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

    #[test]
    fn required_names_cover_real_whisper_tiny_en_families() {
        let cfg = tiny_config();
        let names = required_whisper_tensor_names(&cfg);

        assert!(names.contains(&"encoder.conv1.weight".to_string()));
        assert!(names.contains(&"encoder.conv2.weight".to_string()));
        assert!(names.contains(&"encoder.positional_embedding".to_string()));
        assert!(names.contains(&"encoder.blocks.0.attn.query.weight".to_string()));
        assert!(names.contains(&"encoder.blocks.0.mlp.0.weight".to_string()));
        assert!(names.contains(&"decoder.token_embedding.weight".to_string()));
        assert!(names.contains(&"decoder.positional_embedding".to_string()));
        assert!(names.contains(&"decoder.blocks.0.attn.key.weight".to_string()));
        assert!(names.contains(&"decoder.blocks.0.cross_attn.key.weight".to_string()));
        assert!(names.contains(&"decoder.blocks.0.mlp.2.weight".to_string()));
        assert!(names.contains(&"decoder.ln.weight".to_string()));
        assert!(!names.contains(&"decoder.proj_out.weight".to_string()));
        assert_eq!(
            names.len(),
            5 + cfg.audio_layers * 15 + 2 + cfg.text_layers * 24 + 2
        );
    }

    #[test]
    fn untied_projection_requires_extra_projection_tensor() {
        let mut cfg = tiny_config();
        cfg.tie_word_embeddings = false;

        let names = required_whisper_tensor_names(&cfg);

        assert!(names.contains(&"decoder.proj_out.weight".to_string()));
    }

    #[test]
    fn validate_accepts_complete_manifest() {
        let cfg = tiny_config();
        let names = required_whisper_tensor_names(&cfg);
        let manifest = manifest_with(&names, |name| shape_of(name, &cfg));

        validate_whisper_tensors(&manifest, &cfg, None)
            .expect("complete Whisper manifest must validate");
    }

    #[test]
    fn validate_rejects_invalid_config_before_manifest_walk() {
        let mut cfg = tiny_config();
        cfg.audio_attention_heads = 0;
        let manifest = SafetensorsManifest {
            tensors: Vec::new(),
            data_len: 0,
        };

        let err = validate_whisper_tensors(&manifest, &cfg, None)
            .expect_err("invalid config must fail before tensor validation");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("audio_attention_heads"));
            }
            other => panic!("expected InvalidModel, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_missing_cross_attention_tensor() {
        let cfg = tiny_config();
        let dropped = "decoder.blocks.0.cross_attn.key.weight".to_string();
        let mut names = required_whisper_tensor_names(&cfg);
        names.retain(|name| name != &dropped);
        let manifest = manifest_with(&names, |name| shape_of(name, &cfg));

        let err =
            validate_whisper_tensors(&manifest, &cfg, None).expect_err("missing tensor must fail");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some(dropped.as_str()));
                assert!(invalid.message.contains(&dropped));
            }
            other => panic!("expected InvalidModel, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_wrong_shape() {
        let cfg = tiny_config();
        let bad = "encoder.conv1.weight".to_string();
        let names = required_whisper_tensor_names(&cfg);
        let manifest = manifest_with(&names, |name| {
            if name == bad {
                vec![cfg.mel_bins, cfg.audio_state_size, CONV_KERNEL_WIDTH]
            } else {
                shape_of(name, &cfg)
            }
        });

        let err = validate_whisper_tensors(&manifest, &cfg, None)
            .expect_err("wrong tensor shape must fail");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some(bad.as_str()));
                assert!(invalid.message.contains("expected"));
            }
            other => panic!("expected InvalidModel, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_wrong_dtype() {
        let cfg = tiny_config();
        let names = required_whisper_tensor_names(&cfg);
        let mut manifest = manifest_with(&names, |name| shape_of(name, &cfg));
        manifest
            .tensors
            .iter_mut()
            .find(|entry| entry.name == "decoder.token_embedding.weight")
            .expect("token embedding entry exists")
            .dtype = SupportedDtype::BF16;

        let err =
            validate_whisper_tensors(&manifest, &cfg, None).expect_err("wrong dtype must fail");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(
                    invalid.field.as_deref(),
                    Some("decoder.token_embedding.weight")
                );
                assert!(invalid.message.contains("dtype"));
            }
            other => panic!("expected InvalidModel, got {other:?}"),
        }
    }

    #[test]
    fn validate_accepts_f16_manifest_when_config_is_f16() {
        let mut cfg = tiny_config();
        cfg.dtype = DType::F16;
        let names = required_whisper_tensor_names(&cfg);
        let mut manifest = manifest_with(&names, |name| shape_of(name, &cfg));
        for entry in &mut manifest.tensors {
            entry.dtype = SupportedDtype::F16;
        }

        validate_whisper_tensors(&manifest, &cfg, None)
            .expect("F16 config must accept F16 tensors");
    }
}
