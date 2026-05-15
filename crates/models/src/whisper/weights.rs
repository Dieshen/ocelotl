//! Whisper weight bundle, safetensors-to-Whisper adapter, and tensor-shape rules.
//!
//! `WhisperWeights` owns every tensor the encoder and decoder consume. The
//! `expected_shape` / `block_shape` helpers encode the OpenAI Whisper layout
//! contract (per-architecture-block tensor naming) and are also exercised by
//! validation tests. CONV kernel width comes from the shared `super::`.

use std::collections::{BTreeMap, btree_map::Entry};

use ocelotl_core::Result;
use ocelotl_loader::LoadedTensor;

use super::{WhisperConfig, required_whisper_tensor_names};
use super::{
    CONV_KERNEL_WIDTH, checked_len_product, dtype_matches, invalid_model, supported_dtype_name,
};

#[derive(Debug, Clone)]
pub struct WhisperWeights {
    tensors: BTreeMap<String, Vec<f32>>,
}

impl WhisperWeights {
    pub fn from_loaded_tensors(config: &WhisperConfig, tensors: Vec<LoadedTensor>) -> Result<Self> {
        config.clone().validate()?;

        let mut by_name = BTreeMap::new();
        for tensor in tensors {
            match by_name.entry(tensor.name.clone()) {
                Entry::Vacant(entry) => {
                    entry.insert(tensor);
                }
                Entry::Occupied(_) => {
                    return Err(invalid_model(
                        &tensor.name,
                        "duplicate Whisper tensor supplied",
                    ));
                }
            }
        }

        let mut values_by_name = BTreeMap::new();
        for name in required_whisper_tensor_names(config) {
            let tensor = by_name
                .remove(&name)
                .ok_or_else(|| invalid_model(&name, "required Whisper tensor is missing"))?;
            let expected = expected_shape(&name, config)?;
            if tensor.shape != expected {
                return Err(invalid_model(
                    &name,
                    &format!(
                        "tensor `{name}` has shape {:?}, expected {:?}",
                        tensor.shape, expected
                    ),
                ));
            }
            if !dtype_matches(tensor.dtype, &config.dtype) {
                return Err(invalid_model(
                    &name,
                    &format!(
                        "tensor `{name}` has dtype {}, expected {:?}",
                        supported_dtype_name(tensor.dtype),
                        config.dtype
                    ),
                ));
            }
            let expected_len = checked_len_product(&name, &expected)?;
            if tensor.values.len() != expected_len {
                return Err(invalid_model(
                    &name,
                    &format!(
                        "tensor `{name}` has {} values, expected {expected_len}",
                        tensor.values.len()
                    ),
                ));
            }
            values_by_name.insert(name, tensor.values);
        }

        Ok(Self {
            tensors: values_by_name,
        })
    }

    pub(super) fn get(&self, name: &str) -> &[f32] {
        self.tensors
            .get(name)
            .map(Vec::as_slice)
            .expect("WhisperWeights validation guarantees required tensors")
    }

    /// Iterate over every weight name in deterministic key order. Used at
    /// `WhisperModel` construction to upload every weight to the device side
    /// once, so the forward path can pass `DeviceTensor` handles directly
    /// into `linear_d` / `layer_norm_d` / etc.
    pub(super) fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
}

pub(super) fn expected_shape(name: &str, config: &WhisperConfig) -> Result<Vec<usize>> {
    if name == "encoder.conv1.weight" {
        return Ok(vec![
            config.audio_state_size,
            config.mel_bins,
            CONV_KERNEL_WIDTH,
        ]);
    }
    if name == "encoder.conv1.bias" || name == "encoder.conv2.bias" {
        return Ok(vec![config.audio_state_size]);
    }
    if name == "encoder.conv2.weight" {
        return Ok(vec![
            config.audio_state_size,
            config.audio_state_size,
            CONV_KERNEL_WIDTH,
        ]);
    }
    if name == "encoder.positional_embedding" {
        return Ok(vec![config.audio_context_length, config.audio_state_size]);
    }
    if name == "encoder.ln_post.weight" || name == "encoder.ln_post.bias" {
        return Ok(vec![config.audio_state_size]);
    }
    if name == "decoder.token_embedding.weight" || name == "decoder.proj_out.weight" {
        return Ok(vec![config.vocab_size, config.text_state_size]);
    }
    if name == "decoder.positional_embedding" {
        return Ok(vec![config.text_context_length, config.text_state_size]);
    }
    if name == "decoder.ln.weight" || name == "decoder.ln.bias" {
        return Ok(vec![config.text_state_size]);
    }

    if let Some(rest) = name.strip_prefix("encoder.blocks.") {
        return block_shape(rest, config.audio_state_size, config.audio_ffn_size, None)
            .ok_or_else(|| invalid_model(name, "unknown encoder tensor name"));
    }
    if let Some(rest) = name.strip_prefix("decoder.blocks.") {
        return block_shape(
            rest,
            config.text_state_size,
            config.text_ffn_size,
            Some(config.audio_state_size),
        )
        .ok_or_else(|| invalid_model(name, "unknown decoder tensor name"));
    }
    Err(invalid_model(name, "unknown Whisper tensor name"))
}

fn block_shape(
    rest: &str,
    state: usize,
    ffn: usize,
    audio_state: Option<usize>,
) -> Option<Vec<usize>> {
    let dot = rest.find('.')?;
    let suffix = &rest[dot + 1..];
    if let Some(cross) = suffix.strip_prefix("cross_attn.") {
        let audio = audio_state?;
        return match cross {
            "query.weight" => Some(vec![state, state]),
            "query.bias" => Some(vec![state]),
            "key.weight" => Some(vec![state, audio]),
            "value.weight" => Some(vec![state, audio]),
            "value.bias" => Some(vec![state]),
            "out.weight" => Some(vec![state, state]),
            "out.bias" => Some(vec![state]),
            _ => None,
        };
    }
    match suffix {
        "attn.query.weight" => Some(vec![state, state]),
        "attn.query.bias" => Some(vec![state]),
        "attn.key.weight" => Some(vec![state, state]),
        "attn.value.weight" => Some(vec![state, state]),
        "attn.value.bias" => Some(vec![state]),
        "attn.out.weight" => Some(vec![state, state]),
        "attn.out.bias" => Some(vec![state]),
        "attn_ln.weight" | "attn_ln.bias" => Some(vec![state]),
        "cross_attn_ln.weight" | "cross_attn_ln.bias" => Some(vec![state]),
        "mlp.0.weight" => Some(vec![ffn, state]),
        "mlp.0.bias" => Some(vec![ffn]),
        "mlp.2.weight" => Some(vec![state, ffn]),
        "mlp.2.bias" => Some(vec![state]),
        "mlp_ln.weight" | "mlp_ln.bias" => Some(vec![state]),
        _ => None,
    }
}
