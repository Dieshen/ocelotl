//! `WhisperModel` struct, constructors, accessors, and request/state validation.
//!
//! Forward composition for the encoder lives in `encode.rs`; decoder forward
//! and incremental decode live in `decode.rs`. The validation helpers in this
//! file are `pub(super)` because both encoder and decoder code paths consume
//! them; this keeps the validation invariants in one place.

use std::{collections::BTreeMap, path::Path, sync::Arc};

use ocelotl_core::{Result, TokenId};
use ocelotl_kernels::{DeviceTensor, KernelBackend, default_kernel_backend};
use ocelotl_loader::{LoadedTensor, inspect_safetensors, load_safetensors_tensors_f32};

use super::{WhisperConfig, parse_whisper_config_json, required_whisper_tensor_names};
use super::state::{WhisperDecoderState, WhisperEncodedAudio};
use super::weights::WhisperWeights;
use super::{checked_len_product, invalid_model, invalid_request};

#[derive(Debug, Clone)]
pub struct WhisperModel {
    pub(super) config: WhisperConfig,
    pub(super) weights: WhisperWeights,
    pub(super) kernels: Arc<dyn KernelBackend>,
    /// Device-resident mirror of every host weight tensor. Populated once at
    /// construction so the encoder/decoder forward path can pass weight
    /// `DeviceTensor` handles directly to `linear_d` / `layer_norm_d` / etc.
    /// without re-uploading every call. GW.4-2B: the upload cost is amortised
    /// once per model load, which is a cold-path event.
    pub(super) device_weights: Arc<BTreeMap<String, DeviceTensor>>,
}

impl WhisperModel {
    /// Load a local Whisper artifact directory.
    ///
    /// This reads `config.json` and `model.safetensors` from `dir`, delegates
    /// file inspection/value loading to `ocelotl-loader`, and never downloads
    /// artifacts.
    pub fn load_from_dir<P: AsRef<Path>>(dir: P) -> Result<Self> {
        Self::load_from_dir_with_kernel_backend(dir, default_kernel_backend())
    }

    /// Load a local Whisper artifact directory with an explicit kernel backend.
    pub fn load_from_dir_with_kernel_backend<P: AsRef<Path>>(
        dir: P,
        kernels: Arc<dyn KernelBackend>,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        Self::load_from_paths_with_kernel_backend(
            dir.join("config.json"),
            dir.join("model.safetensors"),
            kernels,
        )
    }

    /// Load a local Whisper config/model pair from explicit paths.
    pub fn load_from_paths<P: AsRef<Path>, Q: AsRef<Path>>(
        config_path: P,
        model_path: Q,
    ) -> Result<Self> {
        Self::load_from_paths_with_kernel_backend(config_path, model_path, default_kernel_backend())
    }

    /// Load a local Whisper config/model pair from explicit paths with an
    /// explicit kernel backend.
    pub fn load_from_paths_with_kernel_backend<P: AsRef<Path>, Q: AsRef<Path>>(
        config_path: P,
        model_path: Q,
        kernels: Arc<dyn KernelBackend>,
    ) -> Result<Self> {
        let config_path = config_path.as_ref();
        let model_path = model_path.as_ref();
        let raw = std::fs::read_to_string(config_path).map_err(|source| {
            ocelotl_core::OcelotlError::Io(ocelotl_core::IoError {
                path: Some(config_path.to_path_buf()),
                source,
            })
        })?;
        let config = parse_whisper_config_json(&raw)?;
        let manifest = inspect_safetensors(model_path)?;
        super::validate_whisper_tensors(&manifest, &config, Some(model_path))?;
        let names = required_whisper_tensor_names(&config);
        let tensors = load_safetensors_tensors_f32(model_path, &names)?;
        Self::with_kernel_backend(config, tensors, kernels)
    }

    pub fn new(config: WhisperConfig, tensors: Vec<LoadedTensor>) -> Result<Self> {
        Self::with_kernel_backend(config, tensors, default_kernel_backend())
    }

    pub fn with_kernel_backend(
        config: WhisperConfig,
        tensors: Vec<LoadedTensor>,
        kernels: Arc<dyn KernelBackend>,
    ) -> Result<Self> {
        let config = config.validate()?;
        let weights = WhisperWeights::from_loaded_tensors(&config, tensors)?;
        let device_weights = upload_device_weights(kernels.as_ref(), &weights)?;
        Ok(Self {
            config,
            weights,
            kernels,
            device_weights: Arc::new(device_weights),
        })
    }

    pub fn config(&self) -> &WhisperConfig {
        &self.config
    }

    pub fn kernel_backend(&self) -> &dyn KernelBackend {
        self.kernels.as_ref()
    }

    /// Borrow the device-resident `DeviceTensor` for a named weight. Errors
    /// if the name is not in the uploaded set — that should be impossible
    /// because `upload_device_weights` walks `required_whisper_tensor_names`,
    /// but we surface a typed error rather than panic so a future schema
    /// addition is loud.
    pub(super) fn device_weight(&self, name: &str) -> Result<&DeviceTensor> {
        self.device_weights
            .get(name)
            .ok_or_else(|| invalid_model(name, "missing device-resident weight upload"))
    }
}

fn upload_device_weights(
    kernels: &dyn KernelBackend,
    weights: &WhisperWeights,
) -> Result<BTreeMap<String, DeviceTensor>> {
    let mut device = BTreeMap::new();
    for name in weights.names() {
        let host = weights.get(name);
        let tensor = kernels.upload(host)?;
        device.insert(name.to_string(), tensor);
    }
    Ok(device)
}

pub(super) fn validate_forward_request(
    config: &WhisperConfig,
    log_mel: &[f32],
    mel_frames: usize,
    decoder_tokens: &[TokenId],
) -> Result<()> {
    validate_audio_request(config, log_mel, mel_frames)?;
    validate_decoder_tokens(config, decoder_tokens)
}

pub(super) fn validate_audio_request(
    config: &WhisperConfig,
    log_mel: &[f32],
    mel_frames: usize,
) -> Result<()> {
    if mel_frames == 0 {
        return Err(invalid_request("mel_frames", "must be > 0"));
    }
    let mel_len = checked_len_product("log_mel", &[mel_frames, config.mel_bins])?;
    if log_mel.len() != mel_len {
        return Err(invalid_request(
            "log_mel",
            &format!("expected length {mel_len}, got {}", log_mel.len()),
        ));
    }
    Ok(())
}

pub(super) fn validate_encoded_audio(
    config: &WhisperConfig,
    audio: &WhisperEncodedAudio,
) -> Result<()> {
    if audio.frames == 0 {
        return Err(invalid_request("encoded_audio.frames", "must be > 0"));
    }
    if audio.state_size != config.audio_state_size {
        return Err(invalid_request(
            "encoded_audio.state_size",
            &format!(
                "expected audio_state_size {}, got {}",
                config.audio_state_size, audio.state_size
            ),
        ));
    }
    let expected = checked_len_product("encoded_audio", &[audio.frames, audio.state_size])?;
    if audio.values.len() != expected {
        return Err(invalid_request(
            "encoded_audio.values",
            &format!("expected length {expected}, got {}", audio.values.len()),
        ));
    }
    if audio.cross_attention.len() != config.text_layers {
        return Err(invalid_request(
            "encoded_audio.cross_attention",
            &format!(
                "expected {} decoder-layer cross-attention caches, got {}",
                config.text_layers,
                audio.cross_attention.len()
            ),
        ));
    }
    let expected_cross = checked_len_product(
        "encoded_audio.cross_attention",
        &[audio.frames, config.text_state_size],
    )?;
    for (layer, cache) in audio.cross_attention.iter().enumerate() {
        if cache.key.len() != expected_cross {
            return Err(invalid_request(
                "encoded_audio.cross_attention.key",
                &format!(
                    "layer {layer} expected length {expected_cross}, got {}",
                    cache.key.len()
                ),
            ));
        }
        if cache.value.len() != expected_cross {
            return Err(invalid_request(
                "encoded_audio.cross_attention.value",
                &format!(
                    "layer {layer} expected length {expected_cross}, got {}",
                    cache.value.len()
                ),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_decoder_tokens(
    config: &WhisperConfig,
    decoder_tokens: &[TokenId],
) -> Result<()> {
    if decoder_tokens.is_empty() {
        return Err(invalid_request("decoder_tokens", "must be non-empty"));
    }
    if decoder_tokens.len() > config.text_context_length {
        return Err(invalid_request(
            "decoder_tokens",
            &format!(
                "decoder token length {} exceeds text_context_length {}",
                decoder_tokens.len(),
                config.text_context_length
            ),
        ));
    }
    for (idx, token) in decoder_tokens.iter().enumerate() {
        if token.0 as usize >= config.vocab_size {
            return Err(invalid_request(
                "decoder_tokens",
                &format!(
                    "token id {} at position {} is out of range for vocab_size {}",
                    token.0, idx, config.vocab_size
                ),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_decoder_token(
    config: &WhisperConfig,
    token: TokenId,
    position: usize,
) -> Result<()> {
    if token.0 as usize >= config.vocab_size {
        return Err(invalid_request(
            "decoder_tokens",
            &format!(
                "token id {} at position {} is out of range for vocab_size {}",
                token.0, position, config.vocab_size
            ),
        ));
    }
    Ok(())
}

pub(super) fn validate_decoder_state_for_append(
    config: &WhisperConfig,
    state: &WhisperDecoderState,
) -> Result<()> {
    if state.tokens.is_empty() {
        return Err(invalid_request("decoder_state.tokens", "must be non-empty"));
    }
    if state.tokens.len() >= config.text_context_length {
        return Err(invalid_request(
            "decoder_context_length",
            &format!(
                "decoder token length {} cannot accept another token because text_context_length is {}",
                state.tokens.len(),
                config.text_context_length
            ),
        ));
    }
    if state.self_attention.len() != config.text_layers {
        return Err(invalid_request(
            "decoder_state.self_attention",
            &format!(
                "expected {} decoder-layer self-attention caches, got {}",
                config.text_layers,
                state.self_attention.len()
            ),
        ));
    }
    let expected_cache = checked_len_product(
        "decoder_state.self_attention",
        &[state.tokens.len(), config.text_state_size],
    )?;
    for (layer, cache) in state.self_attention.iter().enumerate() {
        if cache.key.len() != expected_cache {
            return Err(invalid_request(
                "decoder_state.self_attention.key",
                &format!(
                    "layer {layer} expected length {expected_cache}, got {}",
                    cache.key.len()
                ),
            ));
        }
        if cache.value.len() != expected_cache {
            return Err(invalid_request(
                "decoder_state.self_attention.value",
                &format!(
                    "layer {layer} expected length {expected_cache}, got {}",
                    cache.value.len()
                ),
            ));
        }
    }
    if state.next_token_logits.len() != config.vocab_size {
        return Err(invalid_request(
            "decoder_state.next_token_logits",
            &format!(
                "expected length {}, got {}",
                config.vocab_size,
                state.next_token_logits.len()
            ),
        ));
    }
    Ok(())
}
