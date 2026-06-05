//! Gemma4 GGUF metadata contract.
//!
//! Gemma4 support starts as an inspect/reject path. `ocelotl-loader` owns the
//! GGUF parser; this module projects the loader-owned manifest into the
//! Gemma4-specific facts the model layer must preserve before any execution
//! path is allowed to run.

use ocelotl_core::{
    InvalidModelError, InvalidRequestError, OcelotlError, Result, TokenId, UnsupportedError,
};
use ocelotl_kernels::{KernelBackend, default_kernel_backend};
use ocelotl_loader::{
    GgmlTensorType, GgufManifest, GgufMetadataType, GgufMetadataValue, GgufTensorEntry,
    LoadedTensor, SupportedDtype, inspect_gguf, load_gguf_tensors_dequantized_f32,
    load_gguf_tensors_f32,
};
use std::{
    collections::{BTreeMap, btree_map::Entry},
    path::Path,
    sync::Arc,
};

const GEMMA4_ARCHITECTURE: &str = "gemma4";
const GGUF_FILE_TYPE_Q4_K_M: u32 = 15;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gemma4Quantization {
    Unquantized,
    Q4KM,
    FileType(u32),
}

impl Gemma4Quantization {
    fn from_file_type(file_type: u32) -> Self {
        match file_type {
            0 => Self::Unquantized,
            GGUF_FILE_TYPE_Q4_K_M => Self::Q4KM,
            other => Self::FileType(other),
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::Unquantized => "unquantized".to_string(),
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
    pub embedding_length_per_layer_input: usize,
    pub feed_forward_length: usize,
    pub attention_head_count: usize,
    pub attention_head_count_kv: usize,
    pub attention_key_length: usize,
    pub attention_value_length: usize,
    pub attention_key_length_swa: usize,
    pub attention_value_length_swa: usize,
    pub rope_dimension_count: usize,
    pub rope_dimension_count_swa: usize,
    pub rope_freq_base: f32,
    pub rope_freq_base_swa: f32,
    pub rms_norm_eps: f32,
    pub attention_sliding_window: Option<usize>,
    pub attention_shared_kv_layers: Option<usize>,
    pub attention_sliding_window_pattern_len: Option<usize>,
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

    /// Return `Ok(())` only for the explicitly supported MF.7 text-forward
    /// subset. Real Gemma4 GGUF artifacts still fail here because they carry
    /// multimodal, sliding-window/shared-KV, softcap, and quantized-origin
    /// features whose execution semantics are not implemented yet.
    pub fn ensure_supported_for_text_forward(&self) -> Result<()> {
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
        if self.attention_sliding_window_pattern_len.is_some() {
            requested.push("sliding_window_pattern".to_string());
        }
        if self.final_logit_softcap.is_some() {
            requested.push("final_logit_softcap".to_string());
        }
        if self.has_quantized_tensors {
            requested.push("quantized_tensors".to_string());
        }
        if self.quantization != Gemma4Quantization::Unquantized {
            requested.push(format!("quantization={}", self.quantization.label()));
        }
        if self.attention_key_length != self.attention_key_length_swa {
            requested.push("mixed_swa_global_key_widths".to_string());
        }
        if self.attention_value_length != self.attention_value_length_swa {
            requested.push("mixed_swa_global_value_widths".to_string());
        }
        if self.attention_key_length != self.attention_value_length {
            requested.push("distinct_key_value_widths".to_string());
        }
        if self.rope_dimension_count != self.attention_key_length
            || self.rope_dimension_count_swa != self.attention_key_length_swa
        {
            requested.push("rope_frequency_table_or_partial_rope".to_string());
        }

        if requested.is_empty() {
            return Ok(());
        }

        Err(OcelotlError::from(UnsupportedError {
            feature: "gemma4.text_forward_features".to_string(),
            requested: Some(requested.join(",")),
            supported: vec![
                "text-only unquantized dense F32 synthetic subset with uniform full attention"
                    .to_string(),
            ],
        }))
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
        let embedding_length_per_layer_input =
            required_usize(manifest, "gemma4.embedding_length_per_layer_input")?;
        let feed_forward_length = required_usize(manifest, "gemma4.feed_forward_length")?;
        let attention_head_count = required_usize(manifest, "gemma4.attention.head_count")?;
        let attention_head_count_kv = required_usize(manifest, "gemma4.attention.head_count_kv")?;
        let attention_key_length = required_usize(manifest, "gemma4.attention.key_length")?;
        let attention_value_length = required_usize(manifest, "gemma4.attention.value_length")?;
        let attention_key_length_swa = required_usize(manifest, "gemma4.attention.key_length_swa")?;
        let attention_value_length_swa =
            required_usize(manifest, "gemma4.attention.value_length_swa")?;
        let rope_dimension_count = required_usize(manifest, "gemma4.rope.dimension_count")?;
        let rope_dimension_count_swa = required_usize(manifest, "gemma4.rope.dimension_count_swa")?;
        let rope_freq_base = required_f32(manifest, "gemma4.rope.freq_base")?;
        let rope_freq_base_swa = required_f32(manifest, "gemma4.rope.freq_base_swa")?;
        let rms_norm_eps = required_f32(manifest, "gemma4.attention.layer_norm_rms_epsilon")?;
        let quantization =
            Gemma4Quantization::from_file_type(required_u32(manifest, "general.file_type")?);

        validate_positive("gemma4.context_length", context_length)?;
        validate_positive("gemma4.block_count", block_count)?;
        validate_positive("gemma4.embedding_length", embedding_length)?;
        validate_positive(
            "gemma4.embedding_length_per_layer_input",
            embedding_length_per_layer_input,
        )?;
        validate_positive("gemma4.feed_forward_length", feed_forward_length)?;
        validate_positive("gemma4.attention.head_count", attention_head_count)?;
        validate_positive("gemma4.attention.head_count_kv", attention_head_count_kv)?;
        validate_positive("gemma4.attention.key_length", attention_key_length)?;
        validate_positive("gemma4.attention.value_length", attention_value_length)?;
        validate_positive("gemma4.attention.key_length_swa", attention_key_length_swa)?;
        validate_positive(
            "gemma4.attention.value_length_swa",
            attention_value_length_swa,
        )?;
        validate_positive("gemma4.rope.dimension_count", rope_dimension_count)?;
        validate_positive("gemma4.rope.dimension_count_swa", rope_dimension_count_swa)?;
        validate_finite_positive("gemma4.rope.freq_base", rope_freq_base)?;
        validate_finite_positive("gemma4.rope.freq_base_swa", rope_freq_base_swa)?;
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
        let attention_sliding_window_pattern_len = optional_array_len(
            manifest,
            "gemma4.attention.sliding_window_pattern",
            GgufMetadataType::Bool,
        )?;
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
            embedding_length_per_layer_input,
            feed_forward_length,
            attention_head_count,
            attention_head_count_kv,
            attention_key_length,
            attention_value_length,
            attention_key_length_swa,
            attention_value_length_swa,
            rope_dimension_count,
            rope_dimension_count_swa,
            rope_freq_base,
            rope_freq_base_swa,
            rms_norm_eps,
            attention_sliding_window,
            attention_shared_kv_layers,
            attention_sliding_window_pattern_len,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gemma4TensorKind {
    F32,
    BF16,
    KQuantized,
}

struct Gemma4TensorSpec {
    name: String,
    shape: Vec<usize>,
    kind: Gemma4TensorKind,
}

impl Gemma4TensorKind {
    fn is_dense(self) -> bool {
        matches!(self, Self::F32 | Self::BF16)
    }
}

/// Build the canonical, ordered list of GGUF tensor names required by the
/// selected Gemma4 E4B Q4_K_M artifact.
pub fn required_gemma4_tensor_names(config: &Gemma4Config) -> Vec<String> {
    required_gemma4_tensor_specs(config)
        .into_iter()
        .map(|spec| spec.name)
        .collect()
}

/// Build the subset of required Gemma4 tensor names whose GGUF payloads can be
/// loaded losslessly into `LoadedTensor` today.
pub fn required_gemma4_dense_tensor_names(config: &Gemma4Config) -> Vec<String> {
    required_gemma4_tensor_specs(config)
        .into_iter()
        .filter(|spec| spec.kind.is_dense())
        .map(|spec| spec.name)
        .collect()
}

/// Load the dense F32/BF16 Gemma4 tensors from a local GGUF artifact. This is
/// the first value-loading step toward Gemma4 execution; block-quantized matrix
/// tensors remain behind the explicit dequant policy gate.
pub fn load_gemma4_dense_tensors_from_gguf(
    path: impl AsRef<Path>,
) -> Result<(Gemma4Config, Vec<LoadedTensor>)> {
    let path = path.as_ref();
    let manifest = inspect_gguf(path)?;
    let config = Gemma4Config::try_from(&manifest)?;
    validate_gemma4_tensor_inventory(&manifest, &config, Some(path))?;

    let names = required_gemma4_dense_tensor_names(&config);
    let tensors = load_gguf_tensors_f32(path, &names)?;
    validate_gemma4_dense_tensors(&config, &tensors, Some(path))?;
    Ok((config, tensors))
}

/// Load all required Gemma4 GGUF tensors into F32 value space, dequantizing
/// Q4_K/Q5_K/Q6_K matrices through the explicit loader API.
///
/// This proves the artifact can be read as values; it does not make Gemma4
/// executable. `Gemma4Config::ensure_supported_for_execution` remains the
/// execution gate for multimodal, sliding-window/shared-KV, softcap, and family
/// forward-path support.
pub fn load_gemma4_dequantized_tensors_from_gguf(
    path: impl AsRef<Path>,
) -> Result<(Gemma4Config, Vec<LoadedTensor>)> {
    let path = path.as_ref();
    let manifest = inspect_gguf(path)?;
    let config = Gemma4Config::try_from(&manifest)?;
    validate_gemma4_tensor_inventory(&manifest, &config, Some(path))?;

    let names = required_gemma4_tensor_names(&config);
    let tensors = load_gguf_tensors_dequantized_f32(path, &names)?;
    validate_gemma4_dequantized_tensors(&config, &tensors, Some(path))?;
    Ok((config, tensors))
}

/// Validate the selected Gemma4 GGUF header's required tensor inventory without
/// claiming those tensors can execute yet.
pub fn validate_gemma4_tensor_inventory(
    manifest: &GgufManifest,
    config: &Gemma4Config,
    path: Option<&Path>,
) -> Result<()> {
    validate_gemma4_tensor_dimensions(config, path)?;
    for spec in required_gemma4_tensor_specs(config) {
        check_gguf_tensor(manifest, &spec, path)?;
    }
    Ok(())
}

/// Validate loaded dense Gemma4 values against the same model-family contract
/// used for GGUF header inspection.
pub fn validate_gemma4_dense_tensors(
    config: &Gemma4Config,
    tensors: &[LoadedTensor],
    path: Option<&Path>,
) -> Result<()> {
    validate_gemma4_tensor_dimensions(config, path)?;

    let mut by_name = BTreeMap::new();
    for tensor in tensors {
        match by_name.entry(tensor.name.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(tensor);
            }
            Entry::Occupied(_) => {
                return Err(invalid_at(
                    path,
                    &tensor.name,
                    "duplicate Gemma4 dense tensor supplied",
                ));
            }
        }
    }

    for spec in required_gemma4_tensor_specs(config)
        .into_iter()
        .filter(|spec| spec.kind.is_dense())
    {
        let tensor = by_name.remove(&spec.name).ok_or_else(|| {
            invalid_at(path, &spec.name, "required Gemma4 dense tensor is missing")
        })?;
        check_loaded_dense_tensor(tensor, &spec, path)?;
    }

    Ok(())
}

/// Validate all loaded Gemma4 values after explicit K-quant dequantization.
pub fn validate_gemma4_dequantized_tensors(
    config: &Gemma4Config,
    tensors: &[LoadedTensor],
    path: Option<&Path>,
) -> Result<()> {
    validate_gemma4_tensor_dimensions(config, path)?;

    let mut by_name = BTreeMap::new();
    for tensor in tensors {
        match by_name.entry(tensor.name.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(tensor);
            }
            Entry::Occupied(_) => {
                return Err(invalid_at(
                    path,
                    &tensor.name,
                    "duplicate Gemma4 tensor supplied",
                ));
            }
        }
    }

    for spec in required_gemma4_tensor_specs(config) {
        let tensor = by_name
            .remove(&spec.name)
            .ok_or_else(|| invalid_at(path, &spec.name, "required Gemma4 tensor is missing"))?;
        check_loaded_dequantized_tensor(tensor, &spec, path)?;
    }

    Ok(())
}

/// Validate tensors for execution. This is intentionally stricter than
/// inventory validation: Q4_K_M carries quantized block tensors, and Ocelotl
/// does not have a Gemma4 dequant policy yet.
pub fn validate_gemma4_tensors(
    manifest: &GgufManifest,
    config: &Gemma4Config,
    path: Option<&Path>,
) -> Result<()> {
    validate_gemma4_tensor_inventory(manifest, config, path)?;
    if required_gemma4_tensor_specs(config)
        .into_iter()
        .any(|spec| spec.kind == Gemma4TensorKind::KQuantized)
    {
        return Err(OcelotlError::from(UnsupportedError {
            feature: "gemma4.quantized_tensors".to_string(),
            requested: Some(config.quantization.label()),
            supported: vec!["no Gemma4 dequant policy yet".to_string()],
        }));
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct Gemma4TextLayerWeights {
    /// `[hidden]`.
    pub attn_norm_w: Vec<f32>,
    /// `[hidden, num_attention_heads * attention_key_length]`.
    pub attn_q_w: Vec<f32>,
    /// `[hidden, num_key_value_heads * attention_key_length]`.
    pub attn_k_w: Vec<f32>,
    /// `[hidden, num_key_value_heads * attention_value_length]`.
    pub attn_v_w: Vec<f32>,
    /// `[num_attention_heads * attention_key_length, hidden]`.
    pub attn_o_w: Vec<f32>,
    /// `[attention_key_length]`.
    pub attn_q_norm_w: Vec<f32>,
    /// `[attention_key_length]`.
    pub attn_k_norm_w: Vec<f32>,
    /// `[hidden]`.
    pub ffn_norm_w: Vec<f32>,
    /// `[hidden, feed_forward_length]`.
    pub ffn_gate_w: Vec<f32>,
    /// `[hidden, feed_forward_length]`.
    pub ffn_up_w: Vec<f32>,
    /// `[feed_forward_length, hidden]`.
    pub ffn_down_w: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Gemma4TextWeights {
    /// `[vocab, hidden]`. `from_loaded_tensors` transposes GGUF's
    /// `[hidden, vocab]` embedding table into this lookup layout.
    pub token_embd: Vec<f32>,
    pub layers: Vec<Gemma4TextLayerWeights>,
    /// `[hidden]`.
    pub output_norm_w: Vec<f32>,
    /// `[hidden, vocab]`. For the MF.7 subset this is tied to
    /// `token_embd.weight`; no separate Gemma4 output projection is claimed.
    pub lm_head_w: Vec<f32>,
    pub tie_word_embeddings: bool,
}

impl Gemma4TextWeights {
    /// Build the MF.7 text-forward weight bundle from loader-owned tensor
    /// values. This intentionally consumes only the decoder-core subset; PLE,
    /// per-layer token embeddings, output scale, post norms, softcap, and
    /// sliding/shared KV semantics remain outside the executable subset.
    pub fn from_loaded_tensors(config: &Gemma4Config, tensors: Vec<LoadedTensor>) -> Result<Self> {
        validate_gemma4_text_config_for_model(config)?;

        let mut by_name = BTreeMap::new();
        for tensor in tensors {
            match by_name.entry(tensor.name.clone()) {
                Entry::Vacant(entry) => {
                    entry.insert(tensor);
                }
                Entry::Occupied(_) => {
                    return Err(invalid_at(
                        None,
                        &tensor.name,
                        "duplicate Gemma4 text tensor supplied",
                    ));
                }
            }
        }

        let h = config.embedding_length;
        let v = config.tokenizer_token_count;
        let head_dim = config.attention_key_length;
        let q_out = checked_dim_product(
            "gemma4.attention.head_count * gemma4.attention.key_length",
            config.attention_head_count,
            head_dim,
            None,
        )?;
        let kv_out = checked_dim_product(
            "gemma4.attention.head_count_kv * gemma4.attention.key_length",
            config.attention_head_count_kv,
            head_dim,
            None,
        )?;
        let f = config.feed_forward_length;

        let token_embd_gguf = take_text_tensor(
            &mut by_name,
            tensor_spec("token_embd.weight", &[h, v], Gemma4TensorKind::KQuantized),
        )?;
        let lm_head_w = token_embd_gguf.clone();
        let token_embd = transpose_2d(&token_embd_gguf, h, v);
        let output_norm_w = take_text_tensor(
            &mut by_name,
            tensor_spec("output_norm.weight", &[h], Gemma4TensorKind::F32),
        )?;

        let mut layers = Vec::with_capacity(config.block_count);
        for layer in 0..config.block_count {
            let prefix = format!("blk.{layer}");
            let name = |suffix: &str| format!("{prefix}.{suffix}");
            layers.push(Gemma4TextLayerWeights {
                attn_norm_w: take_text_tensor(
                    &mut by_name,
                    tensor_spec(&name("attn_norm.weight"), &[h], Gemma4TensorKind::F32),
                )?,
                attn_q_w: take_text_tensor(
                    &mut by_name,
                    tensor_spec(
                        &name("attn_q.weight"),
                        &[h, q_out],
                        Gemma4TensorKind::KQuantized,
                    ),
                )?,
                attn_k_w: take_text_tensor(
                    &mut by_name,
                    tensor_spec(
                        &name("attn_k.weight"),
                        &[h, kv_out],
                        Gemma4TensorKind::KQuantized,
                    ),
                )?,
                attn_v_w: take_text_tensor(
                    &mut by_name,
                    tensor_spec(
                        &name("attn_v.weight"),
                        &[h, kv_out],
                        Gemma4TensorKind::KQuantized,
                    ),
                )?,
                attn_o_w: take_text_tensor(
                    &mut by_name,
                    tensor_spec(
                        &name("attn_output.weight"),
                        &[q_out, h],
                        Gemma4TensorKind::KQuantized,
                    ),
                )?,
                attn_q_norm_w: take_text_tensor(
                    &mut by_name,
                    tensor_spec(
                        &name("attn_q_norm.weight"),
                        &[head_dim],
                        Gemma4TensorKind::F32,
                    ),
                )?,
                attn_k_norm_w: take_text_tensor(
                    &mut by_name,
                    tensor_spec(
                        &name("attn_k_norm.weight"),
                        &[head_dim],
                        Gemma4TensorKind::F32,
                    ),
                )?,
                ffn_norm_w: take_text_tensor(
                    &mut by_name,
                    tensor_spec(&name("ffn_norm.weight"), &[h], Gemma4TensorKind::F32),
                )?,
                ffn_gate_w: take_text_tensor(
                    &mut by_name,
                    tensor_spec(
                        &name("ffn_gate.weight"),
                        &[h, f],
                        Gemma4TensorKind::KQuantized,
                    ),
                )?,
                ffn_up_w: take_text_tensor(
                    &mut by_name,
                    tensor_spec(
                        &name("ffn_up.weight"),
                        &[h, f],
                        Gemma4TensorKind::KQuantized,
                    ),
                )?,
                ffn_down_w: take_text_tensor(
                    &mut by_name,
                    tensor_spec(
                        &name("ffn_down.weight"),
                        &[f, h],
                        Gemma4TensorKind::KQuantized,
                    ),
                )?,
            });
        }

        Ok(Self {
            token_embd,
            layers,
            output_norm_w,
            lm_head_w,
            tie_word_embeddings: true,
        })
    }
}

#[derive(Debug)]
pub struct Gemma4TextModel {
    config: Gemma4Config,
    weights: Gemma4TextWeights,
    kernels: Arc<dyn KernelBackend>,
}

impl Gemma4TextModel {
    pub fn new(config: Gemma4Config, weights: Gemma4TextWeights) -> Result<Self> {
        Self::with_kernel_backend(config, weights, default_kernel_backend())
    }

    pub fn with_kernel_backend(
        config: Gemma4Config,
        weights: Gemma4TextWeights,
        kernels: Arc<dyn KernelBackend>,
    ) -> Result<Self> {
        validate_gemma4_text_config_for_model(&config)?;
        validate_gemma4_text_weight_lengths(&config, &weights)?;
        Ok(Self {
            config,
            weights,
            kernels,
        })
    }

    pub fn config(&self) -> &Gemma4Config {
        &self.config
    }

    pub fn kernel_backend(&self) -> &dyn KernelBackend {
        self.kernels.as_ref()
    }

    pub fn execution_backend(&self) -> &dyn KernelBackend {
        self.kernels.as_ref()
    }

    pub fn prefill(&self, tokens: &[TokenId]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(OcelotlError::InvalidRequest(InvalidRequestError {
                field: "tokens".to_string(),
                message: "Gemma4TextModel::prefill requires at least one token".to_string(),
            }));
        }

        let cfg = &self.config;
        if tokens.len() > cfg.context_length {
            return Err(OcelotlError::InvalidRequest(InvalidRequestError {
                field: "tokens".to_string(),
                message: format!(
                    "prompt length {} exceeds context_length {}",
                    tokens.len(),
                    cfg.context_length,
                ),
            }));
        }
        for (idx, token) in tokens.iter().enumerate() {
            if (token.0 as usize) >= cfg.tokenizer_token_count {
                return Err(OcelotlError::InvalidRequest(InvalidRequestError {
                    field: "tokens".to_string(),
                    message: format!(
                        "token id {} at position {} is out of range for tokenizer_token_count {}",
                        token.0, idx, cfg.tokenizer_token_count,
                    ),
                }));
            }
        }

        let seq = tokens.len();
        let h = cfg.embedding_length;
        let q_heads = cfg.attention_head_count;
        let kv_heads = cfg.attention_head_count_kv;
        let head_dim = cfg.attention_key_length;
        let q_out = q_heads * head_dim;
        let kv_out = kv_heads * head_dim;
        let f = cfg.feed_forward_length;
        let vocab = cfg.tokenizer_token_count;
        let eps = cfg.rms_norm_eps;
        let theta = cfg.rope_freq_base;

        let mut hidden = vec![0.0_f32; seq * h];
        for (pos, token) in tokens.iter().enumerate() {
            let src = (token.0 as usize) * h;
            let dst = pos * h;
            hidden[dst..dst + h].copy_from_slice(&self.weights.token_embd[src..src + h]);
        }

        let mut norm_buf = vec![0.0_f32; seq * h];
        let mut q_buf = vec![0.0_f32; seq * q_out];
        let mut k_buf = vec![0.0_f32; seq * kv_out];
        let mut q_norm_buf = vec![0.0_f32; seq * q_out];
        let mut k_norm_buf = vec![0.0_f32; seq * kv_out];
        let mut v_buf = vec![0.0_f32; seq * kv_out];
        let mut attn_out = vec![0.0_f32; seq * q_out];
        let mut o_buf = vec![0.0_f32; seq * h];
        let mut residual_buf = vec![0.0_f32; seq * h];
        let mut gate_buf = vec![0.0_f32; seq * f];
        let mut up_buf = vec![0.0_f32; seq * f];
        let mut mlp_out = vec![0.0_f32; seq * h];

        for layer in &self.weights.layers {
            residual_buf.copy_from_slice(&hidden);

            self.kernels
                .rmsnorm(&hidden, seq, h, &layer.attn_norm_w, eps, &mut norm_buf)?;
            self.kernels
                .matmul(&norm_buf, (seq, h), &layer.attn_q_w, (h, q_out), &mut q_buf)?;
            self.kernels.matmul(
                &norm_buf,
                (seq, h),
                &layer.attn_k_w,
                (h, kv_out),
                &mut k_buf,
            )?;
            self.kernels.matmul(
                &norm_buf,
                (seq, h),
                &layer.attn_v_w,
                (h, kv_out),
                &mut v_buf,
            )?;

            self.kernels.rmsnorm(
                &q_buf,
                seq * q_heads,
                head_dim,
                &layer.attn_q_norm_w,
                eps,
                &mut q_norm_buf,
            )?;
            self.kernels.rmsnorm(
                &k_buf,
                seq * kv_heads,
                head_dim,
                &layer.attn_k_norm_w,
                eps,
                &mut k_norm_buf,
            )?;

            for pos in 0..seq {
                let q_start = pos * q_out;
                self.kernels.rope_apply_inplace(
                    &mut q_norm_buf[q_start..q_start + q_out],
                    head_dim,
                    pos,
                    theta,
                )?;
                let k_start = pos * kv_out;
                self.kernels.rope_apply_inplace(
                    &mut k_norm_buf[k_start..k_start + kv_out],
                    head_dim,
                    pos,
                    theta,
                )?;
            }

            self.kernels.scaled_dot_product_attention(
                &q_norm_buf,
                &k_norm_buf,
                &v_buf,
                seq,
                q_heads,
                kv_heads,
                head_dim,
                &mut attn_out,
            )?;

            self.kernels.matmul(
                &attn_out,
                (seq, q_out),
                &layer.attn_o_w,
                (q_out, h),
                &mut o_buf,
            )?;
            self.kernels.vec_add(&residual_buf, &o_buf, &mut hidden)?;

            residual_buf.copy_from_slice(&hidden);
            self.kernels
                .rmsnorm(&hidden, seq, h, &layer.ffn_norm_w, eps, &mut norm_buf)?;
            self.kernels.mlp_gated_silu(
                &norm_buf,
                seq,
                h,
                f,
                &layer.ffn_gate_w,
                &layer.ffn_up_w,
                &layer.ffn_down_w,
                &mut gate_buf,
                &mut up_buf,
                &mut mlp_out,
            )?;
            self.kernels.vec_add(&residual_buf, &mlp_out, &mut hidden)?;
        }

        self.kernels.rmsnorm(
            &hidden,
            seq,
            h,
            &self.weights.output_norm_w,
            eps,
            &mut norm_buf,
        )?;

        let last_start = (seq - 1) * h;
        let last_row = &norm_buf[last_start..last_start + h];
        let mut logits = vec![0.0_f32; vocab];
        self.kernels.matmul(
            last_row,
            (1, h),
            &self.weights.lm_head_w,
            (h, vocab),
            &mut logits,
        )?;

        Ok(logits)
    }
}

fn validate_gemma4_text_config_for_model(config: &Gemma4Config) -> Result<()> {
    config.ensure_supported_for_text_forward()?;

    validate_positive("gemma4.context_length", config.context_length)?;
    validate_positive("gemma4.block_count", config.block_count)?;
    validate_positive("gemma4.embedding_length", config.embedding_length)?;
    validate_positive(
        "gemma4.embedding_length_per_layer_input",
        config.embedding_length_per_layer_input,
    )?;
    validate_positive("gemma4.feed_forward_length", config.feed_forward_length)?;
    validate_positive("gemma4.attention.head_count", config.attention_head_count)?;
    validate_positive(
        "gemma4.attention.head_count_kv",
        config.attention_head_count_kv,
    )?;
    validate_positive("gemma4.attention.key_length", config.attention_key_length)?;
    validate_positive(
        "gemma4.attention.value_length",
        config.attention_value_length,
    )?;
    validate_positive(
        "gemma4.attention.key_length_swa",
        config.attention_key_length_swa,
    )?;
    validate_positive(
        "gemma4.attention.value_length_swa",
        config.attention_value_length_swa,
    )?;
    validate_positive("gemma4.rope.dimension_count", config.rope_dimension_count)?;
    validate_positive(
        "gemma4.rope.dimension_count_swa",
        config.rope_dimension_count_swa,
    )?;
    validate_positive("tokenizer.ggml.tokens", config.tokenizer_token_count)?;
    validate_finite_positive("gemma4.rope.freq_base", config.rope_freq_base)?;
    validate_finite_positive("gemma4.rope.freq_base_swa", config.rope_freq_base_swa)?;
    validate_finite_positive(
        "gemma4.attention.layer_norm_rms_epsilon",
        config.rms_norm_eps,
    )?;

    if config.attention_head_count % config.attention_head_count_kv != 0 {
        return Err(invalid(
            "gemma4.attention.head_count",
            &format!(
                "must be divisible by gemma4.attention.head_count_kv ({})",
                config.attention_head_count_kv
            ),
        ));
    }

    checked_dim_product(
        "gemma4.attention.head_count * gemma4.attention.key_length",
        config.attention_head_count,
        config.attention_key_length,
        None,
    )?;
    checked_dim_product(
        "gemma4.attention.head_count_kv * gemma4.attention.key_length",
        config.attention_head_count_kv,
        config.attention_key_length,
        None,
    )?;
    checked_dim_product(
        "gemma4.embedding_length * tokenizer.ggml.tokens",
        config.embedding_length,
        config.tokenizer_token_count,
        None,
    )?;
    Ok(())
}

fn validate_gemma4_text_weight_lengths(
    config: &Gemma4Config,
    weights: &Gemma4TextWeights,
) -> Result<()> {
    let h = config.embedding_length;
    let vocab = config.tokenizer_token_count;
    let head_dim = config.attention_key_length;
    let q_out = checked_dim_product(
        "gemma4.attention.head_count * gemma4.attention.key_length",
        config.attention_head_count,
        head_dim,
        None,
    )?;
    let kv_out = checked_dim_product(
        "gemma4.attention.head_count_kv * gemma4.attention.key_length",
        config.attention_head_count_kv,
        head_dim,
        None,
    )?;
    let f = config.feed_forward_length;

    check_text_len(
        "token_embd.weight",
        weights.token_embd.len(),
        checked_dim_product("token_embd.weight", vocab, h, None)?,
    )?;
    check_text_len("output_norm.weight", weights.output_norm_w.len(), h)?;
    check_text_len(
        "lm_head.weight",
        weights.lm_head_w.len(),
        checked_dim_product("lm_head.weight", h, vocab, None)?,
    )?;
    if weights.layers.len() != config.block_count {
        return Err(invalid(
            "layers",
            &format!(
                "expected {} Gemma4 text layer weight bundles, got {}",
                config.block_count,
                weights.layers.len()
            ),
        ));
    }

    for (idx, layer) in weights.layers.iter().enumerate() {
        let prefix = format!("layers[{idx}]");
        check_text_len(&format!("{prefix}.attn_norm_w"), layer.attn_norm_w.len(), h)?;
        check_text_len(
            &format!("{prefix}.attn_q_w"),
            layer.attn_q_w.len(),
            checked_dim_product("attn_q_w", h, q_out, None)?,
        )?;
        check_text_len(
            &format!("{prefix}.attn_k_w"),
            layer.attn_k_w.len(),
            checked_dim_product("attn_k_w", h, kv_out, None)?,
        )?;
        check_text_len(
            &format!("{prefix}.attn_v_w"),
            layer.attn_v_w.len(),
            checked_dim_product("attn_v_w", h, kv_out, None)?,
        )?;
        check_text_len(
            &format!("{prefix}.attn_o_w"),
            layer.attn_o_w.len(),
            checked_dim_product("attn_o_w", q_out, h, None)?,
        )?;
        check_text_len(
            &format!("{prefix}.attn_q_norm_w"),
            layer.attn_q_norm_w.len(),
            head_dim,
        )?;
        check_text_len(
            &format!("{prefix}.attn_k_norm_w"),
            layer.attn_k_norm_w.len(),
            head_dim,
        )?;
        check_text_len(&format!("{prefix}.ffn_norm_w"), layer.ffn_norm_w.len(), h)?;
        check_text_len(
            &format!("{prefix}.ffn_gate_w"),
            layer.ffn_gate_w.len(),
            checked_dim_product("ffn_gate_w", h, f, None)?,
        )?;
        check_text_len(
            &format!("{prefix}.ffn_up_w"),
            layer.ffn_up_w.len(),
            checked_dim_product("ffn_up_w", h, f, None)?,
        )?;
        check_text_len(
            &format!("{prefix}.ffn_down_w"),
            layer.ffn_down_w.len(),
            checked_dim_product("ffn_down_w", f, h, None)?,
        )?;
    }

    Ok(())
}

fn take_text_tensor(
    by_name: &mut BTreeMap<String, LoadedTensor>,
    spec: Gemma4TensorSpec,
) -> Result<Vec<f32>> {
    let tensor = by_name
        .remove(&spec.name)
        .ok_or_else(|| invalid_at(None, &spec.name, "required Gemma4 text tensor is missing"))?;
    check_loaded_dequantized_tensor(&tensor, &spec, None)?;
    Ok(tensor.values)
}

fn transpose_2d(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), rows * cols);
    let mut dst = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
}

fn check_text_len(field: &str, got: usize, expected: usize) -> Result<()> {
    if got == expected {
        Ok(())
    } else {
        Err(invalid(
            field,
            &format!("expected length {expected}, got {got}"),
        ))
    }
}

fn required_gemma4_tensor_specs(config: &Gemma4Config) -> Vec<Gemma4TensorSpec> {
    let per_layer_width = config
        .block_count
        .saturating_mul(config.embedding_length_per_layer_input);
    let mut specs =
        Vec::with_capacity(6usize.saturating_add(config.block_count.saturating_mul(17)));

    specs.push(tensor_spec(
        "token_embd.weight",
        &[config.embedding_length, config.tokenizer_token_count],
        Gemma4TensorKind::KQuantized,
    ));
    specs.push(tensor_spec(
        "output_norm.weight",
        &[config.embedding_length],
        Gemma4TensorKind::F32,
    ));
    specs.push(tensor_spec(
        "rope_freqs.weight",
        &[config.rope_dimension_count_swa],
        Gemma4TensorKind::F32,
    ));
    specs.push(tensor_spec(
        "per_layer_model_proj.weight",
        &[config.embedding_length, per_layer_width],
        Gemma4TensorKind::BF16,
    ));
    specs.push(tensor_spec(
        "per_layer_proj_norm.weight",
        &[config.embedding_length_per_layer_input],
        Gemma4TensorKind::F32,
    ));
    specs.push(tensor_spec(
        "per_layer_token_embd.weight",
        &[per_layer_width, config.tokenizer_token_count],
        Gemma4TensorKind::KQuantized,
    ));

    for layer in 0..config.block_count {
        let is_global_attention = gemma4_uses_global_attention(layer);
        let (key_length, value_length) = if is_global_attention {
            (config.attention_key_length, config.attention_value_length)
        } else {
            (
                config.attention_key_length_swa,
                config.attention_value_length_swa,
            )
        };
        let q_width = config.attention_head_count.saturating_mul(key_length);
        let k_width = config.attention_head_count_kv.saturating_mul(key_length);
        let v_width = config.attention_head_count_kv.saturating_mul(value_length);

        specs.push(tensor_spec(
            &format!("blk.{layer}.attn_norm.weight"),
            &[config.embedding_length],
            Gemma4TensorKind::F32,
        ));
        specs.push(tensor_spec(
            &format!("blk.{layer}.attn_q.weight"),
            &[config.embedding_length, q_width],
            Gemma4TensorKind::KQuantized,
        ));
        specs.push(tensor_spec(
            &format!("blk.{layer}.attn_k.weight"),
            &[config.embedding_length, k_width],
            Gemma4TensorKind::KQuantized,
        ));
        specs.push(tensor_spec(
            &format!("blk.{layer}.attn_v.weight"),
            &[config.embedding_length, v_width],
            Gemma4TensorKind::KQuantized,
        ));
        specs.push(tensor_spec(
            &format!("blk.{layer}.attn_output.weight"),
            &[q_width, config.embedding_length],
            Gemma4TensorKind::KQuantized,
        ));
        specs.push(tensor_spec(
            &format!("blk.{layer}.attn_q_norm.weight"),
            &[key_length],
            Gemma4TensorKind::F32,
        ));
        specs.push(tensor_spec(
            &format!("blk.{layer}.attn_k_norm.weight"),
            &[key_length],
            Gemma4TensorKind::F32,
        ));
        specs.push(tensor_spec(
            &format!("blk.{layer}.ffn_norm.weight"),
            &[config.embedding_length],
            Gemma4TensorKind::F32,
        ));
        specs.push(tensor_spec(
            &format!("blk.{layer}.ffn_gate.weight"),
            &[config.embedding_length, config.feed_forward_length],
            Gemma4TensorKind::KQuantized,
        ));
        specs.push(tensor_spec(
            &format!("blk.{layer}.ffn_up.weight"),
            &[config.embedding_length, config.feed_forward_length],
            Gemma4TensorKind::KQuantized,
        ));
        specs.push(tensor_spec(
            &format!("blk.{layer}.ffn_down.weight"),
            &[config.feed_forward_length, config.embedding_length],
            Gemma4TensorKind::KQuantized,
        ));
        specs.push(tensor_spec(
            &format!("blk.{layer}.inp_gate.weight"),
            &[
                config.embedding_length,
                config.embedding_length_per_layer_input,
            ],
            Gemma4TensorKind::KQuantized,
        ));
        specs.push(tensor_spec(
            &format!("blk.{layer}.proj.weight"),
            &[
                config.embedding_length_per_layer_input,
                config.embedding_length,
            ],
            Gemma4TensorKind::KQuantized,
        ));
        specs.push(tensor_spec(
            &format!("blk.{layer}.layer_output_scale.weight"),
            &[1],
            Gemma4TensorKind::F32,
        ));
        specs.push(tensor_spec(
            &format!("blk.{layer}.post_attention_norm.weight"),
            &[config.embedding_length],
            Gemma4TensorKind::F32,
        ));
        specs.push(tensor_spec(
            &format!("blk.{layer}.post_ffw_norm.weight"),
            &[config.embedding_length],
            Gemma4TensorKind::F32,
        ));
        specs.push(tensor_spec(
            &format!("blk.{layer}.post_norm.weight"),
            &[config.embedding_length],
            Gemma4TensorKind::F32,
        ));
    }

    specs
}

fn tensor_spec(name: &str, shape: &[usize], kind: Gemma4TensorKind) -> Gemma4TensorSpec {
    Gemma4TensorSpec {
        name: name.to_string(),
        shape: shape.to_vec(),
        kind,
    }
}

fn gemma4_uses_global_attention(layer: usize) -> bool {
    layer % 6 == 5
}

fn validate_gemma4_tensor_dimensions(config: &Gemma4Config, path: Option<&Path>) -> Result<()> {
    checked_dim_product(
        "gemma4.block_count * gemma4.embedding_length_per_layer_input",
        config.block_count,
        config.embedding_length_per_layer_input,
        path,
    )?;
    checked_dim_product(
        "gemma4.attention.head_count * gemma4.attention.key_length",
        config.attention_head_count,
        config.attention_key_length,
        path,
    )?;
    checked_dim_product(
        "gemma4.attention.head_count_kv * gemma4.attention.key_length",
        config.attention_head_count_kv,
        config.attention_key_length,
        path,
    )?;
    checked_dim_product(
        "gemma4.attention.head_count_kv * gemma4.attention.value_length",
        config.attention_head_count_kv,
        config.attention_value_length,
        path,
    )?;
    checked_dim_product(
        "gemma4.attention.head_count * gemma4.attention.key_length_swa",
        config.attention_head_count,
        config.attention_key_length_swa,
        path,
    )?;
    checked_dim_product(
        "gemma4.attention.head_count_kv * gemma4.attention.key_length_swa",
        config.attention_head_count_kv,
        config.attention_key_length_swa,
        path,
    )?;
    checked_dim_product(
        "gemma4.attention.head_count_kv * gemma4.attention.value_length_swa",
        config.attention_head_count_kv,
        config.attention_value_length_swa,
        path,
    )?;
    Ok(())
}

fn checked_dim_product(
    field: &str,
    left: usize,
    right: usize,
    path: Option<&Path>,
) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_at(path, field, &format!("{left} * {right} overflows usize")))
}

fn check_gguf_tensor(
    manifest: &GgufManifest,
    spec: &Gemma4TensorSpec,
    path: Option<&Path>,
) -> Result<()> {
    let tensor = manifest
        .tensors
        .iter()
        .find(|tensor| tensor.name == spec.name)
        .ok_or_else(|| {
            OcelotlError::from(InvalidModelError {
                path: path.map(|p| p.to_path_buf()),
                field: Some(spec.name.clone()),
                message: format!("tensor `{}` not found in GGUF header", spec.name),
            })
        })?;
    if tensor.shape != spec.shape {
        return Err(OcelotlError::from(InvalidModelError {
            path: path.map(|p| p.to_path_buf()),
            field: Some(spec.name.clone()),
            message: format!(
                "tensor `{}` has shape {:?}, expected {:?}",
                spec.name, tensor.shape, spec.shape,
            ),
        }));
    }
    if !tensor_kind_matches(tensor, spec.kind) {
        return Err(OcelotlError::from(InvalidModelError {
            path: path.map(|p| p.to_path_buf()),
            field: Some(spec.name.clone()),
            message: format!(
                "tensor `{}` has type {:?}, expected {}",
                spec.name,
                tensor.tensor_type,
                tensor_kind_label(spec.kind),
            ),
        }));
    }
    Ok(())
}

fn check_loaded_dense_tensor(
    tensor: &LoadedTensor,
    spec: &Gemma4TensorSpec,
    path: Option<&Path>,
) -> Result<()> {
    if tensor.shape != spec.shape {
        return Err(invalid_at(
            path,
            &spec.name,
            &format!(
                "tensor `{}` has shape {:?}, expected {:?}",
                spec.name, tensor.shape, spec.shape,
            ),
        ));
    }
    if !loaded_dense_kind_matches(tensor.dtype, spec.kind) {
        return Err(invalid_at(
            path,
            &spec.name,
            &format!(
                "tensor `{}` has dtype {:?}, expected {}",
                spec.name,
                tensor.dtype,
                tensor_kind_label(spec.kind),
            ),
        ));
    }

    let expected_len = checked_shape_len(&spec.name, &spec.shape, path)?;
    if tensor.values.len() != expected_len {
        return Err(invalid_at(
            path,
            &spec.name,
            &format!(
                "tensor `{}` has {} values, expected {expected_len}",
                spec.name,
                tensor.values.len(),
            ),
        ));
    }
    Ok(())
}

fn check_loaded_dequantized_tensor(
    tensor: &LoadedTensor,
    spec: &Gemma4TensorSpec,
    path: Option<&Path>,
) -> Result<()> {
    if tensor.shape != spec.shape {
        return Err(invalid_at(
            path,
            &spec.name,
            &format!(
                "tensor `{}` has shape {:?}, expected {:?}",
                spec.name, tensor.shape, spec.shape,
            ),
        ));
    }
    if !loaded_dequantized_kind_matches(tensor.dtype, spec.kind) {
        return Err(invalid_at(
            path,
            &spec.name,
            &format!(
                "tensor `{}` has dtype {:?}, expected {}",
                spec.name,
                tensor.dtype,
                tensor_dequantized_kind_label(spec.kind),
            ),
        ));
    }

    let expected_len = checked_shape_len(&spec.name, &spec.shape, path)?;
    if tensor.values.len() != expected_len {
        return Err(invalid_at(
            path,
            &spec.name,
            &format!(
                "tensor `{}` has {} values, expected {expected_len}",
                spec.name,
                tensor.values.len(),
            ),
        ));
    }
    Ok(())
}

fn checked_shape_len(field: &str, shape: &[usize], path: Option<&Path>) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| invalid_at(path, field, "tensor shape product overflows usize"))
    })
}

fn loaded_dense_kind_matches(dtype: SupportedDtype, expected: Gemma4TensorKind) -> bool {
    matches!(
        (dtype, expected),
        (SupportedDtype::F32, Gemma4TensorKind::F32)
            | (SupportedDtype::BF16, Gemma4TensorKind::BF16)
    )
}

fn loaded_dequantized_kind_matches(dtype: SupportedDtype, expected: Gemma4TensorKind) -> bool {
    matches!(
        (dtype, expected),
        (SupportedDtype::F32, Gemma4TensorKind::F32)
            | (SupportedDtype::BF16, Gemma4TensorKind::BF16)
            | (SupportedDtype::F32, Gemma4TensorKind::KQuantized)
    )
}

fn tensor_kind_matches(tensor: &GgufTensorEntry, expected: Gemma4TensorKind) -> bool {
    match expected {
        Gemma4TensorKind::F32 => tensor.tensor_type == GgmlTensorType::F32,
        Gemma4TensorKind::BF16 => tensor.tensor_type == GgmlTensorType::BF16,
        Gemma4TensorKind::KQuantized => matches!(
            tensor.tensor_type,
            GgmlTensorType::Q4K | GgmlTensorType::Q5K | GgmlTensorType::Q6K
        ),
    }
}

fn tensor_kind_label(kind: Gemma4TensorKind) -> &'static str {
    match kind {
        Gemma4TensorKind::F32 => "F32",
        Gemma4TensorKind::BF16 => "BF16",
        Gemma4TensorKind::KQuantized => "one of Q4K, Q5K, Q6K",
    }
}

fn tensor_dequantized_kind_label(kind: Gemma4TensorKind) -> &'static str {
    match kind {
        Gemma4TensorKind::F32 => "F32",
        Gemma4TensorKind::BF16 => "BF16",
        Gemma4TensorKind::KQuantized => "F32 dequantized from one of Q4K, Q5K, Q6K",
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

fn optional_array_len(
    manifest: &GgufManifest,
    key: &str,
    expected_element_type: GgufMetadataType,
) -> Result<Option<usize>> {
    match manifest.metadata_value(key) {
        None => Ok(None),
        Some(GgufMetadataValue::Array { element_type, len })
            if *element_type == expected_element_type =>
        {
            (*len)
                .try_into()
                .map(Some)
                .map_err(|_| invalid(key, &format!("array length {len} does not fit in usize")))
        }
        Some(GgufMetadataValue::Array { element_type, .. }) => Err(invalid(
            key,
            &format!(
                "must be a GGUF {expected_element_type:?} array metadata value, got {element_type:?} array",
            ),
        )),
        Some(other) => Err(invalid(
            key,
            &format!("must be a GGUF array metadata value, got {other:?}"),
        )),
    }
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
    invalid_at(None, field, message)
}

fn invalid_at(path: Option<&Path>, field: &str, message: &str) -> OcelotlError {
    OcelotlError::from(InvalidModelError {
        path: path.map(|p| p.to_path_buf()),
        field: Some(field.to_string()),
        message: message.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ocelotl_loader::{GgufMetadataEntry, GgufTensorEntry};
    use serde::Deserialize;
    use std::path::{Path, PathBuf};

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
                            let element_type = match element_type.as_str() {
                                "bool" => GgufMetadataType::Bool,
                                "string" => GgufMetadataType::String,
                                other => panic!(
                                    "unexpected metadata array element type in fixture: {other}"
                                ),
                            };
                            GgufMetadataValue::Array { element_type, len }
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
                        "bf16" => GgmlTensorType::BF16,
                        "q4_k" => GgmlTensorType::Q4K,
                        "q5_k" => GgmlTensorType::Q5K,
                        "q6_k" => GgmlTensorType::Q6K,
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

    fn fixture_config() -> Gemma4Config {
        Gemma4Config::try_from(&fixture_manifest()).expect("Gemma4 GGUF fixture must convert")
    }

    fn tiny_config() -> Gemma4Config {
        Gemma4Config {
            context_length: 16,
            block_count: 1,
            embedding_length: 16,
            embedding_length_per_layer_input: 16,
            feed_forward_length: 16,
            attention_head_count: 2,
            attention_head_count_kv: 1,
            attention_key_length: 16,
            attention_value_length: 16,
            attention_key_length_swa: 16,
            attention_value_length_swa: 16,
            rope_dimension_count: 16,
            rope_dimension_count_swa: 16,
            rope_freq_base: 1_000_000.0,
            rope_freq_base_swa: 10_000.0,
            rms_norm_eps: 1e-6,
            attention_sliding_window: Some(4),
            attention_shared_kv_layers: Some(1),
            attention_sliding_window_pattern_len: Some(1),
            final_logit_softcap: Some(30.0),
            tokenizer_model: Some("gemma4".to_string()),
            tokenizer_token_count: 16,
            quantization: Gemma4Quantization::Q4KM,
            has_quantized_tensors: true,
            tensor_count: 23,
            multimodal: true,
        }
    }

    fn tiny_text_config() -> Gemma4Config {
        Gemma4Config {
            context_length: 16,
            block_count: 1,
            embedding_length: 4,
            embedding_length_per_layer_input: 4,
            feed_forward_length: 8,
            attention_head_count: 2,
            attention_head_count_kv: 1,
            attention_key_length: 2,
            attention_value_length: 2,
            attention_key_length_swa: 2,
            attention_value_length_swa: 2,
            rope_dimension_count: 2,
            rope_dimension_count_swa: 2,
            rope_freq_base: 10_000.0,
            rope_freq_base_swa: 10_000.0,
            rms_norm_eps: 1e-6,
            attention_sliding_window: None,
            attention_shared_kv_layers: None,
            attention_sliding_window_pattern_len: None,
            final_logit_softcap: None,
            tokenizer_model: Some("gemma4".to_string()),
            tokenizer_token_count: 8,
            quantization: Gemma4Quantization::Unquantized,
            has_quantized_tensors: false,
            tensor_count: 13,
            multimodal: false,
        }
    }

    fn complete_inventory_manifest(config: &Gemma4Config) -> GgufManifest {
        let mut manifest = fixture_manifest();
        manifest.tensors = required_gemma4_tensor_specs(config)
            .into_iter()
            .enumerate()
            .map(|(index, spec)| GgufTensorEntry {
                name: spec.name,
                shape: spec.shape,
                tensor_type: tensor_type_for_spec(spec.kind, index),
                offset: 0,
                file_offset: 0,
                byte_len: None,
            })
            .collect();
        manifest
    }

    fn tensor_type_for_spec(kind: Gemma4TensorKind, index: usize) -> GgmlTensorType {
        match kind {
            Gemma4TensorKind::F32 => GgmlTensorType::F32,
            Gemma4TensorKind::BF16 => GgmlTensorType::BF16,
            Gemma4TensorKind::KQuantized => match index % 3 {
                0 => GgmlTensorType::Q4K,
                1 => GgmlTensorType::Q5K,
                _ => GgmlTensorType::Q6K,
            },
        }
    }

    fn complete_dense_loaded_tensors(config: &Gemma4Config) -> Vec<LoadedTensor> {
        required_gemma4_tensor_specs(config)
            .into_iter()
            .filter(|spec| spec.kind.is_dense())
            .map(|spec| loaded_tensor_for_dense_spec(&spec))
            .collect()
    }

    fn complete_dequantized_loaded_tensors(config: &Gemma4Config) -> Vec<LoadedTensor> {
        required_gemma4_tensor_specs(config)
            .into_iter()
            .map(|spec| loaded_tensor_for_dequantized_spec(&spec))
            .collect()
    }

    fn complete_text_loaded_tensors(config: &Gemma4Config) -> Vec<LoadedTensor> {
        let h = config.embedding_length;
        let vocab = config.tokenizer_token_count;
        let head_dim = config.attention_key_length;
        let q_out = config.attention_head_count * head_dim;
        let kv_out = config.attention_head_count_kv * head_dim;
        let f = config.feed_forward_length;
        let mut tensors = vec![
            text_loaded_tensor("token_embd.weight", &[h, vocab], 0.01),
            text_loaded_tensor("output_norm.weight", &[h], 1.0),
        ];
        for layer in 0..config.block_count {
            let prefix = format!("blk.{layer}");
            let name = |suffix: &str| format!("{prefix}.{suffix}");
            tensors.extend([
                text_loaded_tensor(&name("attn_norm.weight"), &[h], 1.0),
                text_loaded_tensor(&name("attn_q.weight"), &[h, q_out], 0.02),
                text_loaded_tensor(&name("attn_k.weight"), &[h, kv_out], 0.03),
                text_loaded_tensor(&name("attn_v.weight"), &[h, kv_out], 0.04),
                text_loaded_tensor(&name("attn_output.weight"), &[q_out, h], 0.05),
                text_loaded_tensor(&name("attn_q_norm.weight"), &[head_dim], 1.0),
                text_loaded_tensor(&name("attn_k_norm.weight"), &[head_dim], 1.0),
                text_loaded_tensor(&name("ffn_norm.weight"), &[h], 1.0),
                text_loaded_tensor(&name("ffn_gate.weight"), &[h, f], 0.06),
                text_loaded_tensor(&name("ffn_up.weight"), &[h, f], 0.07),
                text_loaded_tensor(&name("ffn_down.weight"), &[f, h], 0.08),
            ]);
        }
        tensors
    }

    fn loaded_tensor_for_dense_spec(spec: &Gemma4TensorSpec) -> LoadedTensor {
        let len = checked_shape_len(&spec.name, &spec.shape, None).unwrap();
        LoadedTensor {
            name: spec.name.clone(),
            shape: spec.shape.clone(),
            dtype: match spec.kind {
                Gemma4TensorKind::F32 => SupportedDtype::F32,
                Gemma4TensorKind::BF16 => SupportedDtype::BF16,
                Gemma4TensorKind::KQuantized => panic!("dense helper received quantized spec"),
            },
            values: vec![1.0; len],
        }
    }

    fn text_loaded_tensor(name: &str, shape: &[usize], scale: f32) -> LoadedTensor {
        let len = checked_shape_len(name, shape, None).unwrap();
        LoadedTensor {
            name: name.to_string(),
            shape: shape.to_vec(),
            dtype: SupportedDtype::F32,
            values: (0..len).map(|idx| scale + idx as f32 * 0.01).collect(),
        }
    }

    fn loaded_tensor_for_dequantized_spec(spec: &Gemma4TensorSpec) -> LoadedTensor {
        let len = checked_shape_len(&spec.name, &spec.shape, None).unwrap();
        LoadedTensor {
            name: spec.name.clone(),
            shape: spec.shape.clone(),
            dtype: match spec.kind {
                Gemma4TensorKind::F32 | Gemma4TensorKind::KQuantized => SupportedDtype::F32,
                Gemma4TensorKind::BF16 => SupportedDtype::BF16,
            },
            values: vec![
                match spec.kind {
                    Gemma4TensorKind::KQuantized => 0.0,
                    _ => 1.0,
                };
                len
            ],
        }
    }

    fn tmp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "ocelotl_gemma4_{}_{}.gguf",
            std::process::id(),
            name
        ));
        path
    }

    fn write_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn write_u64(out: &mut Vec<u8>, value: u64) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn write_f32(out: &mut Vec<u8>, value: f32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn write_string(out: &mut Vec<u8>, value: &str) {
        write_u64(out, value.len() as u64);
        out.extend_from_slice(value.as_bytes());
    }

    fn write_string_metadata(out: &mut Vec<u8>, key: &str, value: &str) {
        write_string(out, key);
        write_u32(out, 8);
        write_string(out, value);
    }

    fn write_u32_metadata(out: &mut Vec<u8>, key: &str, value: u32) {
        write_string(out, key);
        write_u32(out, 4);
        write_u32(out, value);
    }

    fn write_f32_metadata(out: &mut Vec<u8>, key: &str, value: f32) {
        write_string(out, key);
        write_u32(out, 6);
        write_f32(out, value);
    }

    fn write_bool_array_metadata(out: &mut Vec<u8>, key: &str, len: usize) {
        write_string(out, key);
        write_u32(out, 9);
        write_u32(out, 7);
        write_u64(out, len as u64);
        out.extend(std::iter::repeat_n(1u8, len));
    }

    fn write_string_array_metadata(out: &mut Vec<u8>, key: &str, len: usize) {
        write_string(out, key);
        write_u32(out, 9);
        write_u32(out, 8);
        write_u64(out, len as u64);
        for token in 0..len {
            write_string(out, &format!("<tok{token}>"));
        }
    }

    fn align_len(len: usize) -> usize {
        len.next_multiple_of(32)
    }

    fn dense_payload(spec: &Gemma4TensorSpec) -> Vec<u8> {
        let len = checked_shape_len(&spec.name, &spec.shape, None).unwrap();
        match spec.kind {
            Gemma4TensorKind::F32 => (0..len).flat_map(|_| 1.0f32.to_le_bytes()).collect(),
            Gemma4TensorKind::BF16 => (0..len).flat_map(|_| 0x3f80u16.to_le_bytes()).collect(),
            Gemma4TensorKind::KQuantized => {
                assert_eq!(
                    len % 256,
                    0,
                    "tiny Gemma4 GGUF fixture K-quant tensor `{}` must use whole Q4_K blocks",
                    spec.name
                );
                vec![0; (len / 256) * 144]
            }
        }
    }

    fn write_tiny_gemma4_gguf(path: &Path, config: &Gemma4Config) {
        let specs = required_gemma4_tensor_specs(config);
        let payloads: Vec<Vec<u8>> = specs.iter().map(dense_payload).collect();
        let mut offsets = Vec::with_capacity(payloads.len());
        let mut next_offset = 0usize;
        for payload in &payloads {
            next_offset = align_len(next_offset);
            offsets.push(next_offset);
            next_offset += payload.len();
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, specs.len() as u64);
        write_u64(&mut bytes, 24);

        write_string_metadata(&mut bytes, "general.architecture", "gemma4");
        write_u32_metadata(&mut bytes, "general.file_type", 15);
        write_u32_metadata(&mut bytes, "gemma4.block_count", config.block_count as u32);
        write_u32_metadata(
            &mut bytes,
            "gemma4.context_length",
            config.context_length as u32,
        );
        write_u32_metadata(
            &mut bytes,
            "gemma4.embedding_length",
            config.embedding_length as u32,
        );
        write_u32_metadata(
            &mut bytes,
            "gemma4.embedding_length_per_layer_input",
            config.embedding_length_per_layer_input as u32,
        );
        write_u32_metadata(
            &mut bytes,
            "gemma4.feed_forward_length",
            config.feed_forward_length as u32,
        );
        write_u32_metadata(
            &mut bytes,
            "gemma4.attention.head_count",
            config.attention_head_count as u32,
        );
        write_u32_metadata(
            &mut bytes,
            "gemma4.attention.head_count_kv",
            config.attention_head_count_kv as u32,
        );
        write_u32_metadata(
            &mut bytes,
            "gemma4.attention.key_length",
            config.attention_key_length as u32,
        );
        write_u32_metadata(
            &mut bytes,
            "gemma4.attention.value_length",
            config.attention_value_length as u32,
        );
        write_u32_metadata(
            &mut bytes,
            "gemma4.attention.key_length_swa",
            config.attention_key_length_swa as u32,
        );
        write_u32_metadata(
            &mut bytes,
            "gemma4.attention.value_length_swa",
            config.attention_value_length_swa as u32,
        );
        write_u32_metadata(
            &mut bytes,
            "gemma4.rope.dimension_count",
            config.rope_dimension_count as u32,
        );
        write_u32_metadata(
            &mut bytes,
            "gemma4.rope.dimension_count_swa",
            config.rope_dimension_count_swa as u32,
        );
        write_f32_metadata(&mut bytes, "gemma4.rope.freq_base", config.rope_freq_base);
        write_f32_metadata(
            &mut bytes,
            "gemma4.rope.freq_base_swa",
            config.rope_freq_base_swa,
        );
        write_f32_metadata(
            &mut bytes,
            "gemma4.attention.layer_norm_rms_epsilon",
            config.rms_norm_eps,
        );
        write_u32_metadata(
            &mut bytes,
            "gemma4.attention.sliding_window",
            config.attention_sliding_window.unwrap() as u32,
        );
        write_u32_metadata(
            &mut bytes,
            "gemma4.attention.shared_kv_layers",
            config.attention_shared_kv_layers.unwrap() as u32,
        );
        write_bool_array_metadata(
            &mut bytes,
            "gemma4.attention.sliding_window_pattern",
            config.attention_sliding_window_pattern_len.unwrap(),
        );
        write_f32_metadata(
            &mut bytes,
            "gemma4.final_logit_softcapping",
            config.final_logit_softcap.unwrap(),
        );
        write_string_metadata(&mut bytes, "tokenizer.ggml.model", "gemma4");
        write_string_array_metadata(
            &mut bytes,
            "tokenizer.ggml.tokens",
            config.tokenizer_token_count,
        );

        for (spec, offset) in specs.iter().zip(offsets.iter()) {
            write_string(&mut bytes, &spec.name);
            write_u32(&mut bytes, spec.shape.len() as u32);
            for dim in &spec.shape {
                write_u64(&mut bytes, *dim as u64);
            }
            write_u32(
                &mut bytes,
                match spec.kind {
                    Gemma4TensorKind::F32 => 0,
                    Gemma4TensorKind::BF16 => 30,
                    Gemma4TensorKind::KQuantized => 12,
                },
            );
            write_u64(&mut bytes, *offset as u64);
        }

        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        let data_start = bytes.len();
        bytes.resize(data_start + next_offset, 0);
        for (offset, payload) in offsets.iter().zip(payloads.iter()) {
            let start = data_start + offset;
            bytes[start..start + payload.len()].copy_from_slice(payload);
        }

        std::fs::write(path, bytes).expect("write tiny Gemma4 GGUF fixture");
    }

    #[test]
    fn try_from_gguf_manifest_accepts_gemma4_fixture_and_preserves_features() {
        let manifest = fixture_manifest();

        let cfg = Gemma4Config::try_from(&manifest).expect("Gemma4 GGUF fixture must convert");

        assert_eq!(cfg.context_length, 131_072);
        assert_eq!(cfg.block_count, 42);
        assert_eq!(cfg.embedding_length, 2_560);
        assert_eq!(cfg.embedding_length_per_layer_input, 256);
        assert_eq!(cfg.feed_forward_length, 10_240);
        assert_eq!(cfg.attention_head_count, 8);
        assert_eq!(cfg.attention_head_count_kv, 2);
        assert_eq!(cfg.attention_key_length, 512);
        assert_eq!(cfg.attention_value_length, 512);
        assert_eq!(cfg.attention_key_length_swa, 256);
        assert_eq!(cfg.attention_value_length_swa, 256);
        assert_eq!(cfg.rope_dimension_count, 512);
        assert_eq!(cfg.rope_dimension_count_swa, 256);
        assert_eq!(cfg.rope_freq_base, 1_000_000.0);
        assert_eq!(cfg.rope_freq_base_swa, 10_000.0);
        assert_eq!(cfg.attention_sliding_window, Some(512));
        assert_eq!(cfg.attention_shared_kv_layers, Some(18));
        assert_eq!(cfg.attention_sliding_window_pattern_len, Some(42));
        assert_eq!(cfg.final_logit_softcap, Some(30.0));
        assert_eq!(cfg.tokenizer_model.as_deref(), Some("gemma4"));
        assert_eq!(cfg.tokenizer_token_count, 262_144);
        assert_eq!(cfg.quantization, Gemma4Quantization::Q4KM);
        assert!(cfg.has_quantized_tensors);
        assert_eq!(cfg.tensor_count, 2);
        assert!(cfg.multimodal);
    }

    #[test]
    fn required_gemma4_tensor_names_enumerates_selected_q4_k_m_inventory() {
        let cfg = fixture_config();

        let names = required_gemma4_tensor_names(&cfg);

        assert_eq!(names.len(), 720);
        assert_eq!(names[0], "token_embd.weight");
        assert_eq!(names[1], "output_norm.weight");
        assert_eq!(names[2], "rope_freqs.weight");
        assert_eq!(names[3], "per_layer_model_proj.weight");
        assert_eq!(names[4], "per_layer_proj_norm.weight");
        assert_eq!(names[5], "per_layer_token_embd.weight");
        assert_eq!(names[7], "blk.0.attn_q.weight");
        assert!(names.contains(&"blk.5.attn_q.weight".to_string()));
        assert!(names.contains(&"blk.41.proj.weight".to_string()));
        assert_eq!(names[719], "blk.41.post_norm.weight");
    }

    #[test]
    fn required_gemma4_tensor_specs_model_swa_and_global_attention_widths() {
        let cfg = fixture_config();
        let specs = required_gemma4_tensor_specs(&cfg);

        let swa_q = specs
            .iter()
            .find(|spec| spec.name == "blk.0.attn_q.weight")
            .unwrap();
        let global_q = specs
            .iter()
            .find(|spec| spec.name == "blk.5.attn_q.weight")
            .unwrap();
        let swa_k_norm = specs
            .iter()
            .find(|spec| spec.name == "blk.0.attn_k_norm.weight")
            .unwrap();
        let global_k_norm = specs
            .iter()
            .find(|spec| spec.name == "blk.5.attn_k_norm.weight")
            .unwrap();

        assert_eq!(swa_q.shape, vec![2_560, 2_048]);
        assert_eq!(global_q.shape, vec![2_560, 4_096]);
        assert_eq!(swa_k_norm.shape, vec![256]);
        assert_eq!(global_k_norm.shape, vec![512]);
    }

    #[test]
    fn validate_gemma4_tensor_inventory_accepts_complete_synthetic_header() {
        let cfg = fixture_config();
        let manifest = complete_inventory_manifest(&cfg);

        validate_gemma4_tensor_inventory(&manifest, &cfg, None)
            .expect("complete synthetic Gemma4 inventory must validate");

        assert_eq!(manifest.tensors.len(), 720);
    }

    #[test]
    fn validate_gemma4_tensor_inventory_rejects_missing_tensor_with_field_name() {
        let cfg = fixture_config();
        let mut manifest = complete_inventory_manifest(&cfg);
        manifest
            .tensors
            .retain(|tensor| tensor.name != "blk.0.attn_q.weight");

        let err = validate_gemma4_tensor_inventory(&manifest, &cfg, None)
            .expect_err("missing tensor must fail inventory validation");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("blk.0.attn_q.weight"));
                assert!(invalid.message.contains("not found"));
            }
            other => panic!("expected InvalidModel for missing tensor, got {other:?}"),
        }
    }

    #[test]
    fn validate_gemma4_tensor_inventory_rejects_wrong_shape_with_field_name() {
        let cfg = fixture_config();
        let mut manifest = complete_inventory_manifest(&cfg);
        let tensor = manifest
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == "blk.5.attn_q.weight")
            .unwrap();
        tensor.shape = vec![
            cfg.embedding_length,
            cfg.attention_head_count * cfg.attention_key_length_swa,
        ];

        let err = validate_gemma4_tensor_inventory(&manifest, &cfg, None)
            .expect_err("wrong tensor shape must fail inventory validation");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("blk.5.attn_q.weight"));
                assert!(invalid.message.contains("expected"));
            }
            other => panic!("expected InvalidModel for wrong tensor shape, got {other:?}"),
        }
    }

    #[test]
    fn validate_gemma4_tensor_inventory_rejects_wrong_tensor_type_with_field_name() {
        let cfg = fixture_config();
        let mut manifest = complete_inventory_manifest(&cfg);
        let tensor = manifest
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == "token_embd.weight")
            .unwrap();
        tensor.tensor_type = GgmlTensorType::F32;

        let err = validate_gemma4_tensor_inventory(&manifest, &cfg, None)
            .expect_err("wrong tensor type must fail inventory validation");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("token_embd.weight"));
                assert!(invalid.message.contains("expected one of Q4K, Q5K, Q6K"));
            }
            other => panic!("expected InvalidModel for wrong tensor type, got {other:?}"),
        }
    }

    #[test]
    fn validate_gemma4_tensors_rejects_quantized_without_dequant_policy() {
        let cfg = fixture_config();
        let manifest = complete_inventory_manifest(&cfg);

        let err = validate_gemma4_tensors(&manifest, &cfg, None)
            .expect_err("Gemma4 quantized tensors must stay execution-rejected");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(unsupported.feature, "gemma4.quantized_tensors");
                assert_eq!(unsupported.requested.as_deref(), Some("q4_k_m"));
            }
            other => panic!("expected Unsupported for quantized tensors, got {other:?}"),
        }
    }

    #[test]
    fn required_gemma4_dense_tensor_names_enumerates_loadable_subset() {
        let cfg = fixture_config();

        let names = required_gemma4_dense_tensor_names(&cfg);

        assert_eq!(names.len(), 340);
        assert!(names.contains(&"output_norm.weight".to_string()));
        assert!(names.contains(&"per_layer_model_proj.weight".to_string()));
        assert!(names.contains(&"blk.0.attn_q_norm.weight".to_string()));
        assert!(names.contains(&"blk.41.post_norm.weight".to_string()));
        assert!(!names.contains(&"token_embd.weight".to_string()));
        assert!(!names.contains(&"blk.0.attn_q.weight".to_string()));
    }

    #[test]
    fn validate_gemma4_dense_tensors_accepts_complete_synthetic_values() {
        let cfg = tiny_config();
        let tensors = complete_dense_loaded_tensors(&cfg);

        validate_gemma4_dense_tensors(&cfg, &tensors, None)
            .expect("complete dense Gemma4 tensors must validate");
    }

    #[test]
    fn validate_gemma4_dequantized_tensors_accepts_complete_synthetic_values() {
        let cfg = tiny_config();
        let tensors = complete_dequantized_loaded_tensors(&cfg);

        validate_gemma4_dequantized_tensors(&cfg, &tensors, None)
            .expect("complete dequantized Gemma4 tensors must validate");
    }

    #[test]
    fn gemma4_text_weights_from_loaded_tensors_accepts_tiny_synthetic_subset() {
        let cfg = tiny_text_config();
        let tensors = complete_text_loaded_tensors(&cfg);
        let source_embedding = tensors
            .iter()
            .find(|tensor| tensor.name == "token_embd.weight")
            .unwrap()
            .values
            .clone();

        let weights = Gemma4TextWeights::from_loaded_tensors(&cfg, tensors)
            .expect("tiny Gemma4 text tensors must map into weights");

        assert_eq!(weights.layers.len(), cfg.block_count);
        assert_eq!(
            weights.token_embd.len(),
            cfg.tokenizer_token_count * cfg.embedding_length
        );
        assert_eq!(
            weights.lm_head_w.len(),
            cfg.embedding_length * cfg.tokenizer_token_count
        );
        assert_eq!(weights.lm_head_w, source_embedding);
        assert_eq!(
            &weights.token_embd[0..cfg.embedding_length],
            &[
                source_embedding[0],
                source_embedding[8],
                source_embedding[16],
                source_embedding[24]
            ]
        );
    }

    #[test]
    fn gemma4_text_weights_from_loaded_tensors_rejects_missing_tensor_with_field_name() {
        let cfg = tiny_text_config();
        let mut tensors = complete_text_loaded_tensors(&cfg);
        tensors.retain(|tensor| tensor.name != "blk.0.attn_q.weight");

        let err = Gemma4TextWeights::from_loaded_tensors(&cfg, tensors)
            .expect_err("missing Gemma4 text tensor must fail");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("blk.0.attn_q.weight"));
                assert!(invalid.message.contains("missing"));
            }
            other => panic!("expected InvalidModel for missing text tensor, got {other:?}"),
        }
    }

    #[test]
    fn gemma4_text_model_new_rejects_real_q4_k_m_execution_features() {
        let cfg = fixture_config();
        let weights = Gemma4TextWeights {
            token_embd: Vec::new(),
            layers: Vec::new(),
            output_norm_w: Vec::new(),
            lm_head_w: Vec::new(),
            tie_word_embeddings: true,
        };

        let err = Gemma4TextModel::new(cfg, weights)
            .expect_err("real Gemma4 Q4_K_M features must stay text-forward rejected");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(unsupported.feature, "gemma4.text_forward_features");
                let requested = unsupported.requested.unwrap();
                assert!(requested.contains("multimodal"));
                assert!(requested.contains("sliding_window_attention"));
                assert!(requested.contains("shared_kv_layers"));
                assert!(requested.contains("final_logit_softcap"));
                assert!(requested.contains("quantized_tensors"));
                assert!(requested.contains("quantization=q4_k_m"));
            }
            other => panic!("expected Unsupported for real Gemma4 text forward, got {other:?}"),
        }
    }

    #[test]
    fn gemma4_text_prefill_returns_one_finite_logit_per_token() {
        let cfg = tiny_text_config();
        let tensors = complete_text_loaded_tensors(&cfg);
        let weights = Gemma4TextWeights::from_loaded_tensors(&cfg, tensors)
            .expect("tiny Gemma4 text tensors must map into weights");
        let model = Gemma4TextModel::new(cfg.clone(), weights)
            .expect("tiny Gemma4 text model must construct");

        let logits = model
            .prefill(&[TokenId(1), TokenId(2)])
            .expect("tiny Gemma4 text prefill must succeed");

        assert_eq!(logits.len(), cfg.tokenizer_token_count);
        assert!(logits.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn validate_gemma4_dequantized_tensors_rejects_quantized_tensor_not_dequantized_to_f32() {
        let cfg = tiny_config();
        let mut tensors = complete_dequantized_loaded_tensors(&cfg);
        let tensor = tensors
            .iter_mut()
            .find(|tensor| tensor.name == "token_embd.weight")
            .unwrap();
        tensor.dtype = SupportedDtype::BF16;

        let err = validate_gemma4_dequantized_tensors(&cfg, &tensors, None)
            .expect_err("quantized Gemma4 tensor must be dequantized to F32");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("token_embd.weight"));
                assert!(invalid.message.contains("F32 dequantized"));
            }
            other => panic!("expected InvalidModel for wrong dequant dtype, got {other:?}"),
        }
    }

    #[test]
    fn validate_gemma4_dense_tensors_rejects_missing_tensor_with_field_name() {
        let cfg = tiny_config();
        let mut tensors = complete_dense_loaded_tensors(&cfg);
        tensors.retain(|tensor| tensor.name != "output_norm.weight");

        let err = validate_gemma4_dense_tensors(&cfg, &tensors, None)
            .expect_err("missing dense tensor must fail");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("output_norm.weight"));
                assert!(invalid.message.contains("missing"));
            }
            other => panic!("expected InvalidModel for missing dense tensor, got {other:?}"),
        }
    }

    #[test]
    fn validate_gemma4_dense_tensors_rejects_wrong_dtype_with_field_name() {
        let cfg = tiny_config();
        let mut tensors = complete_dense_loaded_tensors(&cfg);
        let tensor = tensors
            .iter_mut()
            .find(|tensor| tensor.name == "output_norm.weight")
            .unwrap();
        tensor.dtype = SupportedDtype::BF16;

        let err = validate_gemma4_dense_tensors(&cfg, &tensors, None)
            .expect_err("wrong dense dtype must fail");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("output_norm.weight"));
                assert!(invalid.message.contains("expected F32"));
            }
            other => panic!("expected InvalidModel for wrong dense dtype, got {other:?}"),
        }
    }

    #[test]
    fn validate_gemma4_dense_tensors_rejects_wrong_value_count() {
        let cfg = tiny_config();
        let mut tensors = complete_dense_loaded_tensors(&cfg);
        let tensor = tensors
            .iter_mut()
            .find(|tensor| tensor.name == "output_norm.weight")
            .unwrap();
        tensor.values.pop();

        let err = validate_gemma4_dense_tensors(&cfg, &tensors, None)
            .expect_err("wrong dense value count must fail");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("output_norm.weight"));
                assert!(invalid.message.contains("expected"));
            }
            other => panic!("expected InvalidModel for wrong dense value count, got {other:?}"),
        }
    }

    #[test]
    fn load_gemma4_dense_tensors_from_gguf_loads_tiny_synthetic_dense_subset() {
        let cfg = tiny_config();
        let path = tmp_path("dense_values");
        write_tiny_gemma4_gguf(&path, &cfg);

        let (loaded_cfg, tensors) = load_gemma4_dense_tensors_from_gguf(&path)
            .expect("tiny Gemma4 GGUF dense subset must load");

        assert_eq!(loaded_cfg.embedding_length, cfg.embedding_length);
        assert_eq!(
            tensors.len(),
            required_gemma4_dense_tensor_names(&cfg).len()
        );
        let output_norm = tensors
            .iter()
            .find(|tensor| tensor.name == "output_norm.weight")
            .unwrap();
        assert_eq!(output_norm.dtype, SupportedDtype::F32);
        assert_eq!(output_norm.values, vec![1.0; cfg.embedding_length]);
        let per_layer_model_proj = tensors
            .iter()
            .find(|tensor| tensor.name == "per_layer_model_proj.weight")
            .unwrap();
        assert_eq!(per_layer_model_proj.dtype, SupportedDtype::BF16);
        assert!(
            per_layer_model_proj
                .values
                .iter()
                .all(|value| *value == 1.0)
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_gemma4_dequantized_tensors_from_gguf_loads_tiny_synthetic_all_tensors() {
        let cfg = tiny_config();
        let path = tmp_path("dequantized_values");
        write_tiny_gemma4_gguf(&path, &cfg);

        let (loaded_cfg, tensors) = load_gemma4_dequantized_tensors_from_gguf(&path)
            .expect("tiny Gemma4 GGUF dequantized tensor set must load");

        assert_eq!(loaded_cfg.embedding_length, cfg.embedding_length);
        assert_eq!(tensors.len(), required_gemma4_tensor_names(&cfg).len());

        let token_embd = tensors
            .iter()
            .find(|tensor| tensor.name == "token_embd.weight")
            .unwrap();
        assert_eq!(token_embd.dtype, SupportedDtype::F32);
        assert_eq!(
            token_embd.values.len(),
            cfg.embedding_length * cfg.tokenizer_token_count
        );
        assert!(token_embd.values.iter().all(|value| *value == 0.0));

        let per_layer_model_proj = tensors
            .iter()
            .find(|tensor| tensor.name == "per_layer_model_proj.weight")
            .unwrap();
        assert_eq!(per_layer_model_proj.dtype, SupportedDtype::BF16);
        assert!(
            per_layer_model_proj
                .values
                .iter()
                .all(|value| *value == 1.0)
        );

        loaded_cfg
            .ensure_supported_for_execution()
            .expect_err("value loading must not enable Gemma4 execution");

        let _ = std::fs::remove_file(path);
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
        validate_gemma4_tensor_inventory(&manifest, &cfg, Some(&path))
            .expect("local Gemma4 Q4_K_M tensor inventory must validate");
        validate_gemma4_tensors(&manifest, &cfg, Some(&path))
            .expect_err("local Gemma4 Q4_K_M tensors must stay rejected before dequant policy");
        cfg.ensure_supported_for_execution()
            .expect_err("local Gemma4 Q4_K_M execution must be rejected before compute");
    }

    #[test]
    #[ignore = "requires local-artifacts/gemma4_e4b_it_q4_k_m/google_gemma-4-E4B-it-Q4_K_M.gguf or OCELOTL_GEMMA4_GGUF_PATH"]
    fn local_gemma4_q4_k_m_gguf_dense_subset_loads() {
        let path = local_gemma4_gguf_path();
        assert!(
            path.exists(),
            "missing Gemma4 GGUF artifact at {}; set OCELOTL_GEMMA4_GGUF_PATH or see docs/artifact-preparation.md",
            path.display()
        );

        let (cfg, tensors) = load_gemma4_dense_tensors_from_gguf(&path)
            .expect("local Gemma4 dense GGUF tensors must load");

        assert_eq!(cfg.quantization, Gemma4Quantization::Q4KM);
        assert_eq!(tensors.len(), 340);
        let per_layer_model_proj = tensors
            .iter()
            .find(|tensor| tensor.name == "per_layer_model_proj.weight")
            .expect("PLE projection tensor must load");
        assert_eq!(per_layer_model_proj.dtype, SupportedDtype::BF16);
        assert_eq!(per_layer_model_proj.shape, vec![2_560, 10_752]);
        let output_norm = tensors
            .iter()
            .find(|tensor| tensor.name == "output_norm.weight")
            .expect("output norm tensor must load");
        assert_eq!(output_norm.dtype, SupportedDtype::F32);
        assert_eq!(output_norm.shape, vec![2_560]);
    }
}
