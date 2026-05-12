//! Real Whisper-shaped CPU reference adapter.
//!
//! This is W-ASR.9's correctness-first CPU path. It consumes the canonical
//! OpenAI-style tensor names from `whisper::tensors` after W-ASR.8 validation
//! and keeps all tensor values in owned `f32` buffers.

use std::collections::{BTreeMap, btree_map::Entry};

use ocelotl_core::{DType, InvalidModelError, InvalidRequestError, OcelotlError, Result, TokenId};
use ocelotl_kernels::{CpuKernelBackend, softmax};
use ocelotl_loader::{LoadedTensor, SupportedDtype};

use super::{WhisperConfig, required_whisper_tensor_names};

const CONV_KERNEL_WIDTH: usize = 3;
const LAYER_NORM_EPS: f32 = 1.0e-5;

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

    fn get(&self, name: &str) -> &[f32] {
        self.tensors
            .get(name)
            .map(Vec::as_slice)
            .expect("WhisperWeights validation guarantees required tensors")
    }
}

#[derive(Debug, Clone)]
pub struct WhisperModel {
    config: WhisperConfig,
    weights: WhisperWeights,
    kernels: CpuKernelBackend,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhisperEncodedAudio {
    frames: usize,
    state_size: usize,
    values: Vec<f32>,
    cross_attention: Vec<WhisperCrossAttentionCache>,
}

#[derive(Debug, Clone, PartialEq)]
struct WhisperCrossAttentionCache {
    key: Vec<f32>,
    value: Vec<f32>,
}

impl WhisperEncodedAudio {
    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn state_size(&self) -> usize {
        self.state_size
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

impl WhisperModel {
    pub fn new(config: WhisperConfig, tensors: Vec<LoadedTensor>) -> Result<Self> {
        Self::with_cpu_kernel_backend(config, tensors, CpuKernelBackend::default())
    }

    pub fn with_cpu_kernel_backend(
        config: WhisperConfig,
        tensors: Vec<LoadedTensor>,
        kernels: CpuKernelBackend,
    ) -> Result<Self> {
        let config = config.validate()?;
        let weights = WhisperWeights::from_loaded_tensors(&config, tensors)?;
        Ok(Self {
            config,
            weights,
            kernels,
        })
    }

    pub fn config(&self) -> &WhisperConfig {
        &self.config
    }

    pub fn kernel_backend(&self) -> &CpuKernelBackend {
        &self.kernels
    }

    pub fn forward_next_token_logits(
        &self,
        log_mel: &[f32],
        mel_frames: usize,
        decoder_tokens: &[TokenId],
    ) -> Result<Vec<f32>> {
        validate_forward_request(&self.config, log_mel, mel_frames, decoder_tokens)?;

        let audio = self.encode_audio_features(log_mel, mel_frames)?;
        self.forward_next_token_logits_from_audio(&audio, decoder_tokens)
    }

    pub fn encode_audio_features(
        &self,
        log_mel: &[f32],
        mel_frames: usize,
    ) -> Result<WhisperEncodedAudio> {
        validate_audio_request(&self.config, log_mel, mel_frames)?;

        let values = self.encode_audio(log_mel, mel_frames)?;
        let state_size = self.config.audio_state_size;
        if values.len() % state_size != 0 {
            return Err(invalid_model(
                "encoded_audio",
                &format!(
                    "encoded audio length {} is not divisible by audio_state_size {state_size}",
                    values.len()
                ),
            ));
        }
        let frames = values.len() / state_size;
        if frames == 0 {
            return Err(invalid_model(
                "encoded_audio",
                "encoder produced zero audio frames",
            ));
        }

        Ok(WhisperEncodedAudio {
            frames,
            state_size,
            cross_attention: self.precompute_cross_attention(&values, frames)?,
            values,
        })
    }

    pub fn forward_next_token_logits_from_audio(
        &self,
        audio: &WhisperEncodedAudio,
        decoder_tokens: &[TokenId],
    ) -> Result<Vec<f32>> {
        validate_encoded_audio(&self.config, audio)?;
        validate_decoder_tokens(&self.config, decoder_tokens)?;

        let decoded = self.decode_tokens(decoder_tokens, audio)?;
        let state = self.config.text_state_size;
        let last_start = (decoder_tokens.len() - 1) * state;
        let last = &decoded[last_start..last_start + state];

        let projection = if self.config.tie_word_embeddings {
            self.weights.get("decoder.token_embedding.weight")
        } else {
            self.weights.get("decoder.proj_out.weight")
        };

        linear(
            &self.kernels,
            last,
            1,
            state,
            projection,
            self.config.vocab_size,
            None,
        )
    }

    fn encode_audio(&self, log_mel: &[f32], mel_frames: usize) -> Result<Vec<f32>> {
        let conv1 = conv1d(
            log_mel,
            mel_frames,
            self.config.mel_bins,
            self.weights.get("encoder.conv1.weight"),
            self.weights.get("encoder.conv1.bias"),
            self.config.audio_state_size,
            CONV_KERNEL_WIDTH,
            1,
            1,
        )?;
        let conv1_frames = conv_output_len(mel_frames, CONV_KERNEL_WIDTH, 1, 1)?;
        let mut conv1 = conv1;
        gelu_inplace(&mut conv1);

        let mut conv2 = conv1d(
            &conv1,
            conv1_frames,
            self.config.audio_state_size,
            self.weights.get("encoder.conv2.weight"),
            self.weights.get("encoder.conv2.bias"),
            self.config.audio_state_size,
            CONV_KERNEL_WIDTH,
            2,
            1,
        )?;
        gelu_inplace(&mut conv2);

        let seq = conv_output_len(conv1_frames, CONV_KERNEL_WIDTH, 2, 1)?;
        if seq > self.config.audio_context_length {
            return Err(invalid_request(
                "mel_frames",
                &format!(
                    "convolution output length {seq} exceeds audio_context_length {}",
                    self.config.audio_context_length
                ),
            ));
        }

        add_positional_embedding(
            &mut conv2,
            seq,
            self.config.audio_state_size,
            self.weights.get("encoder.positional_embedding"),
            self.config.audio_context_length,
        )?;

        let mut x = conv2;
        for layer in 0..self.config.audio_layers {
            let prefix = format!("encoder.blocks.{layer}");
            let attn_ln = layer_norm(
                &x,
                seq,
                self.config.audio_state_size,
                self.weights.get(&format!("{prefix}.attn_ln.weight")),
                self.weights.get(&format!("{prefix}.attn_ln.bias")),
                LAYER_NORM_EPS,
            )?;
            let attn = attention(
                &self.kernels,
                &attn_ln,
                seq,
                self.config.audio_state_size,
                self.config.audio_attention_heads,
                self.weights.get(&format!("{prefix}.attn.query.weight")),
                self.weights.get(&format!("{prefix}.attn.query.bias")),
                self.weights.get(&format!("{prefix}.attn.key.weight")),
                self.weights.get(&format!("{prefix}.attn.value.weight")),
                self.weights.get(&format!("{prefix}.attn.value.bias")),
                self.weights.get(&format!("{prefix}.attn.out.weight")),
                self.weights.get(&format!("{prefix}.attn.out.bias")),
                None,
                false,
            )?;
            add_inplace(&mut x, &attn);

            let mlp_ln = layer_norm(
                &x,
                seq,
                self.config.audio_state_size,
                self.weights.get(&format!("{prefix}.mlp_ln.weight")),
                self.weights.get(&format!("{prefix}.mlp_ln.bias")),
                LAYER_NORM_EPS,
            )?;
            let mlp = mlp_gelu(
                &self.kernels,
                &mlp_ln,
                seq,
                self.config.audio_state_size,
                self.config.audio_ffn_size,
                self.weights.get(&format!("{prefix}.mlp.0.weight")),
                self.weights.get(&format!("{prefix}.mlp.0.bias")),
                self.weights.get(&format!("{prefix}.mlp.2.weight")),
                self.weights.get(&format!("{prefix}.mlp.2.bias")),
            )?;
            add_inplace(&mut x, &mlp);
        }

        layer_norm(
            &x,
            seq,
            self.config.audio_state_size,
            self.weights.get("encoder.ln_post.weight"),
            self.weights.get("encoder.ln_post.bias"),
            LAYER_NORM_EPS,
        )
    }

    fn decode_tokens(
        &self,
        decoder_tokens: &[TokenId],
        audio: &WhisperEncodedAudio,
    ) -> Result<Vec<f32>> {
        let seq = decoder_tokens.len();
        let text_state = self.config.text_state_size;
        let audio_seq = audio.frames();
        let mut x = vec![0.0_f32; seq * text_state];
        let token_embedding = self.weights.get("decoder.token_embedding.weight");
        let positional_embedding = self.weights.get("decoder.positional_embedding");

        for (pos, token) in decoder_tokens.iter().enumerate() {
            let token_start = token.0 as usize * text_state;
            let row_start = pos * text_state;
            for dim in 0..text_state {
                x[row_start + dim] =
                    token_embedding[token_start + dim] + positional_embedding[row_start + dim];
            }
        }

        for layer in 0..self.config.text_layers {
            let prefix = format!("decoder.blocks.{layer}");
            let attn_ln = layer_norm(
                &x,
                seq,
                text_state,
                self.weights.get(&format!("{prefix}.attn_ln.weight")),
                self.weights.get(&format!("{prefix}.attn_ln.bias")),
                LAYER_NORM_EPS,
            )?;
            let self_attn = attention(
                &self.kernels,
                &attn_ln,
                seq,
                text_state,
                self.config.text_attention_heads,
                self.weights.get(&format!("{prefix}.attn.query.weight")),
                self.weights.get(&format!("{prefix}.attn.query.bias")),
                self.weights.get(&format!("{prefix}.attn.key.weight")),
                self.weights.get(&format!("{prefix}.attn.value.weight")),
                self.weights.get(&format!("{prefix}.attn.value.bias")),
                self.weights.get(&format!("{prefix}.attn.out.weight")),
                self.weights.get(&format!("{prefix}.attn.out.bias")),
                None,
                true,
            )?;
            add_inplace(&mut x, &self_attn);

            let cross_ln = layer_norm(
                &x,
                seq,
                text_state,
                self.weights.get(&format!("{prefix}.cross_attn_ln.weight")),
                self.weights.get(&format!("{prefix}.cross_attn_ln.bias")),
                LAYER_NORM_EPS,
            )?;
            let cross_cache = &audio.cross_attention[layer];
            let cross_attn = attention_with_precomputed_kv(
                &self.kernels,
                &cross_ln,
                seq,
                text_state,
                self.config.text_attention_heads,
                self.weights
                    .get(&format!("{prefix}.cross_attn.query.weight")),
                self.weights.get(&format!("{prefix}.cross_attn.query.bias")),
                &cross_cache.key,
                &cross_cache.value,
                audio_seq,
                self.weights.get(&format!("{prefix}.cross_attn.out.weight")),
                self.weights.get(&format!("{prefix}.cross_attn.out.bias")),
                false,
            )?;
            add_inplace(&mut x, &cross_attn);

            let mlp_ln = layer_norm(
                &x,
                seq,
                text_state,
                self.weights.get(&format!("{prefix}.mlp_ln.weight")),
                self.weights.get(&format!("{prefix}.mlp_ln.bias")),
                LAYER_NORM_EPS,
            )?;
            let mlp = mlp_gelu(
                &self.kernels,
                &mlp_ln,
                seq,
                text_state,
                self.config.text_ffn_size,
                self.weights.get(&format!("{prefix}.mlp.0.weight")),
                self.weights.get(&format!("{prefix}.mlp.0.bias")),
                self.weights.get(&format!("{prefix}.mlp.2.weight")),
                self.weights.get(&format!("{prefix}.mlp.2.bias")),
            )?;
            add_inplace(&mut x, &mlp);
        }

        layer_norm(
            &x,
            seq,
            text_state,
            self.weights.get("decoder.ln.weight"),
            self.weights.get("decoder.ln.bias"),
            LAYER_NORM_EPS,
        )
    }

    fn precompute_cross_attention(
        &self,
        encoded_audio: &[f32],
        audio_seq: usize,
    ) -> Result<Vec<WhisperCrossAttentionCache>> {
        let state = self.config.text_state_size;
        let mut caches = Vec::with_capacity(self.config.text_layers);
        for layer in 0..self.config.text_layers {
            let prefix = format!("decoder.blocks.{layer}.cross_attn");
            let key = linear(
                &self.kernels,
                encoded_audio,
                audio_seq,
                state,
                self.weights.get(&format!("{prefix}.key.weight")),
                state,
                None,
            )?;
            let value = linear(
                &self.kernels,
                encoded_audio,
                audio_seq,
                state,
                self.weights.get(&format!("{prefix}.value.weight")),
                state,
                Some(self.weights.get(&format!("{prefix}.value.bias"))),
            )?;
            caches.push(WhisperCrossAttentionCache { key, value });
        }
        Ok(caches)
    }
}

#[allow(clippy::too_many_arguments)]
fn conv1d(
    input: &[f32],
    time: usize,
    in_channels: usize,
    weight: &[f32],
    bias: &[f32],
    out_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<Vec<f32>> {
    if stride == 0 {
        return Err(invalid_model("conv1d.stride", "must be > 0"));
    }
    if kernel == 0 {
        return Err(invalid_model("conv1d.kernel", "must be > 0"));
    }
    let input_len = checked_len_product("conv1d.input", &[time, in_channels])?;
    if input.len() != input_len {
        return Err(invalid_request(
            "conv1d.input",
            &format!("expected input length {input_len}, got {}", input.len()),
        ));
    }
    let weight_len = checked_len_product("conv1d.weight", &[out_channels, in_channels, kernel])?;
    if weight.len() != weight_len {
        return Err(invalid_model(
            "conv1d.weight",
            &format!("expected weight length {weight_len}, got {}", weight.len()),
        ));
    }
    if bias.len() != out_channels {
        return Err(invalid_model(
            "conv1d.bias",
            &format!("expected bias length {out_channels}, got {}", bias.len()),
        ));
    }

    let out_time = conv_output_len(time, kernel, stride, padding)?;
    let mut out = vec![0.0_f32; out_time * out_channels];
    for t_out in 0..out_time {
        for oc in 0..out_channels {
            let mut acc = bias[oc];
            for ic in 0..in_channels {
                for k in 0..kernel {
                    let padded_t = t_out * stride + k;
                    if padded_t < padding {
                        continue;
                    }
                    let t_in = padded_t - padding;
                    if t_in >= time {
                        continue;
                    }
                    let input_idx = t_in * in_channels + ic;
                    let weight_idx = (oc * in_channels + ic) * kernel + k;
                    acc += input[input_idx] * weight[weight_idx];
                }
            }
            out[t_out * out_channels + oc] = acc;
        }
    }
    Ok(out)
}

fn conv_output_len(time: usize, kernel: usize, stride: usize, padding: usize) -> Result<usize> {
    let padded = time
        .checked_add(
            padding.checked_mul(2).ok_or_else(|| {
                invalid_model("conv1d.padding", "padding product overflows usize")
            })?,
        )
        .ok_or_else(|| invalid_model("conv1d.padding", "padded length overflows usize"))?;
    if padded < kernel {
        return Err(invalid_request(
            "mel_frames",
            &format!("padded input length {padded} is smaller than kernel width {kernel}"),
        ));
    }
    Ok(((padded - kernel) / stride) + 1)
}

fn layer_norm(
    x: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Result<Vec<f32>> {
    let expected = checked_len_product("layer_norm.x", &[rows, cols])?;
    if x.len() != expected {
        return Err(invalid_request(
            "layer_norm.x",
            &format!("expected input length {expected}, got {}", x.len()),
        ));
    }
    if weight.len() != cols {
        return Err(invalid_model(
            "layer_norm.weight",
            &format!("expected weight length {cols}, got {}", weight.len()),
        ));
    }
    if bias.len() != cols {
        return Err(invalid_model(
            "layer_norm.bias",
            &format!("expected bias length {cols}, got {}", bias.len()),
        ));
    }

    let mut out = vec![0.0_f32; x.len()];
    for row in 0..rows {
        let start = row * cols;
        let values = &x[start..start + cols];
        let mean = values.iter().sum::<f32>() / cols as f32;
        let variance = values
            .iter()
            .map(|v| {
                let delta = *v - mean;
                delta * delta
            })
            .sum::<f32>()
            / cols as f32;
        let inv_std = 1.0_f32 / (variance + eps).sqrt();
        for col in 0..cols {
            out[start + col] = ((x[start + col] - mean) * inv_std) * weight[col] + bias[col];
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn mlp_gelu(
    kernels: &CpuKernelBackend,
    x: &[f32],
    rows: usize,
    hidden: usize,
    ffn: usize,
    fc1_w: &[f32],
    fc1_b: &[f32],
    fc2_w: &[f32],
    fc2_b: &[f32],
) -> Result<Vec<f32>> {
    let mut hidden_act = linear(kernels, x, rows, hidden, fc1_w, ffn, Some(fc1_b))?;
    gelu_inplace(&mut hidden_act);
    linear(kernels, &hidden_act, rows, ffn, fc2_w, hidden, Some(fc2_b))
}

#[allow(clippy::too_many_arguments)]
fn attention(
    kernels: &CpuKernelBackend,
    x: &[f32],
    q_seq: usize,
    state: usize,
    heads: usize,
    query_w: &[f32],
    query_b: &[f32],
    key_w: &[f32],
    value_w: &[f32],
    value_b: &[f32],
    out_w: &[f32],
    out_b: &[f32],
    cross: Option<(&[f32], usize)>,
    causal: bool,
) -> Result<Vec<f32>> {
    if heads == 0 {
        return Err(invalid_model("attention.heads", "must be > 0"));
    }
    if state % heads != 0 {
        return Err(invalid_model(
            "attention.state",
            &format!("state {state} must be divisible by heads {heads}"),
        ));
    }
    let (kv_source, kv_seq) = cross.unwrap_or((x, q_seq));
    let q = linear(kernels, x, q_seq, state, query_w, state, Some(query_b))?;
    let k = linear(kernels, kv_source, kv_seq, state, key_w, state, None)?;
    let v = linear(
        kernels,
        kv_source,
        kv_seq,
        state,
        value_w,
        state,
        Some(value_b),
    )?;

    attention_from_projected(
        kernels, &q, q_seq, &k, &v, kv_seq, state, heads, out_w, out_b, causal,
    )
}

#[allow(clippy::too_many_arguments)]
fn attention_with_precomputed_kv(
    kernels: &CpuKernelBackend,
    x: &[f32],
    q_seq: usize,
    state: usize,
    heads: usize,
    query_w: &[f32],
    query_b: &[f32],
    key: &[f32],
    value: &[f32],
    kv_seq: usize,
    out_w: &[f32],
    out_b: &[f32],
    causal: bool,
) -> Result<Vec<f32>> {
    let q = linear(kernels, x, q_seq, state, query_w, state, Some(query_b))?;
    attention_from_projected(
        kernels, &q, q_seq, key, value, kv_seq, state, heads, out_w, out_b, causal,
    )
}

#[allow(clippy::too_many_arguments)]
fn attention_from_projected(
    kernels: &CpuKernelBackend,
    q: &[f32],
    q_seq: usize,
    k: &[f32],
    v: &[f32],
    kv_seq: usize,
    state: usize,
    heads: usize,
    out_w: &[f32],
    out_b: &[f32],
    causal: bool,
) -> Result<Vec<f32>> {
    if heads == 0 {
        return Err(invalid_model("attention.heads", "must be > 0"));
    }
    if state % heads != 0 {
        return Err(invalid_model(
            "attention.state",
            &format!("state {state} must be divisible by heads {heads}"),
        ));
    }
    let q_expected = checked_len_product("attention.q", &[q_seq, state])?;
    if q.len() != q_expected {
        return Err(invalid_request(
            "attention.q",
            &format!("expected length {q_expected}, got {}", q.len()),
        ));
    }
    let kv_expected = checked_len_product("attention.kv", &[kv_seq, state])?;
    if k.len() != kv_expected {
        return Err(invalid_request(
            "attention.key",
            &format!("expected length {kv_expected}, got {}", k.len()),
        ));
    }
    if v.len() != kv_expected {
        return Err(invalid_request(
            "attention.value",
            &format!("expected length {kv_expected}, got {}", v.len()),
        ));
    }

    let head_dim = state / heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut context = vec![0.0_f32; q_seq * state];
    let mut scores = vec![0.0_f32; kv_seq];

    for qi in 0..q_seq {
        for head in 0..heads {
            let visible = if causal { qi + 1 } else { kv_seq };
            if visible > kv_seq {
                return Err(invalid_request(
                    "attention.causal",
                    "causal self-attention query length exceeds key length",
                ));
            }
            let q_base = qi * state + head * head_dim;
            for (ki, score) in scores.iter_mut().enumerate().take(visible) {
                let k_base = ki * state + head * head_dim;
                let mut acc = 0.0_f32;
                for dim in 0..head_dim {
                    acc += q[q_base + dim] * k[k_base + dim];
                }
                *score = acc * scale;
            }
            softmax(&mut scores[..visible]);
            for dim in 0..head_dim {
                let mut acc = 0.0_f32;
                for (ki, &p) in scores.iter().enumerate().take(visible) {
                    let v_base = ki * state + head * head_dim;
                    acc += p * v[v_base + dim];
                }
                context[qi * state + head * head_dim + dim] = acc;
            }
        }
    }

    linear(kernels, &context, q_seq, state, out_w, state, Some(out_b))
}

fn linear(
    kernels: &CpuKernelBackend,
    x: &[f32],
    rows: usize,
    in_features: usize,
    weight_out_by_in: &[f32],
    out_features: usize,
    bias: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let x_expected = checked_len_product("linear.x", &[rows, in_features])?;
    if x.len() != x_expected {
        return Err(invalid_request(
            "linear.x",
            &format!("expected input length {x_expected}, got {}", x.len()),
        ));
    }
    let weight_expected = checked_len_product("linear.weight", &[out_features, in_features])?;
    if weight_out_by_in.len() != weight_expected {
        return Err(invalid_model(
            "linear.weight",
            &format!(
                "expected [out,in] weight length {weight_expected}, got {}",
                weight_out_by_in.len()
            ),
        ));
    }
    if let Some(bias) = bias {
        if bias.len() != out_features {
            return Err(invalid_model(
                "linear.bias",
                &format!("expected bias length {out_features}, got {}", bias.len()),
            ));
        }
    }

    let out_len = checked_len_product("linear.out", &[rows, out_features])?;
    let mut out = vec![0.0_f32; out_len];
    kernels.linear_out_by_in(
        x,
        rows,
        in_features,
        weight_out_by_in,
        out_features,
        bias,
        &mut out,
    )?;
    Ok(out)
}

fn gelu_inplace(x: &mut [f32]) {
    for v in x {
        *v = gelu(*v);
    }
}

fn gelu(x: f32) -> f32 {
    // OpenAI Whisper uses PyTorch's default GELU, which is the exact-erf
    // formulation rather than the tanh approximation.
    0.5 * x * (1.0 + erf_approx(x / std::f32::consts::SQRT_2))
}

fn erf_approx(x: f32) -> f32 {
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp());
    sign * y
}

fn add_positional_embedding(
    x: &mut [f32],
    rows: usize,
    cols: usize,
    positional_embedding: &[f32],
    max_rows: usize,
) -> Result<()> {
    if rows > max_rows {
        return Err(invalid_request(
            "positional_embedding",
            &format!("rows {rows} exceeds max rows {max_rows}"),
        ));
    }
    if positional_embedding.len() != max_rows * cols {
        return Err(invalid_model(
            "positional_embedding",
            &format!(
                "expected positional embedding length {}, got {}",
                max_rows * cols,
                positional_embedding.len()
            ),
        ));
    }
    for row in 0..rows {
        let start = row * cols;
        for col in 0..cols {
            x[start + col] += positional_embedding[start + col];
        }
    }
    Ok(())
}

fn add_inplace(lhs: &mut [f32], rhs: &[f32]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
        *lhs += rhs;
    }
}

fn validate_forward_request(
    config: &WhisperConfig,
    log_mel: &[f32],
    mel_frames: usize,
    decoder_tokens: &[TokenId],
) -> Result<()> {
    validate_audio_request(config, log_mel, mel_frames)?;
    validate_decoder_tokens(config, decoder_tokens)
}

fn validate_audio_request(
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

fn validate_encoded_audio(config: &WhisperConfig, audio: &WhisperEncodedAudio) -> Result<()> {
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

fn validate_decoder_tokens(config: &WhisperConfig, decoder_tokens: &[TokenId]) -> Result<()> {
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

fn expected_shape(name: &str, config: &WhisperConfig) -> Result<Vec<usize>> {
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

fn checked_len_product(label: &str, dims: &[usize]) -> Result<usize> {
    dims.iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| invalid_model(label, &format!("shape product overflows usize: {:?}", dims)))
}

fn invalid_model(field: &str, message: &str) -> OcelotlError {
    OcelotlError::from(InvalidModelError {
        path: None,
        field: Some(field.to_string()),
        message: message.to_string(),
    })
}

fn invalid_request(field: &str, message: &str) -> OcelotlError {
    OcelotlError::from(InvalidRequestError {
        field: field.to_string(),
        message: message.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ocelotl_core::DType;

    #[test]
    fn conv1d_applies_padding_and_stride() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let weight = [10.0_f32, 1.0, -1.0];
        let bias = [0.5_f32];

        let out = conv1d(&input, 4, 1, &weight, &bias, 1, 3, 2, 1).expect("conv1d");

        assert_eq!(out, vec![-0.5, 19.5]);
    }

    #[test]
    fn layer_norm_applies_weight_bias_and_epsilon_per_row() {
        let input = [1.0_f32, 3.0];
        let weight = [2.0_f32, 0.5];
        let bias = [0.1_f32, -0.2];

        let out = layer_norm(&input, 1, 2, &weight, &bias, 1.0e-5).expect("layer norm");

        assert_close(&out, &[-1.89999, 0.2999975], 1.0e-5);
    }

    #[test]
    fn gelu_mlp_projection_path_matches_hand_checked_fixture() {
        let x = [1.0_f32, 0.0];
        let fc1_w = [1.0_f32, 0.0, 0.0, 1.0];
        let fc1_b = [0.0_f32, 0.0];
        let fc2_w = [1.0_f32, 0.0, 0.0, 1.0];
        let fc2_b = [0.0_f32, 0.0];
        let kernels = CpuKernelBackend::default();

        let out = mlp_gelu(&kernels, &x, 1, 2, 2, &fc1_w, &fc1_b, &fc2_w, &fc2_b).expect("mlp");

        assert_close(&out, &[0.841_344_7, 0.0], 1.0e-6);
    }

    #[test]
    fn gelu_pins_exact_erf_variant_used_by_openai_whisper() {
        assert_close(&[gelu(1.0)], &[0.841_344_7], 1.0e-6);
        assert_close(&[gelu(-1.0)], &[-0.158_655_26], 1.0e-6);
    }

    #[test]
    fn encoder_self_attention_does_not_apply_causal_mask() {
        let x = [1.0_f32, 2.0, 7.0];
        let identity = [1.0_f32];
        let zero = [0.0_f32];
        let kernels = CpuKernelBackend::default();
        let out = attention(
            &kernels, &x, 3, 1, 1, &zero, &zero, &zero, &identity, &zero, &identity, &zero, None,
            false,
        )
        .expect("encoder self attention");

        assert_close(&out, &[10.0 / 3.0, 10.0 / 3.0, 10.0 / 3.0], 1.0e-5);
    }

    #[test]
    fn decoder_self_attention_applies_causal_mask() {
        let x = [1.0_f32, 2.0, 7.0];
        let identity = [1.0_f32];
        let zero = [0.0_f32];
        let kernels = CpuKernelBackend::default();
        let out = attention(
            &kernels, &x, 3, 1, 1, &zero, &zero, &zero, &identity, &zero, &identity, &zero, None,
            true,
        )
        .expect("decoder self attention");

        assert_close(&out, &[1.0, 1.5, 10.0 / 3.0], 1.0e-5);
    }

    #[test]
    fn decoder_cross_attention_does_not_apply_causal_mask() {
        let text = [1.0_f32, 1.0];
        let audio = [1.0_f32, 2.0, 7.0];
        let identity = [1.0_f32];
        let zero = [0.0_f32];
        let kernels = CpuKernelBackend::default();
        let out = attention(
            &kernels,
            &text,
            2,
            1,
            1,
            &zero,
            &zero,
            &zero,
            &identity,
            &zero,
            &identity,
            &zero,
            Some((&audio, 3)),
            false,
        )
        .expect("decoder cross attention");

        assert_close(&out, &[10.0 / 3.0, 10.0 / 3.0], 1.0e-5);
    }

    #[test]
    fn model_construction_rejects_missing_weight_before_compute() {
        let cfg = tiny_config();
        let mut weights = tiny_weight_tensors(&cfg);
        weights.retain(|tensor| tensor.name != "encoder.conv1.weight");

        let err = WhisperModel::new(cfg, weights).expect_err("missing tensor must fail");

        match err {
            ocelotl_core::OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("encoder.conv1.weight"));
            }
            other => panic!("expected InvalidModel, got {other:?}"),
        }
    }

    #[test]
    fn model_construction_rejects_wrong_loaded_shape_before_compute() {
        let cfg = tiny_config();
        let mut weights = tiny_weight_tensors(&cfg);
        let tensor = weights
            .iter_mut()
            .find(|tensor| tensor.name == "decoder.token_embedding.weight")
            .expect("token embedding test tensor");
        tensor.shape = vec![cfg.text_state_size, cfg.vocab_size];

        let err = WhisperModel::new(cfg, weights).expect_err("wrong tensor shape must fail");

        match err {
            ocelotl_core::OcelotlError::InvalidModel(invalid) => {
                assert_eq!(
                    invalid.field.as_deref(),
                    Some("decoder.token_embedding.weight")
                );
                assert!(invalid.message.contains("expected"));
            }
            other => panic!("expected InvalidModel, got {other:?}"),
        }
    }

    #[test]
    fn cached_audio_logits_match_legacy_forward_path() {
        let cfg = tiny_config();
        let model = WhisperModel::new(cfg.clone(), tiny_weight_tensors(&cfg)).expect("model");
        let mel = vec![0.0_f32; 4 * cfg.mel_bins];
        let tokens = [TokenId(0), TokenId(2)];

        let legacy = model
            .forward_next_token_logits(&mel, 4, &tokens)
            .expect("legacy forward");
        let audio = model.encode_audio_features(&mel, 4).expect("encoded audio");
        let cached = model
            .forward_next_token_logits_from_audio(&audio, &tokens)
            .expect("cached forward");

        assert_eq!(audio.frames(), 2);
        assert_eq!(audio.state_size(), cfg.audio_state_size);
        assert_eq!(audio.values().len(), audio.frames() * audio.state_size());
        assert_eq!(audio.cross_attention.len(), cfg.text_layers);
        for cache in &audio.cross_attention {
            assert_eq!(cache.key.len(), audio.frames() * cfg.text_state_size);
            assert_eq!(cache.value.len(), audio.frames() * cfg.text_state_size);
        }
        assert_close(&cached, &legacy, 0.0);
    }

    #[test]
    fn precomputed_cross_attention_matches_projected_cross_attention() {
        let text = [1.0_f32, 1.0];
        let audio = [1.0_f32, 2.0, 7.0];
        let identity = [1.0_f32];
        let zero = [0.0_f32];
        let kernels = CpuKernelBackend::default();

        let projected = attention(
            &kernels,
            &text,
            2,
            1,
            1,
            &zero,
            &zero,
            &zero,
            &identity,
            &zero,
            &identity,
            &zero,
            Some((&audio, 3)),
            false,
        )
        .expect("projected cross attention");
        let precomputed = attention_with_precomputed_kv(
            &kernels, &text, 2, 1, 1, &zero, &zero, &audio, &audio, 3, &identity, &zero, false,
        )
        .expect("precomputed cross attention");

        assert_close(&precomputed, &projected, 0.0);
    }

    #[test]
    fn optimized_cpu_backend_preserves_forward_logits() {
        let cfg = tiny_config();
        let scalar =
            WhisperModel::new(cfg.clone(), tiny_weight_tensors(&cfg)).expect("scalar model");
        let optimized = WhisperModel::with_cpu_kernel_backend(
            cfg.clone(),
            tiny_weight_tensors(&cfg),
            CpuKernelBackend::optimized(),
        )
        .expect("optimized model");
        let mel = vec![0.0_f32; 4 * cfg.mel_bins];
        let tokens = [TokenId(0), TokenId(2)];

        assert_eq!(
            optimized.kernel_backend().mode(),
            ocelotl_kernels::CpuKernelMode::Optimized
        );
        let scalar_logits = scalar
            .forward_next_token_logits(&mel, 4, &tokens)
            .expect("scalar logits");
        let optimized_logits = optimized
            .forward_next_token_logits(&mel, 4, &tokens)
            .expect("optimized logits");

        assert_close(&optimized_logits, &scalar_logits, 1.0e-5);
    }

    #[test]
    fn cached_audio_forward_rejects_wrong_state_size_before_compute() {
        let cfg = tiny_config();
        let model = WhisperModel::new(cfg.clone(), tiny_weight_tensors(&cfg)).expect("model");
        let audio = WhisperEncodedAudio {
            frames: 1,
            state_size: cfg.audio_state_size + 1,
            values: vec![0.0; cfg.audio_state_size + 1],
            cross_attention: Vec::new(),
        };

        let err = model
            .forward_next_token_logits_from_audio(&audio, &[TokenId(0)])
            .expect_err("wrong encoded audio shape must fail");

        match err {
            ocelotl_core::OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "encoded_audio.state_size");
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    fn tiny_config() -> super::super::WhisperConfig {
        super::super::WhisperConfig {
            vocab_size: 4,
            mel_bins: 2,
            audio_context_length: 2,
            audio_state_size: 2,
            audio_attention_heads: 1,
            audio_layers: 1,
            audio_ffn_size: 2,
            text_context_length: 3,
            text_state_size: 2,
            text_attention_heads: 1,
            text_layers: 1,
            text_ffn_size: 2,
            dtype: DType::F32,
            tie_word_embeddings: true,
        }
    }

    fn tiny_weight_tensors(cfg: &super::super::WhisperConfig) -> Vec<ocelotl_loader::LoadedTensor> {
        synthetic_weight_tensors(cfg)
    }

    fn synthetic_weight_tensors(
        cfg: &super::super::WhisperConfig,
    ) -> Vec<ocelotl_loader::LoadedTensor> {
        required_whisper_tensor_names(cfg)
            .into_iter()
            .map(|name| {
                let shape = expected_shape(&name, cfg).expect("known test tensor shape");
                let len = shape.iter().product();
                let mut values = vec![0.0_f32; len];
                if name == "decoder.token_embedding.weight" {
                    values = vec![
                        0.5, 0.0, // token 0
                        0.0, 0.0, // token 1
                        0.25, 0.0, // token 2
                        -0.25, 0.0, // token 3
                    ];
                } else if name == "decoder.ln.bias" {
                    values = vec![1.0, 0.0];
                }
                ocelotl_loader::LoadedTensor {
                    name,
                    shape,
                    dtype: ocelotl_loader::SupportedDtype::F32,
                    values,
                }
            })
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let delta = (actual - expected).abs();
            assert!(
                delta <= tolerance,
                "index {idx}: expected {expected}, got {actual}, delta {delta}"
            );
        }
    }
}
