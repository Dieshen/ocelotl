//! Tiny synthetic Whisper-shaped encoder/decoder path.
//!
//! This is a default-on W-ASR.4 fixture path, not a real Whisper implementation.
//! It proves the wiring shape Ocelotl needs next: log-mel frames enter an
//! encoder, decoder token IDs produce a decoder state, cross-attention reads the
//! encoder states, and a token projection returns logits over an Ocelotl-owned
//! vocabulary.

use ocelotl_core::{InvalidModelError, InvalidRequestError, OcelotlError, Result, TokenId};
use ocelotl_kernels::{dot, matmul, softmax};

use super::audio::LogMelSpectrogram;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperTinyConfig {
    pub mel_bins: usize,
    pub max_audio_frames: usize,
    pub decoder_context_length: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhisperTinyWeights {
    /// `[mel_bins, hidden_size]`.
    pub encoder_projection: Vec<f32>,
    /// `[hidden_size]`.
    pub encoder_bias: Vec<f32>,
    /// `[vocab_size, hidden_size]`.
    pub decoder_token_embedding: Vec<f32>,
    /// `[hidden_size, hidden_size]`.
    pub decoder_query_projection: Vec<f32>,
    /// `[hidden_size, hidden_size]`.
    pub cross_attention_key_projection: Vec<f32>,
    /// `[hidden_size, hidden_size]`.
    pub cross_attention_value_projection: Vec<f32>,
    /// `[hidden_size, hidden_size]`.
    pub cross_attention_output_projection: Vec<f32>,
    /// `[hidden_size, vocab_size]`.
    pub token_projection: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct WhisperTinyModel {
    config: WhisperTinyConfig,
    weights: WhisperTinyWeights,
}

impl WhisperTinyModel {
    pub fn new(config: WhisperTinyConfig, weights: WhisperTinyWeights) -> Result<Self> {
        validate_config(&config)?;

        let h = config.hidden_size;
        let v = config.vocab_size;
        let mel = config.mel_bins;
        let mel_hidden = checked_len_product("mel_bins*hidden_size", &[mel, h])?;
        let vocab_hidden = checked_len_product("vocab_size*hidden_size", &[v, h])?;
        let hidden_hidden = checked_len_product("hidden_size*hidden_size", &[h, h])?;
        let hidden_vocab = checked_len_product("hidden_size*vocab_size", &[h, v])?;

        check_len(
            "encoder_projection",
            weights.encoder_projection.len(),
            mel_hidden,
        )?;
        check_len("encoder_bias", weights.encoder_bias.len(), h)?;
        check_len(
            "decoder_token_embedding",
            weights.decoder_token_embedding.len(),
            vocab_hidden,
        )?;
        check_len(
            "decoder_query_projection",
            weights.decoder_query_projection.len(),
            hidden_hidden,
        )?;
        check_len(
            "cross_attention_key_projection",
            weights.cross_attention_key_projection.len(),
            hidden_hidden,
        )?;
        check_len(
            "cross_attention_value_projection",
            weights.cross_attention_value_projection.len(),
            hidden_hidden,
        )?;
        check_len(
            "cross_attention_output_projection",
            weights.cross_attention_output_projection.len(),
            hidden_hidden,
        )?;
        check_len(
            "token_projection",
            weights.token_projection.len(),
            hidden_vocab,
        )?;

        Ok(Self { config, weights })
    }

    pub fn config(&self) -> &WhisperTinyConfig {
        &self.config
    }

    /// Run the tiny synthetic Whisper-shaped forward path.
    ///
    /// Returns final-position logits with length `config.vocab_size`.
    pub fn forward(&self, mel: &LogMelSpectrogram, decoder_tokens: &[TokenId]) -> Result<Vec<f32>> {
        validate_forward_request(&self.config, mel, decoder_tokens)?;

        let h = self.config.hidden_size;
        let frames = mel.frames;
        let vocab = self.config.vocab_size;

        // Encoder: `[frames, mel_bins] @ [mel_bins, hidden] -> [frames, hidden]`.
        let mut encoder = vec![0.0_f32; frames * h];
        matmul(
            &mel.values,
            (frames, mel.mel_bins),
            &self.weights.encoder_projection,
            (mel.mel_bins, h),
            &mut encoder,
        )?;
        add_bias_per_row(&mut encoder, &self.weights.encoder_bias, frames, h);

        // Decoder seed: last decoder token embedding. W-ASR.4 intentionally
        // accepts full startup-prompt-shaped input while leaving autoregressive
        // decode policy to later runtime work.
        let last_token = decoder_tokens
            .last()
            .copied()
            .expect("non-empty checked above");
        let embed_start = (last_token.0 as usize) * h;
        let decoder_state = &self.weights.decoder_token_embedding[embed_start..embed_start + h];

        let mut query = vec![0.0_f32; h];
        matmul(
            decoder_state,
            (1, h),
            &self.weights.decoder_query_projection,
            (h, h),
            &mut query,
        )?;

        let mut keys = vec![0.0_f32; frames * h];
        matmul(
            &encoder,
            (frames, h),
            &self.weights.cross_attention_key_projection,
            (h, h),
            &mut keys,
        )?;

        let mut values = vec![0.0_f32; frames * h];
        matmul(
            &encoder,
            (frames, h),
            &self.weights.cross_attention_value_projection,
            (h, h),
            &mut values,
        )?;

        let mut scores = vec![0.0_f32; frames];
        let scale = (h as f32).sqrt();
        for (frame, score) in scores.iter_mut().enumerate() {
            let row_start = frame * h;
            *score = dot(&query, &keys[row_start..row_start + h])? / scale;
        }
        softmax(&mut scores);

        let mut context = vec![0.0_f32; h];
        for (frame, score) in scores.iter().enumerate() {
            let row_start = frame * h;
            for dim in 0..h {
                context[dim] += *score * values[row_start + dim];
            }
        }

        let mut cross_output = vec![0.0_f32; h];
        matmul(
            &context,
            (1, h),
            &self.weights.cross_attention_output_projection,
            (h, h),
            &mut cross_output,
        )?;

        let mut final_decoder_state = decoder_state.to_vec();
        for dim in 0..h {
            final_decoder_state[dim] += cross_output[dim];
        }

        let mut logits = vec![0.0_f32; vocab];
        matmul(
            &final_decoder_state,
            (1, h),
            &self.weights.token_projection,
            (h, vocab),
            &mut logits,
        )?;

        Ok(logits)
    }
}

/// Transpose a `[rows, cols]` row-major slice into `[cols, rows]` row-major.
pub fn transpose_2d(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), rows * cols);
    let mut dst = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
}

fn validate_config(config: &WhisperTinyConfig) -> Result<()> {
    if config.mel_bins == 0 {
        return Err(invalid_model("mel_bins", "must be > 0"));
    }
    if config.max_audio_frames == 0 {
        return Err(invalid_model("max_audio_frames", "must be > 0"));
    }
    if config.decoder_context_length == 0 {
        return Err(invalid_model("decoder_context_length", "must be > 0"));
    }
    if config.vocab_size < 2 {
        return Err(invalid_model("vocab_size", "must be >= 2"));
    }
    if config.hidden_size == 0 {
        return Err(invalid_model("hidden_size", "must be > 0"));
    }
    Ok(())
}

fn validate_forward_request(
    config: &WhisperTinyConfig,
    mel: &LogMelSpectrogram,
    decoder_tokens: &[TokenId],
) -> Result<()> {
    if mel.frames == 0 {
        return Err(invalid_request(
            "mel.frames",
            "WhisperTinyModel::forward requires at least one mel frame",
        ));
    }
    if mel.frames > config.max_audio_frames {
        return Err(invalid_request(
            "mel.frames",
            &format!(
                "mel frames {} exceeds max_audio_frames {}",
                mel.frames, config.max_audio_frames
            ),
        ));
    }
    if mel.mel_bins != config.mel_bins {
        return Err(invalid_request(
            "mel.mel_bins",
            &format!(
                "expected {} mel bins, got {}",
                config.mel_bins, mel.mel_bins
            ),
        ));
    }
    let expected_values = checked_len_product("mel.frames*mel_bins", &[mel.frames, mel.mel_bins])?;
    if mel.values.len() != expected_values {
        return Err(invalid_request(
            "mel.values",
            &format!(
                "expected length {expected_values} for {}x{} mel input, got {}",
                mel.frames,
                mel.mel_bins,
                mel.values.len()
            ),
        ));
    }
    if decoder_tokens.is_empty() {
        return Err(invalid_request(
            "decoder_tokens",
            "WhisperTinyModel::forward requires at least one decoder token",
        ));
    }
    if decoder_tokens.len() > config.decoder_context_length {
        return Err(invalid_request(
            "decoder_tokens",
            &format!(
                "decoder token length {} exceeds decoder_context_length {}",
                decoder_tokens.len(),
                config.decoder_context_length
            ),
        ));
    }
    for (idx, token) in decoder_tokens.iter().enumerate() {
        if (token.0 as usize) >= config.vocab_size {
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

fn add_bias_per_row(x: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
    debug_assert_eq!(x.len(), rows * cols);
    debug_assert_eq!(bias.len(), cols);
    for row in 0..rows {
        let row_start = row * cols;
        for col in 0..cols {
            x[row_start + col] += bias[col];
        }
    }
}

fn check_len(name: &str, got: usize, expected: usize) -> Result<()> {
    if got == expected {
        Ok(())
    } else {
        Err(invalid_model(
            name,
            &format!("expected length {expected}, got {got}"),
        ))
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
