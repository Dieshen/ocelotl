//! End-to-end Parakeet TDT: audio samples in, transcript out.
//!
//! Assembles the four stages that were built and gated independently — mel
//! frontend, 8x depthwise-striding subsampler, 24-block FastConformer encoder,
//! TDT prediction/joint/greedy decode — behind one call. Each stage keeps its
//! own parity test against the reference graph; this type only wires them.
//!
//! # Long-form audio
//!
//! The encoder uses full self-attention (`att_context_size = [-1, -1]`), so
//! attention cost is quadratic in the *subsampled* frame count with no
//! KV-cache shortcut available. Roughly 12.5 encoder frames per second of
//! audio, and each block materializes a `[T, 2T-1]` score matrix — so ten
//! minutes of audio is on the order of a gigabyte of transient scores per
//! block. [`ParakeetModel::encode_audio`] therefore refuses inputs beyond
//! [`MAX_AUDIO_SECONDS`] rather than attempting an allocation that will fail
//! somewhere less legible. Chunking with overlap is the fix, and it is not
//! implemented here.

use std::path::Path;

use ocelotl_core::{OcelotlError, Result, RuntimeError};
use ocelotl_kernels::KernelBackend;
use ocelotl_loader::load_safetensors_tensors_f32;

use super::audio::{PARAKEET_SAMPLE_RATE_HZ, parakeet_log_mel};
use super::decoder::{
    JointWeights, LstmLayerWeights, PRED_LAYERS, PredNetWeights, TdtConfig, TdtDecode,
    greedy_decode,
};
use super::encoder::{BlockWeights, EncoderShape, encode};
use super::subsample::{SubsampleWeights, subsample};

/// Conformer blocks in `parakeet-tdt-0.6b-v3`.
pub const NUM_BLOCKS: usize = 24;

/// Longest audio [`ParakeetModel::encode_audio`] will attempt, in seconds.
///
/// Not a model limit — a memory one. See the module docs: the encoder's
/// quadratic attention makes the failure mode a doomed allocation deep inside
/// block 0, which is far harder to read than an explicit refusal.
pub const MAX_AUDIO_SECONDS: usize = 60;

fn rt<S: Into<String>>(m: S) -> OcelotlError {
    OcelotlError::Runtime(RuntimeError { message: m.into() })
}

/// Every weight the model needs, already in the layouts the kernels consume.
#[derive(Debug, Clone)]
pub struct ParakeetWeights {
    pub subsample: SubsampleWeights,
    pub blocks: Vec<BlockWeights>,
    pub prednet: PredNetWeights,
    pub joint: JointWeights,
}

const SUBSAMPLE_TENSORS: [&str; 12] = [
    "encoder.subsampling.layers.0.weight",
    "encoder.subsampling.layers.0.bias",
    "encoder.subsampling.layers.2.weight",
    "encoder.subsampling.layers.2.bias",
    "encoder.subsampling.layers.3.weight",
    "encoder.subsampling.layers.3.bias",
    "encoder.subsampling.layers.5.weight",
    "encoder.subsampling.layers.5.bias",
    "encoder.subsampling.layers.6.weight",
    "encoder.subsampling.layers.6.bias",
    "encoder.subsampling.linear.weight",
    "encoder.subsampling.linear.bias",
];

const BLOCK_TENSORS: [&str; 28] = [
    "norm_feed_forward1.weight",
    "norm_feed_forward1.bias",
    "feed_forward1.linear1.weight",
    "feed_forward1.linear2.weight",
    "norm_self_att.weight",
    "norm_self_att.bias",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.relative_k_proj.weight",
    "self_attn.bias_u",
    "self_attn.bias_v",
    "norm_conv.weight",
    "norm_conv.bias",
    "conv.pointwise_conv1.weight",
    "conv.depthwise_conv.weight",
    "conv.norm.running_mean",
    "conv.norm.running_var",
    "conv.norm.weight",
    "conv.norm.bias",
    "conv.pointwise_conv2.weight",
    "norm_feed_forward2.weight",
    "norm_feed_forward2.bias",
    "feed_forward2.linear1.weight",
    "feed_forward2.linear2.weight",
    "norm_out.weight",
    "norm_out.bias",
];

const JOINT_TENSORS: [&str; 6] = [
    "encoder_projector.weight",
    "encoder_projector.bias",
    "decoder.decoder_projector.weight",
    "decoder.decoder_projector.bias",
    "joint.head.weight",
    "joint.head.bias",
];

impl ParakeetWeights {
    /// Load every tensor from a `model.safetensors` in **one** pass.
    ///
    /// One call rather than one per module is not a micro-optimization: the
    /// loader reads and parses the whole 2.5 GB archive per invocation, so
    /// loading 26 modules separately would read 65 GB.
    pub fn from_safetensors(path: &Path) -> Result<Self> {
        let mut names: Vec<String> = SUBSAMPLE_TENSORS.iter().map(|s| s.to_string()).collect();
        for i in 0..NUM_BLOCKS {
            names.extend(
                BLOCK_TENSORS
                    .iter()
                    .map(|s| format!("encoder.layers.{i}.{s}")),
            );
        }
        names.push("decoder.embedding.weight".to_string());
        for l in 0..PRED_LAYERS {
            for t in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"] {
                names.push(format!("decoder.lstm.{t}_l{l}"));
            }
        }
        names.extend(JOINT_TENSORS.iter().map(|s| s.to_string()));

        let loaded = load_safetensors_tensors_f32(path, &names)?;
        if loaded.len() != names.len() {
            return Err(rt(format!(
                "loaded {} tensors, expected {}",
                loaded.len(),
                names.len()
            )));
        }
        let mut it = loaded.into_iter().map(|t| t.values);
        let mut n = || it.next().expect("tensor count already checked");

        let subsample = SubsampleWeights {
            conv0_w: n(),
            conv0_b: n(),
            dw1_w: n(),
            dw1_b: n(),
            pw1_w: n(),
            pw1_b: n(),
            dw2_w: n(),
            dw2_b: n(),
            pw2_w: n(),
            pw2_b: n(),
            linear_w: n(),
            linear_b: n(),
        };
        let blocks = (0..NUM_BLOCKS)
            .map(|_| BlockWeights {
                norm_ff1_w: n(),
                norm_ff1_b: n(),
                ff1_l1: n(),
                ff1_l2: n(),
                norm_attn_w: n(),
                norm_attn_b: n(),
                q_proj: n(),
                k_proj: n(),
                v_proj: n(),
                o_proj: n(),
                pos_proj: n(),
                bias_u: n(),
                bias_v: n(),
                norm_conv_w: n(),
                norm_conv_b: n(),
                pw1: n(),
                dw: n(),
                bn_mean: n(),
                bn_var: n(),
                bn_w: n(),
                bn_b: n(),
                pw2: n(),
                norm_ff2_w: n(),
                norm_ff2_b: n(),
                ff2_l1: n(),
                ff2_l2: n(),
                norm_out_w: n(),
                norm_out_b: n(),
            })
            .collect();
        let embedding = n();
        let layers = (0..PRED_LAYERS)
            .map(|_| LstmLayerWeights {
                weight_ih: n(),
                weight_hh: n(),
                bias_ih: n(),
                bias_hh: n(),
            })
            .collect();
        let joint = JointWeights {
            enc_proj_w: n(),
            enc_proj_b: n(),
            pred_proj_w: n(),
            pred_proj_b: n(),
            head_w: n(),
            head_b: n(),
        };
        Ok(Self {
            subsample,
            blocks,
            prednet: PredNetWeights { embedding, layers },
            joint,
        })
    }
}

/// A loaded Parakeet TDT model.
#[derive(Debug, Clone)]
pub struct ParakeetModel {
    weights: ParakeetWeights,
    shape: EncoderShape,
    config: TdtConfig,
}

impl ParakeetModel {
    pub fn new(weights: ParakeetWeights) -> Self {
        Self {
            weights,
            shape: EncoderShape::default(),
            config: TdtConfig::default(),
        }
    }

    pub fn from_safetensors(path: &Path) -> Result<Self> {
        Ok(Self::new(ParakeetWeights::from_safetensors(path)?))
    }

    pub fn weights(&self) -> &ParakeetWeights {
        &self.weights
    }

    /// Encoder hidden states for `audio` (16 kHz mono f32), `[frames][d_model]`.
    ///
    /// Exposed separately from [`Self::decode_audio`] so the encoder can be
    /// benchmarked and diffed without the decode, and so a caller doing its own
    /// decoding does not pay for the greedy loop.
    pub fn encode_audio(
        &self,
        audio: &[f32],
        kernels: &dyn KernelBackend,
    ) -> Result<(Vec<f32>, usize)> {
        let max_samples = MAX_AUDIO_SECONDS * PARAKEET_SAMPLE_RATE_HZ as usize;
        if audio.len() > max_samples {
            return Err(rt(format!(
                "audio is {:.1}s; this encoder uses full self-attention with no \
                 KV cache, so inputs over {MAX_AUDIO_SECONDS}s need chunking \
                 (not implemented) rather than a larger allocation",
                audio.len() as f32 / PARAKEET_SAMPLE_RATE_HZ as f32
            )));
        }
        let features = parakeet_log_mel(audio)?;
        let subsampled = subsample(
            &features,
            &self.weights.subsample,
            self.shape.d_model,
            kernels,
        )?;
        let mut hidden = subsampled.values;
        let frames = subsampled.frames;
        encode(
            &mut hidden,
            frames,
            self.shape,
            &self.weights.blocks,
            kernels,
        )?;
        Ok((hidden, frames))
    }

    /// Greedy TDT decode of `audio`, returning token ids and frame indices.
    pub fn decode_audio(&self, audio: &[f32], kernels: &dyn KernelBackend) -> Result<TdtDecode> {
        let (hidden, frames) = self.encode_audio(audio, kernels)?;
        greedy_decode(
            &hidden,
            frames,
            &self.weights.prednet,
            &self.weights.joint,
            self.config,
            kernels,
        )
    }
}
