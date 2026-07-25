//! End-to-end Parakeet TDT: audio samples in, transcript out.
//!
//! Assembles the four stages that were built and gated independently — mel
//! frontend, 8x depthwise-striding subsampler, 24-block FastConformer encoder,
//! TDT prediction/joint/greedy decode — behind one call. Each stage keeps its
//! own parity test against the reference graph; this type only wires them.
//!
//! # Long-form audio
//!
//! The encoder uses full self-attention (`att_context_size = [-1, -1]`) with no
//! KV-cache shortcut, so its cost is quadratic in the subsampled frame count
//! (~12.5 frames per second of audio).
//!
//! **The binding constraint is memory, not time** — though it took fixing a
//! kernel to make that true. With attention vectorized and running on the thread
//! pool, the measured quadratic share is 0.3% at 11 s, 3% at 121 s and 14% at
//! 600 s, so RTF stays roughly flat (0.095 at 11 s and 0.09 at 121 s on 12
//! threads). What eventually bites is the `[T, 2T-1]` score transient: ~0.7 GB at
//! 600 s and ~2.7 GB at 20 minutes.
//!
//! [`ParakeetModel::encode_audio`] therefore guards on [`MAX_AUDIO_SECONDS`] for
//! allocation size, and [`ParakeetModel::encode_audio_chunked`] has no limit.
//! Note that chunking is **slower** than a single window below roughly 40
//! minutes of audio — see [`super::chunk`] for why, and for what that says about
//! benchmarking a workaround against a broken baseline.

use std::path::Path;

use ocelotl_core::{OcelotlError, Result, RuntimeError};
use ocelotl_kernels::KernelBackend;
use ocelotl_loader::load_safetensors_tensors_f32;

use super::audio::{PARAKEET_SAMPLE_RATE_HZ, ParakeetFeatures, parakeet_log_mel};
use super::chunk::plan_chunks;
use super::decoder::{
    JointWeights, LstmLayerWeights, PRED_LAYERS, PredNetWeights, TdtConfig, TdtDecode,
    greedy_decode,
};
use super::encoder::{BlockWeights, EncoderShape, encode};
use super::subsample::{SubsampleWeights, subsample, subsampled_frames};

/// Conformer blocks in `parakeet-tdt-0.6b-v3`.
pub const NUM_BLOCKS: usize = 24;

/// Longest audio the **single-window** path will attempt, in seconds.
///
/// A **memory** guard, and only that. Attention materializes a `[T, 2T-1]` score
/// transient per block: ~0.7 GB at 600 s, ~2.7 GB at 20 minutes, quadratic
/// thereafter.
///
/// This was 60 s, sized when the encoder's quadratic term was ~94% of runtime at
/// that length. It is not any more: with attention vectorized and threaded the
/// quadratic share at 600 s is 14%, and single-window beats chunking out to
/// roughly 40 minutes of audio. A 60 s ceiling would now push callers onto the
/// slower, lossier path for no reason.
///
/// [`ParakeetModel::encode_audio_chunked`] is not bounded by this and is the
/// right choice past it — for memory, not for speed.
pub const MAX_AUDIO_SECONDS: usize = 600;

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
    /// **Single-window**: encodes the whole input under full attention. Cost is
    /// quadratic in length, so prefer [`Self::encode_audio_chunked`] for
    /// anything long. Kept public because parity tests need the unchunked path
    /// as the thing chunking is checked against.
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
                "audio is {:.1}s, over the {MAX_AUDIO_SECONDS}s single-window \
                 limit. This encoder uses full self-attention with no KV cache, \
                 so cost grows quadratically; use encode_audio_chunked or \
                 decode_audio_chunked instead.",
                audio.len() as f32 / PARAKEET_SAMPLE_RATE_HZ as f32
            )));
        }
        self.encode_window(audio, kernels)
    }

    /// Encode one window with no length check. The chunked path drives this.
    fn encode_window(
        &self,
        audio: &[f32],
        kernels: &dyn KernelBackend,
    ) -> Result<(Vec<f32>, usize)> {
        let features = parakeet_log_mel(audio)?;
        self.encode_features(&features, kernels)
    }

    /// Subsample and encode already-computed mel features.
    ///
    /// Split out from [`Self::encode_window`] so the chunked path can normalize
    /// once and slice, rather than re-running the frontend per window.
    fn encode_features(
        &self,
        features: &ParakeetFeatures,
        kernels: &dyn KernelBackend,
    ) -> Result<(Vec<f32>, usize)> {
        let subsampled = subsample(
            features,
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

    /// Encoder hidden states via overlap-and-trim chunking, with **no length
    /// limit**.
    ///
    /// The mel frontend runs **once** over the whole utterance and chunking
    /// slices its normalized output — not the waveform. Per-feature
    /// normalization reduces over every frame of its input, so waveform-level
    /// chunking would give each window its own statistics and perturb every
    /// frame inside it, which no amount of context trimming can undo. See
    /// [`super::chunk`] for the measurement that established this.
    ///
    /// Each window is then encoded with `context_frames` of extra frames on both
    /// sides, which are discarded; the surviving bodies tile the utterance
    /// exactly.
    ///
    /// Returns the same `(hidden, frames)` shape as [`Self::encode_audio`], so
    /// the two are directly diffable — which is exactly how the chunking gate is
    /// written.
    pub fn encode_audio_chunked(
        &self,
        audio: &[f32],
        chunk_frames: usize,
        context_frames: usize,
        kernels: &dyn KernelBackend,
    ) -> Result<(Vec<f32>, usize)> {
        let features = parakeet_log_mel(audio)?;
        let total = subsampled_frames(features.frames);
        let plan = plan_chunks(total, chunk_frames, context_frames)?;
        let d = self.shape.d_model;
        let mut out: Vec<f32> = Vec::with_capacity(total * d);

        for window in &plan {
            let range = window.mel_range(features.frames);
            let slice = features.slice_frames(range.clone());
            let (hidden, produced) = self.encode_features(&slice, kernels)?;
            let keep = window.keep_range();
            // The closed-form frame count and what the encoder actually emits
            // for a slice must agree. If they ever do not, stitching would
            // silently shift the time axis from here on — so fail loudly.
            if keep.end > produced {
                return Err(rt(format!(
                    "chunk [{}, {}) needed local frames {keep:?} but the encoder \
                     produced only {produced} from {} mel frames — the frame \
                     arithmetic and the encoder disagree",
                    window.ctx_start,
                    window.ctx_end,
                    range.len()
                )));
            }
            out.extend_from_slice(&hidden[keep.start * d..keep.end * d]);
        }

        let frames = out.len() / d;
        if frames != total {
            return Err(rt(format!("stitched {frames} frames but planned {total}")));
        }
        Ok((out, frames))
    }

    /// Greedy TDT decode of `audio`, returning token ids and frame indices.
    ///
    /// Single-window; see [`Self::encode_audio`] for the length limit.
    pub fn decode_audio(&self, audio: &[f32], kernels: &dyn KernelBackend) -> Result<TdtDecode> {
        let (hidden, frames) = self.encode_audio(audio, kernels)?;
        self.decode_hidden(&hidden, frames, kernels)
    }

    /// Chunked encode followed by **one** greedy decode over the stitched
    /// output, with no length limit.
    ///
    /// The decode deliberately is not chunked. Running it per window would reset
    /// the prediction network's recurrent state at every seam; a single pass
    /// over the stitched encoder output keeps that state continuous, which is
    /// why this can be token-identical to the unchunked path.
    pub fn decode_audio_chunked(
        &self,
        audio: &[f32],
        chunk_frames: usize,
        context_frames: usize,
        kernels: &dyn KernelBackend,
    ) -> Result<TdtDecode> {
        let (hidden, frames) =
            self.encode_audio_chunked(audio, chunk_frames, context_frames, kernels)?;
        self.decode_hidden(&hidden, frames, kernels)
    }

    /// Greedy decode over encoder hidden states the caller already has.
    pub fn decode_hidden(
        &self,
        hidden: &[f32],
        frames: usize,
        kernels: &dyn KernelBackend,
    ) -> Result<TdtDecode> {
        greedy_decode(
            hidden,
            frames,
            &self.weights.prednet,
            &self.weights.joint,
            self.config,
            kernels,
        )
    }
}
