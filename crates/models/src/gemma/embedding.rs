//! EmbeddingGemma-300M — a Gemma-3 **bidirectional text encoder** that produces
//! mean-pooled, L2-normalized sentence embeddings.
//!
//! It reuses ocelotl's Gemma kernels (RMSNorm, RoPE, gated-tanh-GELU MLP) with
//! **bidirectional** (non-causal) attention plus a mean-pool tail. Unlike the
//! generative [`super::gemma4`] path there is no KV cache, no incremental
//! decode, no sampling — a single forward pass over the whole sequence.
//!
//! Spec (from `llama.cpp` `src/models/gemma-embedding.cpp` + the target GGUF):
//! - 24 layers, n_embd 768, FFN 1152, 3 Q heads / 1 KV head (GQA), head_dim 256.
//! - Per layer: `attn_norm → q/k/v proj → QK-norm (per-head, no V-norm) →
//!   RoPE(NEOX) → bidirectional attention (Q pre-scaled 1/√256) → O-proj →
//!   post_attention_norm → +residual → ffn_norm → gated-tanh-GELU →
//!   post_ffw_norm → +residual`. Then `output_norm → mean-pool → L2`.
//! - **Dual RoPE base**: global layers (il % 6 == 5) use `rope.freq_base` (1e6);
//!   the sliding-window layers fall back to the hardcoded `10000.0`.
//! - Mean pooling includes BOS and EOS. L2 normalization is applied here
//!   (llama.cpp does it as a post-graph CLI convention).
//!
//! NOTE (first cut): attention is fully bidirectional for every layer. This is
//! exact for sequences up to `sliding_window` tokens (≈256); windowed
//! bidirectional attention for longer inputs is a follow-up (see `is_global_layer`).

use std::collections::BTreeMap;
use std::sync::Arc;

use ocelotl_core::{OcelotlError, Result, RuntimeError, TokenId};
use ocelotl_kernels::KernelBackend;
use ocelotl_kernels::attention::scaled_dot_product_attention_bidirectional_with_scale;
use ocelotl_kernels::mlp::gelu_tanh_inplace;
use ocelotl_kernels::pooling::{l2_normalize, mean_pool};
use ocelotl_kernels::rmsnorm::rmsnorm;
use ocelotl_kernels::rope::rope_apply_inplace;
use ocelotl_loader::{
    GgufManifest, GgufMetadataValue, LoadedTensor, inspect_gguf, load_gguf_tensors_f32,
};

use crate::cpu_embedding_backend;

fn rt<S: Into<String>>(message: S) -> OcelotlError {
    OcelotlError::Runtime(RuntimeError {
        message: message.into(),
    })
}

/// EmbeddingGemma runtime config, parsed from the GGUF `gemma-embedding.*` keys.
#[derive(Debug, Clone)]
pub struct EmbeddingGemmaConfig {
    pub context_length: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,
    pub sliding_window: usize,
    pub rms_norm_eps: f32,
    pub rope_freq_base: f32,
    /// SWA layers use this; the GGUF omits a `_swa` key so it defaults to 10000.
    pub rope_freq_base_swa: f32,
}

/// Global-attention layers are every 6th (il % 6 == 5 → {5,11,17,23}); the rest
/// are sliding-window. (Only the RoPE base differs while seq ≤ window.)
fn is_global_layer(il: usize) -> bool {
    il % 6 == 5
}

fn md_u32(m: &GgufManifest, key: &str) -> Result<usize> {
    match m.metadata_value(key) {
        Some(GgufMetadataValue::U32(v)) => Ok(*v as usize),
        Some(GgufMetadataValue::U64(v)) => Ok(*v as usize),
        Some(GgufMetadataValue::I32(v)) if *v >= 0 => Ok(*v as usize),
        _ => Err(rt(format!("EmbeddingGemma GGUF missing u32 key `{key}`"))),
    }
}

fn md_f32(m: &GgufManifest, key: &str) -> Result<f32> {
    match m.metadata_value(key) {
        Some(GgufMetadataValue::F32(v)) => Ok(*v),
        _ => Err(rt(format!("EmbeddingGemma GGUF missing f32 key `{key}`"))),
    }
}

impl EmbeddingGemmaConfig {
    pub fn from_manifest(m: &GgufManifest) -> Result<Self> {
        Ok(Self {
            context_length: md_u32(m, "gemma-embedding.context_length")?,
            block_count: md_u32(m, "gemma-embedding.block_count")?,
            embedding_length: md_u32(m, "gemma-embedding.embedding_length")?,
            feed_forward_length: md_u32(m, "gemma-embedding.feed_forward_length")?,
            head_count: md_u32(m, "gemma-embedding.attention.head_count")?,
            head_count_kv: md_u32(m, "gemma-embedding.attention.head_count_kv")?,
            head_dim: md_u32(m, "gemma-embedding.attention.key_length")?,
            sliding_window: md_u32(m, "gemma-embedding.attention.sliding_window")?,
            rms_norm_eps: md_f32(m, "gemma-embedding.attention.layer_norm_rms_epsilon")?,
            rope_freq_base: md_f32(m, "gemma-embedding.rope.freq_base")?,
            rope_freq_base_swa: 10_000.0,
        })
    }
}

struct Layer {
    attn_norm: Vec<f32>,
    attn_q: Vec<f32>,
    attn_k: Vec<f32>,
    attn_v: Vec<f32>,
    attn_q_norm: Vec<f32>,
    attn_k_norm: Vec<f32>,
    attn_output: Vec<f32>,
    post_attention_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_down: Vec<f32>,
    post_ffw_norm: Vec<f32>,
}

/// A loaded EmbeddingGemma model. `embed` runs the encoder + mean-pool + L2.
pub struct EmbeddingGemmaModel {
    config: EmbeddingGemmaConfig,
    token_embd: Vec<f32>,
    output_norm: Vec<f32>,
    layers: Vec<Layer>,
    kernels: Arc<dyn KernelBackend>,
}

fn take(map: &mut BTreeMap<String, LoadedTensor>, name: &str) -> Result<Vec<f32>> {
    map.remove(name)
        .map(|t| t.values)
        .ok_or_else(|| rt(format!("EmbeddingGemma GGUF missing tensor `{name}`")))
}

/// Take a 2D linear weight in the raw GGUF `[out_features][in_features]` layout,
/// validating its length. This is exactly the layout `linear_out_by_in` (and
/// its AVX2 microkernel) consume — `out[o] = Σ_i x[i]·W[o,i]` — so no transpose
/// is needed (unlike the `matmul` path, which wants `[in][out]`).
fn take_linear(
    map: &mut BTreeMap<String, LoadedTensor>,
    name: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    let raw = take(map, name)?;
    if raw.len() != out_dim * in_dim {
        return Err(rt(format!(
            "EmbeddingGemma tensor `{name}` len {} != out*in = {out_dim}*{in_dim}",
            raw.len()
        )));
    }
    Ok(raw)
}

impl EmbeddingGemmaModel {
    /// Load an EmbeddingGemma F32 GGUF.
    pub fn load_from_gguf(path: &std::path::Path) -> Result<Self> {
        let manifest = inspect_gguf(path)?;
        let config = EmbeddingGemmaConfig::from_manifest(&manifest)?;

        // Collect every tensor name we need, load them all as F32.
        let mut names: Vec<String> = vec!["token_embd.weight".into(), "output_norm.weight".into()];
        for il in 0..config.block_count {
            for suffix in [
                "attn_norm",
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_q_norm",
                "attn_k_norm",
                "attn_output",
                "post_attention_norm",
                "ffn_norm",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
                "post_ffw_norm",
            ] {
                names.push(format!("blk.{il}.{suffix}.weight"));
            }
        }
        let loaded = load_gguf_tensors_f32(path, &names)?;
        let mut map: BTreeMap<String, LoadedTensor> =
            loaded.into_iter().map(|t| (t.name.clone(), t)).collect();

        let token_embd = take(&mut map, "token_embd.weight")?;
        let output_norm = take(&mut map, "output_norm.weight")?;
        // Dimensions for the raw [out][in] linear layouts.
        let h = config.embedding_length;
        let hd = config.head_dim;
        let q_dim = config.head_count * hd;
        let kv_dim = config.head_count_kv * hd;
        let f = config.feed_forward_length;
        let mut layers = Vec::with_capacity(config.block_count);
        for il in 0..config.block_count {
            let g = |s: &str| format!("blk.{il}.{s}.weight");
            // 1D norms load raw; 2D linears transpose GGUF [out][in] → [in][out].
            layers.push(Layer {
                attn_norm: take(&mut map, &g("attn_norm"))?,
                attn_q: take_linear(&mut map, &g("attn_q"), q_dim, h)?,
                attn_k: take_linear(&mut map, &g("attn_k"), kv_dim, h)?,
                attn_v: take_linear(&mut map, &g("attn_v"), kv_dim, h)?,
                attn_q_norm: take(&mut map, &g("attn_q_norm"))?,
                attn_k_norm: take(&mut map, &g("attn_k_norm"))?,
                attn_output: take_linear(&mut map, &g("attn_output"), h, q_dim)?,
                post_attention_norm: take(&mut map, &g("post_attention_norm"))?,
                ffn_norm: take(&mut map, &g("ffn_norm"))?,
                ffn_gate: take_linear(&mut map, &g("ffn_gate"), f, h)?,
                ffn_up: take_linear(&mut map, &g("ffn_up"), f, h)?,
                ffn_down: take_linear(&mut map, &g("ffn_down"), h, f)?,
                post_ffw_norm: take(&mut map, &g("post_ffw_norm"))?,
            });
        }

        Ok(Self {
            config,
            token_embd,
            output_norm,
            layers,
            kernels: cpu_embedding_backend(),
        })
    }

    pub fn config(&self) -> &EmbeddingGemmaConfig {
        &self.config
    }

    /// Run the encoder over `tokens` (which must already include BOS/EOS and any
    /// task prefix) and return the mean-pooled, L2-normalized embedding.
    pub fn embed(&self, tokens: &[TokenId]) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let seq = tokens.len();
        if seq == 0 {
            return Err(rt("EmbeddingGemma embed requires at least one token"));
        }
        let h = cfg.embedding_length;
        let hd = cfg.head_dim;
        let nq = cfg.head_count;
        let nkv = cfg.head_count_kv;
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;
        let f = cfg.feed_forward_length;
        let eps = cfg.rms_norm_eps;
        let attn_scale = 1.0_f32 / (hd as f32).sqrt();

        // Token embedding + Gemma sqrt(n_embd) scaling.
        let mut hidden = vec![0.0_f32; seq * h];
        for (pos, tok) in tokens.iter().enumerate() {
            let src = (tok.0 as usize) * h;
            if src + h > self.token_embd.len() {
                return Err(rt(format!("token id {} out of vocab range", tok.0)));
            }
            hidden[pos * h..pos * h + h].copy_from_slice(&self.token_embd[src..src + h]);
        }
        let embed_scale = (h as f32).sqrt();
        for x in &mut hidden {
            *x *= embed_scale;
        }

        // Reused scratch buffers.
        let mut norm = vec![0.0_f32; seq * h];
        let mut q = vec![0.0_f32; seq * q_dim];
        let mut k = vec![0.0_f32; seq * kv_dim];
        let mut v = vec![0.0_f32; seq * kv_dim];
        let mut qn = vec![0.0_f32; seq * q_dim];
        let mut kn = vec![0.0_f32; seq * kv_dim];
        let mut attn = vec![0.0_f32; seq * q_dim];
        let mut o = vec![0.0_f32; seq * h];
        let mut post = vec![0.0_f32; seq * h];
        let mut ffn_in = vec![0.0_f32; seq * h];
        let mut gate_buf = vec![0.0_f32; seq * f];
        let mut up_buf = vec![0.0_f32; seq * f];
        let mut mlp_out = vec![0.0_f32; seq * h];

        for (il, layer) in self.layers.iter().enumerate() {
            let theta = if is_global_layer(il) {
                cfg.rope_freq_base
            } else {
                cfg.rope_freq_base_swa
            };

            // --- attention sublayer ---
            // Projections use `linear_out_by_in` on the raw GGUF [out][in]
            // weights so the AVX2 microkernel (contiguous dot products) applies.
            rmsnorm(&hidden, seq, h, &layer.attn_norm, eps, &mut norm)?;
            let kern = self.kernels.as_ref();
            kern.linear_out_by_in(&norm, seq, h, &layer.attn_q, q_dim, None, &mut q)?;
            kern.linear_out_by_in(&norm, seq, h, &layer.attn_k, kv_dim, None, &mut k)?;
            kern.linear_out_by_in(&norm, seq, h, &layer.attn_v, kv_dim, None, &mut v)?;
            // QK-norm: per-head RMSNorm over head_dim, weight shared across heads.
            rmsnorm(&q, seq * nq, hd, &layer.attn_q_norm, eps, &mut qn)?;
            rmsnorm(&k, seq * nkv, hd, &layer.attn_k_norm, eps, &mut kn)?;
            // RoPE (NEOX) per (position, head).
            for pos in 0..seq {
                for head in 0..nq {
                    let base = pos * q_dim + head * hd;
                    rope_apply_inplace(&mut qn[base..base + hd], hd, pos, theta)?;
                }
                for head in 0..nkv {
                    let base = pos * kv_dim + head * hd;
                    rope_apply_inplace(&mut kn[base..base + hd], hd, pos, theta)?;
                }
            }
            // Bidirectional attention (Q pre-scaled via explicit scale).
            scaled_dot_product_attention_bidirectional_with_scale(
                &qn, &kn, &v, seq, nq, nkv, hd, attn_scale, &mut attn,
            )?;
            kern.linear_out_by_in(&attn, seq, q_dim, &layer.attn_output, h, None, &mut o)?;
            rmsnorm(&o, seq, h, &layer.post_attention_norm, eps, &mut post)?;
            for i in 0..seq * h {
                hidden[i] += post[i];
            }

            // --- FFN sublayer (gated-tanh-GELU, [out][in] projections) ---
            rmsnorm(&hidden, seq, h, &layer.ffn_norm, eps, &mut ffn_in)?;
            kern.linear_out_by_in(&ffn_in, seq, h, &layer.ffn_gate, f, None, &mut gate_buf)?;
            kern.linear_out_by_in(&ffn_in, seq, h, &layer.ffn_up, f, None, &mut up_buf)?;
            gelu_tanh_inplace(&mut gate_buf);
            for (g, u) in gate_buf.iter_mut().zip(up_buf.iter()) {
                *g *= *u;
            }
            kern.linear_out_by_in(&gate_buf, seq, f, &layer.ffn_down, h, None, &mut mlp_out)?;
            rmsnorm(&mlp_out, seq, h, &layer.post_ffw_norm, eps, &mut post)?;
            for i in 0..seq * h {
                hidden[i] += post[i];
            }
        }

        // Final norm → mean-pool (includes BOS/EOS) → L2.
        rmsnorm(&hidden, seq, h, &self.output_norm, eps, &mut norm)?;
        let mut emb = mean_pool(&norm, seq, h)?;
        l2_normalize(&mut emb);
        Ok(emb)
    }
}

// ---------------------------------------------------------------------------
// Device-resident (GPU) forward. Uploads all weights once, chains every layer
// on-device via the `_d` kernels (rmsnorm/linear/rope/GQA-expand/encoder-attn/
// gelu/mul/add), and only reads back the final pre-pool hidden state. Mean-pool
// and L2 stay on host (a single small vector). This is the batched-friendly
// path the design note argued for: the encoder is a stack of GEMMs.
// ---------------------------------------------------------------------------
#[cfg(feature = "cubecl-wgpu")]
mod gpu {
    use super::*;
    use ocelotl_kernels::{CubeClKernelBackend, DeviceTensor, KernelBackend};

    /// Per-position NEOX cos/sin tables `[n_positions * head_dim/2]`.
    fn rope_tables(head_dim: usize, n_positions: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(n_positions * half);
        let mut sin = Vec::with_capacity(n_positions * half);
        for pos in 0..n_positions {
            for i in 0..half {
                let inv_freq = theta.powf(-2.0 * (i as f32) / head_dim as f32);
                let angle = pos as f32 * inv_freq;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        (cos, sin)
    }

    struct GpuLayer {
        attn_norm: DeviceTensor,
        attn_q: DeviceTensor,
        attn_k: DeviceTensor,
        attn_v: DeviceTensor,
        attn_q_norm: DeviceTensor,
        attn_k_norm: DeviceTensor,
        attn_output: DeviceTensor,
        post_attention_norm: DeviceTensor,
        ffn_norm: DeviceTensor,
        ffn_gate: DeviceTensor,
        ffn_up: DeviceTensor,
        ffn_down: DeviceTensor,
        post_ffw_norm: DeviceTensor,
    }

    /// EmbeddingGemma with all weights resident on the GPU.
    pub struct EmbeddingGemmaGpu {
        config: EmbeddingGemmaConfig,
        backend: CubeClKernelBackend,
        token_embd: Vec<f32>,
        output_norm: DeviceTensor,
        layers: Vec<GpuLayer>,
    }

    impl EmbeddingGemmaGpu {
        /// Load the GGUF on CPU, then upload every weight to GPU ordinal 0.
        pub fn load_from_gguf(path: &std::path::Path) -> Result<Self> {
            let cpu = EmbeddingGemmaModel::load_from_gguf(path)?;
            let backend = CubeClKernelBackend::new_gpu(0);
            let up = |v: &[f32]| backend.upload(v);
            let output_norm = up(&cpu.output_norm)?;
            let mut layers = Vec::with_capacity(cpu.layers.len());
            for l in &cpu.layers {
                layers.push(GpuLayer {
                    attn_norm: up(&l.attn_norm)?,
                    attn_q: up(&l.attn_q)?,
                    attn_k: up(&l.attn_k)?,
                    attn_v: up(&l.attn_v)?,
                    attn_q_norm: up(&l.attn_q_norm)?,
                    attn_k_norm: up(&l.attn_k_norm)?,
                    attn_output: up(&l.attn_output)?,
                    post_attention_norm: up(&l.post_attention_norm)?,
                    ffn_norm: up(&l.ffn_norm)?,
                    ffn_gate: up(&l.ffn_gate)?,
                    ffn_up: up(&l.ffn_up)?,
                    ffn_down: up(&l.ffn_down)?,
                    post_ffw_norm: up(&l.post_ffw_norm)?,
                });
            }
            Ok(Self {
                config: cpu.config,
                backend,
                token_embd: cpu.token_embd,
                output_norm,
                layers,
            })
        }

        pub fn config(&self) -> &EmbeddingGemmaConfig {
            &self.config
        }

        /// Device-resident encoder forward → mean-pooled, L2-normalized vector.
        pub fn embed(&self, tokens: &[TokenId]) -> Result<Vec<f32>> {
            let cfg = &self.config;
            let seq = tokens.len();
            if seq == 0 {
                return Err(rt("EmbeddingGemma embed requires at least one token"));
            }
            let h = cfg.embedding_length;
            let hd = cfg.head_dim;
            let nq = cfg.head_count;
            let nkv = cfg.head_count_kv;
            let q_dim = nq * hd;
            let kv_dim = nkv * hd;
            let f = cfg.feed_forward_length;
            let eps = cfg.rms_norm_eps;
            let scale = 1.0_f32 / (hd as f32).sqrt();
            let b = &self.backend;

            // Token gather + Gemma √h scale on host, then upload.
            let mut hidden_host = vec![0.0_f32; seq * h];
            for (pos, tok) in tokens.iter().enumerate() {
                let src = (tok.0 as usize) * h;
                if src + h > self.token_embd.len() {
                    return Err(rt(format!("token id {} out of vocab range", tok.0)));
                }
                hidden_host[pos * h..pos * h + h].copy_from_slice(&self.token_embd[src..src + h]);
            }
            let embed_scale = (h as f32).sqrt();
            for x in &mut hidden_host {
                *x *= embed_scale;
            }
            let hidden = b.upload(&hidden_host)?;

            // Dual-base RoPE tables (global vs sliding-window layers).
            let (cg, sg) = rope_tables(hd, seq, cfg.rope_freq_base);
            let (cs, ss) = rope_tables(hd, seq, cfg.rope_freq_base_swa);
            let (cg, sg) = (b.upload(&cg)?, b.upload(&sg)?);
            let (cs, ss) = (b.upload(&cs)?, b.upload(&ss)?);

            // Reusable device scratch.
            let norm = b.alloc(seq * h)?;
            let q = b.alloc(seq * q_dim)?;
            let k = b.alloc(seq * kv_dim)?;
            let v = b.alloc(seq * kv_dim)?;
            let qn = b.alloc(seq * q_dim)?;
            let kn = b.alloc(seq * kv_dim)?;
            let kn_exp = b.alloc(seq * q_dim)?;
            let v_exp = b.alloc(seq * q_dim)?;
            let attn = b.alloc(seq * q_dim)?;
            let o = b.alloc(seq * h)?;
            let post = b.alloc(seq * h)?;
            let ffn_in = b.alloc(seq * h)?;
            let gate = b.alloc(seq * f)?;
            let up = b.alloc(seq * f)?;
            let mlp_out = b.alloc(seq * h)?;

            for (il, layer) in self.layers.iter().enumerate() {
                let (cos_d, sin_d) = if is_global_layer(il) {
                    (&cg, &sg)
                } else {
                    (&cs, &ss)
                };
                // Attention.
                b.rmsnorm_d(&hidden, seq, h, &layer.attn_norm, eps, &norm)?;
                b.linear_d(&norm, seq, h, &layer.attn_q, q_dim, None, &q)?;
                b.linear_d(&norm, seq, h, &layer.attn_k, kv_dim, None, &k)?;
                b.linear_d(&norm, seq, h, &layer.attn_v, kv_dim, None, &v)?;
                b.rmsnorm_d(&q, seq * nq, hd, &layer.attn_q_norm, eps, &qn)?;
                b.rmsnorm_d(&k, seq * nkv, hd, &layer.attn_k_norm, eps, &kn)?;
                b.rope_tables_d(&qn, cos_d, sin_d, hd, nq)?;
                b.rope_tables_d(&kn, cos_d, sin_d, hd, nkv)?;
                b.expand_kv_heads_d(&kn, &kn_exp, hd, nq, nkv)?;
                b.expand_kv_heads_d(&v, &v_exp, hd, nq, nkv)?;
                b.attention_encoder_d(&qn, &kn_exp, &v_exp, seq, nq, hd, scale, &attn)?;
                b.linear_d(&attn, seq, q_dim, &layer.attn_output, h, None, &o)?;
                b.rmsnorm_d(&o, seq, h, &layer.post_attention_norm, eps, &post)?;
                b.add_inplace_d(&hidden, &post)?;
                // FFN (gated-tanh-GELU).
                b.rmsnorm_d(&hidden, seq, h, &layer.ffn_norm, eps, &ffn_in)?;
                b.linear_d(&ffn_in, seq, h, &layer.ffn_gate, f, None, &gate)?;
                b.linear_d(&ffn_in, seq, h, &layer.ffn_up, f, None, &up)?;
                b.gelu_inplace_d(&gate)?;
                b.mul_inplace_d(&gate, &up)?;
                b.linear_d(&gate, seq, f, &layer.ffn_down, h, None, &mlp_out)?;
                b.rmsnorm_d(&mlp_out, seq, h, &layer.post_ffw_norm, eps, &post)?;
                b.add_inplace_d(&hidden, &post)?;
            }

            b.rmsnorm_d(&hidden, seq, h, &self.output_norm, eps, &norm)?;
            let normed = norm.to_host_owned()?;
            let mut emb = mean_pool(&normed, seq, h)?;
            l2_normalize(&mut emb);
            Ok(emb)
        }
    }
}

#[cfg(feature = "cubecl-wgpu")]
pub use gpu::EmbeddingGemmaGpu;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires OCELOTL_EMBGEMMA_GGUF; dumps the q1 embedding for external cosine vs llama-embedding"]
    fn embeddinggemma_dump_q1_embedding() {
        let gguf = std::env::var("OCELOTL_EMBGEMMA_GGUF")
            .expect("set OCELOTL_EMBGEMMA_GGUF to the EmbeddingGemma F32 GGUF");
        let model = EmbeddingGemmaModel::load_from_gguf(std::path::Path::new(&gguf))
            .expect("EmbeddingGemma must load");
        // Token IDs come from `llama-tokenize` on a full_prompt ([BOS ... EOS]),
        // isolating the embedding math from tokenization. Default = q1
        // ("task: search result | query: How do I reset my forgotten password?");
        // override with OCELOTL_EMBGEMMA_TOKENS="2 8071 ..." to drive any prompt.
        let default_q1 =
            "2 8071 236787 3927 1354 1109 7609 236787 2088 776 564 14724 1041 27971 8918 236881 1";
        let tok_str =
            std::env::var("OCELOTL_EMBGEMMA_TOKENS").unwrap_or_else(|_| default_q1.into());
        let toks: Vec<TokenId> = tok_str
            .split_whitespace()
            .map(|x| TokenId(x.parse().expect("token id must be u32")))
            .collect();
        let emb = model.embed(&toks).expect("embed must run");
        assert_eq!(emb.len(), model.config().embedding_length);
        eprintln!(
            "EMBGEMMA_Q1 {}",
            emb.iter()
                .map(|x| format!("{x:.6}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    /// Load-once, time-many CPU throughput bench. `OCELOTL_BENCH_TOKENS` points
    /// at a file with one space-separated token-id line per prompt; the corpus
    /// is repeated to `OCELOTL_BENCH_ITERS` (default 200) embeds.
    #[test]
    #[ignore = "requires OCELOTL_EMBGEMMA_GGUF + OCELOTL_BENCH_TOKENS; CPU throughput bench"]
    fn embeddinggemma_bench() {
        let gguf = std::env::var("OCELOTL_EMBGEMMA_GGUF").expect("set OCELOTL_EMBGEMMA_GGUF");
        let tokens_path = std::env::var("OCELOTL_BENCH_TOKENS").expect("set OCELOTL_BENCH_TOKENS");
        let iters: usize = std::env::var("OCELOTL_BENCH_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(200);
        let prompts: Vec<Vec<TokenId>> = std::fs::read_to_string(&tokens_path)
            .expect("read tokens file")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                l.split_whitespace()
                    .map(|x| TokenId(x.parse().unwrap()))
                    .collect()
            })
            .collect();
        let parallel = std::env::var("OCELOTL_BENCH_PARALLEL").is_ok();
        let model = EmbeddingGemmaModel::load_from_gguf(std::path::Path::new(&gguf))
            .expect("model must load");
        // Warm up.
        let _ = model.embed(&prompts[0]).unwrap();
        let total_tokens: usize = (0..iters).map(|i| prompts[i % prompts.len()].len()).sum();
        let start = std::time::Instant::now();
        if parallel {
            use rayon::prelude::*;
            (0..iters).into_par_iter().for_each(|i| {
                let _ = model.embed(&prompts[i % prompts.len()]).expect("embed");
            });
        } else {
            for i in 0..iters {
                let _ = model.embed(&prompts[i % prompts.len()]).expect("embed");
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        eprintln!(
            "EMBGEMMA_BENCH parallel={parallel} embeds={iters} elapsed_s={elapsed:.3} ms_per_embed={:.3} embeds_per_s={:.1} tokens_per_s={:.1}",
            elapsed / iters as f64 * 1000.0,
            iters as f64 / elapsed,
            total_tokens as f64 / elapsed,
        );
    }

    /// GPU device-resident embedding dump, for external cosine vs the reference.
    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    #[ignore = "requires OCELOTL_EMBGEMMA_GGUF + a WGPU GPU; dumps the GPU embedding"]
    fn embeddinggemma_gpu_dump_embedding() {
        let gguf = std::env::var("OCELOTL_EMBGEMMA_GGUF").expect("set OCELOTL_EMBGEMMA_GGUF");
        let model = EmbeddingGemmaGpu::load_from_gguf(std::path::Path::new(&gguf))
            .expect("GPU EmbeddingGemma must load");
        let default_q1 =
            "2 8071 236787 3927 1354 1109 7609 236787 2088 776 564 14724 1041 27971 8918 236881 1";
        let tok_str =
            std::env::var("OCELOTL_EMBGEMMA_TOKENS").unwrap_or_else(|_| default_q1.into());
        let toks: Vec<TokenId> = tok_str
            .split_whitespace()
            .map(|x| TokenId(x.parse().expect("token id must be u32")))
            .collect();
        let emb = model.embed(&toks).expect("gpu embed must run");
        assert_eq!(emb.len(), model.config().embedding_length);
        eprintln!(
            "EMBGEMMA_GPU {}",
            emb.iter()
                .map(|x| format!("{x:.6}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    /// GPU throughput bench (load-once, time-many). Weights resident on device.
    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    #[ignore = "requires OCELOTL_EMBGEMMA_GGUF + OCELOTL_BENCH_TOKENS + a WGPU GPU"]
    fn embeddinggemma_gpu_bench() {
        let gguf = std::env::var("OCELOTL_EMBGEMMA_GGUF").expect("set OCELOTL_EMBGEMMA_GGUF");
        let tokens_path = std::env::var("OCELOTL_BENCH_TOKENS").expect("set OCELOTL_BENCH_TOKENS");
        let iters: usize = std::env::var("OCELOTL_BENCH_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64);
        let prompts: Vec<Vec<TokenId>> = std::fs::read_to_string(&tokens_path)
            .expect("read tokens file")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                l.split_whitespace()
                    .map(|x| TokenId(x.parse().unwrap()))
                    .collect()
            })
            .collect();
        let model = EmbeddingGemmaGpu::load_from_gguf(std::path::Path::new(&gguf))
            .expect("model must load");
        let _ = model.embed(&prompts[0]).unwrap();
        let total_tokens: usize = (0..iters).map(|i| prompts[i % prompts.len()].len()).sum();
        let start = std::time::Instant::now();
        for i in 0..iters {
            let _ = model.embed(&prompts[i % prompts.len()]).expect("embed");
        }
        let elapsed = start.elapsed().as_secs_f64();
        eprintln!(
            "EMBGEMMA_GPU_BENCH embeds={iters} elapsed_s={elapsed:.3} ms_per_embed={:.3} embeds_per_s={:.1} tokens_per_s={:.1}",
            elapsed / iters as f64 * 1000.0,
            iters as f64 / elapsed,
            total_tokens as f64 / elapsed,
        );
    }
}
