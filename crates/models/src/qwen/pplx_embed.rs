//! pplx-embed-v1 — Perplexity's **Qwen3-based bidirectional text encoder**.
//!
//! A Qwen3 backbone (the GGUF carries `general.architecture = qwen3`) run
//! *non-causally* (`qwen3.attention.causal = false`) as an embedding model:
//! one bidirectional forward pass, mean-pool over all tokens, L2-normalize.
//! There is no KV cache, no decode loop, no sampling.
//!
//! It reuses the same kernels as [`crate::gemma::embedding`] — RMSNorm, NEOX
//! RoPE, per-head QK-norm, the bidirectional attention kernel, mean-pool, L2 —
//! but the Qwen3 block is *structurally simpler* than EmbeddingGemma's Gemma-3
//! block:
//! - **No sliding window** — every layer is global attention with a single RoPE
//!   base (`qwen3.rope.freq_base`, 1e6).
//! - **No Gemma "sandwich" norms** — a Qwen3 block is a plain pre-norm
//!   transformer: `h += attn(norm(h))` then `h += mlp(norm(h))`, with no
//!   post-attention / post-FFN normalization on the residual branch.
//! - **SwiGLU MLP** (`silu(gate)·up`) rather than gated-tanh-GELU.
//! - **No embedding scale** — Qwen does not multiply token embeddings by
//!   √n_embd (Gemma does).
//! - **Decoupled head_dim** — `key_length = value_length = 128` while
//!   `embedding_length = 1024`, so `q_dim = 16·128 = 2048 ≠ n_embd`.
//! - Per-head **QK-norm** (RMSNorm over head_dim, weight shared across heads);
//!   no QKV bias (Qwen3 dropped the Qwen2 biases).
//!
//! Parity is validated against `llama.cpp` `llama-embedding` on the same GGUF
//! (which honors `causal = false`), cosine/ranking rather than bit-exact.

use std::collections::BTreeMap;
use std::sync::Arc;

use ocelotl_core::{OcelotlError, Result, RuntimeError, TokenId};
use ocelotl_kernels::KernelBackend;
use ocelotl_kernels::attention::scaled_dot_product_attention_bidirectional_with_scale;
use ocelotl_kernels::mlp::silu_inplace;
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

/// pplx-embed runtime config, parsed from the GGUF `qwen3.*` keys.
#[derive(Debug, Clone)]
pub struct PplxEmbedConfig {
    pub context_length: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_freq_base: f32,
}

fn md_u32(m: &GgufManifest, key: &str) -> Result<usize> {
    match m.metadata_value(key) {
        Some(GgufMetadataValue::U32(v)) => Ok(*v as usize),
        Some(GgufMetadataValue::U64(v)) => Ok(*v as usize),
        Some(GgufMetadataValue::I32(v)) if *v >= 0 => Ok(*v as usize),
        _ => Err(rt(format!("pplx-embed GGUF missing u32 key `{key}`"))),
    }
}

fn md_f32(m: &GgufManifest, key: &str) -> Result<f32> {
    match m.metadata_value(key) {
        Some(GgufMetadataValue::F32(v)) => Ok(*v),
        _ => Err(rt(format!("pplx-embed GGUF missing f32 key `{key}`"))),
    }
}

impl PplxEmbedConfig {
    pub fn from_manifest(m: &GgufManifest) -> Result<Self> {
        Ok(Self {
            context_length: md_u32(m, "qwen3.context_length")?,
            block_count: md_u32(m, "qwen3.block_count")?,
            embedding_length: md_u32(m, "qwen3.embedding_length")?,
            feed_forward_length: md_u32(m, "qwen3.feed_forward_length")?,
            head_count: md_u32(m, "qwen3.attention.head_count")?,
            head_count_kv: md_u32(m, "qwen3.attention.head_count_kv")?,
            head_dim: md_u32(m, "qwen3.attention.key_length")?,
            rms_norm_eps: md_f32(m, "qwen3.attention.layer_norm_rms_epsilon")?,
            rope_freq_base: md_f32(m, "qwen3.rope.freq_base")?,
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
    ffn_norm: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_down: Vec<f32>,
}

/// A loaded pplx-embed model. `embed` runs the encoder + mean-pool + L2.
pub struct PplxEmbedModel {
    config: PplxEmbedConfig,
    token_embd: Vec<f32>,
    output_norm: Vec<f32>,
    layers: Vec<Layer>,
    kernels: Arc<dyn KernelBackend>,
}

fn take(map: &mut BTreeMap<String, LoadedTensor>, name: &str) -> Result<Vec<f32>> {
    map.remove(name)
        .map(|t| t.values)
        .ok_or_else(|| rt(format!("pplx-embed GGUF missing tensor `{name}`")))
}

/// Take a 2D linear weight in the raw GGUF `[out_features][in_features]` layout,
/// validating its length. `linear_out_by_in` (and its AVX2 microkernel) consume
/// this layout directly — `out[o] = Σ_i x[i]·W[o,i]` — so no transpose is needed.
fn take_linear(
    map: &mut BTreeMap<String, LoadedTensor>,
    name: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    let raw = take(map, name)?;
    if raw.len() != out_dim * in_dim {
        return Err(rt(format!(
            "pplx-embed tensor `{name}` len {} != out*in = {out_dim}*{in_dim}",
            raw.len()
        )));
    }
    Ok(raw)
}

impl PplxEmbedModel {
    /// Load a pplx-embed GGUF (F16/F32 dense tensors).
    pub fn load_from_gguf(path: &std::path::Path) -> Result<Self> {
        let manifest = inspect_gguf(path)?;
        let config = PplxEmbedConfig::from_manifest(&manifest)?;

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
                "ffn_norm",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ] {
                names.push(format!("blk.{il}.{suffix}.weight"));
            }
        }
        let loaded = load_gguf_tensors_f32(path, &names)?;
        let mut map: BTreeMap<String, LoadedTensor> =
            loaded.into_iter().map(|t| (t.name.clone(), t)).collect();

        let token_embd = take(&mut map, "token_embd.weight")?;
        let output_norm = take(&mut map, "output_norm.weight")?;
        let h = config.embedding_length;
        let hd = config.head_dim;
        let q_dim = config.head_count * hd;
        let kv_dim = config.head_count_kv * hd;
        let f = config.feed_forward_length;
        let mut layers = Vec::with_capacity(config.block_count);
        for il in 0..config.block_count {
            let g = |s: &str| format!("blk.{il}.{s}.weight");
            layers.push(Layer {
                attn_norm: take(&mut map, &g("attn_norm"))?,
                attn_q: take_linear(&mut map, &g("attn_q"), q_dim, h)?,
                attn_k: take_linear(&mut map, &g("attn_k"), kv_dim, h)?,
                attn_v: take_linear(&mut map, &g("attn_v"), kv_dim, h)?,
                attn_q_norm: take(&mut map, &g("attn_q_norm"))?,
                attn_k_norm: take(&mut map, &g("attn_k_norm"))?,
                attn_output: take_linear(&mut map, &g("attn_output"), h, q_dim)?,
                ffn_norm: take(&mut map, &g("ffn_norm"))?,
                ffn_gate: take_linear(&mut map, &g("ffn_gate"), f, h)?,
                ffn_up: take_linear(&mut map, &g("ffn_up"), f, h)?,
                ffn_down: take_linear(&mut map, &g("ffn_down"), h, f)?,
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

    pub fn config(&self) -> &PplxEmbedConfig {
        &self.config
    }

    /// Run the encoder over `tokens` and return the mean-pooled, L2-normalized
    /// embedding. `tokens` must already carry whatever BOS/EOS the reference
    /// tokenizer emits (Qwen3: `add_bos = false`).
    pub fn embed(&self, tokens: &[TokenId]) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let seq = tokens.len();
        if seq == 0 {
            return Err(rt("pplx-embed embed requires at least one token"));
        }
        let h = cfg.embedding_length;
        let hd = cfg.head_dim;
        let nq = cfg.head_count;
        let nkv = cfg.head_count_kv;
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;
        let f = cfg.feed_forward_length;
        let eps = cfg.rms_norm_eps;
        let theta = cfg.rope_freq_base;
        let attn_scale = 1.0_f32 / (hd as f32).sqrt();

        // Token embedding — no Gemma-style √n_embd scaling for Qwen.
        let mut hidden = vec![0.0_f32; seq * h];
        for (pos, tok) in tokens.iter().enumerate() {
            let src = (tok.0 as usize) * h;
            if src + h > self.token_embd.len() {
                return Err(rt(format!("token id {} out of vocab range", tok.0)));
            }
            hidden[pos * h..pos * h + h].copy_from_slice(&self.token_embd[src..src + h]);
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
        let mut ffn_in = vec![0.0_f32; seq * h];
        let mut gate_buf = vec![0.0_f32; seq * f];
        let mut up_buf = vec![0.0_f32; seq * f];
        let mut mlp_out = vec![0.0_f32; seq * h];

        for layer in &self.layers {
            // --- attention sublayer (pre-norm, no post-norm) ---
            // Projections use `linear_out_by_in` on the raw GGUF [out][in]
            // weights so the AVX2 microkernel applies.
            rmsnorm(&hidden, seq, h, &layer.attn_norm, eps, &mut norm)?;
            let kern = self.kernels.as_ref();
            kern.linear_out_by_in(&norm, seq, h, &layer.attn_q, q_dim, None, &mut q)?;
            kern.linear_out_by_in(&norm, seq, h, &layer.attn_k, kv_dim, None, &mut k)?;
            kern.linear_out_by_in(&norm, seq, h, &layer.attn_v, kv_dim, None, &mut v)?;
            // QK-norm: per-head RMSNorm over head_dim, weight shared across heads.
            rmsnorm(&q, seq * nq, hd, &layer.attn_q_norm, eps, &mut qn)?;
            rmsnorm(&k, seq * nkv, hd, &layer.attn_k_norm, eps, &mut kn)?;
            // RoPE (NEOX) per (position, head); single global base.
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
            scaled_dot_product_attention_bidirectional_with_scale(
                &qn, &kn, &v, seq, nq, nkv, hd, attn_scale, &mut attn,
            )?;
            kern.linear_out_by_in(&attn, seq, q_dim, &layer.attn_output, h, None, &mut o)?;
            for i in 0..seq * h {
                hidden[i] += o[i];
            }

            // --- FFN sublayer (SwiGLU, pre-norm, no post-norm, [out][in]) ---
            rmsnorm(&hidden, seq, h, &layer.ffn_norm, eps, &mut ffn_in)?;
            kern.linear_out_by_in(&ffn_in, seq, h, &layer.ffn_gate, f, None, &mut gate_buf)?;
            kern.linear_out_by_in(&ffn_in, seq, h, &layer.ffn_up, f, None, &mut up_buf)?;
            silu_inplace(&mut gate_buf);
            for (g, u) in gate_buf.iter_mut().zip(up_buf.iter()) {
                *g *= *u;
            }
            kern.linear_out_by_in(&gate_buf, seq, f, &layer.ffn_down, h, None, &mut mlp_out)?;
            for i in 0..seq * h {
                hidden[i] += mlp_out[i];
            }
        }

        // Final norm → mean-pool (all tokens) → L2.
        rmsnorm(&hidden, seq, h, &self.output_norm, eps, &mut norm)?;
        let mut emb = mean_pool(&norm, seq, h)?;
        l2_normalize(&mut emb);
        Ok(emb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires OCELOTL_PPLX_GGUF; dumps an embedding for external cosine vs llama-embedding"]
    fn pplx_embed_dump_embedding() {
        let gguf = std::env::var("OCELOTL_PPLX_GGUF")
            .expect("set OCELOTL_PPLX_GGUF to the pplx-embed F16 GGUF");
        let model = PplxEmbedModel::load_from_gguf(std::path::Path::new(&gguf))
            .expect("pplx-embed must load");
        // Token IDs come from `llama-tokenize` on the prompt (Qwen3: add_bos=false).
        // Override with OCELOTL_PPLX_TOKENS="151643 ..." to drive any prompt.
        let tok_str = std::env::var("OCELOTL_PPLX_TOKENS").expect("set OCELOTL_PPLX_TOKENS");
        let toks: Vec<TokenId> = tok_str
            .split_whitespace()
            .map(|x| TokenId(x.parse().expect("token id must be u32")))
            .collect();
        let emb = model.embed(&toks).expect("embed must run");
        assert_eq!(emb.len(), model.config().embedding_length);
        eprintln!(
            "PPLX_EMB {}",
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
    #[ignore = "requires OCELOTL_PPLX_GGUF + OCELOTL_BENCH_TOKENS; CPU throughput bench"]
    fn pplx_embed_bench() {
        let gguf = std::env::var("OCELOTL_PPLX_GGUF").expect("set OCELOTL_PPLX_GGUF");
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
        let model =
            PplxEmbedModel::load_from_gguf(std::path::Path::new(&gguf)).expect("model must load");
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
            "PPLX_BENCH parallel={parallel} embeds={iters} elapsed_s={elapsed:.3} ms_per_embed={:.3} embeds_per_s={:.1} tokens_per_s={:.1}",
            elapsed / iters as f64 * 1000.0,
            iters as f64 / elapsed,
            total_tokens as f64 / elapsed,
        );
    }
}
