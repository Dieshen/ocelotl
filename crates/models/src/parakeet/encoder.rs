//! Parakeet FastConformer encoder block and stack.
//!
//! One block, in the order the exported graph runs it (five LayerNorms, not two
//! — the "macaron" sandwich):
//!
//! ```text
//! x += 0.5 * feed_forward1(norm_feed_forward1(x))
//! x +=       self_attn    (norm_self_att      (x), pos)
//! x +=       conv         (norm_conv          (x))
//! x += 0.5 * feed_forward2(norm_feed_forward2(x))
//! out = norm_out(x)
//! ```
//!
//! The checkpoint carries **no biases** on any of the projections or FFN linears
//! (`use_bias: false`), only on the norms and the convolution module's pointwise
//! stages — so every `linear_out_by_in` here passes `None`.

use ocelotl_core::{OcelotlError, Result, RuntimeError};
use ocelotl_kernels::activation::{add_scaled_inplace, glu_halves};
use ocelotl_kernels::conv::{batch_norm_inference, depthwise_conv1d};
use ocelotl_kernels::mlp::silu_inplace;
use ocelotl_kernels::relpos::rel_shift;
use ocelotl_kernels::{KernelBackend, layer_norm_whisper_scalar, transpose_2d};

/// LayerNorm epsilon used throughout the encoder.
pub const LN_EPS: f32 = 1e-5;
/// Conformer convolution-module depthwise kernel.
pub const CONV_KERNEL: usize = 9;
/// BatchNorm epsilon (PyTorch default).
pub const BN_EPS: f32 = 1e-5;

fn rt<S: Into<String>>(m: S) -> OcelotlError {
    OcelotlError::Runtime(RuntimeError { message: m.into() })
}

/// Weights for one Conformer block, in the checkpoint's native layouts.
#[derive(Debug, Clone, Default)]
pub struct BlockWeights {
    pub norm_ff1_w: Vec<f32>,
    pub norm_ff1_b: Vec<f32>,
    pub ff1_l1: Vec<f32>,
    pub ff1_l2: Vec<f32>,
    pub norm_attn_w: Vec<f32>,
    pub norm_attn_b: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub pos_proj: Vec<f32>,
    /// `[heads][head_dim]`, one pair **per block** (`untie_biases: true`).
    pub bias_u: Vec<f32>,
    pub bias_v: Vec<f32>,
    pub norm_conv_w: Vec<f32>,
    pub norm_conv_b: Vec<f32>,
    pub pw1: Vec<f32>,
    pub dw: Vec<f32>,
    pub bn_mean: Vec<f32>,
    pub bn_var: Vec<f32>,
    pub bn_w: Vec<f32>,
    pub bn_b: Vec<f32>,
    pub pw2: Vec<f32>,
    pub norm_ff2_w: Vec<f32>,
    pub norm_ff2_b: Vec<f32>,
    pub ff2_l1: Vec<f32>,
    pub ff2_l2: Vec<f32>,
    pub norm_out_w: Vec<f32>,
    pub norm_out_b: Vec<f32>,
}

/// Encoder shape constants.
#[derive(Debug, Clone, Copy)]
pub struct EncoderShape {
    pub d_model: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
}

impl Default for EncoderShape {
    fn default() -> Self {
        Self {
            d_model: 1024,
            heads: 8,
            head_dim: 128,
            ffn: 4096,
        }
    }
}

/// Macaron half-step FFN: `linear1 -> SiLU -> linear2`, no biases.
fn feed_forward(
    x: &[f32],
    rows: usize,
    shape: EncoderShape,
    l1: &[f32],
    l2: &[f32],
    kernels: &dyn KernelBackend,
) -> Result<Vec<f32>> {
    let mut hidden = vec![0.0_f32; rows * shape.ffn];
    kernels.linear_out_by_in(x, rows, shape.d_model, l1, shape.ffn, None, &mut hidden)?;
    silu_inplace(&mut hidden);
    let mut out = vec![0.0_f32; rows * shape.d_model];
    kernels.linear_out_by_in(&hidden, rows, shape.ffn, l2, shape.d_model, None, &mut out)?;
    Ok(out)
}

/// Relative-position multi-head self-attention (Transformer-XL form).
///
/// `score = ((q + bias_u)·kᵀ + rel_shift((q + bias_v)·pᵀ)) / √head_dim`
///
/// `pos` is the `[2T-1, d_model]` sinusoidal table; `pos_proj` is the block's
/// `relative_k_proj`. `bias_u`/`bias_v` are **per block** — a shared pair loads
/// without complaint and silently changes every score, which is why they are
/// stored per `BlockWeights` rather than once for the stack.
fn self_attention(
    x: &[f32],
    rows: usize,
    pos: &[f32],
    shape: EncoderShape,
    w: &BlockWeights,
    kernels: &dyn KernelBackend,
) -> Result<Vec<f32>> {
    let (d, h, hd) = (shape.d_model, shape.heads, shape.head_dim);
    let pos_rows = 2 * rows - 1;
    let mut q = vec![0.0_f32; rows * d];
    let mut k = vec![0.0_f32; rows * d];
    let mut v = vec![0.0_f32; rows * d];
    kernels.linear_out_by_in(x, rows, d, &w.q_proj, d, None, &mut q)?;
    kernels.linear_out_by_in(x, rows, d, &w.k_proj, d, None, &mut k)?;
    kernels.linear_out_by_in(x, rows, d, &w.v_proj, d, None, &mut v)?;
    let mut p = vec![0.0_f32; pos_rows * d];
    kernels.linear_out_by_in(pos, pos_rows, d, &w.pos_proj, d, None, &mut p)?;

    let scale = 1.0_f32 / (hd as f32).sqrt();
    let mut context = vec![0.0_f32; rows * d];
    let mut ac = vec![0.0_f32; rows * rows];
    let mut bd_wide = vec![0.0_f32; rows * pos_rows];
    let mut bd = vec![0.0_f32; rows * rows];

    for head in 0..h {
        let off = head * hd;
        // AC: (q + bias_u) . k^T ; BD: (q + bias_v) . p^T
        for i in 0..rows {
            let qi = &q[i * d + off..i * d + off + hd];
            for j in 0..rows {
                let kj = &k[j * d + off..j * d + off + hd];
                let mut acc = 0.0_f32;
                for t in 0..hd {
                    acc += (qi[t] + w.bias_u[off + t]) * kj[t];
                }
                ac[i * rows + j] = acc;
            }
            for j in 0..pos_rows {
                let pj = &p[j * d + off..j * d + off + hd];
                let mut acc = 0.0_f32;
                for t in 0..hd {
                    acc += (qi[t] + w.bias_v[off + t]) * pj[t];
                }
                bd_wide[i * pos_rows + j] = acc;
            }
        }
        rel_shift(&bd_wide, rows, &mut bd)?;

        for i in 0..rows {
            let row = &mut ac[i * rows..(i + 1) * rows];
            for (j, s) in row.iter_mut().enumerate() {
                *s = (*s + bd[i * rows + j]) * scale;
            }
            // Softmax, max-subtracted.
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for s in row.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            let inv = 1.0 / sum;
            for s in row.iter_mut() {
                *s *= inv;
            }
            for t in 0..hd {
                let mut acc = 0.0_f32;
                for (j, s) in row.iter().enumerate() {
                    acc += *s * v[j * d + off + t];
                }
                context[i * d + off + t] = acc;
            }
        }
    }

    let mut out = vec![0.0_f32; rows * d];
    kernels.linear_out_by_in(&context, rows, d, &w.o_proj, d, None, &mut out)?;
    Ok(out)
}

/// Conformer convolution module:
/// `pointwise(2d) -> GLU -> depthwise(k9) -> BatchNorm -> SiLU -> pointwise`.
///
/// The pointwise stages are `k = 1` convolutions, i.e. per-timestep matmuls, so
/// they route through `linear_out_by_in`. Only the depthwise stage and the
/// BatchNorm need the channel-major `[channels][time]` layout, which is why this
/// transposes in and back out.
fn conv_module(
    x: &[f32],
    rows: usize,
    shape: EncoderShape,
    w: &BlockWeights,
    kernels: &dyn KernelBackend,
) -> Result<Vec<f32>> {
    let d = shape.d_model;
    // pointwise1: d -> 2d, then GLU halves back to d (time-major rows).
    let mut wide = vec![0.0_f32; rows * 2 * d];
    kernels.linear_out_by_in(x, rows, d, &w.pw1, 2 * d, None, &mut wide)?;
    let mut gated = vec![0.0_f32; rows * d];
    glu_halves(&wide, rows, d, &mut gated)?;

    // depthwise + BatchNorm want [channels][time].
    let mut chan = transpose_2d(&gated, rows, d);
    let mut dw_out = vec![0.0_f32; d * rows];
    depthwise_conv1d(
        &chan,
        d,
        rows,
        &w.dw,
        None,
        CONV_KERNEL,
        1,
        CONV_KERNEL / 2,
        &mut dw_out,
    )?;
    batch_norm_inference(
        &mut dw_out,
        d,
        rows,
        &w.bn_mean,
        &w.bn_var,
        &w.bn_w,
        &w.bn_b,
        BN_EPS,
    )?;
    silu_inplace(&mut dw_out);
    chan = transpose_2d(&dw_out, d, rows);

    let mut out = vec![0.0_f32; rows * d];
    kernels.linear_out_by_in(&chan, rows, d, &w.pw2, d, None, &mut out)?;
    Ok(out)
}

/// Run one Conformer block in place on `x` (`[rows][d_model]`).
pub fn block_forward(
    x: &mut [f32],
    rows: usize,
    pos: &[f32],
    shape: EncoderShape,
    w: &BlockWeights,
    kernels: &dyn KernelBackend,
) -> Result<()> {
    let d = shape.d_model;
    if x.len() != rows * d {
        return Err(rt(format!(
            "block_forward x.len()={} does not match rows*d_model={}",
            x.len(),
            rows * d
        )));
    }
    let mut norm = vec![0.0_f32; rows * d];

    layer_norm_whisper_scalar(x, rows, d, &w.norm_ff1_w, &w.norm_ff1_b, LN_EPS, &mut norm);
    let ff1 = feed_forward(&norm, rows, shape, &w.ff1_l1, &w.ff1_l2, kernels)?;
    add_scaled_inplace(x, &ff1, 0.5)?;

    layer_norm_whisper_scalar(
        x,
        rows,
        d,
        &w.norm_attn_w,
        &w.norm_attn_b,
        LN_EPS,
        &mut norm,
    );
    let attn = self_attention(&norm, rows, pos, shape, w, kernels)?;
    add_scaled_inplace(x, &attn, 1.0)?;

    layer_norm_whisper_scalar(
        x,
        rows,
        d,
        &w.norm_conv_w,
        &w.norm_conv_b,
        LN_EPS,
        &mut norm,
    );
    let cv = conv_module(&norm, rows, shape, w, kernels)?;
    add_scaled_inplace(x, &cv, 1.0)?;

    layer_norm_whisper_scalar(x, rows, d, &w.norm_ff2_w, &w.norm_ff2_b, LN_EPS, &mut norm);
    let ff2 = feed_forward(&norm, rows, shape, &w.ff2_l1, &w.ff2_l2, kernels)?;
    add_scaled_inplace(x, &ff2, 0.5)?;

    layer_norm_whisper_scalar(x, rows, d, &w.norm_out_w, &w.norm_out_b, LN_EPS, &mut norm);
    x.copy_from_slice(&norm);
    Ok(())
}
