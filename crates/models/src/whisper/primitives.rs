//! Numerical primitives composed by the Whisper encoder/decoder forward paths.
//!
//! These are deliberately family-specific (Whisper's conv1d shapes, exact-erf
//! GELU, scalar attention with optional causal mask) — they don't generalize
//! to other model families and would only obscure the shared kernel crate.
//! When two families want the same op, that's the signal to promote it into
//! `ocelotl-kernels`; until then, it stays here.

use ocelotl_core::Result;
use ocelotl_kernels::{KernelBackend, softmax};

use super::{checked_len_product, invalid_model, invalid_request};

#[allow(clippy::too_many_arguments)]
pub(super) fn conv1d(
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

pub(super) fn conv_output_len(
    time: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<usize> {
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

pub(super) fn layer_norm(
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

/// Caller-supplied scratch + output variant of the Whisper GELU MLP.
///
/// `hidden_act_buf` is the `rows * ffn` intermediate (post-fc1, post-GELU).
/// `out` is the `rows * hidden` projection back to model width.
/// Both are passed in so the encoder and decoder forward paths allocate them
/// once outside their per-layer loops and reuse across all layers, which
/// dominates allocator pressure on the encoder pass at tiny (4 layers, 1500
/// rows, 1536 ffn = ~37 MB per layer in the old vec-returning version).
/// This matches the kernel-crate `mlp_gated_silu` convention and prepares
/// the call sites for the upcoming `linear_d` device-resident migration.
#[allow(clippy::too_many_arguments)]
pub(super) fn mlp_gelu(
    kernels: &dyn KernelBackend,
    x: &[f32],
    rows: usize,
    hidden: usize,
    ffn: usize,
    fc1_w: &[f32],
    fc1_b: &[f32],
    fc2_w: &[f32],
    fc2_b: &[f32],
    hidden_act_buf: &mut [f32],
    out: &mut [f32],
) -> Result<()> {
    let hidden_act_expected = checked_len_product("mlp_gelu.hidden_act", &[rows, ffn])?;
    if hidden_act_buf.len() != hidden_act_expected {
        return Err(invalid_request(
            "mlp_gelu.hidden_act",
            &format!(
                "expected hidden_act buffer length {hidden_act_expected}, got {}",
                hidden_act_buf.len()
            ),
        ));
    }
    let out_expected = checked_len_product("mlp_gelu.out", &[rows, hidden])?;
    if out.len() != out_expected {
        return Err(invalid_request(
            "mlp_gelu.out",
            &format!(
                "expected out buffer length {out_expected}, got {}",
                out.len()
            ),
        ));
    }

    linear_into(
        kernels,
        x,
        rows,
        hidden,
        fc1_w,
        ffn,
        Some(fc1_b),
        hidden_act_buf,
    )?;
    gelu_inplace(hidden_act_buf);
    linear_into(
        kernels,
        hidden_act_buf,
        rows,
        ffn,
        fc2_w,
        hidden,
        Some(fc2_b),
        out,
    )
}

/// Caller-supplied output variant of `linear`. Same validation contract,
/// no internal allocation. Used by `mlp_gelu` to write into pre-allocated
/// scratch and output buffers, and available for future call sites that
/// want to skip the `Vec` round-trip.
#[allow(clippy::too_many_arguments)]
pub(super) fn linear_into(
    kernels: &dyn KernelBackend,
    x: &[f32],
    rows: usize,
    in_features: usize,
    weight_out_by_in: &[f32],
    out_features: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> Result<()> {
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
    let out_expected = checked_len_product("linear.out", &[rows, out_features])?;
    if out.len() != out_expected {
        return Err(invalid_request(
            "linear.out",
            &format!("expected output length {out_expected}, got {}", out.len()),
        ));
    }

    kernels.linear_out_by_in(
        x,
        rows,
        in_features,
        weight_out_by_in,
        out_features,
        bias,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attention(
    kernels: &dyn KernelBackend,
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
pub(super) fn attention_with_precomputed_kv(
    kernels: &dyn KernelBackend,
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

/// Below this Q-sequence length, rayon dispatch overhead exceeds the
/// per-row compute cost. Encoder self-attention hits q_seq = audio_ctx
/// (>=1500 for all classic Whisper sizes) and always parallelizes; full-
/// context decoder attention has short q_seq and stays serial.
const PARALLEL_ATTENTION_MIN_Q: usize = 32;

#[allow(clippy::too_many_arguments)]
pub(super) fn attention_from_projected(
    kernels: &dyn KernelBackend,
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
    if causal && q_seq > kv_seq {
        return Err(invalid_request(
            "attention.causal",
            "causal self-attention query length exceeds key length",
        ));
    }

    let head_dim = state / heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut context = vec![0.0_f32; q_seq * state];

    // Parallel dispatch when a pool is available and the workload is large
    // enough to amortize rayon overhead. The Q range is split into one
    // chunk per worker thread; each chunk owns a single `scores` scratch
    // buffer that is reused across all rows it processes, which is the
    // serial path's allocation discipline applied per worker. Disjoint
    // context-row writes plus identical per-row K-loop order keep the
    // result bit-identical to the serial path.
    match kernels.cpu_thread_pool() {
        Some(pool) if q_seq >= PARALLEL_ATTENTION_MIN_Q => {
            use rayon::prelude::*;
            let threads = pool.current_num_threads().max(1);
            let rows_per_chunk = q_seq.div_ceil(threads);
            let chunk_len = rows_per_chunk * state;
            pool.install(|| {
                context
                    .par_chunks_mut(chunk_len)
                    .enumerate()
                    .for_each(|(idx, ctx_chunk)| {
                        let row_start = idx * rows_per_chunk;
                        let chunk_rows = ctx_chunk.len() / state;
                        let mut scores = vec![0.0_f32; kv_seq];
                        for local_row in 0..chunk_rows {
                            let qi = row_start + local_row;
                            let ctx_row =
                                &mut ctx_chunk[local_row * state..(local_row + 1) * state];
                            compute_attention_row_with_scratch(
                                qi, q, k, v, kv_seq, state, heads, head_dim, scale, causal,
                                ctx_row, &mut scores,
                            );
                        }
                    });
            });
        }
        _ => {
            let mut scores = vec![0.0_f32; kv_seq];
            for qi in 0..q_seq {
                let ctx_row = &mut context[qi * state..(qi + 1) * state];
                compute_attention_row_with_scratch(
                    qi, q, k, v, kv_seq, state, heads, head_dim, scale, causal, ctx_row,
                    &mut scores,
                );
            }
        }
    }

    linear(kernels, &context, q_seq, state, out_w, state, Some(out_b))
}

/// Per-Q-row attention body that borrows its `scores` scratch from the
/// caller. Used by both serial and parallel paths so the scratch buffer is
/// reused across rows without reallocation; the parallel path allocates one
/// scratch per worker chunk.
#[allow(clippy::too_many_arguments)]
fn compute_attention_row_with_scratch(
    qi: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kv_seq: usize,
    state: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
    ctx_row: &mut [f32],
    scores: &mut [f32],
) {
    debug_assert_eq!(ctx_row.len(), state);
    debug_assert_eq!(scores.len(), kv_seq);
    for head in 0..heads {
        let visible = if causal { qi + 1 } else { kv_seq };
        let q_base = qi * state + head * head_dim;
        let q_row = &q[q_base..q_base + head_dim];
        for (ki, score) in scores.iter_mut().enumerate().take(visible) {
            let k_base = ki * state + head * head_dim;
            let k_row = &k[k_base..k_base + head_dim];
            let acc = dot_unrolled_4(q_row, k_row);
            *score = acc * scale;
        }
        softmax(&mut scores[..visible]);
        let context_base = head * head_dim;
        for dim in 0..head_dim {
            ctx_row[context_base + dim] = 0.0;
        }
        for (ki, &p) in scores.iter().enumerate().take(visible) {
            let v_base = ki * state + head * head_dim;
            for dim in 0..head_dim {
                ctx_row[context_base + dim] += p * v[v_base + dim];
            }
        }
    }
}

fn dot_unrolled_4(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let unrolled = a.len() - (a.len() % 4);
    let mut acc = 0.0_f32;

    for idx in (0..unrolled).step_by(4) {
        acc += a[idx] * b[idx];
        acc += a[idx + 1] * b[idx + 1];
        acc += a[idx + 2] * b[idx + 2];
        acc += a[idx + 3] * b[idx + 3];
    }
    for idx in unrolled..a.len() {
        acc += a[idx] * b[idx];
    }

    acc
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attention_incremental_from_projected(
    kernels: &dyn KernelBackend,
    q: &[f32],
    new_k: &[f32],
    new_v: &[f32],
    past_k: &[f32],
    past_v: &[f32],
    past_seq: usize,
    state: usize,
    heads: usize,
    out_w: &[f32],
    out_b: &[f32],
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
    if q.len() != state {
        return Err(invalid_request(
            "attention.q",
            &format!("expected length {state}, got {}", q.len()),
        ));
    }
    if new_k.len() != state {
        return Err(invalid_request(
            "attention.new_key",
            &format!("expected length {state}, got {}", new_k.len()),
        ));
    }
    if new_v.len() != state {
        return Err(invalid_request(
            "attention.new_value",
            &format!("expected length {state}, got {}", new_v.len()),
        ));
    }
    let past_expected = checked_len_product("attention.past", &[past_seq, state])?;
    if past_k.len() != past_expected {
        return Err(invalid_request(
            "attention.past_key",
            &format!("expected length {past_expected}, got {}", past_k.len()),
        ));
    }
    if past_v.len() != past_expected {
        return Err(invalid_request(
            "attention.past_value",
            &format!("expected length {past_expected}, got {}", past_v.len()),
        ));
    }

    let head_dim = state / heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let visible = past_seq
        .checked_add(1)
        .ok_or_else(|| invalid_request("attention.past", "past sequence length overflows usize"))?;
    let mut scores = vec![0.0_f32; visible];
    let mut context = vec![0.0_f32; state];

    for head in 0..heads {
        let q_base = head * head_dim;
        for (ki, score) in scores.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for dim in 0..head_dim {
                let key_value = if ki < past_seq {
                    past_k[ki * state + head * head_dim + dim]
                } else {
                    new_k[head * head_dim + dim]
                };
                acc += q[q_base + dim] * key_value;
            }
            *score = acc * scale;
        }
        softmax(&mut scores);
        for dim in 0..head_dim {
            let mut acc = 0.0_f32;
            for (ki, &p) in scores.iter().enumerate() {
                let value = if ki < past_seq {
                    past_v[ki * state + head * head_dim + dim]
                } else {
                    new_v[head * head_dim + dim]
                };
                acc += p * value;
            }
            context[head * head_dim + dim] = acc;
        }
    }

    linear(kernels, &context, 1, state, out_w, state, Some(out_b))
}

pub(super) fn linear(
    kernels: &dyn KernelBackend,
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

pub(super) fn gelu_inplace(x: &mut [f32]) {
    for v in x {
        *v = gelu(*v);
    }
}

pub(super) fn gelu(x: f32) -> f32 {
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

pub(super) fn add_positional_embedding(
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

pub(super) fn add_inplace(lhs: &mut [f32], rhs: &[f32]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
        *lhs += rhs;
    }
}
