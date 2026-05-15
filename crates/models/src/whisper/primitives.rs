//! Numerical primitives composed by the Whisper encoder/decoder forward paths.
//!
//! These are deliberately family-specific (Whisper's conv1d shapes, exact-erf
//! GELU, scalar attention with optional causal mask) — they don't generalize
//! to other model families and would only obscure the shared kernel crate.
//! When two families want the same op, that's the signal to promote it into
//! `ocelotl-kernels`; until then, it stays here.

use ocelotl_core::Result;
use ocelotl_kernels::{DeviceTensor, KernelBackend, softmax};

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

/// Host-resident `layer_norm`. Kept as the parity oracle for `layer_norm_d`
/// and exercised by the existing scalar tests.
#[allow(dead_code)]
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
#[allow(clippy::too_many_arguments, dead_code)]
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
#[allow(clippy::too_many_arguments, dead_code)]
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

#[allow(clippy::too_many_arguments, dead_code)]
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

#[allow(clippy::too_many_arguments, dead_code)]
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

#[allow(clippy::too_many_arguments, dead_code)]
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
    let context = attention_body_host(kernels, q, q_seq, k, v, kv_seq, state, heads, causal)?;
    linear(kernels, &context, q_seq, state, out_w, state, Some(out_b))
}

/// Host-only attention body: produces the `context` activation (length
/// `q_seq * state`) without the trailing out-projection. Used by both
/// `attention_from_projected` and the GW.4-2B device-resident encoder/decoder
/// paths — the device path does its own `linear_d` for the out projection on
/// the device side so the only host bounce is the scalar attention math.
#[allow(clippy::too_many_arguments)]
pub(super) fn attention_body_host(
    kernels: &dyn KernelBackend,
    q: &[f32],
    q_seq: usize,
    k: &[f32],
    v: &[f32],
    kv_seq: usize,
    state: usize,
    heads: usize,
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

    Ok(context)
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

#[allow(clippy::too_many_arguments, dead_code)]
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
    let context = attention_incremental_body_host(
        q, new_k, new_v, past_k, past_v, past_seq, state, heads,
    )?;
    linear(kernels, &context, 1, state, out_w, state, Some(out_b))
}

/// Host-only single-token incremental attention body: produces the `context`
/// activation (length `state`) without the trailing out-projection. GW.4-2B
/// device-resident decoder calls this from the host bounce point and then
/// runs the out-projection as a `linear_d` on the device side.
#[allow(clippy::too_many_arguments)]
pub(super) fn attention_incremental_body_host(
    q: &[f32],
    new_k: &[f32],
    new_v: &[f32],
    past_k: &[f32],
    past_v: &[f32],
    past_seq: usize,
    state: usize,
    heads: usize,
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

    Ok(context)
}

/// Host-resident `linear`. Kept as the parity oracle for `linear_d` and
/// exercised by the legacy attention helpers below; not on the GW.4-2B
/// device-resident forward path.
#[allow(dead_code)]
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

#[allow(dead_code)]
pub(super) fn add_inplace(lhs: &mut [f32], rhs: &[f32]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
        *lhs += rhs;
    }
}

// =====================================================================
// GW.4-2B device-resident wrappers
// =====================================================================
//
// The wrappers below add the same shape-checking layer that the host
// `linear` / `layer_norm` / `mlp_gelu` primitives apply, and then hand off
// to `KernelBackend`'s device-resident `*_d` methods. The encoder/decoder
// forward paths in `encode.rs` / `decode.rs` call these so every linear,
// layer norm, residual add, and MLP step stays on the device for the
// duration of a layer. The only host bounces are at the scalar attention
// bodies (`attention_body_host`, `attention_incremental_body_host`) and at
// the final logits readback for sampling. Search for `to_host_owned()` in
// `decode.rs` / `encode.rs` to grep every bounce point.

/// Device-resident linear projection. Caller-supplied `out` lets the
/// encoder/decoder pool scratch across layers.
#[allow(clippy::too_many_arguments)]
pub(super) fn linear_d(
    kernels: &dyn KernelBackend,
    x: &DeviceTensor,
    rows: usize,
    in_features: usize,
    weight_out_by_in: &DeviceTensor,
    out_features: usize,
    bias: Option<&DeviceTensor>,
    out: &DeviceTensor,
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

    kernels.linear_d(
        x,
        rows,
        in_features,
        weight_out_by_in,
        out_features,
        bias,
        out,
    )
}

/// Device-resident LayerNorm with caller-supplied output.
#[allow(clippy::too_many_arguments)]
pub(super) fn layer_norm_d(
    kernels: &dyn KernelBackend,
    x: &DeviceTensor,
    rows: usize,
    cols: usize,
    weight: &DeviceTensor,
    bias: &DeviceTensor,
    eps: f32,
    out: &DeviceTensor,
) -> Result<()> {
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
    if out.len() != expected {
        return Err(invalid_request(
            "layer_norm.out",
            &format!("expected output length {expected}, got {}", out.len()),
        ));
    }
    kernels.layer_norm_d(x, rows, cols, weight, bias, eps, out)
}

/// Device-resident `lhs += rhs`. Lengths must match.
pub(super) fn add_inplace_d(
    kernels: &dyn KernelBackend,
    lhs: &DeviceTensor,
    rhs: &DeviceTensor,
) -> Result<()> {
    if lhs.len() != rhs.len() {
        return Err(invalid_request(
            "add_inplace.rhs",
            &format!(
                "length mismatch: lhs={} rhs={}",
                lhs.len(),
                rhs.len()
            ),
        ));
    }
    kernels.add_inplace_d(lhs, rhs)
}

/// Device-resident elementwise GELU.
pub(super) fn gelu_inplace_d(kernels: &dyn KernelBackend, x: &DeviceTensor) -> Result<()> {
    kernels.gelu_inplace_d(x)
}

/// Device-resident positional-embedding add. `pe` is the full positional
/// table; `start_pos` is the offset into that table. `pe_rows` must equal
/// the table height so the kernel can bounds-check.
///
/// Reserved for the GW.4-2.5+ encoder/decoder migration that moves
/// embedding gather and positional add onto the device. GW.4-2B keeps both
/// on host (cold path, one-shot per 30 s window) and uploads the post-add
/// activation in a single pass.
#[allow(dead_code)]
pub(super) fn add_positional_embedding_d(
    kernels: &dyn KernelBackend,
    x: &DeviceTensor,
    rows: usize,
    cols: usize,
    pe: &DeviceTensor,
    pe_rows: usize,
    start_pos: usize,
) -> Result<()> {
    if rows > pe_rows {
        return Err(invalid_request(
            "positional_embedding",
            &format!("rows {rows} exceeds max rows {pe_rows}"),
        ));
    }
    let expected_pe = checked_len_product("positional_embedding", &[pe_rows, cols])?;
    if pe.len() != expected_pe {
        return Err(invalid_model(
            "positional_embedding",
            &format!(
                "expected positional embedding length {expected_pe}, got {}",
                pe.len()
            ),
        ));
    }
    kernels.add_positional_embedding_d(x, rows, cols, pe, pe_rows, start_pos)
}

/// Device-resident Whisper encoder self-attention. Encoder-only: no
/// causal mask, no GQA, all of `q`/`k`/`v` are `[seq, state]` row-major
/// where `state == n_head * head_dim`. Computes the same math as
/// `attention_body_host` with `causal == false` but keeps everything on
/// device — no host bounce between Q/K/V projections and the out
/// projection.
///
/// Decoder paths (causal self-attention with KV cache, cross-attention)
/// stay on host for now — their shapes and scratch patterns are different
/// enough that a fused kernel was deferred until the encoder bottleneck
/// closes.
#[allow(clippy::too_many_arguments)]
pub(super) fn attention_encoder_d(
    kernels: &dyn KernelBackend,
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    seq: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
    output: &DeviceTensor,
) -> Result<()> {
    if n_head == 0 {
        return Err(invalid_model("attention.heads", "must be > 0"));
    }
    if head_dim == 0 {
        return Err(invalid_model("attention.head_dim", "must be > 0"));
    }
    let state = checked_len_product("attention.state", &[n_head, head_dim])?;
    let expected = checked_len_product("attention.q", &[seq, state])?;
    for (label, len) in [
        ("attention.q", q.len()),
        ("attention.k", k.len()),
        ("attention.v", v.len()),
        ("attention.out", output.len()),
    ] {
        if len != expected {
            return Err(invalid_request(
                label,
                &format!("expected length {expected}, got {len}"),
            ));
        }
    }
    kernels.attention_encoder_d(q, k, v, seq, n_head, head_dim, scale, output)
}

/// Device-resident `mlp_gelu`: `linear_d → gelu_inplace_d → linear_d`.
/// `hidden_act` is a caller-supplied scratch of length `rows * ffn`; `out`
/// is the `rows * hidden` projection back to model width. Both are reused
/// across encoder / decoder layers to keep scratch allocation off the hot
/// path.
#[allow(clippy::too_many_arguments)]
pub(super) fn mlp_gelu_d(
    kernels: &dyn KernelBackend,
    x: &DeviceTensor,
    rows: usize,
    hidden: usize,
    ffn: usize,
    fc1_w: &DeviceTensor,
    fc1_b: &DeviceTensor,
    fc2_w: &DeviceTensor,
    fc2_b: &DeviceTensor,
    hidden_act: &DeviceTensor,
    out: &DeviceTensor,
) -> Result<()> {
    linear_d(
        kernels,
        x,
        rows,
        hidden,
        fc1_w,
        ffn,
        Some(fc1_b),
        hidden_act,
    )?;
    gelu_inplace_d(kernels, hidden_act)?;
    linear_d(
        kernels,
        hidden_act,
        rows,
        ffn,
        fc2_w,
        hidden,
        Some(fc2_b),
        out,
    )
}
