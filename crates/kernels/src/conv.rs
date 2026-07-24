//! Convolution kernels for the Conformer family (Parakeet).
//!
//! The tree previously had exactly one convolution — Whisper's `conv1d`, which
//! unconditionally sums over every input channel and therefore cannot express a
//! grouped or depthwise convolution at all. Parakeet needs three shapes:
//!
//! - [`conv2d`] — the dense `Conv2d(1 -> 256, k3, s2, p1)` that opens the
//!   `dw_striding` subsampler, over a `(freq, time)` image.
//! - [`depthwise_conv2d`] — subsampler stages 2 and 3, `groups = channels`.
//! - [`depthwise_conv1d`] — the Conformer convolution module's depthwise stage
//!   (`kernel 9`, `padding 4`, `groups = channels`).
//!
//! Plus [`batch_norm_inference`], which the convolution module applies with the
//! checkpoint's **fixed running statistics** — it is not a LayerNorm and it does
//! not compute batch statistics at inference.
//!
//! **Layouts** are channel-major throughout, matching the reference graphs:
//! - 2-D activations: `[channels][height][width]`, row-major, `height` = freq.
//! - 2-D dense weights: `[out_ch][in_ch][kh][kw]`.
//! - 2-D depthwise weights: `[channels][kh][kw]`.
//! - 1-D activations: `[channels][time]`; depthwise weights `[channels][kernel]`.
//!
//! The pointwise (`k = 1`) convolutions in the convolution module are deliberately
//! **not** here: those are per-timestep matmuls and route through
//! `linear_out_by_in`, which already has the AVX2 and register-blocked paths.

use crate::{Result, kernel_err, linear_out_by_in};

/// Output extent of a convolution along one axis.
#[inline]
pub fn conv_out_len(input: usize, kernel: usize, stride: usize, padding: usize) -> usize {
    (input + 2 * padding).saturating_sub(kernel) / stride + 1
}

/// Dense 2-D convolution, computed as `im2col` + [`linear_out_by_in`].
///
/// Building the patch matrix costs one pass and a `k*k*in_ch` copy per output
/// cell, but it buys the existing AVX2 / register-blocked GEMM for the actual
/// arithmetic — which is where essentially all of the time goes for the
/// `1 -> 256` opening layer. `weight` is `[out_ch][in_ch*kh*kw]` when flattened,
/// which is exactly the raw `[out][in]` layout `linear_out_by_in` consumes, so
/// no transpose is needed.
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    input: &[f32],
    in_channels: usize,
    height: usize,
    width: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    out_channels: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    out: &mut [f32],
) -> Result<()> {
    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    if kh == 0 || kw == 0 || sh == 0 || sw == 0 {
        return Err(kernel_err("conv2d kernel and stride must be non-zero"));
    }
    if input.len() != in_channels * height * width {
        return Err(kernel_err(format!(
            "conv2d input.len()={} does not match in_channels*height*width={}",
            input.len(),
            in_channels * height * width
        )));
    }
    let patch = in_channels * kh * kw;
    if weight.len() != out_channels * patch {
        return Err(kernel_err(format!(
            "conv2d weight.len()={} does not match out_channels*in_channels*kh*kw={}",
            weight.len(),
            out_channels * patch
        )));
    }
    let oh = conv_out_len(height, kh, sh, ph);
    let ow = conv_out_len(width, kw, sw, pw);
    if out.len() != out_channels * oh * ow {
        return Err(kernel_err(format!(
            "conv2d out.len()={} does not match out_channels*out_h*out_w={}",
            out.len(),
            out_channels * oh * ow
        )));
    }

    // im2col: [oh*ow, in_ch*kh*kw], zero outside the padded border.
    let cells = oh * ow;
    let mut cols = vec![0.0_f32; cells * patch];
    for oy in 0..oh {
        for ox in 0..ow {
            let dst = (oy * ow + ox) * patch;
            for ic in 0..in_channels {
                for ky in 0..kh {
                    let iy = (oy * sh + ky) as isize - ph as isize;
                    if iy < 0 || iy >= height as isize {
                        continue;
                    }
                    for kx in 0..kw {
                        let ix = (ox * sw + kx) as isize - pw as isize;
                        if ix < 0 || ix >= width as isize {
                            continue;
                        }
                        cols[dst + (ic * kh + ky) * kw + kx] =
                            input[(ic * height + iy as usize) * width + ix as usize];
                    }
                }
            }
        }
    }

    // [cells, patch] x [out_ch, patch]^T -> [cells, out_ch]
    let mut cell_major = vec![0.0_f32; cells * out_channels];
    linear_out_by_in(
        &cols,
        cells,
        patch,
        weight,
        out_channels,
        bias,
        &mut cell_major,
    )?;

    // Transpose to channel-major [out_ch][oh][ow].
    for cell in 0..cells {
        for oc in 0..out_channels {
            out[oc * cells + cell] = cell_major[cell * out_channels + oc];
        }
    }
    Ok(())
}

/// Depthwise 2-D convolution: output channel `c` sees only input channel `c`.
///
/// `weight` is `[channels][kh][kw]` — one `kh*kw` filter per channel, with no
/// summation across channels. That absence is the whole point: a dense conv with
/// `in_channels = 1` per group is a *different* computation from a dense conv
/// over all channels, and the existing `conv1d` can only express the latter.
#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv2d(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    out: &mut [f32],
) -> Result<()> {
    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    if kh == 0 || kw == 0 || sh == 0 || sw == 0 {
        return Err(kernel_err(
            "depthwise_conv2d kernel and stride must be non-zero",
        ));
    }
    if input.len() != channels * height * width {
        return Err(kernel_err(format!(
            "depthwise_conv2d input.len()={} does not match channels*height*width={}",
            input.len(),
            channels * height * width
        )));
    }
    if weight.len() != channels * kh * kw {
        return Err(kernel_err(format!(
            "depthwise_conv2d weight.len()={} does not match channels*kh*kw={}",
            weight.len(),
            channels * kh * kw
        )));
    }
    if let Some(b) = bias {
        if b.len() != channels {
            return Err(kernel_err(format!(
                "depthwise_conv2d bias.len()={} does not match channels={channels}",
                b.len()
            )));
        }
    }
    let oh = conv_out_len(height, kh, sh, ph);
    let ow = conv_out_len(width, kw, sw, pw);
    if out.len() != channels * oh * ow {
        return Err(kernel_err(format!(
            "depthwise_conv2d out.len()={} does not match channels*out_h*out_w={}",
            out.len(),
            channels * oh * ow
        )));
    }

    for c in 0..channels {
        let plane = c * height * width;
        let wbase = c * kh * kw;
        let obase = c * oh * ow;
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = bias.map_or(0.0, |b| b[c]);
                for ky in 0..kh {
                    let iy = (oy * sh + ky) as isize - ph as isize;
                    if iy < 0 || iy >= height as isize {
                        continue;
                    }
                    for kx in 0..kw {
                        let ix = (ox * sw + kx) as isize - pw as isize;
                        if ix < 0 || ix >= width as isize {
                            continue;
                        }
                        acc += input[plane + iy as usize * width + ix as usize]
                            * weight[wbase + ky * kw + kx];
                    }
                }
                out[obase + oy * ow + ox] = acc;
            }
        }
    }
    Ok(())
}

/// Depthwise 1-D convolution over `[channels][time]`, `weight` `[channels][kernel]`.
///
/// The Conformer convolution module uses `kernel 9`, `stride 1`, `padding 4`,
/// which is length-preserving ("same") — asserted by a test rather than assumed,
/// because an off-by-one in the padding silently shortens the time axis and
/// desynchronizes everything downstream.
#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv1d(
    input: &[f32],
    channels: usize,
    time: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    kernel: usize,
    stride: usize,
    padding: usize,
    out: &mut [f32],
) -> Result<()> {
    if kernel == 0 || stride == 0 {
        return Err(kernel_err(
            "depthwise_conv1d kernel and stride must be non-zero",
        ));
    }
    if input.len() != channels * time {
        return Err(kernel_err(format!(
            "depthwise_conv1d input.len()={} does not match channels*time={}",
            input.len(),
            channels * time
        )));
    }
    if weight.len() != channels * kernel {
        return Err(kernel_err(format!(
            "depthwise_conv1d weight.len()={} does not match channels*kernel={}",
            weight.len(),
            channels * kernel
        )));
    }
    let ot = conv_out_len(time, kernel, stride, padding);
    if out.len() != channels * ot {
        return Err(kernel_err(format!(
            "depthwise_conv1d out.len()={} does not match channels*out_time={}",
            out.len(),
            channels * ot
        )));
    }

    for c in 0..channels {
        let base = c * time;
        let wbase = c * kernel;
        let obase = c * ot;
        for o in 0..ot {
            let mut acc = bias.map_or(0.0, |b| b[c]);
            for k in 0..kernel {
                let i = (o * stride + k) as isize - padding as isize;
                if i < 0 || i >= time as isize {
                    continue;
                }
                acc += input[base + i as usize] * weight[wbase + k];
            }
            out[obase + o] = acc;
        }
    }
    Ok(())
}

/// Inference-time BatchNorm over `[channels][spatial]`:
/// `y = (x - running_mean) / sqrt(running_var + eps) * gamma + beta`.
///
/// The statistics are the checkpoint's **fixed** running values — nothing is
/// computed from the input. This is emphatically not interchangeable with
/// LayerNorm (which normalizes across features per position, using statistics of
/// the data), and the tree's `rmsnorm`/`layer_norm` cannot express it.
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_inference(
    x: &mut [f32],
    channels: usize,
    spatial: usize,
    running_mean: &[f32],
    running_var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Result<()> {
    if x.len() != channels * spatial {
        return Err(kernel_err(format!(
            "batch_norm_inference x.len()={} does not match channels*spatial={}",
            x.len(),
            channels * spatial
        )));
    }
    for (name, v) in [
        ("running_mean", running_mean),
        ("running_var", running_var),
        ("gamma", gamma),
        ("beta", beta),
    ] {
        if v.len() != channels {
            return Err(kernel_err(format!(
                "batch_norm_inference {name}.len()={} does not match channels={channels}",
                v.len()
            )));
        }
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(kernel_err(format!(
            "batch_norm_inference eps must be finite and non-negative, got {eps}"
        )));
    }

    for c in 0..channels {
        // eps goes UNDER the square root here (unlike the Parakeet frontend's
        // normalization, which adds it to the std) — this follows the BatchNorm
        // definition the checkpoint was trained with.
        let scale = gamma[c] / (running_var[c] + eps).sqrt();
        let shift = beta[c] - running_mean[c] * scale;
        let plane = &mut x[c * spatial..(c + 1) * spatial];
        for value in plane.iter_mut() {
            *value = *value * scale + shift;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_out_len_matches_the_standard_formula() {
        // "same" padding for k=9,s=1,p=4 must preserve length — the Conformer
        // convolution module depends on this exactly.
        assert_eq!(conv_out_len(100, 9, 1, 4), 100);
        // The subsampler's k=3,s=2,p=1 halves (rounding up).
        assert_eq!(conv_out_len(128, 3, 2, 1), 64);
        assert_eq!(conv_out_len(1101, 3, 2, 1), 551);
        assert_eq!(conv_out_len(551, 3, 2, 1), 276);
        assert_eq!(conv_out_len(276, 3, 2, 1), 138);
    }

    #[test]
    fn conv2d_matches_hand_computed_output() {
        // 1 channel, 3x3 input, 2x2 kernel, stride 1, no padding -> 2x2 out.
        //   input  [[1,2,3],[4,5,6],[7,8,9]]   kernel [[1,0],[0,1]]  (trace)
        //   out[0,0] = 1*1 + 5*1 = 6      out[0,1] = 2 + 6 = 8
        //   out[1,0] = 4 + 8 = 12         out[1,1] = 5 + 9 = 14
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let weight = [1.0_f32, 0.0, 0.0, 1.0];
        let mut out = [0.0_f32; 4];
        conv2d(
            &input,
            1,
            3,
            3,
            &weight,
            None,
            1,
            (2, 2),
            (1, 1),
            (0, 0),
            &mut out,
        )
        .expect("conv2d");
        assert_eq!(out, [6.0, 8.0, 12.0, 14.0]);
    }

    #[test]
    fn conv2d_applies_bias_and_zero_padding() {
        // 1x1 input, 3x3 kernel of ones, padding 1 -> the single cell sees the
        // input once (centre) and zeros elsewhere: out = 5 + bias.
        let input = [5.0_f32];
        let weight = [1.0_f32; 9];
        let bias = [0.25_f32];
        let mut out = [0.0_f32; 1];
        conv2d(
            &input,
            1,
            1,
            1,
            &weight,
            Some(&bias),
            1,
            (3, 3),
            (1, 1),
            (1, 1),
            &mut out,
        )
        .expect("conv2d");
        assert_eq!(out, [5.25]);
    }

    #[test]
    fn conv2d_sums_across_input_channels() {
        // Discriminating vs a depthwise conv: 2 in-channels must be SUMMED.
        // in ch0 = [1], ch1 = [10]; weight ch0 = 1, ch1 = 2 -> 1 + 20 = 21.
        let input = [1.0_f32, 10.0];
        let weight = [1.0_f32, 2.0];
        let mut out = [0.0_f32; 1];
        conv2d(
            &input,
            2,
            1,
            1,
            &weight,
            None,
            1,
            (1, 1),
            (1, 1),
            (0, 0),
            &mut out,
        )
        .expect("conv2d");
        assert_eq!(out, [21.0]);
    }

    #[test]
    fn depthwise_conv2d_does_not_mix_channels() {
        // The discriminating test: same inputs as the dense case above, but
        // depthwise must keep them SEPARATE -> [1*1, 10*2] = [1, 20], never 21.
        let input = [1.0_f32, 10.0];
        let weight = [1.0_f32, 2.0];
        let mut out = [0.0_f32; 2];
        depthwise_conv2d(
            &input,
            2,
            1,
            1,
            &weight,
            None,
            (1, 1),
            (1, 1),
            (0, 0),
            &mut out,
        )
        .expect("depthwise");
        assert_eq!(out, [1.0, 20.0]);
    }

    #[test]
    fn depthwise_conv2d_matches_hand_computed_stride_two() {
        // 1 channel, 4x4 ramp 1..16, 2x2 kernel of ones, stride 2 -> 2x2 out.
        //   out[0,0] = 1+2+5+6 = 14     out[0,1] = 3+4+7+8 = 22
        //   out[1,0] = 9+10+13+14 = 46  out[1,1] = 11+12+15+16 = 54
        let input: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let weight = [1.0_f32; 4];
        let mut out = [0.0_f32; 4];
        depthwise_conv2d(
            &input,
            1,
            4,
            4,
            &weight,
            None,
            (2, 2),
            (2, 2),
            (0, 0),
            &mut out,
        )
        .expect("depthwise");
        assert_eq!(out, [14.0, 22.0, 46.0, 54.0]);
    }

    #[test]
    fn depthwise_conv1d_is_length_preserving_with_same_padding() {
        // k=9, s=1, p=4 over T=20 must give exactly 20 back.
        let input = vec![1.0_f32; 20];
        let weight = vec![1.0_f32; 9];
        let mut out = vec![0.0_f32; 20];
        depthwise_conv1d(&input, 1, 20, &weight, None, 9, 1, 4, &mut out).expect("dw1d");
        // Interior taps see all 9 ones; the edges see fewer (zero padding).
        assert_eq!(out[10], 9.0);
        assert_eq!(out[0], 5.0); // taps -4..4 -> 5 valid
        assert_eq!(out[19], 5.0);
    }

    #[test]
    fn depthwise_conv1d_matches_hand_computed_per_channel_filters() {
        // 2 channels, T=3. ch0 = [1,2,3] with filter [1,1]; ch1 = [4,5,6] with
        // filter [0,2]. k=2, s=1, p=0 -> out T=2.
        //   ch0: [1+2, 2+3] = [3, 5]
        //   ch1: [0*4+2*5, 0*5+2*6] = [10, 12]
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = [1.0_f32, 1.0, 0.0, 2.0];
        let mut out = [0.0_f32; 4];
        depthwise_conv1d(&input, 2, 3, &weight, None, 2, 1, 0, &mut out).expect("dw1d");
        assert_eq!(out, [3.0, 5.0, 10.0, 12.0]);
    }

    #[test]
    fn batch_norm_inference_matches_hand_computed_affine() {
        // 1 channel: mean=1, var=4 (std=2), gamma=3, beta=0.5, eps=0.
        // x=5 -> (5-1)/2*3 + 0.5 = 6.5 ; x=1 -> 0*3 + 0.5 = 0.5
        let mut x = [5.0_f32, 1.0];
        batch_norm_inference(&mut x, 1, 2, &[1.0], &[4.0], &[3.0], &[0.5], 0.0).expect("bn");
        assert!((x[0] - 6.5).abs() < 1e-6, "got {}", x[0]);
        assert!((x[1] - 0.5).abs() < 1e-6, "got {}", x[1]);
    }

    #[test]
    fn batch_norm_inference_uses_fixed_stats_not_the_input() {
        // The discriminating test: if this computed statistics FROM the data
        // (LayerNorm-style), both channels would come out identically
        // standardized. With fixed running stats they must not.
        let mut x = [10.0_f32, 20.0, 10.0, 20.0];
        batch_norm_inference(
            &mut x,
            2,
            2,
            &[0.0, 100.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
            &[0.0, 0.0],
            0.0,
        )
        .expect("bn");
        assert_eq!(x, [10.0, 20.0, -90.0, -80.0]);
    }

    #[test]
    fn batch_norm_inference_rejects_bad_shapes_and_eps() {
        let mut x = [1.0_f32, 2.0];
        assert!(
            batch_norm_inference(&mut x, 1, 2, &[0.0, 0.0], &[1.0], &[1.0], &[0.0], 0.0).is_err()
        );
        assert!(batch_norm_inference(&mut x, 1, 2, &[0.0], &[1.0], &[1.0], &[0.0], -1.0).is_err());
    }
}
