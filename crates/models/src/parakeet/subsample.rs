//! Parakeet `dw_striding` convolutional subsampler (8x on both axes).
//!
//! Structure, read off the checkpoint's tensor names
//! (`encoder.subsampling.layers.{0,2,3,5,6}` + `encoder.subsampling.linear`):
//!
//! ```text
//! [1, T, F=128]
//!   layers.0  Conv2d(1 -> 256, k3, s2, p1)  dense      + ReLU
//!   layers.2  Conv2d(256, groups=256, k3, s2, p1)      depthwise
//!   layers.3  Conv2d(256 -> 256, k1)                   pointwise + ReLU
//!   layers.5  Conv2d(256, groups=256, k3, s2, p1)      depthwise
//!   layers.6  Conv2d(256 -> 256, k1)                   pointwise + ReLU
//! [256, T/8, F/8=16]
//!   flatten per timestep, channel-major -> [T/8, 4096]
//!   linear    4096 -> 1024                             + bias
//! ```
//!
//! **Axis order is load-bearing.** NeMo's `ConvSubsampling` does
//! `x.unsqueeze(1)` on `[B, T, F]`, giving `[B, 1, T, F]` — so the convolution's
//! *height* is TIME and its *width* is FREQUENCY, not the other way round. Both
//! are subsampled by 8, which makes the mistake invisible in the output shape
//! when `T` happens to relate to `F`; it is only visible in the values. The
//! flatten that follows is `transpose(1,2).reshape(b, t, -1)`, i.e. **channel-
//! major within each timestep** (`c0f0..c0f15, c1f0..`), which is the other half
//! of the same trap.

use ocelotl_core::{OcelotlError, Result, RuntimeError};
use ocelotl_kernels::KernelBackend;
use ocelotl_kernels::activation::relu_inplace;
use ocelotl_kernels::conv::{conv_out_len, conv2d, depthwise_conv2d};

use super::audio::ParakeetFeatures;

/// Channels the subsampler works in.
pub const SUBSAMPLE_CHANNELS: usize = 256;
/// Per-axis stride of each of the three convolution stages.
pub const SUBSAMPLE_STRIDE: usize = 2;
pub const SUBSAMPLE_KERNEL: usize = 3;
pub const SUBSAMPLE_PADDING: usize = 1;

/// Subsampler weights, in the checkpoint's native `[out][in][kh][kw]` order.
#[derive(Debug, Clone)]
pub struct SubsampleWeights {
    /// `layers.0`: dense `[256, 1, 3, 3]` + `[256]`.
    pub conv0_w: Vec<f32>,
    pub conv0_b: Vec<f32>,
    /// `layers.2`: depthwise `[256, 1, 3, 3]` + `[256]`.
    pub dw1_w: Vec<f32>,
    pub dw1_b: Vec<f32>,
    /// `layers.3`: pointwise `[256, 256, 1, 1]` + `[256]`.
    pub pw1_w: Vec<f32>,
    pub pw1_b: Vec<f32>,
    /// `layers.5`: depthwise `[256, 1, 3, 3]` + `[256]`.
    pub dw2_w: Vec<f32>,
    pub dw2_b: Vec<f32>,
    /// `layers.6`: pointwise `[256, 256, 1, 1]` + `[256]`.
    pub pw2_w: Vec<f32>,
    pub pw2_b: Vec<f32>,
    /// `linear`: `[1024, 4096]` + `[1024]`.
    pub linear_w: Vec<f32>,
    pub linear_b: Vec<f32>,
}

/// Subsampled encoder input, row-major `[frames][d_model]`.
#[derive(Debug, Clone)]
pub struct Subsampled {
    pub frames: usize,
    pub d_model: usize,
    pub values: Vec<f32>,
}

fn rt<S: Into<String>>(message: S) -> OcelotlError {
    OcelotlError::Runtime(RuntimeError {
        message: message.into(),
    })
}

/// Time frames surviving the three stride-2 stages.
pub fn subsampled_frames(frames: usize) -> usize {
    let mut t = frames;
    for _ in 0..3 {
        t = conv_out_len(t, SUBSAMPLE_KERNEL, SUBSAMPLE_STRIDE, SUBSAMPLE_PADDING);
    }
    t
}

/// Run the subsampler over mel features, producing `[frames/8, d_model]`.
pub fn subsample(
    features: &ParakeetFeatures,
    weights: &SubsampleWeights,
    d_model: usize,
    kernels: &dyn KernelBackend,
) -> Result<Subsampled> {
    let time = features.frames;
    let freq = features.mel_bins;
    let c = SUBSAMPLE_CHANNELS;

    // Mel features arrive mel-major `[mel][frame]`; the convolution wants
    // `[1][time][freq]`, so this transpose is the axis-order decision made
    // explicit rather than left to an index expression.
    let mut image = vec![0.0_f32; time * freq];
    for m in 0..freq {
        for t in 0..time {
            image[t * freq + m] = features.values[m * time + t];
        }
    }

    let (t1, f1) = (stage_len(time), stage_len(freq));
    let mut buf1 = vec![0.0_f32; c * t1 * f1];
    conv2d(
        &image,
        1,
        time,
        freq,
        &weights.conv0_w,
        Some(&weights.conv0_b),
        c,
        (SUBSAMPLE_KERNEL, SUBSAMPLE_KERNEL),
        (SUBSAMPLE_STRIDE, SUBSAMPLE_STRIDE),
        (SUBSAMPLE_PADDING, SUBSAMPLE_PADDING),
        &mut buf1,
    )?;
    relu_inplace(&mut buf1);

    let (t2, f2) = (stage_len(t1), stage_len(f1));
    let mut dw = vec![0.0_f32; c * t2 * f2];
    depthwise_conv2d(
        &buf1,
        c,
        t1,
        f1,
        &weights.dw1_w,
        Some(&weights.dw1_b),
        (SUBSAMPLE_KERNEL, SUBSAMPLE_KERNEL),
        (SUBSAMPLE_STRIDE, SUBSAMPLE_STRIDE),
        (SUBSAMPLE_PADDING, SUBSAMPLE_PADDING),
        &mut dw,
    )?;
    let mut buf2 = vec![0.0_f32; c * t2 * f2];
    pointwise(
        &dw,
        c,
        t2 * f2,
        &weights.pw1_w,
        &weights.pw1_b,
        kernels,
        &mut buf2,
    )?;
    relu_inplace(&mut buf2);

    let (t3, f3) = (stage_len(t2), stage_len(f2));
    let mut dw2 = vec![0.0_f32; c * t3 * f3];
    depthwise_conv2d(
        &buf2,
        c,
        t2,
        f2,
        &weights.dw2_w,
        Some(&weights.dw2_b),
        (SUBSAMPLE_KERNEL, SUBSAMPLE_KERNEL),
        (SUBSAMPLE_STRIDE, SUBSAMPLE_STRIDE),
        (SUBSAMPLE_PADDING, SUBSAMPLE_PADDING),
        &mut dw2,
    )?;
    let mut buf3 = vec![0.0_f32; c * t3 * f3];
    pointwise(
        &dw2,
        c,
        t3 * f3,
        &weights.pw2_w,
        &weights.pw2_b,
        kernels,
        &mut buf3,
    )?;
    relu_inplace(&mut buf3);

    // Flatten per timestep, CHANNEL-MAJOR: row t is
    // [c0f0..c0f_{F-1}, c1f0..], matching transpose(1,2).reshape(b, t, -1).
    let flat = c * f3;
    if weights.linear_w.len() != d_model * flat {
        return Err(rt(format!(
            "subsample linear weight len {} does not match d_model*channels*freq = {}*{}",
            weights.linear_w.len(),
            d_model,
            flat
        )));
    }
    let mut rows = vec![0.0_f32; t3 * flat];
    for t in 0..t3 {
        for ch in 0..c {
            for f in 0..f3 {
                rows[t * flat + ch * f3 + f] = buf3[(ch * t3 + t) * f3 + f];
            }
        }
    }

    let mut values = vec![0.0_f32; t3 * d_model];
    kernels.linear_out_by_in(
        &rows,
        t3,
        flat,
        &weights.linear_w,
        d_model,
        Some(&weights.linear_b),
        &mut values,
    )?;

    Ok(Subsampled {
        frames: t3,
        d_model,
        values,
    })
}

#[inline]
fn stage_len(n: usize) -> usize {
    conv_out_len(n, SUBSAMPLE_KERNEL, SUBSAMPLE_STRIDE, SUBSAMPLE_PADDING)
}

/// A `k=1` convolution is a per-position matmul — routed through
/// `linear_out_by_in` so it gets the AVX2 / register-blocked path.
fn pointwise(
    input: &[f32],
    channels: usize,
    positions: usize,
    weight: &[f32],
    bias: &[f32],
    kernels: &dyn KernelBackend,
    out: &mut [f32],
) -> Result<()> {
    // [ch][pos] -> [pos][ch] so each position is a contiguous input row.
    let mut pos_major = vec![0.0_f32; positions * channels];
    for ch in 0..channels {
        for p in 0..positions {
            pos_major[p * channels + ch] = input[ch * positions + p];
        }
    }
    let mut result = vec![0.0_f32; positions * channels];
    kernels.linear_out_by_in(
        &pos_major,
        positions,
        channels,
        weight,
        channels,
        Some(bias),
        &mut result,
    )?;
    for ch in 0..channels {
        for p in 0..positions {
            out[ch * positions + p] = result[p * channels + ch];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsampled_frames_matches_the_reference_encoder_length() {
        // The reference encoder reports T_enc = 138 for 1101 mel frames; the
        // chain is 1101 -> 551 -> 276 -> 138. Integer equality, never a
        // tolerance: a desynchronized time axis is exactly what this catches.
        assert_eq!(subsampled_frames(1101), 138);
        assert_eq!(subsampled_frames(76), 10);
        assert_eq!(subsampled_frames(12101), 1513);
        // Frequency takes the same path: 128 -> 64 -> 32 -> 16, and 256*16
        // = 4096 is the subsampler linear's input width.
        assert_eq!(subsampled_frames(128), 16);
        assert_eq!(SUBSAMPLE_CHANNELS * subsampled_frames(128), 4096);
    }
}
