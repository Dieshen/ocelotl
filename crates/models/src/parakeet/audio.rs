//! Parakeet (NeMo `AudioToMelSpectrogramPreprocessor`) audio frontend.
//!
//! Deliberately a FORK of [`crate::whisper::audio`], not a generalization of it:
//! the Whisper frontend is parity-clean and pinned by fixture tests, and it is
//! the one ASR path that already works. The two frontends share only the Slaney
//! mel helpers (re-derived here to keep the modules independent).
//!
//! Every constant and every ordering decision below was read out of the exported
//! reference graph (`nemo128.onnx` from `istupakov/parakeet-tdt-0.6b-v3-onnx`),
//! not from prose documentation — two published descriptions of this frontend
//! disagree with the graph (see NOTES). The graph is the oracle because it is
//! what `parakeet.cpp` and `onnx-asr` agree with.
//!
//! Pipeline, in exact order:
//! 1. **Pre-emphasis** `y[0] = x[0]`, `y[i] = x[i] - 0.97·x[i-1]` — before padding.
//! 2. **Reflect** pad `n_fft/2 = 256` samples each side.
//! 3. **STFT** hop 160, `n_fft` 512 → 257 bins, with a **400-tap symmetric Hann**
//!    window centre-placed inside the 512-point frame (zero outside `56..456`).
//! 4. **Power** spectrum (sum of squares, not magnitude).
//! 5. **Mel** projection: 128 Slaney-scale, Slaney **area-normalized** triangles.
//! 6. **Natural log** of `mel + 2^-24`.
//! 7. **Per-feature normalization** across time: subtract the per-mel mean, divide
//!    by `std + 1e-5` where the variance uses the **unbiased `T-1`** denominator.
//!
//! NOTES — where the graph contradicts the written descriptions:
//! - Padding is **reflect**, not `constant`/zero as the HF feature-extractor doc
//!   states. (Verified: the graph's `Pad` node carries `mode = "reflect"`.)
//! - **Every** frame is valid: `T = floor(L/hop) + 1` equals the exported
//!   `features_lens` exactly, so the graph's validity mask is a no-op and no
//!   trailing frame is zeroed or excluded from the statistics. Descriptions
//!   claiming `valid == T - 1` do not apply to this export.
//! - The normalization epsilon is added to the **standard deviation**
//!   (`x / (sqrt(var) + 1e-5)`), not to the variance under the square root.

use std::f32::consts::PI;
use std::sync::OnceLock;

use ocelotl_core::{InvalidRequestError, OcelotlError, Result, RuntimeError};

pub const PARAKEET_SAMPLE_RATE_HZ: u32 = 16_000;
/// DFT length. Note this is **not** the window length — see [`PARAKEET_WIN_LENGTH`].
pub const PARAKEET_N_FFT: usize = 512;
/// Hann window length, centre-placed inside the `PARAKEET_N_FFT` frame.
pub const PARAKEET_WIN_LENGTH: usize = 400;
pub const PARAKEET_HOP_LENGTH: usize = 160;
pub const PARAKEET_MEL_BINS: usize = 128;
/// Number of real-FFT bins: `n_fft/2 + 1`.
pub const PARAKEET_FFT_BINS: usize = PARAKEET_N_FFT / 2 + 1;

const PREEMPHASIS: f32 = 0.97;
/// `2^-24`, the exported graph's `log_zero_guard_value`.
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;
/// Added to the standard deviation (not the variance) during normalization.
const NORM_EPS: f32 = 1e-5;
const SLANEY_LOG_STEP: f32 = 1.856_298 / 27.0;

/// Offset of the 400-tap window inside the 512-point frame: `(512 - 400) / 2`.
const WINDOW_OFFSET: usize = (PARAKEET_N_FFT - PARAKEET_WIN_LENGTH) / 2;

static FOURIER_BASIS: OnceLock<Vec<FourierBin>> = OnceLock::new();
static MEL_FILTERS: OnceLock<Vec<[f32; PARAKEET_FFT_BINS]>> = OnceLock::new();

struct FourierBin {
    cos: [f32; PARAKEET_WIN_LENGTH],
    sin: [f32; PARAKEET_WIN_LENGTH],
}

/// Normalized log-mel features, **mel-major**: `values[mel * frames + frame]`.
///
/// Mel-major (rather than Whisper's frame-major) because the downstream
/// `dw_striding` subsampler consumes this as a `[1, n_mels, T]` image, and
/// because per-feature normalization reduces along time.
#[derive(Debug, Clone, PartialEq)]
pub struct ParakeetFeatures {
    pub frames: usize,
    pub mel_bins: usize,
    pub values: Vec<f32>,
}

impl ParakeetFeatures {
    /// Borrow one mel band across all frames.
    pub fn band(&self, mel: usize) -> &[f32] {
        &self.values[mel * self.frames..(mel + 1) * self.frames]
    }
}

fn invalid(field: impl Into<String>, message: impl Into<String>) -> OcelotlError {
    OcelotlError::InvalidRequest(InvalidRequestError {
        field: field.into(),
        message: message.into(),
    })
}

/// Frame count for `samples` input samples: `floor(L / hop) + 1`.
///
/// Every frame this returns is valid — see the module NOTES.
pub fn frame_count(samples: usize) -> usize {
    samples / PARAKEET_HOP_LENGTH + 1
}

/// Compute Parakeet log-mel features from 16 kHz mono f32 PCM.
pub fn parakeet_log_mel(audio: &[f32]) -> Result<ParakeetFeatures> {
    let pad = PARAKEET_N_FFT / 2;
    if audio.len() <= pad {
        return Err(invalid(
            "audio.samples",
            format!("Parakeet preprocessing requires more than {pad} samples for reflect padding"),
        ));
    }

    // 1. Pre-emphasis, applied to the raw waveform BEFORE padding.
    let mut emphasized = Vec::new();
    emphasized
        .try_reserve_exact(audio.len())
        .map_err(|source| reserve_failed("pre-emphasis", source))?;
    emphasized.push(audio[0]);
    for idx in 1..audio.len() {
        emphasized.push(audio[idx] - PREEMPHASIS * audio[idx - 1]);
    }

    // 2. Reflect padding, n_fft/2 each side.
    let centered = reflect_pad_centered(&emphasized, pad)?;

    // 3-6. STFT -> power -> mel -> natural log, accumulated mel-major.
    let frames = frame_count(audio.len());
    let window = hann_window();
    let filters = mel_filterbank();
    let value_count = frames
        .checked_mul(PARAKEET_MEL_BINS)
        .ok_or_else(|| invalid("audio.samples", "Parakeet feature length overflows usize"))?;
    let mut values = vec![0.0_f32; value_count];

    for frame_idx in 0..frames {
        let start = frame_idx * PARAKEET_HOP_LENGTH;
        let power = power_spectrum(&centered, start, &window);
        for (mel_idx, filter) in filters.iter().enumerate() {
            // Same reasoning as the DFT accumulator: 257 terms, and the low
            // bands land near the log guard where error is amplified.
            let energy = power
                .iter()
                .zip(filter.iter())
                .map(|(bin, weight)| (*bin as f64) * (*weight as f64))
                .sum::<f64>() as f32;
            values[mel_idx * frames + frame_idx] = (energy + LOG_ZERO_GUARD).ln();
        }
    }

    // 7. Per-feature normalization across time, unbiased (T-1) variance.
    normalize_per_feature(&mut values, frames, PARAKEET_MEL_BINS);

    Ok(ParakeetFeatures {
        frames,
        mel_bins: PARAKEET_MEL_BINS,
        values,
    })
}

/// `x = (x - mean_t) / (std_t + eps)` per mel band, variance over `T - 1`.
///
/// A single frame leaves the band centred but unscaled: the unbiased variance is
/// undefined at `T = 1`, and the exported graph divides by `sqrt(0) + eps` there,
/// which this reproduces exactly rather than special-casing.
fn normalize_per_feature(values: &mut [f32], frames: usize, mel_bins: usize) {
    if frames == 0 {
        return;
    }
    // f64 accumulators. This is the step that actually needs them: a band whose
    // energy barely moves (mel 0, where pre-emphasis' 0.03 DC gain pins every
    // frame at the 2^-24 log guard) has all its values clustered near -16.6 with
    // near-zero variance. Centering subtracts two nearly-equal large numbers —
    // catastrophic cancellation — and dividing by the resulting tiny std then
    // amplifies whatever error survived. In f32 that alone costs ~1.6e-4.
    let denom = if frames > 1 { (frames - 1) as f64 } else { 1.0 };
    for mel in 0..mel_bins {
        let band = &mut values[mel * frames..(mel + 1) * frames];
        let mean = band.iter().map(|v| *v as f64).sum::<f64>() / frames as f64;
        let mut sum_sq = 0.0_f64;
        for value in band.iter_mut() {
            let centered = (*value as f64) - mean;
            sum_sq += centered * centered;
            *value = centered as f32;
        }
        let scale = 1.0 / ((sum_sq / denom).sqrt() + NORM_EPS as f64);
        for value in band.iter_mut() {
            *value = ((*value as f64) * scale) as f32;
        }
    }
}

fn reflect_pad_centered(audio: &[f32], pad: usize) -> Result<Vec<f32>> {
    let centered_len = audio
        .len()
        .checked_add(2 * pad)
        .ok_or_else(|| invalid("audio.samples", "reflect padding overflows usize"))?;
    let mut centered = Vec::new();
    centered
        .try_reserve_exact(centered_len)
        .map_err(|source| reserve_failed("reflect padding", source))?;

    centered.extend(audio[1..=pad].iter().rev().copied());
    centered.extend_from_slice(audio);
    centered.extend(
        audio[(audio.len() - pad - 1)..(audio.len() - 1)]
            .iter()
            .rev()
            .copied(),
    );
    Ok(centered)
}

/// 400-tap **symmetric** Hann (`denominator N-1`), as the graph's window shows.
///
/// The exported window is a 512-length vector that is zero outside `56..456`;
/// only the 400 non-zero taps are stored here and the frame is indexed with
/// [`WINDOW_OFFSET`], which is both exact and 22% less work than multiplying
/// through the zeros.
fn hann_window() -> [f32; PARAKEET_WIN_LENGTH] {
    let mut window = [0.0; PARAKEET_WIN_LENGTH];
    for (idx, value) in window.iter_mut().enumerate() {
        let phase = 2.0 * PI * (idx as f32) / ((PARAKEET_WIN_LENGTH - 1) as f32);
        *value = 0.5 - 0.5 * phase.cos();
    }
    window
}

fn power_spectrum(
    audio: &[f32],
    frame_start: usize,
    window: &[f32; PARAKEET_WIN_LENGTH],
) -> [f32; PARAKEET_FFT_BINS] {
    let mut power = [0.0; PARAKEET_FFT_BINS];
    let basis = fourier_basis();

    // Accumulate in f64. The terms are f32, but a 400-term naive sum in f32
    // leaves ~2e-4 of error in the *lowest* mel bands: pre-emphasis is a
    // high-pass (0.03 gain at DC), so their energy sits near the 2^-24 log
    // guard, where `ln` amplifies a small absolute error into a large one.
    // The reference FFT's butterfly summation is better conditioned than a
    // flat sum, so matching it needs the wider accumulator, not a looser gate.
    for (freq_bin, bin_power) in power.iter_mut().enumerate() {
        let mut real = 0.0_f64;
        let mut imag = 0.0_f64;
        let fourier = &basis[freq_bin];
        for (tap, &window_value) in window.iter().enumerate() {
            let sample = audio
                .get(frame_start + WINDOW_OFFSET + tap)
                .copied()
                .unwrap_or(0.0)
                * window_value;
            real += (sample as f64) * (fourier.cos[tap] as f64);
            imag += (sample as f64) * (fourier.sin[tap] as f64);
        }
        *bin_power = (real * real + imag * imag) as f32;
    }
    power
}

/// DFT basis folded to the window's non-zero span: the phase still uses the full
/// `n_fft` period and the absolute sample offset `WINDOW_OFFSET + tap`, so this
/// is the 512-point transform of the zero-padded frame, just without the zeros.
///
/// The phase index is reduced `mod n_fft` in **exact integer arithmetic** before
/// it ever reaches a trig call, and the trig itself is evaluated in `f64`. Both
/// matter: `k·n` reaches ~116_000 here, and evaluating `cos` on an argument that
/// large in `f32` loses most of the mantissa to range reduction. Doing it naively
/// costs ~3e-2 in the top mel bands — 300x the parity tolerance — while the
/// filterbank, the obvious suspect, is accurate to 3e-7 either way.
fn fourier_basis() -> &'static [FourierBin] {
    FOURIER_BASIS.get_or_init(|| {
        (0..PARAKEET_FFT_BINS)
            .map(|freq_bin| {
                let mut cos = [0.0_f32; PARAKEET_WIN_LENGTH];
                let mut sin = [0.0_f32; PARAKEET_WIN_LENGTH];
                for tap in 0..PARAKEET_WIN_LENGTH {
                    // Exact: (k * n) mod N, integers, no rounding yet.
                    let phase_index = (freq_bin * (WINDOW_OFFSET + tap)) % PARAKEET_N_FFT;
                    let angle = -2.0 * std::f64::consts::PI * (phase_index as f64)
                        / (PARAKEET_N_FFT as f64);
                    cos[tap] = angle.cos() as f32;
                    sin[tap] = angle.sin() as f32;
                }
                FourierBin { cos, sin }
            })
            .collect()
    })
}

/// 128 Slaney-scale triangles with Slaney area normalization (`2 / (right-left)`).
fn mel_filterbank() -> &'static [[f32; PARAKEET_FFT_BINS]] {
    MEL_FILTERS.get_or_init(|| {
        let min_mel = hz_to_slaney_mel(0.0);
        let max_mel = hz_to_slaney_mel((PARAKEET_SAMPLE_RATE_HZ / 2) as f32);
        let mel_step = (max_mel - min_mel) / ((PARAKEET_MEL_BINS + 1) as f32);
        let points = (0..PARAKEET_MEL_BINS + 2)
            .map(|idx| slaney_mel_to_hz(min_mel + mel_step * (idx as f32)))
            .collect::<Vec<_>>();

        (0..PARAKEET_MEL_BINS)
            .map(|mel_idx| {
                let (left, center, right) =
                    (points[mel_idx], points[mel_idx + 1], points[mel_idx + 2]);
                let mut filter = [0.0; PARAKEET_FFT_BINS];
                for (bin_idx, weight) in filter.iter_mut().enumerate() {
                    let hz = (bin_idx as f32) * (PARAKEET_SAMPLE_RATE_HZ as f32)
                        / (PARAKEET_N_FFT as f32);
                    *weight = if hz <= left || hz >= right {
                        0.0
                    } else if hz <= center {
                        (hz - left) / (center - left)
                    } else {
                        (right - hz) / (right - center)
                    };
                    *weight *= 2.0 / (right - left);
                }
                filter
            })
            .collect()
    })
}

fn hz_to_slaney_mel(hz: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    if hz >= MIN_LOG_HZ {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / SLANEY_LOG_STEP
    } else {
        hz / F_SP
    }
}

fn slaney_mel_to_hz(mel: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (SLANEY_LOG_STEP * (mel - MIN_LOG_MEL)).exp()
    } else {
        mel * F_SP
    }
}

fn reserve_failed(what: &str, source: std::collections::TryReserveError) -> OcelotlError {
    OcelotlError::Runtime(RuntimeError {
        message: format!("failed to reserve Parakeet {what}: {source}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_matches_floor_plus_one() {
        // Hand-checked against the exported graph's features_lens on the pinned
        // fixtures: 176000 -> 1101, 12000 -> 76, 1936000 -> 12101.
        assert_eq!(frame_count(176_000), 1101);
        assert_eq!(frame_count(12_000), 76);
        assert_eq!(frame_count(1_936_000), 12101);
    }

    #[test]
    fn hann_window_is_symmetric_and_zero_at_both_ends() {
        let w = hann_window();
        assert_eq!(w[0], 0.0);
        assert!(w[PARAKEET_WIN_LENGTH - 1].abs() < 1e-7);
        // Symmetric (N-1 denominator) => w[n] == w[N-1-n].
        for n in 0..PARAKEET_WIN_LENGTH {
            assert!((w[n] - w[PARAKEET_WIN_LENGTH - 1 - n]).abs() < 1e-6);
        }
        // Peak of a symmetric 400-tap Hann sits just under 1.0, never at 1.0.
        let max = w.iter().copied().fold(f32::MIN, f32::max);
        assert!((0.999_9..1.0).contains(&max), "peak was {max}");
    }

    #[test]
    fn preemphasis_passes_first_sample_through() {
        // y[0] = x[0]; y[1] = x[1] - 0.97*x[0]. Hand-computed.
        let audio: Vec<f32> = (0..600).map(|i| (i as f32) * 0.001).collect();
        let mut emphasized = vec![audio[0]];
        for i in 1..audio.len() {
            emphasized.push(audio[i] - PREEMPHASIS * audio[i - 1]);
        }
        assert_eq!(emphasized[0], 0.0);
        assert!((emphasized[1] - (0.001 - 0.97 * 0.0)).abs() < 1e-9);
        assert!((emphasized[2] - (0.002 - 0.97 * 0.001)).abs() < 1e-9);
    }

    #[test]
    fn normalize_per_feature_matches_hand_computed_unbiased_std() {
        // One band, T=4: [1,2,3,4]. mean=2.5, centered=[-1.5,-0.5,0.5,1.5],
        // sum_sq=5, var=5/3 (UNBIASED, T-1=3), std=1.290994,
        // scale=1/(1.290994+1e-5).
        let mut values = vec![1.0_f32, 2.0, 3.0, 4.0];
        normalize_per_feature(&mut values, 4, 1);
        let std = (5.0_f32 / 3.0).sqrt();
        let expected: Vec<f32> = [-1.5, -0.5, 0.5, 1.5]
            .iter()
            .map(|c| c / (std + NORM_EPS))
            .collect();
        for (got, want) in values.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got} want {want}");
        }
        // A biased (T) denominator would give std=1.118034 and fail this.
        assert!((std - 1.290_994).abs() < 1e-5);
    }

    #[test]
    fn reflect_pad_mirrors_without_repeating_the_edge() {
        // torch/numpy reflect: front = [x[p]..x[1]], back = [x[L-2]..x[L-p-1]].
        let audio = [10.0_f32, 20.0, 30.0, 40.0, 50.0];
        let padded = reflect_pad_centered(&audio, 2).expect("pad");
        assert_eq!(
            padded,
            vec![30.0, 20.0, 10.0, 20.0, 30.0, 40.0, 50.0, 40.0, 30.0]
        );
    }

    #[test]
    fn rejects_audio_shorter_than_the_pad() {
        assert!(parakeet_log_mel(&vec![0.0_f32; 256]).is_err());
    }
}
