//! Whisper audio preprocessing boundary.

use std::f32::consts::PI;

use ocelotl_core::{OcelotlError, Result, UnsupportedError};

pub const WHISPER_SAMPLE_RATE_HZ: u32 = 16_000;
pub const WHISPER_FFT_SIZE: usize = 400;
pub const WHISPER_HOP_LENGTH: usize = 160;
pub const WHISPER_MEL_BINS: usize = 80;

const WHISPER_POWER_FLOOR: f32 = 1e-10;
const SLANEY_LOG_STEP: f32 = 1.856_298 / 27.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioMetadata {
    pub sample_rate_hz: u32,
    pub channels: u16,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogMelSpectrogram {
    pub frames: usize,
    pub mel_bins: usize,
    pub values: Vec<f32>,
}

pub fn validate_audio_metadata(metadata: AudioMetadata) -> Result<()> {
    if metadata.sample_rate_hz != WHISPER_SAMPLE_RATE_HZ {
        return Err(OcelotlError::Unsupported(UnsupportedError {
            feature: "whisper_audio.sample_rate_hz".to_string(),
            requested: Some(metadata.sample_rate_hz.to_string()),
            supported: vec![WHISPER_SAMPLE_RATE_HZ.to_string()],
        }));
    }

    if metadata.channels != 1 {
        return Err(OcelotlError::Unsupported(UnsupportedError {
            feature: "whisper_audio.channels".to_string(),
            requested: Some(metadata.channels.to_string()),
            supported: vec!["1".to_string()],
        }));
    }

    Ok(())
}

pub fn log_mel_spectrogram(audio: &[f32], metadata: AudioMetadata) -> Result<LogMelSpectrogram> {
    validate_audio_metadata(metadata)?;

    if audio.len() < WHISPER_FFT_SIZE {
        return Err(OcelotlError::InvalidRequest(
            ocelotl_core::InvalidRequestError {
                field: "audio.samples".to_string(),
                message: format!(
                    "Whisper audio preprocessing requires at least {WHISPER_FFT_SIZE} samples"
                ),
            },
        ));
    }

    let centered = reflect_pad_centered(audio);
    let frames = stft_frame_count(centered.len()).saturating_sub(1);
    let window = hann_window();
    let mel_filters = mel_filterbank();
    let mut values = Vec::with_capacity(frames * WHISPER_MEL_BINS);

    for frame_idx in 0..frames {
        let start = frame_idx * WHISPER_HOP_LENGTH;
        let power = power_spectrum(&centered, start, &window);

        for filter in &mel_filters {
            let energy = power
                .iter()
                .zip(filter)
                .map(|(bin_power, weight)| bin_power * weight)
                .sum::<f32>();
            values.push(energy.max(WHISPER_POWER_FLOOR).log10());
        }
    }

    apply_whisper_log_mel_postprocess(&mut values);

    Ok(LogMelSpectrogram {
        frames,
        mel_bins: WHISPER_MEL_BINS,
        values,
    })
}

fn reflect_pad_centered(audio: &[f32]) -> Vec<f32> {
    let pad = WHISPER_FFT_SIZE / 2;
    let mut centered = Vec::with_capacity(audio.len() + 2 * pad);

    centered.extend(audio[1..=pad].iter().rev().copied());
    centered.extend_from_slice(audio);
    centered.extend(
        audio[(audio.len() - pad - 1)..(audio.len() - 1)]
            .iter()
            .rev()
            .copied(),
    );

    centered
}

fn stft_frame_count(samples: usize) -> usize {
    (samples - WHISPER_FFT_SIZE) / WHISPER_HOP_LENGTH + 1
}

fn hann_window() -> [f32; WHISPER_FFT_SIZE] {
    let mut window = [0.0; WHISPER_FFT_SIZE];
    for (idx, value) in window.iter_mut().enumerate() {
        let phase = 2.0 * PI * (idx as f32) / (WHISPER_FFT_SIZE as f32);
        *value = 0.5 - 0.5 * phase.cos();
    }
    window
}

fn power_spectrum(
    audio: &[f32],
    frame_start: usize,
    window: &[f32; WHISPER_FFT_SIZE],
) -> [f32; WHISPER_FFT_SIZE / 2 + 1] {
    let mut power = [0.0; WHISPER_FFT_SIZE / 2 + 1];

    for (freq_bin, bin_power) in power.iter_mut().enumerate() {
        let mut real = 0.0_f32;
        let mut imag = 0.0_f32;

        for (n, &window_value) in window.iter().enumerate() {
            let sample = audio.get(frame_start + n).copied().unwrap_or(0.0) * window_value;
            let angle = -2.0 * PI * (freq_bin as f32) * (n as f32) / (WHISPER_FFT_SIZE as f32);
            real += sample * angle.cos();
            imag += sample * angle.sin();
        }

        *bin_power = real.mul_add(real, imag * imag);
    }

    power
}

fn mel_filterbank() -> Vec<[f32; WHISPER_FFT_SIZE / 2 + 1]> {
    let min_mel = hz_to_slaney_mel(0.0);
    let max_mel = hz_to_slaney_mel((WHISPER_SAMPLE_RATE_HZ / 2) as f32);
    let mel_step = (max_mel - min_mel) / ((WHISPER_MEL_BINS + 1) as f32);

    let mel_points = (0..WHISPER_MEL_BINS + 2)
        .map(|idx| slaney_mel_to_hz(min_mel + mel_step * (idx as f32)))
        .collect::<Vec<_>>();

    (0..WHISPER_MEL_BINS)
        .map(|mel_idx| {
            let left = mel_points[mel_idx];
            let center = mel_points[mel_idx + 1];
            let right = mel_points[mel_idx + 2];
            let mut filter = [0.0; WHISPER_FFT_SIZE / 2 + 1];

            for (bin_idx, weight) in filter.iter_mut().enumerate() {
                let hz =
                    (bin_idx as f32) * (WHISPER_SAMPLE_RATE_HZ as f32) / (WHISPER_FFT_SIZE as f32);
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
}

fn apply_whisper_log_mel_postprocess(values: &mut [f32]) {
    let max_floor = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 8.0;

    for value in values {
        *value = value.max(max_floor);
        *value = (*value + 4.0) / 4.0;
    }
}

fn hz_to_slaney_mel(hz: f32) -> f32 {
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = (MIN_LOG_HZ - F_MIN) / F_SP;

    if hz >= MIN_LOG_HZ {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / SLANEY_LOG_STEP
    } else {
        (hz - F_MIN) / F_SP
    }
}

fn slaney_mel_to_hz(mel: f32) -> f32 {
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = (MIN_LOG_HZ - F_MIN) / F_SP;

    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (SLANEY_LOG_STEP * (mel - MIN_LOG_MEL)).exp()
    } else {
        F_MIN + F_SP * mel
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ocelotl_core::OcelotlError;

    #[test]
    fn whisper_audio_constants_match_expected_boundary() {
        assert_eq!(WHISPER_SAMPLE_RATE_HZ, 16_000);
        assert_eq!(WHISPER_FFT_SIZE, 400);
        assert_eq!(WHISPER_HOP_LENGTH, 160);
        assert_eq!(WHISPER_MEL_BINS, 80);
    }

    #[test]
    fn audio_metadata_accepts_whisper_16khz_mono() {
        let metadata = AudioMetadata {
            sample_rate_hz: 16_000,
            channels: 1,
        };

        validate_audio_metadata(metadata).expect("16 kHz mono audio should be accepted");
    }

    #[test]
    fn audio_metadata_rejects_unsupported_sample_rate_before_compute() {
        let metadata = AudioMetadata {
            sample_rate_hz: 44_100,
            channels: 1,
        };

        let err = validate_audio_metadata(metadata)
            .expect_err("unsupported sample rate should fail before compute");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(unsupported.feature, "whisper_audio.sample_rate_hz");
                assert_eq!(unsupported.requested.as_deref(), Some("44100"));
                assert_eq!(unsupported.supported, vec!["16000"]);
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn audio_metadata_rejects_non_mono_before_compute() {
        let metadata = AudioMetadata {
            sample_rate_hz: 16_000,
            channels: 2,
        };

        let err = validate_audio_metadata(metadata)
            .expect_err("non-mono audio should fail before compute");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(unsupported.feature, "whisper_audio.channels");
                assert_eq!(unsupported.requested.as_deref(), Some("2"));
                assert_eq!(unsupported.supported, vec!["1"]);
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn tiny_waveform_fixture_maps_to_pinned_log_mel_values() {
        let metadata = AudioMetadata {
            sample_rate_hz: 16_000,
            channels: 1,
        };
        let audio = tiny_waveform_fixture();

        let spectrogram = log_mel_spectrogram(&audio, metadata).expect("fixture should preprocess");

        assert_eq!(spectrogram.frames, 2);
        assert_eq!(spectrogram.mel_bins, WHISPER_MEL_BINS);
        assert_eq!(spectrogram.values.len(), 2 * WHISPER_MEL_BINS);

        let expected = [
            0.5390415_f32,
            0.6864456,
            0.6932472,
            0.60004145,
            0.46436572,
            0.50140697,
        ];
        assert_close(&spectrogram.values[..expected.len()], &expected, 1e-5);
    }

    #[test]
    fn whisper_log_mel_postprocess_clamps_to_eight_decades_and_normalizes() {
        let mut values = vec![-20.0, -8.0, -4.0, 0.0];

        apply_whisper_log_mel_postprocess(&mut values);

        assert_close(&values, &[-1.0, -1.0, 0.0, 1.0], 1e-6);
    }

    fn tiny_waveform_fixture() -> Vec<f32> {
        let mut audio = vec![0.0; WHISPER_FFT_SIZE];
        audio[0] = 1.0;
        audio[80] = -0.5;
        audio[160] = 0.25;
        audio[240] = -0.125;
        audio[320] = 0.0625;
        audio
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let delta = (actual - expected).abs();
            assert!(
                delta <= tolerance,
                "index {idx}: expected {expected}, got {actual}, delta {delta}"
            );
        }
    }
}
