use std::{f32::consts::PI, hint::black_box};

use criterion::{Criterion, criterion_group, criterion_main};

// The production Whisper module intentionally keeps preprocessing helpers
// private. Include the owning source module for this package-local benchmark
// so measuring log-mel does not widen Ocelotl's public API.
#[allow(dead_code, unused_imports)]
#[path = "../src/whisper/audio.rs"]
mod whisper_audio;

use whisper_audio::{AudioMetadata, WHISPER_SAMPLE_RATE_HZ, log_mel_spectrogram};

fn benchmark_whisper_log_mel(criterion: &mut Criterion) {
    let metadata = AudioMetadata {
        sample_rate_hz: WHISPER_SAMPLE_RATE_HZ,
        channels: 1,
    };
    let audio = deterministic_audio(WHISPER_SAMPLE_RATE_HZ as usize / 5);

    // Exclude one-time Fourier-basis initialization from steady-state samples.
    log_mel_spectrogram(&audio, metadata).expect("deterministic audio must preprocess");

    criterion.bench_function("whisper/log_mel/200ms_mono_16khz", |bencher| {
        bencher.iter(|| {
            let mel = log_mel_spectrogram(black_box(&audio), metadata)
                .expect("deterministic audio must preprocess");
            black_box(mel);
        });
    });
}

fn deterministic_audio(samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|index| {
            let seconds = index as f32 / WHISPER_SAMPLE_RATE_HZ as f32;
            0.25 * (2.0 * PI * 440.0 * seconds).sin() + 0.05 * (2.0 * PI * 880.0 * seconds).sin()
        })
        .collect()
}

criterion_group!(benches, benchmark_whisper_log_mel);
criterion_main!(benches);
