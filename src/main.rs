use std::path::{Path, PathBuf};
use std::time::Instant;

use ocelotl_core::TokenId;
use ocelotl_loader::{LoadedTensor, inspect_safetensors, load_safetensors_tensor_f32};
use ocelotl_models::whisper::{
    WhisperConfig, WhisperModel,
    audio::{AudioMetadata, LogMelSpectrogram, log_mel_spectrogram},
    parse_whisper_config_json, required_whisper_tensor_names, validate_whisper_tensors,
};
use ocelotl_tokenizer::{
    WhisperDecodeMask, WhisperStartupTokens, WhisperTokenMaskDecision,
    whisper_multilingual_english_transcribe_tokens,
};
use serde::Deserialize;

const OPENAI_MULTILINGUAL_VOCAB_THRESHOLD: usize = 51_865;

#[derive(Debug)]
struct BenchWhisperArgs {
    config_path: PathBuf,
    model_path: PathBuf,
    audio_path: PathBuf,
    expected_tokens_path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct ExpectedTokens {
    expected_token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WavMetadata {
    audio_format: u16,
    channels: u16,
    sample_rate_hz: u32,
    bits_per_sample: u16,
    data_bytes: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct WavAudio {
    metadata: WavMetadata,
    samples: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WavLayout {
    metadata: WavMetadata,
    data_offset: usize,
    data_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WhisperArtifactTokenizerFamily {
    Multilingual,
    EnglishOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WhisperArtifactDecodePolicy {
    family: WhisperArtifactTokenizerFamily,
    tokens: WhisperStartupTokens,
}

impl WhisperArtifactDecodePolicy {
    fn for_config(config: &WhisperConfig) -> Self {
        if config.vocab_size >= OPENAI_MULTILINGUAL_VOCAB_THRESHOLD {
            Self::multilingual_english()
        } else {
            Self::english_only()
        }
    }

    fn multilingual_english() -> Self {
        Self {
            family: WhisperArtifactTokenizerFamily::Multilingual,
            tokens: whisper_multilingual_english_transcribe_tokens(),
        }
    }

    fn english_only() -> Self {
        Self {
            family: WhisperArtifactTokenizerFamily::EnglishOnly,
            tokens: WhisperStartupTokens {
                end_of_text: TokenId(50_256),
                start_of_transcript: TokenId(50_257),
                language: TokenId(50_258),
                transcribe_task: TokenId(50_358),
                no_timestamps: TokenId(50_362),
                first_timestamp: TokenId(50_363),
            },
        }
    }

    fn startup_prompt(self) -> Vec<TokenId> {
        match self.family {
            WhisperArtifactTokenizerFamily::Multilingual => {
                self.tokens.transcribe_no_timestamps_prompt()
            }
            WhisperArtifactTokenizerFamily::EnglishOnly => {
                vec![self.tokens.start_of_transcript, self.tokens.no_timestamps]
            }
        }
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("bench-whisper-transcribe") => run_bench_whisper_transcribe(parse_bench_args(args)?),
        Some("--help") | Some("-h") => {
            print_help();
            Ok(())
        }
        Some(other) => Err(format!(
            "unsupported command {other:?}; run `ocelotl --help`"
        )),
        None => {
            println!("ocelotl {}", ocelotl::VERSION);
            Ok(())
        }
    }
}

fn print_help() {
    println!("ocelotl {}", ocelotl::VERSION);
    println!();
    println!("Commands:");
    println!(
        "  bench-whisper-transcribe --config-path <path> --model-path <path> --audio-path <path> --expected-tokens-path <path>"
    );
}

fn parse_bench_args(args: impl IntoIterator<Item = String>) -> Result<BenchWhisperArgs, String> {
    let mut config_path = None;
    let mut model_path = None;
    let mut audio_path = None;
    let mut expected_tokens_path = None;
    let mut iter = args.into_iter();

    while let Some(flag) = iter.next() {
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--config-path" => config_path = Some(PathBuf::from(value)),
            "--model-path" => model_path = Some(PathBuf::from(value)),
            "--audio-path" => audio_path = Some(PathBuf::from(value)),
            "--expected-tokens-path" => expected_tokens_path = Some(PathBuf::from(value)),
            _ => {
                return Err(format!(
                    "unsupported bench-whisper-transcribe flag {flag:?}"
                ));
            }
        }
    }

    Ok(BenchWhisperArgs {
        config_path: config_path.ok_or("missing --config-path")?,
        model_path: model_path.ok_or("missing --model-path")?,
        audio_path: audio_path.ok_or("missing --audio-path")?,
        expected_tokens_path: expected_tokens_path.ok_or("missing --expected-tokens-path")?,
    })
}

fn run_bench_whisper_transcribe(args: BenchWhisperArgs) -> Result<(), String> {
    let started = Instant::now();
    ensure_file(&args.config_path, "config")?;
    ensure_file(&args.model_path, "model")?;
    ensure_file(&args.audio_path, "audio")?;
    ensure_file(&args.expected_tokens_path, "expected tokens")?;

    let config_raw = std::fs::read_to_string(&args.config_path).map_err(|err| {
        format!(
            "failed to read config at {} - {err}",
            args.config_path.display()
        )
    })?;
    let whisper_config = parse_whisper_config_json(&config_raw).map_err(|err| {
        format!(
            "invalid Whisper config at {} - {err:?}",
            args.config_path.display()
        )
    })?;

    let manifest = inspect_safetensors(&args.model_path).map_err(|err| {
        format!(
            "failed to inspect model at {} - {err:?}",
            args.model_path.display()
        )
    })?;
    validate_whisper_tensors(&manifest, &whisper_config, Some(&args.model_path)).map_err(
        |err| {
            format!(
                "model at {} does not match Whisper tensor contract - {err:?}",
                args.model_path.display()
            )
        },
    )?;

    let expected_tokens = read_expected_tokens(&args.expected_tokens_path)?;
    if expected_tokens.is_empty() {
        return Err("expected token sequence must be non-empty".to_string());
    }
    if expected_tokens.len() > whisper_config.text_context_length {
        return Err(format!(
            "expected token sequence length {} exceeds text context length {}",
            expected_tokens.len(),
            whisper_config.text_context_length
        ));
    }

    let model = WhisperModel::new(
        whisper_config.clone(),
        load_required_whisper_tensors(&args.model_path, &whisper_config)?,
    )
    .map_err(|err| format!("failed to construct Whisper model - {err:?}"))?;

    let wav = read_wav_mono_samples(&args.audio_path)?;
    let mel = log_mel_spectrogram(
        &wav.samples,
        AudioMetadata {
            sample_rate_hz: wav.metadata.sample_rate_hz,
            channels: wav.metadata.channels,
        },
    )
    .map_err(|err| format!("failed to compute Whisper log-mel spectrogram - {err:?}"))?;

    let decode_policy = WhisperArtifactDecodePolicy::for_config(&whisper_config);
    let generated =
        generate_tokens_to_expected_length(&model, &mel, expected_tokens.len(), decode_policy)?;
    let elapsed_ms = started.elapsed().as_millis();

    let matches_expected = generated == expected_tokens;
    let token_ids: Vec<u32> = generated.iter().map(|token| token.0).collect();
    let output = serde_json::json!({
        "status": if matches_expected { "completed" } else { "mismatch" },
        "elapsed_ms": elapsed_ms,
        "token_count": generated.len(),
        "tokens": token_ids,
        "matches_expected": matches_expected
    });
    println!("{output}");

    if matches_expected {
        Ok(())
    } else {
        Err("generated tokens did not match expected token fixture".to_string())
    }
}

fn ensure_file(path: &Path, label: &str) -> Result<(), String> {
    if path.is_file() {
        return Ok(());
    }
    Err(format!(
        "missing Ocelotl Whisper {label} artifact at {}; prepare local-artifacts/whisper_tiny_en per docs/artifact-preparation.md and docs/benchmarks/whisper-cpp.md",
        path.display()
    ))
}

fn read_expected_tokens(path: &Path) -> Result<Vec<TokenId>, String> {
    let raw = std::fs::read_to_string(path).map_err(|err| {
        format!(
            "failed to read expected tokens at {} - {err}",
            path.display()
        )
    })?;
    let expected: ExpectedTokens = serde_json::from_str(&raw).map_err(|err| {
        format!(
            "failed to parse expected tokens at {} - {err}",
            path.display()
        )
    })?;
    Ok(expected
        .expected_token_ids
        .into_iter()
        .map(TokenId)
        .collect())
}

fn load_required_whisper_tensors(
    path: &Path,
    config: &WhisperConfig,
) -> Result<Vec<LoadedTensor>, String> {
    required_whisper_tensor_names(config)
        .into_iter()
        .map(|name| {
            load_safetensors_tensor_f32(path, &name).map_err(|err| {
                format!(
                    "failed to load required Whisper tensor {name:?} from {} - {err:?}",
                    path.display()
                )
            })
        })
        .collect()
}

fn generate_tokens_to_expected_length(
    model: &WhisperModel,
    mel: &LogMelSpectrogram,
    expected_len: usize,
    policy: WhisperArtifactDecodePolicy,
) -> Result<Vec<TokenId>, String> {
    let mut generated = policy.startup_prompt();
    let mask = WhisperDecodeMask::transcribe_without_timestamps(policy.tokens);

    while generated.len() < expected_len {
        let logits = model
            .forward_next_token_logits(&mel.values, mel.frames, &generated)
            .map_err(|err| {
                format!(
                    "Whisper forward_next_token_logits failed at generated length {} - {err:?}",
                    generated.len()
                )
            })?;
        let next = masked_greedy_sample(&logits, mask)?;
        generated.push(next);
        if next == policy.tokens.end_of_text {
            break;
        }
    }

    Ok(generated)
}

fn masked_greedy_sample(logits: &[f32], mask: WhisperDecodeMask) -> Result<TokenId, String> {
    if logits.is_empty() {
        return Err("cannot sample from empty logits".to_string());
    }

    let mut best = None;
    for (idx, &logit) in logits.iter().enumerate() {
        let token = TokenId(u32::try_from(idx).map_err(|_| "vocab index exceeds u32")?);
        if mask.mask_token(token) == WhisperTokenMaskDecision::Suppress {
            continue;
        }
        if best.is_none_or(|(_, best_logit)| logit > best_logit) {
            best = Some((idx, logit));
        }
    }

    let (idx, _) = best.ok_or("Whisper decode mask suppressed every logit")?;
    Ok(TokenId(
        u32::try_from(idx).map_err(|_| "vocab index exceeds u32")?,
    ))
}

fn read_wav_mono_samples(path: &Path) -> Result<WavAudio, String> {
    let bytes = std::fs::read(path)
        .map_err(|err| format!("failed to read WAV at {} - {err}", path.display()))?;
    parse_wav_mono_samples(&bytes)
        .map_err(|err| format!("unsupported WAV sample data at {} - {err}", path.display()))
}

fn parse_wav_mono_samples(bytes: &[u8]) -> Result<WavAudio, String> {
    let layout = parse_wav_layout(bytes)?;
    if layout.metadata.channels != 1 {
        return Err(format!(
            "only mono WAV data is supported, got {} channels",
            layout.metadata.channels
        ));
    }

    let data = bytes
        .get(layout.data_offset..layout.data_offset + layout.data_bytes)
        .ok_or_else(|| "data chunk extends beyond file length".to_string())?;
    let samples = match (
        layout.metadata.audio_format,
        layout.metadata.bits_per_sample,
    ) {
        (1, 16) => decode_pcm16_samples(data)?,
        (1, bits) => {
            return Err(format!(
                "unsupported PCM WAV bits_per_sample {bits}; supported PCM format is 16-bit"
            ));
        }
        (3, 32) => decode_float32_samples(data)?,
        (3, bits) => {
            return Err(format!(
                "unsupported IEEE float WAV bits_per_sample {bits}; supported float format is 32-bit"
            ));
        }
        (format, bits) => {
            return Err(format!(
                "unsupported WAV audio_format {format} with bits_per_sample {bits}; supported formats are PCM16 and IEEE float32"
            ));
        }
    };

    Ok(WavAudio {
        metadata: layout.metadata,
        samples,
    })
}

fn parse_wav_layout(bytes: &[u8]) -> Result<WavLayout, String> {
    if bytes.len() < 12 {
        return Err("file is shorter than a RIFF/WAVE header".to_string());
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("file must start with RIFF/WAVE".to_string());
    }

    let mut offset = 12usize;
    let mut fmt = None;
    let mut data = None;

    while offset.checked_add(8).is_some_and(|end| end <= bytes.len()) {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_size = read_u32_le(bytes, offset + 4)? as usize;
        let chunk_start = offset + 8;
        let chunk_end = chunk_start
            .checked_add(chunk_size)
            .ok_or_else(|| "chunk size overflows usize".to_string())?;
        if chunk_end > bytes.len() {
            return Err("chunk size extends beyond file length".to_string());
        }

        match chunk_id {
            b"fmt " => {
                if chunk_size < 16 {
                    return Err("fmt chunk must be at least 16 bytes".to_string());
                }
                fmt = Some((
                    read_u16_le(bytes, chunk_start)?,
                    read_u16_le(bytes, chunk_start + 2)?,
                    read_u32_le(bytes, chunk_start + 4)?,
                    read_u16_le(bytes, chunk_start + 14)?,
                ));
            }
            b"data" => data = Some((chunk_start, chunk_size)),
            _ => {}
        }

        offset = chunk_end
            .checked_add(chunk_size % 2)
            .ok_or_else(|| "chunk padding overflows usize".to_string())?;
    }

    let (audio_format, channels, sample_rate_hz, bits_per_sample) =
        fmt.ok_or_else(|| "missing fmt chunk".to_string())?;
    let (data_offset, data_bytes) = data.ok_or_else(|| "missing data chunk".to_string())?;

    Ok(WavLayout {
        metadata: WavMetadata {
            audio_format,
            channels,
            sample_rate_hz,
            bits_per_sample,
            data_bytes,
        },
        data_offset,
        data_bytes,
    })
}

fn decode_pcm16_samples(data: &[u8]) -> Result<Vec<f32>, String> {
    let chunks = data.chunks_exact(2);
    if !chunks.remainder().is_empty() {
        return Err(format!(
            "PCM16 data chunk has {} bytes, which is not divisible by 2",
            data.len()
        ));
    }

    Ok(chunks
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32768.0)
        .collect())
}

fn decode_float32_samples(data: &[u8]) -> Result<Vec<f32>, String> {
    let chunks = data.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(format!(
            "IEEE float32 data chunk has {} bytes, which is not divisible by 4",
            data.len()
        ));
    }

    Ok(chunks
        .map(|sample| f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]))
        .collect())
}

fn read_u16_le(bytes: &[u8], offset: usize) -> Result<u16, String> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| "u16 offset overflows usize".to_string())?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| "unexpected EOF while reading u16".to_string())?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| "u32 offset overflows usize".to_string())?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| "unexpected EOF while reading u32".to_string())?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}
