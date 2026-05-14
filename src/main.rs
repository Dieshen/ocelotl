use std::time::Instant;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use ocelotl_core::TokenId;
use ocelotl_kernels::{CpuKernelBackend, CpuKernelMode};
use ocelotl_loader::{LoadedTensor, inspect_safetensors, load_safetensors_tensors_f32};
use ocelotl_models::whisper::{
    WhisperAudioEncodeTimings, WhisperConfig, WhisperEncodedAudio, WhisperModel,
    audio::{AudioMetadata, log_mel_spectrogram},
    parse_whisper_config_json, required_whisper_tensor_names, validate_whisper_tensors,
};
use ocelotl_tokenizer::{
    JsonTokenizer, Tokenizer, WhisperDecodeMask, WhisperStartupTokens, WhisperTokenMaskDecision,
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
    tokenizer_path: Option<PathBuf>,
    cpu_kernel_mode: CpuKernelMode,
    cpu_threads: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FetchHfArgs {
    repo: String,
    revision: String,
    local_dir: PathBuf,
    include: Vec<String>,
    execute: bool,
}

#[derive(Debug, Deserialize)]
struct ExpectedTokens {
    expected_token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct BenchWhisperTimings {
    total_ms: u128,
    config_parse_ms: u128,
    manifest_validate_ms: u128,
    expected_tokens_read_ms: u128,
    tensor_load_model_ms: u128,
    tokenizer_load_ms: Option<u128>,
    wav_read_ms: u128,
    log_mel_ms: u128,
    audio_encode_ms: u128,
    audio_encode_detail: WhisperAudioEncodeTimings,
    decode_total_ms: u128,
    text_decode_ms: Option<u128>,
    decode_token_ms: Vec<u128>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct BenchWhisperTextOutput {
    text: Option<String>,
    tokenizer_load_ms: Option<u128>,
    text_decode_ms: Option<u128>,
}

impl BenchWhisperTimings {
    fn resident_audio_to_tokens_ms(&self) -> u128 {
        self.log_mel_ms + self.audio_encode_ms + self.decode_total_ms
    }

    fn resident_mel_to_tokens_ms(&self) -> u128 {
        self.audio_encode_ms + self.decode_total_ms
    }
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
        Some("fetch") => match args.next().as_deref() {
            Some("hf") => run_fetch_hf(parse_fetch_hf_args(args)?),
            Some(other) => Err(format!(
                "unsupported fetch target {other:?}; supported targets: hf"
            )),
            None => Err("missing fetch target; expected `ocelotl fetch hf ...`".to_string()),
        },
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
        "  bench-whisper-transcribe --config-path <path> --model-path <path> --audio-path <path> --expected-tokens-path <path> [--tokenizer-path <path>] [--cpu-kernel-mode scalar|optimized|avx2] [--cpu-threads <n>]"
    );
    println!(
        "  fetch hf --repo <repo> --revision <sha> --local-dir <path> [--include <pattern> ...] [--execute]"
    );
}

fn parse_bench_args(args: impl IntoIterator<Item = String>) -> Result<BenchWhisperArgs, String> {
    let mut config_path = None;
    let mut model_path = None;
    let mut audio_path = None;
    let mut expected_tokens_path = None;
    let mut tokenizer_path = None;
    let mut cpu_kernel_mode = CpuKernelMode::Scalar;
    let mut cpu_threads: usize = 1;
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
            "--tokenizer-path" => tokenizer_path = Some(PathBuf::from(value)),
            "--cpu-kernel-mode" => cpu_kernel_mode = parse_cpu_kernel_mode(&value)?,
            "--cpu-threads" => cpu_threads = parse_cpu_threads(&value)?,
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
        tokenizer_path,
        cpu_kernel_mode,
        cpu_threads,
    })
}

fn parse_cpu_threads(value: &str) -> Result<usize, String> {
    let parsed: usize = value
        .parse()
        .map_err(|_| format!("invalid --cpu-threads value {value:?}; expected a positive integer"))?;
    if parsed == 0 {
        return Err("--cpu-threads must be >= 1".to_string());
    }
    Ok(parsed)
}

fn parse_cpu_kernel_mode(value: &str) -> Result<CpuKernelMode, String> {
    match value {
        "scalar" => Ok(CpuKernelMode::Scalar),
        "optimized" => Ok(CpuKernelMode::Optimized),
        "avx2" => Ok(CpuKernelMode::Avx2),
        other => Err(format!(
            "unsupported --cpu-kernel-mode {other:?}; supported values: scalar, optimized, avx2"
        )),
    }
}

fn parse_fetch_hf_args(args: impl IntoIterator<Item = String>) -> Result<FetchHfArgs, String> {
    let mut repo = None;
    let mut revision = None;
    let mut local_dir = None;
    let mut include = Vec::new();
    let mut execute = false;
    let mut iter = args.into_iter();

    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--repo" => repo = Some(next_arg(&mut iter, "--repo")?),
            "--revision" => revision = Some(next_arg(&mut iter, "--revision")?),
            "--local-dir" => local_dir = Some(PathBuf::from(next_arg(&mut iter, "--local-dir")?)),
            "--include" => include.push(next_arg(&mut iter, "--include")?),
            "--execute" => execute = true,
            _ => return Err(format!("unsupported fetch hf flag {flag:?}")),
        }
    }

    Ok(FetchHfArgs {
        repo: repo.ok_or("missing --repo")?,
        revision: revision.ok_or("missing --revision")?,
        local_dir: local_dir.ok_or("missing --local-dir")?,
        include,
        execute,
    })
}

fn next_arg(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn run_fetch_hf(args: FetchHfArgs) -> Result<(), String> {
    let command_args = huggingface_cli_args(&args);
    if !args.execute {
        println!(
            "{}",
            serde_json::json!({
                "status": "planned",
                "execute": false,
                "command": command_line("huggingface-cli", &command_args),
                "note": "pass --execute to run the explicit artifact fetch"
            })
        );
        return Ok(());
    }

    let status = std::process::Command::new("huggingface-cli")
        .args(&command_args)
        .status()
        .map_err(|err| format!("failed to run huggingface-cli: {err}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("huggingface-cli exited with status {status}"))
    }
}

fn huggingface_cli_args(args: &FetchHfArgs) -> Vec<String> {
    let mut command = vec![
        "download".to_string(),
        args.repo.clone(),
        "--revision".to_string(),
        args.revision.clone(),
        "--local-dir".to_string(),
        args.local_dir.display().to_string(),
        "--local-dir-use-symlinks".to_string(),
        "False".to_string(),
    ];
    if !args.include.is_empty() {
        command.push("--include".to_string());
        command.extend(args.include.iter().cloned());
    }
    command
}

fn command_line(program: &str, args: &[String]) -> String {
    std::iter::once(program.to_string())
        .chain(args.iter().map(|arg| shell_quote(arg)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_quote(arg: &str) -> String {
    if arg
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | '\\' | ':'))
    {
        arg.to_string()
    } else {
        format!("\"{}\"", arg.replace('"', "\\\""))
    }
}

fn run_bench_whisper_transcribe(args: BenchWhisperArgs) -> Result<(), String> {
    let started = Instant::now();
    ensure_file(&args.config_path, "config")?;
    ensure_file(&args.model_path, "model")?;
    ensure_file(&args.audio_path, "audio")?;
    ensure_file(&args.expected_tokens_path, "expected tokens")?;
    if let Some(tokenizer_path) = &args.tokenizer_path {
        ensure_file(tokenizer_path, "tokenizer")?;
    }

    let config_started = Instant::now();
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
    let config_parse_ms = config_started.elapsed().as_millis();

    let manifest_started = Instant::now();
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
    let manifest_validate_ms = manifest_started.elapsed().as_millis();

    let expected_started = Instant::now();
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
    let expected_tokens_read_ms = expected_started.elapsed().as_millis();

    let tensor_started = Instant::now();
    let kernels = CpuKernelBackend::with_mode_and_threads(args.cpu_kernel_mode, args.cpu_threads)
        .map_err(|err| format!("failed to build CPU kernel backend - {err:?}"))?;
    let model = WhisperModel::with_kernel_backend(
        whisper_config.clone(),
        load_required_whisper_tensors(&args.model_path, &whisper_config)?,
        Arc::new(kernels),
    )
    .map_err(|err| format!("failed to construct Whisper model - {err:?}"))?;
    let tensor_load_model_ms = tensor_started.elapsed().as_millis();

    let wav_started = Instant::now();
    let wav = read_wav_mono_samples(&args.audio_path)?;
    let wav_read_ms = wav_started.elapsed().as_millis();

    let log_mel_started = Instant::now();
    let mel = log_mel_spectrogram(
        &wav.samples,
        AudioMetadata {
            sample_rate_hz: wav.metadata.sample_rate_hz,
            channels: wav.metadata.channels,
        },
    )
    .map_err(|err| format!("failed to compute Whisper log-mel spectrogram - {err:?}"))?;
    let log_mel_ms = log_mel_started.elapsed().as_millis();

    let decode_policy = WhisperArtifactDecodePolicy::for_config(&whisper_config);
    let audio_encode_started = Instant::now();
    let (encoded_audio, audio_encode_detail) = model
        .encode_audio_features_with_timings(&mel.values, mel.frames)
        .map_err(|err| format!("failed to encode Whisper audio features - {err:?}"))?;
    let audio_encode_ms = audio_encode_started.elapsed().as_millis();

    let decode_started = Instant::now();
    let mut decode_token_ms = Vec::new();
    let generated = generate_tokens_to_expected_length(
        &model,
        &encoded_audio,
        expected_tokens.len(),
        decode_policy,
        &mut decode_token_ms,
    )?;
    let decode_total_ms = decode_started.elapsed().as_millis();

    let text_output = decode_generated_text(args.tokenizer_path.as_deref(), &generated)?;
    let elapsed_ms = started.elapsed().as_millis();

    let matches_expected = generated == expected_tokens;
    let timings = BenchWhisperTimings {
        total_ms: elapsed_ms,
        config_parse_ms,
        manifest_validate_ms,
        expected_tokens_read_ms,
        tensor_load_model_ms,
        tokenizer_load_ms: text_output.tokenizer_load_ms,
        wav_read_ms,
        log_mel_ms,
        audio_encode_ms,
        audio_encode_detail,
        decode_total_ms,
        text_decode_ms: text_output.text_decode_ms,
        decode_token_ms,
    };
    let output = bench_whisper_output(
        matches_expected,
        args.cpu_kernel_mode,
        args.cpu_threads,
        &generated,
        text_output.text.as_deref(),
        &timings,
    );
    println!("{output}");

    if matches_expected {
        Ok(())
    } else {
        Err("generated tokens did not match expected token fixture".to_string())
    }
}

fn bench_whisper_output(
    matches_expected: bool,
    cpu_kernel_mode: CpuKernelMode,
    cpu_threads: usize,
    generated: &[TokenId],
    transcript_text: Option<&str>,
    timings: &BenchWhisperTimings,
) -> serde_json::Value {
    let token_ids: Vec<u32> = generated.iter().map(|token| token.0).collect();
    serde_json::json!({
        "status": if matches_expected { "completed" } else { "mismatch" },
        "elapsed_ms": timings.total_ms,
        "token_count": generated.len(),
        "tokens": token_ids,
        "text": transcript_text,
        "matches_expected": matches_expected,
        "cpu_kernel_mode": cpu_kernel_mode.as_str(),
        "cpu_threads": cpu_threads,
        "resident_model_ms": {
            "audio_to_tokens": timings.resident_audio_to_tokens_ms(),
            "mel_to_tokens": timings.resident_mel_to_tokens_ms(),
        },
        "timings_ms": {
            "total": timings.total_ms,
            "config_parse": timings.config_parse_ms,
            "manifest_validate": timings.manifest_validate_ms,
            "expected_tokens_read": timings.expected_tokens_read_ms,
            "tensor_load_model": timings.tensor_load_model_ms,
            "tokenizer_load": timings.tokenizer_load_ms,
            "wav_read": timings.wav_read_ms,
            "log_mel": timings.log_mel_ms,
            "audio_encode": timings.audio_encode_ms,
            "audio_encode_detail": {
                "encoder": timings.audio_encode_detail.encoder_ms,
                "cross_attention_precompute": timings.audio_encode_detail.cross_attention_precompute_ms,
            },
            "decode_total": timings.decode_total_ms,
            "text_decode": timings.text_decode_ms,
            "decode_token": timings.decode_token_ms,
        }
    })
}

fn decode_generated_text(
    tokenizer_path: Option<&Path>,
    generated: &[TokenId],
) -> Result<BenchWhisperTextOutput, String> {
    let Some(tokenizer_path) = tokenizer_path else {
        return Ok(BenchWhisperTextOutput::default());
    };

    let tokenizer_started = Instant::now();
    let tokenizer = JsonTokenizer::from_json_path(tokenizer_path).map_err(|err| {
        format!(
            "failed to load tokenizer at {} - {err:?}",
            tokenizer_path.display()
        )
    })?;
    let tokenizer_load_ms = tokenizer_started.elapsed().as_millis();

    let text_started = Instant::now();
    let text = tokenizer.decode(generated).map_err(|err| {
        format!(
            "failed to decode generated tokens with tokenizer at {} - {err:?}",
            tokenizer_path.display()
        )
    })?;
    let text_decode_ms = text_started.elapsed().as_millis();

    Ok(BenchWhisperTextOutput {
        text: Some(text),
        tokenizer_load_ms: Some(tokenizer_load_ms),
        text_decode_ms: Some(text_decode_ms),
    })
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
    let names = required_whisper_tensor_names(config);
    load_safetensors_tensors_f32(path, &names).map_err(|err| {
        format!(
            "failed to load required Whisper tensors from {} - {err:?}",
            path.display()
        )
    })
}

fn generate_tokens_to_expected_length(
    model: &WhisperModel,
    encoded_audio: &WhisperEncodedAudio,
    expected_len: usize,
    policy: WhisperArtifactDecodePolicy,
    decode_token_ms: &mut Vec<u128>,
) -> Result<Vec<TokenId>, String> {
    let mut generated = policy.startup_prompt();
    let mask = WhisperDecodeMask::transcribe_without_timestamps(policy.tokens);
    let mut decoder_state = model
        .prepare_decoder_state_from_audio(encoded_audio, &generated)
        .map_err(|err| {
            format!(
                "Whisper prepare_decoder_state_from_audio failed at generated length {} - {err:?}",
                generated.len()
            )
        })?;

    while generated.len() < expected_len {
        let token_started = Instant::now();
        let logits = decoder_state.next_token_logits();
        let next = masked_greedy_sample(logits, mask)?;
        generated.push(next);
        if next == policy.tokens.end_of_text {
            decode_token_ms.push(token_started.elapsed().as_millis());
            break;
        }
        if generated.len() < expected_len {
            model
                .append_decoder_token_from_audio(encoded_audio, &mut decoder_state, next)
                .map_err(|err| {
                    format!(
                        "Whisper append_decoder_token_from_audio failed at generated length {} - {err:?}",
                        generated.len()
                    )
                })?;
        }
        decode_token_ms.push(token_started.elapsed().as_millis());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_args_default_to_scalar_and_accept_optimized_kernel_mode() {
        let base = [
            "--config-path",
            "config.json",
            "--model-path",
            "model.safetensors",
            "--audio-path",
            "sample.wav",
            "--expected-tokens-path",
            "expected_tokens.json",
        ];

        let default_args =
            parse_bench_args(base.iter().copied().map(str::to_string)).expect("default args");
        assert_eq!(default_args.cpu_kernel_mode, CpuKernelMode::Scalar);
        assert_eq!(default_args.cpu_threads, 1);

        let mut optimized = base.iter().copied().map(str::to_string).collect::<Vec<_>>();
        optimized.extend(["--cpu-kernel-mode".to_string(), "optimized".to_string()]);
        let optimized_args = parse_bench_args(optimized).expect("optimized args");
        assert_eq!(optimized_args.cpu_kernel_mode, CpuKernelMode::Optimized);

        let mut with_threads = base.iter().copied().map(str::to_string).collect::<Vec<_>>();
        with_threads.extend(["--cpu-threads".to_string(), "4".to_string()]);
        let threaded_args = parse_bench_args(with_threads).expect("threaded args");
        assert_eq!(threaded_args.cpu_threads, 4);

        let mut with_tokenizer = base.iter().copied().map(str::to_string).collect::<Vec<_>>();
        with_tokenizer.extend(["--tokenizer-path".to_string(), "tokenizer.json".to_string()]);
        let tokenizer_args = parse_bench_args(with_tokenizer).expect("tokenizer args");
        assert_eq!(
            tokenizer_args.tokenizer_path,
            Some(PathBuf::from("tokenizer.json"))
        );
    }

    #[test]
    fn fetch_hf_args_require_pinned_revision_and_plan_command() {
        let args = parse_fetch_hf_args(
            [
                "--repo",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--revision",
                "7ae557604adf67be50417f59c2c2f167def9a775",
                "--local-dir",
                "local-artifacts/qwen2_5_0_5b_instruct",
                "--include",
                "config.json",
                "--include",
                "model.safetensors",
            ]
            .iter()
            .copied()
            .map(str::to_string),
        )
        .expect("fetch args");

        assert!(!args.execute);
        assert_eq!(args.repo, "Qwen/Qwen2.5-0.5B-Instruct");
        assert_eq!(args.revision, "7ae557604adf67be50417f59c2c2f167def9a775");
        assert_eq!(
            args.local_dir,
            PathBuf::from("local-artifacts/qwen2_5_0_5b_instruct")
        );
        assert_eq!(args.include, vec!["config.json", "model.safetensors"]);
        assert_eq!(
            huggingface_cli_args(&args),
            vec![
                "download",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--revision",
                "7ae557604adf67be50417f59c2c2f167def9a775",
                "--local-dir",
                "local-artifacts/qwen2_5_0_5b_instruct",
                "--local-dir-use-symlinks",
                "False",
                "--include",
                "config.json",
                "model.safetensors",
            ]
        );
    }

    #[test]
    fn fetch_hf_args_reject_missing_revision() {
        let err = parse_fetch_hf_args(
            [
                "--repo",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--local-dir",
                "local-artifacts/qwen2_5_0_5b_instruct",
            ]
            .iter()
            .copied()
            .map(str::to_string),
        )
        .expect_err("unpinned fetch must be rejected");

        assert_eq!(err, "missing --revision");
    }

    #[test]
    fn bench_whisper_output_reports_stage_timings() {
        let timings = BenchWhisperTimings {
            total_ms: 100,
            config_parse_ms: 1,
            manifest_validate_ms: 2,
            expected_tokens_read_ms: 3,
            tensor_load_model_ms: 4,
            tokenizer_load_ms: Some(13),
            wav_read_ms: 5,
            log_mel_ms: 6,
            audio_encode_ms: 7,
            audio_encode_detail: WhisperAudioEncodeTimings {
                encoder_ms: 11,
                cross_attention_precompute_ms: 12,
            },
            decode_total_ms: 8,
            text_decode_ms: Some(14),
            decode_token_ms: vec![9, 10],
        };

        let output = bench_whisper_output(
            true,
            CpuKernelMode::Optimized,
            4,
            &[TokenId(50257), TokenId(50362)],
            Some("example transcript"),
            &timings,
        );

        assert_eq!(output["status"], "completed");
        assert_eq!(output["cpu_kernel_mode"], "optimized");
        assert_eq!(output["cpu_threads"], 4);
        assert_eq!(output["elapsed_ms"], 100);
        assert_eq!(output["token_count"], 2);
        assert_eq!(output["text"], "example transcript");
        assert_eq!(output["resident_model_ms"]["audio_to_tokens"], 21);
        assert_eq!(output["resident_model_ms"]["mel_to_tokens"], 15);
        assert_eq!(output["timings_ms"]["total"], 100);
        assert_eq!(output["timings_ms"]["config_parse"], 1);
        assert_eq!(output["timings_ms"]["manifest_validate"], 2);
        assert_eq!(output["timings_ms"]["expected_tokens_read"], 3);
        assert_eq!(output["timings_ms"]["tensor_load_model"], 4);
        assert_eq!(output["timings_ms"]["tokenizer_load"], 13);
        assert_eq!(output["timings_ms"]["wav_read"], 5);
        assert_eq!(output["timings_ms"]["log_mel"], 6);
        assert_eq!(output["timings_ms"]["audio_encode"], 7);
        assert_eq!(output["timings_ms"]["audio_encode_detail"]["encoder"], 11);
        assert_eq!(
            output["timings_ms"]["audio_encode_detail"]["cross_attention_precompute"],
            12
        );
        assert_eq!(output["timings_ms"]["decode_total"], 8);
        assert_eq!(output["timings_ms"]["text_decode"], 14);
        assert_eq!(
            output["timings_ms"]["decode_token"],
            serde_json::json!([9, 10])
        );
    }
}
