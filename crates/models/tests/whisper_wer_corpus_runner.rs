//! W-ASR.16 local Whisper WER corpus runner.
//!
//! Default tests validate the committed manifest and report schema without
//! touching local artifacts. The ignored test loads local artifacts when present,
//! runs deterministic greedy transcription, decodes text with the local
//! tokenizer, and reports per-sample plus aggregate WER without thresholds.
//!
//! Run the opt-in check with:
//!
//! ```text
//! cargo test -p ocelotl-models --test whisper_wer_corpus_runner -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use ocelotl_core::TokenId;
use ocelotl_loader::{LoadedTensor, load_safetensors_tensor_f32};
use ocelotl_models::whisper::{
    WhisperConfig, WhisperModel,
    audio::{AudioMetadata, LogMelSpectrogram, log_mel_spectrogram},
    parse_whisper_config_json, required_whisper_tensor_names, score_wer_corpus,
};
use ocelotl_tokenizer::{
    JsonTokenizer, Tokenizer, WhisperDecodeMask, WhisperStartupTokens, WhisperTimestampMode,
    WhisperTokenMaskDecision, whisper_english_transcribe_tokens,
    whisper_multilingual_english_transcribe_tokens,
};
use serde::Deserialize;

const MANIFEST_FIXTURE: &str = "../../fixtures/wer/whisper_wer_corpus.example.json";
const OPENAI_MULTILINGUAL_VOCAB_THRESHOLD: usize = 51_865;

#[derive(Debug, Deserialize)]
struct WerCorpusManifest {
    fixture_version: u32,
    name: String,
    description: String,
    model_dir: String,
    config_path: String,
    tokenizer_path: String,
    model_path: String,
    skip_when_missing: bool,
    max_new_tokens: usize,
    cases: Vec<WerCorpusManifestCase>,
}

#[derive(Debug, Deserialize)]
struct WerCorpusManifestCase {
    id: String,
    audio_path: String,
    expected_transcript: String,
    #[serde(default)]
    expected_token_path: Option<String>,
    #[serde(default)]
    skip_when_missing: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ExpectedTokens {
    expected_token_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
struct WerRunnerCaseResult {
    id: String,
    audio_path: String,
    expected_transcript: String,
    recognized_transcript: String,
    generated_token_ids: Vec<TokenId>,
    expected_token_match: Option<bool>,
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
struct WhisperArtifactDecodePolicy {
    tokens: WhisperStartupTokens,
    mode: WhisperTimestampMode,
    multilingual: bool,
}

impl WhisperArtifactDecodePolicy {
    fn for_config(config: &WhisperConfig) -> Self {
        if config.vocab_size >= OPENAI_MULTILINGUAL_VOCAB_THRESHOLD {
            Self {
                tokens: whisper_multilingual_english_transcribe_tokens(),
                mode: WhisperTimestampMode::NoTimestamps,
                multilingual: true,
            }
        } else {
            Self {
                tokens: whisper_english_transcribe_tokens(),
                mode: WhisperTimestampMode::NoTimestamps,
                multilingual: false,
            }
        }
    }

    fn startup_prompt(self) -> Vec<TokenId> {
        if self.multilingual {
            self.tokens.transcribe_no_timestamps_prompt()
        } else {
            self.tokens.english_transcribe_prompt(self.mode)
        }
    }

    fn decode_mask(self) -> WhisperDecodeMask {
        WhisperDecodeMask::transcribe(self.tokens, self.mode)
    }
}

#[test]
fn wer_corpus_manifest_schema_names_local_artifacts_and_cases() {
    let manifest = parse_manifest_fixture();

    validate_manifest(&manifest).expect("committed WER manifest fixture must be valid");
    assert_eq!(manifest.fixture_version, 1);
    assert_eq!(manifest.name, "whisper_tiny_en_local_wer_corpus");
    assert_eq!(manifest.model_dir, "local-artifacts/whisper_tiny_en");
    assert_eq!(
        manifest.config_path,
        "local-artifacts/whisper_tiny_en/config.json"
    );
    assert_eq!(
        manifest.tokenizer_path,
        "local-artifacts/whisper_tiny_en/tokenizer.json"
    );
    assert_eq!(
        manifest.model_path,
        "local-artifacts/whisper_tiny_en/model.safetensors"
    );
    assert!(manifest.skip_when_missing);
    assert_eq!(manifest.max_new_tokens, 64);
    assert_eq!(manifest.cases.len(), 2);
    assert_eq!(manifest.cases[0].id, "sample_16khz_mono");
    assert_eq!(
        manifest.cases[0].expected_token_path.as_deref(),
        Some("local-artifacts/whisper_tiny_en/reference/expected_tokens.json")
    );
    assert_eq!(manifest.cases[1].skip_when_missing, Some(true));
}

#[test]
fn wer_corpus_manifest_rejects_empty_cases_and_missing_skip_policy() {
    let err = validate_manifest(&WerCorpusManifest {
        fixture_version: 1,
        name: "bad".to_string(),
        description: "missing skip policy".to_string(),
        model_dir: "local-artifacts/whisper_tiny_en".to_string(),
        config_path: "local-artifacts/whisper_tiny_en/config.json".to_string(),
        tokenizer_path: "local-artifacts/whisper_tiny_en/tokenizer.json".to_string(),
        model_path: "local-artifacts/whisper_tiny_en/model.safetensors".to_string(),
        skip_when_missing: false,
        max_new_tokens: 64,
        cases: Vec::new(),
    })
    .expect_err("manifest without cases and skip policy must fail schema validation");

    assert!(err.contains("skip_when_missing"));
    assert!(err.contains("cases"));
}

#[test]
fn missing_local_artifacts_are_reported_as_skip_reasons_without_loading() {
    let manifest = parse_manifest_fixture();
    let missing = missing_required_artifacts(
        &manifest,
        Path::new("target/ocelotl-test-missing-artifacts-root"),
    );
    let missing_cases = missing_case_artifacts(
        &manifest,
        Path::new("target/ocelotl-test-missing-artifacts-root"),
    );

    assert!(
        missing
            .iter()
            .any(|path| path.ends_with("local-artifacts/whisper_tiny_en/model.safetensors")),
        "missing-artifact plan should include the local model path: {missing:?}"
    );
    assert!(
        missing_cases.iter().any(|path| path
            .ends_with("local-artifacts/whisper_tiny_en/reference/sample_16khz_mono.wav")),
        "missing-case plan should include manifest WAV paths: {missing_cases:?}"
    );
}

#[test]
fn runner_report_formats_per_sample_and_aggregate_wer_without_thresholds() {
    let cases = vec![
        WerRunnerCaseResult {
            id: "exact".to_string(),
            audio_path: "local-artifacts/whisper_tiny_en/reference/exact.wav".to_string(),
            expected_transcript: "hello world".to_string(),
            recognized_transcript: "Hello, world!".to_string(),
            generated_token_ids: vec![TokenId(1), TokenId(2)],
            expected_token_match: Some(true),
        },
        WerRunnerCaseResult {
            id: "substitution".to_string(),
            audio_path: "local-artifacts/whisper_tiny_en/reference/substitution.wav".to_string(),
            expected_transcript: "the quick fox".to_string(),
            recognized_transcript: "the slow fox".to_string(),
            generated_token_ids: vec![TokenId(3), TokenId(4)],
            expected_token_match: None,
        },
    ];

    let report = format_wer_report("fixture", &cases).expect("WER report should format");

    assert!(report.contains("WER corpus report: fixture"));
    assert!(report.contains("case exact"));
    assert!(report.contains("case substitution"));
    assert!(report.contains("expected_token_match=true"));
    assert!(report.contains("expected_token_match=n/a"));
    assert!(report.contains("aggregate wer=0.200000 errors=1 ref_words=5"));
    assert!(
        !report.contains("threshold"),
        "W-ASR.16 must report WER only, without pass/fail thresholds"
    );
}

#[test]
#[ignore = "requires local Whisper corpus artifacts named by fixtures/wer/whisper_wer_corpus.example.json"]
fn local_whisper_wer_corpus_runner_reports_scores_when_artifacts_exist() {
    let manifest = parse_manifest_fixture();
    validate_manifest(&manifest).expect("committed WER manifest fixture must be valid");
    let repo_root = repo_root();

    let missing = missing_required_artifacts(&manifest, &repo_root);
    if !missing.is_empty() {
        println!(
            "skipping local Whisper WER corpus runner because artifacts are absent:\n{}",
            missing
                .iter()
                .map(|path| format!("  - {path}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
        return;
    }

    let results = run_local_corpus(&manifest, &repo_root);
    if results.is_empty() {
        println!(
            "skipping local Whisper WER corpus runner because no manifest WAV cases are present"
        );
        return;
    }
    let report = format_wer_report(&manifest.name, &results).expect("WER report should format");
    println!("{report}");
}

fn parse_manifest_fixture() -> WerCorpusManifest {
    let raw = std::fs::read_to_string(manifest_fixture_path()).unwrap_or_else(|err| {
        panic!(
            "failed to read WER corpus manifest fixture at {} - {err}",
            manifest_fixture_path().display()
        )
    });
    serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("failed to parse WER corpus manifest fixture - {err}"))
}

fn validate_manifest(manifest: &WerCorpusManifest) -> Result<(), String> {
    let mut errors = Vec::new();

    if manifest.fixture_version != 1 {
        errors.push(format!(
            "fixture_version must be 1, got {}",
            manifest.fixture_version
        ));
    }
    if manifest.name.trim().is_empty() {
        errors.push("name must be non-empty".to_string());
    }
    if manifest.description.trim().is_empty() {
        errors.push("description must be non-empty".to_string());
    }
    if manifest.model_dir.trim().is_empty() {
        errors.push("model_dir must be non-empty".to_string());
    }
    for (field, value) in [
        ("config_path", &manifest.config_path),
        ("tokenizer_path", &manifest.tokenizer_path),
        ("model_path", &manifest.model_path),
    ] {
        if value.trim().is_empty() {
            errors.push(format!("{field} must be non-empty"));
        }
        if Path::new(value).is_absolute() {
            errors.push(format!(
                "{field} must be repository-relative, got {value:?}"
            ));
        }
    }
    if !manifest.skip_when_missing {
        errors.push("skip_when_missing must be true for local-only corpus artifacts".to_string());
    }
    if manifest.max_new_tokens == 0 {
        errors.push("max_new_tokens must be greater than zero".to_string());
    }
    if manifest.cases.is_empty() {
        errors.push("cases must contain at least one sample".to_string());
    }

    for case in &manifest.cases {
        if case.id.trim().is_empty() {
            errors.push("cases[].id must be non-empty".to_string());
        }
        if case.audio_path.trim().is_empty() {
            errors.push(format!("case {:?} audio_path must be non-empty", case.id));
        }
        if Path::new(&case.audio_path).is_absolute() {
            errors.push(format!(
                "case {:?} audio_path must be repository-relative",
                case.id
            ));
        }
        if case.expected_transcript.trim().is_empty() {
            errors.push(format!(
                "case {:?} expected_transcript must be non-empty",
                case.id
            ));
        }
        if let Some(path) = &case.expected_token_path {
            if path.trim().is_empty() {
                errors.push(format!(
                    "case {:?} expected_token_path must be non-empty when present",
                    case.id
                ));
            }
            if Path::new(path).is_absolute() {
                errors.push(format!(
                    "case {:?} expected_token_path must be repository-relative",
                    case.id
                ));
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

fn missing_required_artifacts(manifest: &WerCorpusManifest, root: &Path) -> Vec<String> {
    [
        manifest.config_path.as_str(),
        manifest.tokenizer_path.as_str(),
        manifest.model_path.as_str(),
    ]
    .into_iter()
    .filter(|path| !root.join(path).exists())
    .map(str::to_string)
    .collect()
}

fn missing_case_artifacts(manifest: &WerCorpusManifest, root: &Path) -> Vec<String> {
    manifest
        .cases
        .iter()
        .map(|case| case.audio_path.as_str())
        .filter(|path| !root.join(path).exists())
        .map(str::to_string)
        .collect()
}

fn run_local_corpus(manifest: &WerCorpusManifest, root: &Path) -> Vec<WerRunnerCaseResult> {
    let config_path = root.join(&manifest.config_path);
    let config_raw = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|err| panic!("failed to read {} - {err}", config_path.display()));
    let config = parse_whisper_config_json(&config_raw).unwrap_or_else(|err| {
        panic!(
            "invalid Whisper config at {} - {err:?}",
            config_path.display()
        )
    });
    let policy = WhisperArtifactDecodePolicy::for_config(&config);
    let tokenizer_path = root.join(&manifest.tokenizer_path);
    let tokenizer = JsonTokenizer::from_json_path(&tokenizer_path).unwrap_or_else(|err| {
        panic!(
            "failed to load tokenizer at {} - {err:?}",
            tokenizer_path.display()
        )
    });
    let model_path = root.join(&manifest.model_path);
    let tensors = load_required_whisper_tensors(&model_path, &config);
    let model = WhisperModel::new(config, tensors)
        .unwrap_or_else(|err| panic!("failed to construct Whisper model - {err:?}"));

    manifest
        .cases
        .iter()
        .filter_map(|case| {
            let audio_path = root.join(&case.audio_path);
            if !audio_path.exists() && case.skip_when_missing.unwrap_or(manifest.skip_when_missing)
            {
                println!(
                    "skipping WER case {} because {} is absent",
                    case.id,
                    audio_path.display()
                );
                return None;
            }
            let generated =
                transcribe_case_tokens(&model, &audio_path, policy, manifest.max_new_tokens);
            let recognized = tokenizer.decode(&generated).unwrap_or_else(|err| {
                panic!(
                    "failed to decode generated tokens for WER case {} - {err:?}",
                    case.id
                )
            });
            let expected_token_match = case.expected_token_path.as_ref().and_then(|path| {
                let expected_path = root.join(path);
                expected_path
                    .exists()
                    .then(|| read_expected_token_ids(&expected_path) == generated)
            });

            Some(WerRunnerCaseResult {
                id: case.id.clone(),
                audio_path: case.audio_path.clone(),
                expected_transcript: case.expected_transcript.clone(),
                recognized_transcript: recognized,
                generated_token_ids: generated,
                expected_token_match,
            })
        })
        .collect()
}

fn transcribe_case_tokens(
    model: &WhisperModel,
    audio_path: &Path,
    policy: WhisperArtifactDecodePolicy,
    max_new_tokens: usize,
) -> Vec<TokenId> {
    let wav = read_wav_mono_samples(audio_path);
    assert_eq!(wav.metadata.channels, 1, "WER corpus WAV must be mono");
    assert_eq!(
        wav.metadata.sample_rate_hz, 16_000,
        "WER corpus WAV must be 16 kHz"
    );
    let mel = log_mel_spectrogram(
        &wav.samples,
        AudioMetadata {
            sample_rate_hz: wav.metadata.sample_rate_hz,
            channels: wav.metadata.channels,
        },
    )
    .unwrap_or_else(|err| {
        panic!(
            "failed to compute log-mel spectrogram for {} - {err:?}",
            audio_path.display()
        )
    });

    generate_tokens(model, &mel, policy, max_new_tokens)
}

fn generate_tokens(
    model: &WhisperModel,
    mel: &LogMelSpectrogram,
    policy: WhisperArtifactDecodePolicy,
    max_new_tokens: usize,
) -> Vec<TokenId> {
    let mut generated = policy.startup_prompt();
    let prompt_len = generated.len();
    let mask = policy.decode_mask();

    while generated.len() - prompt_len < max_new_tokens {
        let logits = model
            .forward_next_token_logits(&mel.values, mel.frames, &generated)
            .unwrap_or_else(|err| {
                panic!(
                    "Whisper forward_next_token_logits failed at generated length {} - {err:?}",
                    generated.len()
                )
            });
        let next = masked_greedy_sample(&logits, mask);
        generated.push(next);
        if next == policy.tokens.end_of_text {
            break;
        }
    }

    generated
}

fn format_wer_report(name: &str, results: &[WerRunnerCaseResult]) -> ocelotl_core::Result<String> {
    let cases = results
        .iter()
        .map(|result| ocelotl_models::whisper::WerCorpusCase {
            id: result.id.as_str(),
            expected_transcript: result.expected_transcript.as_str(),
            recognized_transcript: result.recognized_transcript.as_str(),
        })
        .collect::<Vec<_>>();
    let scored = score_wer_corpus(&cases)?;

    let mut lines = vec![format!("WER corpus report: {name}")];
    for (result, score) in results.iter().zip(scored.cases.iter()) {
        let token_match = result
            .expected_token_match
            .map(|matches| matches.to_string())
            .unwrap_or_else(|| "n/a".to_string());
        lines.push(format!(
            "case {} audio={} wer={:.6} errors={} ref_words={} substitutions={} insertions={} deletions={} generated_tokens={} expected_token_match={}",
            result.id,
            result.audio_path,
            score.score.wer,
            score.score.counts.errors(),
            score.score.counts.reference_words,
            score.score.counts.substitutions,
            score.score.counts.insertions,
            score.score.counts.deletions,
            result.generated_token_ids.len(),
            token_match
        ));
        lines.push(format!("  expected: {}", result.expected_transcript));
        lines.push(format!("  recognized: {}", result.recognized_transcript));
    }
    lines.push(format!(
        "aggregate wer={:.6} errors={} ref_words={} substitutions={} insertions={} deletions={}",
        scored.aggregate.wer,
        scored.aggregate.counts.errors(),
        scored.aggregate.counts.reference_words,
        scored.aggregate.counts.substitutions,
        scored.aggregate.counts.insertions,
        scored.aggregate.counts.deletions
    ));

    Ok(lines.join("\n"))
}

fn masked_greedy_sample(logits: &[f32], mask: WhisperDecodeMask) -> TokenId {
    assert!(!logits.is_empty(), "cannot sample from empty logits");

    let mut best = None;
    for (idx, &logit) in logits.iter().enumerate() {
        let token = TokenId(u32::try_from(idx).expect("vocab index must fit in u32"));
        if mask.mask_token(token) == WhisperTokenMaskDecision::Suppress {
            continue;
        }
        if best.is_none_or(|(_, best_logit)| logit > best_logit) {
            best = Some((idx, logit));
        }
    }

    let (idx, _) = best.expect("Whisper decode mask suppressed every logit");
    TokenId(u32::try_from(idx).expect("vocab index must fit in u32"))
}

fn load_required_whisper_tensors(path: &Path, config: &WhisperConfig) -> Vec<LoadedTensor> {
    required_whisper_tensor_names(config)
        .into_iter()
        .map(|name| {
            load_safetensors_tensor_f32(path, &name).unwrap_or_else(|err| {
                panic!(
                    "failed to load required Whisper tensor {name:?} from {} - {err:?}",
                    path.display()
                )
            })
        })
        .collect()
}

fn read_expected_token_ids(path: &Path) -> Vec<TokenId> {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|err| {
        panic!(
            "failed to read expected token file {} - {err}",
            path.display()
        )
    });
    let expected: ExpectedTokens = serde_json::from_str(&raw).unwrap_or_else(|err| {
        panic!(
            "failed to parse expected token file {} - {err}",
            path.display()
        )
    });
    expected
        .expected_token_ids
        .into_iter()
        .map(TokenId)
        .collect()
}

fn read_wav_mono_samples(path: &Path) -> WavAudio {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|err| panic!("failed to read WAV at {} - {err}", path.display()));
    parse_wav_mono_samples(&bytes)
        .unwrap_or_else(|err| panic!("unsupported WAV sample data at {} - {err}", path.display()))
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
            b"data" => {
                data = Some((chunk_start, chunk_size));
            }
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

fn manifest_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(MANIFEST_FIXTURE)
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}
