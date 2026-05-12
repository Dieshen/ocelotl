//! W-ASR.10/W-ASR.19 opt-in local-artifact harness for Whisper classic sizes.
//!
//! The default-on tests in this file validate the harness schema without
//! touching local artifacts. The ignored tests check each local bundle contract
//! independently and, when a bundle is present, run exact output-token parity:
//!
//! ```text
//! local-artifacts/whisper_{tiny_en,base_en,small_en,medium_en,large}/
//!   config.json
//!   tokenizer.json
//!   model.safetensors
//!   reference/sample_16khz_mono.wav
//!   reference/expected_tokens.json
//! ```
//!
//! Run the opt-in check with:
//!
//! ```text
//! cargo test -p ocelotl-models --test whisper_local_artifact_parity -- --ignored
//! ```

use std::path::{Path, PathBuf};

use ocelotl_core::{DType, TokenId};
use ocelotl_loader::{LoadedTensor, inspect_safetensors, load_safetensors_tensor_f32};
use ocelotl_models::whisper::{
    WhisperConfig, WhisperModel,
    audio::{AudioMetadata, LogMelSpectrogram, log_mel_spectrogram},
    parse_whisper_config_json, required_whisper_tensor_names, validate_whisper_tensors,
};
use ocelotl_tokenizer::{
    WhisperDecodeMask, WhisperStartupTokens, WhisperTimestampMode, WhisperTimestampedSegment,
    WhisperTokenMaskDecision, parse_whisper_timestamped_segments,
    whisper_multilingual_english_transcribe_tokens,
};
use serde::Deserialize;

const OPENAI_MULTILINGUAL_VOCAB_THRESHOLD: usize = 51_865;
const LOCAL_ARTIFACT_ROOT_ENV: &str = "OCELOTL_LOCAL_ARTIFACTS_DIR";
const LOCAL_ARTIFACT_ROOT: &str = "local-artifacts";
const CONFIG_JSON: &str = "config.json";
const TOKENIZER_JSON: &str = "tokenizer.json";
const MODEL_SAFETENSORS: &str = "model.safetensors";
const REFERENCE_AUDIO: &str = "reference/sample_16khz_mono.wav";
const EXPECTED_TOKENS_JSON: &str = "reference/expected_tokens.json";
const TIMESTAMPED_EXPECTED_TOKENS_JSON: &str = "reference/expected_tokens_timestamped.json";
const EXPECTED_AUDIO_FIELD: &str = "reference/sample_16khz_mono.wav";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WhisperLocalBundle {
    artifact: &'static str,
    dir_name: &'static str,
    reference_name: &'static str,
}

const WHISPER_TINY_EN_BUNDLE: WhisperLocalBundle = WhisperLocalBundle {
    artifact: "tiny.en",
    dir_name: "whisper_tiny_en",
    reference_name: "whisper_tiny_en_sample_16khz_mono",
};

const CLASSIC_WHISPER_LOCAL_BUNDLES: &[WhisperLocalBundle] = &[
    WHISPER_TINY_EN_BUNDLE,
    WhisperLocalBundle {
        artifact: "base.en",
        dir_name: "whisper_base_en",
        reference_name: "whisper_base_en_sample_16khz_mono",
    },
    WhisperLocalBundle {
        artifact: "small.en",
        dir_name: "whisper_small_en",
        reference_name: "whisper_small_en_sample_16khz_mono",
    },
    WhisperLocalBundle {
        artifact: "medium.en",
        dir_name: "whisper_medium_en",
        reference_name: "whisper_medium_en_sample_16khz_mono",
    },
    WhisperLocalBundle {
        artifact: "large",
        dir_name: "whisper_large",
        reference_name: "whisper_large_sample_16khz_mono",
    },
];

#[derive(Debug, Deserialize)]
struct ExpectedTokens {
    fixture_version: u32,
    name: String,
    source: String,
    audio: String,
    task: String,
    language: String,
    timestamps: bool,
    expected_token_ids: Vec<u32>,
    #[serde(default)]
    expected_text: Option<String>,
    #[serde(default)]
    expected_segments: Vec<ExpectedTimestampedSegment>,
}

#[derive(Debug, Deserialize)]
struct ExpectedTimestampedSegment {
    start_seconds: f32,
    end_seconds: f32,
    text_token_ids: Vec<u32>,
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

    fn startup_prompt(self, mode: WhisperTimestampMode) -> Vec<TokenId> {
        match self.family {
            WhisperArtifactTokenizerFamily::Multilingual => self.tokens.transcribe_prompt(mode),
            WhisperArtifactTokenizerFamily::EnglishOnly => {
                self.tokens.english_transcribe_prompt(mode)
            }
        }
    }

    fn label(self) -> &'static str {
        match self.family {
            WhisperArtifactTokenizerFamily::Multilingual => "multilingual English transcribe",
            WhisperArtifactTokenizerFamily::EnglishOnly => "English-only transcribe",
        }
    }
}

impl WhisperLocalBundle {
    fn relative_dir(self) -> String {
        format!("{LOCAL_ARTIFACT_ROOT}/{}", self.dir_name)
    }

    fn path(self, relative: &str) -> PathBuf {
        local_artifact_root().join(self.dir_name).join(relative)
    }

    fn required_relative_paths(self) -> Vec<String> {
        required_artifacts()
            .into_iter()
            .map(|relative| format!("{}/{}", self.relative_dir(), relative))
            .collect()
    }
}

fn local_artifact_root() -> PathBuf {
    std::env::var_os(LOCAL_ARTIFACT_ROOT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(default_local_artifact_root)
}

fn repo_artifact_path(relative: &str) -> PathBuf {
    WHISPER_TINY_EN_BUNDLE.path(relative)
}

fn default_local_artifact_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(LOCAL_ARTIFACT_ROOT)
}

fn required_artifacts() -> [&'static str; 5] {
    [
        CONFIG_JSON,
        TOKENIZER_JSON,
        MODEL_SAFETENSORS,
        REFERENCE_AUDIO,
        EXPECTED_TOKENS_JSON,
    ]
}

fn tiny_whisper_config() -> WhisperConfig {
    WhisperConfig {
        vocab_size: OPENAI_MULTILINGUAL_VOCAB_THRESHOLD - 1,
        mel_bins: 80,
        audio_context_length: 1500,
        audio_state_size: 384,
        audio_attention_heads: 6,
        audio_layers: 4,
        audio_ffn_size: 1536,
        text_context_length: 448,
        text_state_size: 384,
        text_attention_heads: 6,
        text_layers: 4,
        text_ffn_size: 1536,
        dtype: DType::F32,
        tie_word_embeddings: true,
    }
}

#[test]
fn whisper_local_artifact_contract_lists_exact_required_paths() {
    assert_eq!(
        WHISPER_TINY_EN_BUNDLE.relative_dir(),
        "local-artifacts/whisper_tiny_en"
    );
    assert_eq!(
        required_artifacts(),
        [
            "config.json",
            "tokenizer.json",
            "model.safetensors",
            "reference/sample_16khz_mono.wav",
            "reference/expected_tokens.json",
        ]
    );
}

#[test]
fn classic_whisper_local_artifact_contract_lists_all_size_paths() {
    let cases: Vec<(&str, String, Vec<String>)> = CLASSIC_WHISPER_LOCAL_BUNDLES
        .iter()
        .copied()
        .map(|bundle| {
            (
                bundle.artifact,
                bundle.relative_dir(),
                bundle.required_relative_paths(),
            )
        })
        .collect();

    assert_eq!(
        cases,
        vec![
            (
                "tiny.en",
                "local-artifacts/whisper_tiny_en".to_string(),
                vec![
                    "local-artifacts/whisper_tiny_en/config.json".to_string(),
                    "local-artifacts/whisper_tiny_en/tokenizer.json".to_string(),
                    "local-artifacts/whisper_tiny_en/model.safetensors".to_string(),
                    "local-artifacts/whisper_tiny_en/reference/sample_16khz_mono.wav".to_string(),
                    "local-artifacts/whisper_tiny_en/reference/expected_tokens.json".to_string(),
                ],
            ),
            (
                "base.en",
                "local-artifacts/whisper_base_en".to_string(),
                vec![
                    "local-artifacts/whisper_base_en/config.json".to_string(),
                    "local-artifacts/whisper_base_en/tokenizer.json".to_string(),
                    "local-artifacts/whisper_base_en/model.safetensors".to_string(),
                    "local-artifacts/whisper_base_en/reference/sample_16khz_mono.wav".to_string(),
                    "local-artifacts/whisper_base_en/reference/expected_tokens.json".to_string(),
                ],
            ),
            (
                "small.en",
                "local-artifacts/whisper_small_en".to_string(),
                vec![
                    "local-artifacts/whisper_small_en/config.json".to_string(),
                    "local-artifacts/whisper_small_en/tokenizer.json".to_string(),
                    "local-artifacts/whisper_small_en/model.safetensors".to_string(),
                    "local-artifacts/whisper_small_en/reference/sample_16khz_mono.wav".to_string(),
                    "local-artifacts/whisper_small_en/reference/expected_tokens.json".to_string(),
                ],
            ),
            (
                "medium.en",
                "local-artifacts/whisper_medium_en".to_string(),
                vec![
                    "local-artifacts/whisper_medium_en/config.json".to_string(),
                    "local-artifacts/whisper_medium_en/tokenizer.json".to_string(),
                    "local-artifacts/whisper_medium_en/model.safetensors".to_string(),
                    "local-artifacts/whisper_medium_en/reference/sample_16khz_mono.wav".to_string(),
                    "local-artifacts/whisper_medium_en/reference/expected_tokens.json".to_string(),
                ],
            ),
            (
                "large",
                "local-artifacts/whisper_large".to_string(),
                vec![
                    "local-artifacts/whisper_large/config.json".to_string(),
                    "local-artifacts/whisper_large/tokenizer.json".to_string(),
                    "local-artifacts/whisper_large/model.safetensors".to_string(),
                    "local-artifacts/whisper_large/reference/sample_16khz_mono.wav".to_string(),
                    "local-artifacts/whisper_large/reference/expected_tokens.json".to_string(),
                ],
            ),
        ]
    );
}

#[test]
fn classic_whisper_local_artifact_references_reuse_expected_token_schema() {
    for bundle in CLASSIC_WHISPER_LOCAL_BUNDLES {
        let (expected_token_ids, policy) = if bundle.artifact == "large" {
            (
                "[50258, 50259, 50359, 50363, 42, 50257]",
                WhisperArtifactDecodePolicy::multilingual_english(),
            )
        } else {
            (
                "[50257, 50362, 42, 50256]",
                WhisperArtifactDecodePolicy::english_only(),
            )
        };
        let raw = format!(
            r#"{{
              "fixture_version": 1,
              "name": "{}",
              "source": "local {} converter output",
              "audio": "reference/sample_16khz_mono.wav",
              "task": "transcribe",
              "language": "en",
              "timestamps": false,
              "expected_token_ids": {},
              "expected_text": "hello"
            }}"#,
            bundle.reference_name, bundle.artifact, expected_token_ids,
        );
        let fixture = parse_expected_tokens(&raw);

        validate_expected_tokens(&fixture);
        validate_expected_tokens_for_policy(&fixture, policy);
    }
}

#[test]
fn expected_tokens_schema_accepts_multilingual_documented_shape() {
    let fixture = parse_expected_tokens(
        r#"{
          "fixture_version": 1,
          "name": "whisper_tiny_en_sample_16khz_mono",
          "source": "local whisper_tiny_en converter output",
          "audio": "reference/sample_16khz_mono.wav",
          "task": "transcribe",
          "language": "en",
          "timestamps": false,
          "expected_token_ids": [50258, 50259, 50359, 50363, 42, 50257],
          "expected_text": "hello"
        }"#,
    );

    validate_expected_tokens(&fixture);
    validate_expected_tokens_for_policy(
        &fixture,
        WhisperArtifactDecodePolicy::multilingual_english(),
    );
}

#[test]
fn expected_tokens_schema_accepts_english_only_documented_shape() {
    let fixture = parse_expected_tokens(
        r#"{
          "fixture_version": 1,
          "name": "whisper_tiny_en_sample_16khz_mono",
          "source": "local whisper_tiny_en converter output",
          "audio": "reference/sample_16khz_mono.wav",
          "task": "transcribe",
          "language": "en",
          "timestamps": false,
          "expected_token_ids": [50257, 50362, 42, 50256],
          "expected_text": "hello"
        }"#,
    );

    validate_expected_tokens(&fixture);
    validate_expected_tokens_for_policy(&fixture, WhisperArtifactDecodePolicy::english_only());
}

#[test]
fn decode_policy_uses_english_only_prompt_below_openai_multilingual_threshold() {
    let mut config = tiny_whisper_config();
    config.vocab_size = OPENAI_MULTILINGUAL_VOCAB_THRESHOLD - 1;

    let policy = WhisperArtifactDecodePolicy::for_config(&config);

    assert_eq!(policy.family, WhisperArtifactTokenizerFamily::EnglishOnly);
    assert_eq!(
        policy.startup_prompt(WhisperTimestampMode::NoTimestamps),
        vec![TokenId(50_257), TokenId(50_362)]
    );
    assert_eq!(policy.tokens.end_of_text, TokenId(50_256));
    assert_eq!(policy.tokens.first_timestamp, TokenId(50_363));
}

#[test]
fn decode_policy_uses_multilingual_prompt_at_openai_multilingual_threshold() {
    let mut config = tiny_whisper_config();
    config.vocab_size = OPENAI_MULTILINGUAL_VOCAB_THRESHOLD;

    let policy = WhisperArtifactDecodePolicy::for_config(&config);

    assert_eq!(policy.family, WhisperArtifactTokenizerFamily::Multilingual);
    assert_eq!(
        policy.startup_prompt(WhisperTimestampMode::NoTimestamps),
        vec![
            TokenId(50_258),
            TokenId(50_259),
            TokenId(50_359),
            TokenId(50_363),
        ]
    );
    assert_eq!(policy.tokens.end_of_text, TokenId(50_257));
    assert_eq!(policy.tokens.first_timestamp, TokenId(50_364));
}

#[test]
fn timestamped_expected_tokens_schema_accepts_segments_for_english_only() {
    let fixture = parse_expected_tokens(
        r#"{
          "fixture_version": 1,
          "name": "whisper_tiny_en_sample_16khz_mono_timestamped",
          "source": "local whisper_tiny_en timestamp reference",
          "audio": "reference/sample_16khz_mono.wav",
          "task": "transcribe",
          "language": "en",
          "timestamps": true,
          "expected_token_ids": [50257, 50373, 42, 43, 50388, 50256],
          "expected_segments": [
            {
              "start_seconds": 0.20,
              "end_seconds": 0.50,
              "text_token_ids": [42, 43]
            }
          ],
          "expected_text": "hello"
        }"#,
    );

    let policy = WhisperArtifactDecodePolicy::english_only();

    validate_expected_tokens(&fixture);
    validate_expected_tokens_for_policy(&fixture, policy);
    let segments = expected_timestamped_segments(&fixture, policy);

    assert_eq!(segments.len(), 1);
    assert_seconds(segments[0].start_seconds, 0.20);
    assert_seconds(segments[0].end_seconds, 0.50);
    assert_eq!(segments[0].text_tokens, vec![TokenId(42), TokenId(43)]);
}

#[test]
fn timestamped_expected_tokens_schema_rejects_mismatched_segments() {
    let fixture = parse_expected_tokens(
        r#"{
          "fixture_version": 1,
          "name": "whisper_tiny_en_sample_16khz_mono_timestamped",
          "source": "local whisper_tiny_en timestamp reference",
          "audio": "reference/sample_16khz_mono.wav",
          "task": "transcribe",
          "language": "en",
          "timestamps": true,
          "expected_token_ids": [50257, 50373, 42, 50388, 50256],
          "expected_segments": [
            {
              "start_seconds": 0.20,
              "end_seconds": 0.52,
              "text_token_ids": [42]
            }
          ]
        }"#,
    );

    let err = validate_expected_tokens_for_policy_result(
        &fixture,
        WhisperArtifactDecodePolicy::english_only(),
    )
    .expect_err("mismatched timestamp segment boundary must be rejected");

    assert!(err.contains("expected_segments"));
}

#[test]
fn expected_tokens_schema_rejects_empty_reference_sequence() {
    let fixture = parse_expected_tokens(
        r#"{
          "fixture_version": 1,
          "name": "whisper_tiny_en_sample_16khz_mono",
          "source": "local whisper_tiny_en converter output",
          "audio": "reference/sample_16khz_mono.wav",
          "task": "transcribe",
          "language": "en",
          "timestamps": false,
          "expected_token_ids": []
        }"#,
    );

    let err = validate_expected_tokens_result(&fixture)
        .expect_err("empty expected_token_ids must be rejected");
    assert!(err.contains("expected_token_ids"));
}

#[test]
fn expected_tokens_schema_rejects_sequence_without_artifact_startup_prompt() {
    let fixture = parse_expected_tokens(
        r#"{
          "fixture_version": 1,
          "name": "whisper_tiny_en_sample_16khz_mono",
          "source": "local whisper_tiny_en converter output",
          "audio": "reference/sample_16khz_mono.wav",
          "task": "transcribe",
          "language": "en",
          "timestamps": false,
          "expected_token_ids": [42, 50257]
        }"#,
    );

    let err = validate_expected_tokens_for_policy_result(
        &fixture,
        WhisperArtifactDecodePolicy::english_only(),
    )
    .expect_err("missing artifact startup prompt must be rejected");
    assert!(err.contains("startup prompt"));
}

#[test]
fn wav_sample_reader_decodes_pcm16_mono_values() {
    let bytes = build_test_wav(
        1,
        16,
        &[-32768_i16, 0, 32767]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect::<Vec<_>>(),
    );

    let wav = parse_wav_mono_samples(&bytes).expect("PCM16 mono WAV should decode");

    assert_eq!(
        wav.metadata,
        WavMetadata {
            audio_format: 1,
            channels: 1,
            sample_rate_hz: 16_000,
            bits_per_sample: 16,
            data_bytes: 6,
        }
    );
    assert_close(&wav.samples, &[-1.0, 0.0, 32767.0 / 32768.0], 1.0e-7);
}

#[test]
fn wav_sample_reader_decodes_ieee_float32_mono_values() {
    let bytes = build_test_wav(
        3,
        32,
        &[-0.25_f32, 0.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    );

    let wav = parse_wav_mono_samples(&bytes).expect("IEEE float32 mono WAV should decode");

    assert_eq!(
        wav.metadata,
        WavMetadata {
            audio_format: 3,
            channels: 1,
            sample_rate_hz: 16_000,
            bits_per_sample: 32,
            data_bytes: 8,
        }
    );
    assert_eq!(wav.samples, vec![-0.25, 0.5]);
}

#[test]
#[ignore = "requires local-artifacts/whisper_tiny_en/{config.json,tokenizer.json,model.safetensors,reference/sample_16khz_mono.wav,reference/expected_tokens.json} - see docs/artifact-preparation.md"]
fn local_whisper_tiny_en_artifact_contract_is_well_formed() {
    run_local_whisper_bundle_parity(WHISPER_TINY_EN_BUNDLE);
}

#[test]
#[ignore = "requires local-artifacts/whisper_base_en/{config.json,tokenizer.json,model.safetensors,reference/sample_16khz_mono.wav,reference/expected_tokens.json} - see docs/artifact-preparation.md"]
fn local_whisper_base_en_artifact_contract_is_well_formed() {
    run_local_whisper_bundle_parity(CLASSIC_WHISPER_LOCAL_BUNDLES[1]);
}

#[test]
#[ignore = "requires local-artifacts/whisper_small_en/{config.json,tokenizer.json,model.safetensors,reference/sample_16khz_mono.wav,reference/expected_tokens.json} - see docs/artifact-preparation.md"]
fn local_whisper_small_en_artifact_contract_is_well_formed() {
    run_local_whisper_bundle_parity(CLASSIC_WHISPER_LOCAL_BUNDLES[2]);
}

#[test]
#[ignore = "requires local-artifacts/whisper_medium_en/{config.json,tokenizer.json,model.safetensors,reference/sample_16khz_mono.wav,reference/expected_tokens.json} - see docs/artifact-preparation.md"]
fn local_whisper_medium_en_artifact_contract_is_well_formed() {
    run_local_whisper_bundle_parity(CLASSIC_WHISPER_LOCAL_BUNDLES[3]);
}

#[test]
#[ignore = "requires local-artifacts/whisper_large/{config.json,tokenizer.json,model.safetensors,reference/sample_16khz_mono.wav,reference/expected_tokens.json} - see docs/artifact-preparation.md"]
fn local_whisper_large_artifact_contract_is_well_formed() {
    run_local_whisper_bundle_parity(CLASSIC_WHISPER_LOCAL_BUNDLES[4]);
}

fn run_local_whisper_bundle_parity(bundle: WhisperLocalBundle) {
    if !bundle.path(CONFIG_JSON).exists() {
        eprintln!(
            "skipping Whisper {} local parity: missing bundle at {}; prepare {} with {} or set {} to a directory containing {}; see docs/artifact-preparation.md",
            bundle.artifact,
            bundle.path(CONFIG_JSON).display(),
            bundle.relative_dir(),
            required_artifacts().join(", "),
            LOCAL_ARTIFACT_ROOT_ENV,
            bundle.dir_name,
        );
        return;
    }

    for relative in required_artifacts() {
        let path = bundle.path(relative);
        assert!(
            path.exists(),
            "{} bundle is present but missing artifact at {} - expected {} under {}; see docs/artifact-preparation.md",
            bundle.artifact,
            path.display(),
            relative,
            bundle.relative_dir(),
        );
    }

    run_local_whisper_bundle_expected_tokens(
        bundle,
        EXPECTED_TOKENS_JSON,
        WhisperTimestampMode::NoTimestamps,
    );

    let timestamped_path = bundle.path(TIMESTAMPED_EXPECTED_TOKENS_JSON);
    if timestamped_path.exists() {
        run_local_whisper_bundle_expected_tokens(
            bundle,
            TIMESTAMPED_EXPECTED_TOKENS_JSON,
            WhisperTimestampMode::Timestamps,
        );
    } else {
        eprintln!(
            "skipping optional Whisper {} timestamped local parity: missing {}; see docs/artifact-preparation.md",
            bundle.artifact,
            timestamped_path.display(),
        );
    }
}

fn run_local_whisper_bundle_expected_tokens(
    bundle: WhisperLocalBundle,
    expected_tokens_json: &str,
    mode: WhisperTimestampMode,
) {
    let config_path = bundle.path(CONFIG_JSON);
    let config_raw = std::fs::read_to_string(&config_path).unwrap_or_else(|err| {
        panic!(
            "failed to read config.json at {} - {err}",
            config_path.display()
        )
    });
    let whisper_config = parse_whisper_config_json(&config_raw).unwrap_or_else(|err| {
        panic!(
            "invalid Whisper config contract at {} - {err:?}",
            config_path.display()
        )
    });
    let decode_policy = WhisperArtifactDecodePolicy::for_config(&whisper_config);

    let tokenizer_path = bundle.path(TOKENIZER_JSON);
    assert_json_object(&tokenizer_path, "tokenizer.json");

    let model_path = bundle.path(MODEL_SAFETENSORS);
    let manifest = inspect_safetensors(&model_path).unwrap_or_else(|err| {
        panic!(
            "failed to inspect safetensors header at {} - {err:?}",
            model_path.display()
        )
    });
    assert!(
        !manifest.tensors.is_empty(),
        "model.safetensors header at {} must contain at least one tensor",
        model_path.display(),
    );
    validate_whisper_tensors(&manifest, &whisper_config, Some(&model_path)).unwrap_or_else(|err| {
        panic!(
            "model.safetensors at {} does not match the Whisper tensor contract - {err:?}",
            model_path.display()
        )
    });

    let audio_path = bundle.path(REFERENCE_AUDIO);
    let wav = read_wav_metadata(&audio_path);
    assert!(
        matches!(wav.audio_format, 1 | 3),
        "reference audio must be PCM or IEEE float WAV, got format {}",
        wav.audio_format
    );
    assert_eq!(wav.channels, 1, "reference audio must be mono");
    assert_eq!(wav.sample_rate_hz, 16_000, "reference audio must be 16 kHz");
    assert!(
        wav.bits_per_sample > 0,
        "reference audio must declare a positive bits-per-sample value"
    );
    assert!(
        wav.data_bytes > 0,
        "reference audio at {} must have a non-empty data chunk",
        audio_path.display(),
    );

    let expected_path = bundle.path(expected_tokens_json);
    let expected = parse_expected_tokens_file(&expected_path);
    validate_expected_tokens(&expected);
    validate_expected_tokens_for_policy(&expected, decode_policy);
    assert_eq!(
        expected.timestamps,
        matches!(mode, WhisperTimestampMode::Timestamps),
        "{} must declare timestamps={} for {}",
        expected_path.display(),
        matches!(mode, WhisperTimestampMode::Timestamps),
        bundle.artifact,
    );

    let expected_token_ids = expected_token_ids(&expected);
    if matches!(mode, WhisperTimestampMode::NoTimestamps) {
        assert_startup_prompt(&expected_token_ids, decode_policy);
    }
    assert!(
        expected_token_ids.len() <= whisper_config.text_context_length,
        "{} has {} tokens, but config text_context_length is {}",
        expected_path.display(),
        expected_token_ids.len(),
        whisper_config.text_context_length,
    );
    let expected_segments = if matches!(mode, WhisperTimestampMode::Timestamps) {
        let segments = expected_timestamped_segments(&expected, decode_policy);
        assert!(
            !segments.is_empty(),
            "timestamped reference at {} must include at least one expected segment",
            expected_path.display(),
        );
        Some(segments)
    } else {
        None
    };

    let loaded_tensors = load_required_whisper_tensors(&model_path, &whisper_config);
    let model = WhisperModel::new(whisper_config, loaded_tensors).unwrap_or_else(|err| {
        panic!(
            "failed to construct WhisperModel from loaded tensors at {} - {err:?}",
            model_path.display()
        )
    });

    let wav_audio = read_wav_mono_samples(&audio_path);
    assert_eq!(wav_audio.metadata, wav);
    assert!(
        !wav_audio.samples.is_empty(),
        "reference audio at {} must decode to at least one sample",
        audio_path.display(),
    );
    let mel = log_mel_spectrogram(
        &wav_audio.samples,
        AudioMetadata {
            sample_rate_hz: wav.sample_rate_hz,
            channels: wav.channels,
        },
    )
    .unwrap_or_else(|err| {
        panic!(
            "failed to compute Whisper log-mel spectrogram for {} - {err:?}",
            audio_path.display()
        )
    });

    let generated = generate_tokens_to_expected_length(
        &model,
        &mel,
        expected_token_ids.len(),
        decode_policy,
        mode,
    );
    assert_eq!(
        generated,
        expected_token_ids,
        "generated Whisper {} token IDs must exactly match {} for {}",
        match mode {
            WhisperTimestampMode::NoTimestamps => "no-timestamps",
            WhisperTimestampMode::Timestamps => "timestamped",
        },
        expected_tokens_json,
        bundle.artifact,
    );

    if let Some(expected_segments) = expected_segments {
        assert_eq!(
            expected_timestamped_segments_for_tokens(&generated, &expected, decode_policy),
            expected_segments,
            "generated timestamped Whisper segments must exactly match expected_segments for {}",
            bundle.artifact,
        );
    }
}

#[test]
#[ignore = "requires local-artifacts/whisper_tiny_en/reference/expected_tokens_timestamped.json plus the W-ASR.10 tiny.en bundle; see docs/artifact-preparation.md"]
fn local_whisper_tiny_en_timestamped_artifact_contract_is_well_formed() {
    let expected_path = repo_artifact_path(TIMESTAMPED_EXPECTED_TOKENS_JSON);
    if !expected_path.exists() {
        eprintln!(
            "skipping timestamped Whisper local parity: missing optional reference at {}; see docs/artifact-preparation.md",
            expected_path.display()
        );
        return;
    }

    for relative in required_artifacts() {
        let path = repo_artifact_path(relative);
        assert!(
            path.exists(),
            "timestamped parity reference exists, but missing required base artifact at {} - expected {} under {}; see docs/artifact-preparation.md",
            path.display(),
            relative,
            WHISPER_TINY_EN_BUNDLE.relative_dir(),
        );
    }

    let config_path = repo_artifact_path(CONFIG_JSON);
    let config_raw = std::fs::read_to_string(&config_path).unwrap_or_else(|err| {
        panic!(
            "failed to read config.json at {} - {err}",
            config_path.display()
        )
    });
    let whisper_config = parse_whisper_config_json(&config_raw).unwrap_or_else(|err| {
        panic!(
            "invalid Whisper config contract at {} - {err:?}",
            config_path.display()
        )
    });
    let decode_policy = WhisperArtifactDecodePolicy::for_config(&whisper_config);

    let expected = parse_expected_tokens_file(&expected_path);
    validate_expected_tokens(&expected);
    validate_expected_tokens_for_policy(&expected, decode_policy);
    let expected_token_ids = expected_token_ids(&expected);
    let expected_segments = expected_timestamped_segments(&expected, decode_policy);
    assert!(
        !expected_segments.is_empty(),
        "timestamped reference at {} must include at least one expected segment",
        expected_path.display(),
    );

    let tokenizer_path = repo_artifact_path(TOKENIZER_JSON);
    assert_json_object(&tokenizer_path, "tokenizer.json");

    let model_path = repo_artifact_path(MODEL_SAFETENSORS);
    let manifest = inspect_safetensors(&model_path).unwrap_or_else(|err| {
        panic!(
            "failed to inspect safetensors header at {} - {err:?}",
            model_path.display()
        )
    });
    validate_whisper_tensors(&manifest, &whisper_config, Some(&model_path)).unwrap_or_else(|err| {
        panic!(
            "model.safetensors at {} does not match the Whisper tensor contract - {err:?}",
            model_path.display()
        )
    });

    let audio_path = repo_artifact_path(REFERENCE_AUDIO);
    let wav_audio = read_wav_mono_samples(&audio_path);
    assert_eq!(wav_audio.metadata.sample_rate_hz, 16_000);
    assert_eq!(wav_audio.metadata.channels, 1);

    let mel = log_mel_spectrogram(
        &wav_audio.samples,
        AudioMetadata {
            sample_rate_hz: wav_audio.metadata.sample_rate_hz,
            channels: wav_audio.metadata.channels,
        },
    )
    .unwrap_or_else(|err| {
        panic!(
            "failed to compute Whisper log-mel spectrogram for {} - {err:?}",
            audio_path.display()
        )
    });

    let loaded_tensors = load_required_whisper_tensors(&model_path, &whisper_config);
    let model = WhisperModel::new(whisper_config, loaded_tensors).unwrap_or_else(|err| {
        panic!(
            "failed to construct WhisperModel from loaded tensors at {} - {err:?}",
            model_path.display()
        )
    });

    let generated = generate_tokens_to_expected_length(
        &model,
        &mel,
        expected_token_ids.len(),
        decode_policy,
        WhisperTimestampMode::Timestamps,
    );
    assert_eq!(
        generated, expected_token_ids,
        "generated timestamped Whisper token IDs must exactly match reference/expected_tokens_timestamped.json",
    );
    assert_eq!(
        expected_timestamped_segments_for_tokens(&generated, &expected, decode_policy),
        expected_segments,
        "generated timestamped Whisper segments must exactly match expected_segments",
    );
}

fn assert_json_object(path: &Path, label: &str) {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read {} at {} - {err}", label, path.display()));
    let value: serde_json::Value = serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("failed to parse {} at {} - {err}", label, path.display()));
    assert!(
        value.is_object(),
        "{} at {} must be a JSON object",
        label,
        path.display()
    );
}

fn parse_expected_tokens_file(path: &Path) -> ExpectedTokens {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|err| {
        panic!(
            "failed to read expected_tokens.json at {} - {err}",
            path.display()
        )
    });
    parse_expected_tokens(&raw)
}

fn parse_expected_tokens(raw: &str) -> ExpectedTokens {
    serde_json::from_str(raw)
        .unwrap_or_else(|err| panic!("failed to parse expected_tokens JSON shape - {err}"))
}

fn validate_expected_tokens(fixture: &ExpectedTokens) {
    validate_expected_tokens_result(fixture)
        .unwrap_or_else(|err| panic!("invalid expected_tokens.json contract - {err}"));
}

fn validate_expected_tokens_for_policy(
    fixture: &ExpectedTokens,
    policy: WhisperArtifactDecodePolicy,
) {
    validate_expected_tokens_for_policy_result(fixture, policy)
        .unwrap_or_else(|err| panic!("invalid expected_tokens.json startup contract - {err}"));
}

fn validate_expected_tokens_result(fixture: &ExpectedTokens) -> Result<(), String> {
    if fixture.fixture_version != 1 {
        return Err(format!(
            "fixture_version must be 1, got {}",
            fixture.fixture_version
        ));
    }
    if fixture.name.trim().is_empty() {
        return Err("name must be non-empty".to_string());
    }
    if fixture.source.trim().is_empty() {
        return Err("source must be non-empty".to_string());
    }
    if fixture.audio != EXPECTED_AUDIO_FIELD {
        return Err(format!(
            "audio must be {EXPECTED_AUDIO_FIELD:?}, got {:?}",
            fixture.audio
        ));
    }
    if fixture.task != "transcribe" {
        return Err(format!(
            "task must be \"transcribe\", got {:?}",
            fixture.task
        ));
    }
    if fixture.language != "en" {
        return Err(format!(
            "language must be \"en\", got {:?}",
            fixture.language
        ));
    }
    if fixture.expected_token_ids.is_empty() {
        return Err("expected_token_ids must be non-empty".to_string());
    }
    if let Some(text) = &fixture.expected_text {
        if text.trim().is_empty() {
            return Err("expected_text must be non-empty when present".to_string());
        }
    }
    if !fixture.timestamps && !fixture.expected_segments.is_empty() {
        return Err("expected_segments must be empty when timestamps is false".to_string());
    }
    if fixture.timestamps && fixture.expected_segments.is_empty() {
        return Err("expected_segments must be non-empty when timestamps is true".to_string());
    }
    Ok(())
}

fn validate_expected_tokens_for_policy_result(
    fixture: &ExpectedTokens,
    policy: WhisperArtifactDecodePolicy,
) -> Result<(), String> {
    let mode = if fixture.timestamps {
        WhisperTimestampMode::Timestamps
    } else {
        WhisperTimestampMode::NoTimestamps
    };
    let startup_prompt = startup_prompt_ids(policy, mode);
    if !fixture.expected_token_ids.starts_with(&startup_prompt) {
        return Err(format!(
            "expected_token_ids must start with the {} {mode:?} startup prompt {:?}",
            policy.label(),
            startup_prompt
        ));
    }
    if fixture.expected_token_ids.len() == startup_prompt.len() {
        return Err(
            "expected_token_ids must include at least one generated token after the startup prompt"
                .to_string(),
        );
    }
    if fixture.timestamps {
        validate_expected_timestamped_segments(fixture, policy)?;
    } else if fixture
        .expected_token_ids
        .iter()
        .any(|&token| token >= policy.tokens.first_timestamp.0)
    {
        return Err(
            "no-timestamps expected_token_ids must not contain timestamp tokens".to_string(),
        );
    }
    Ok(())
}

fn startup_prompt_ids(policy: WhisperArtifactDecodePolicy, mode: WhisperTimestampMode) -> Vec<u32> {
    policy
        .startup_prompt(mode)
        .into_iter()
        .map(|token| token.0)
        .collect()
}

fn expected_token_ids(fixture: &ExpectedTokens) -> Vec<TokenId> {
    fixture
        .expected_token_ids
        .iter()
        .copied()
        .map(TokenId)
        .collect()
}

fn assert_startup_prompt(expected_token_ids: &[TokenId], policy: WhisperArtifactDecodePolicy) {
    let startup_prompt = policy.startup_prompt(WhisperTimestampMode::NoTimestamps);
    assert!(
        expected_token_ids.starts_with(&startup_prompt),
        "expected_tokens.json must start with {} no-timestamps prompt {:?}, got prefix {:?}",
        policy.label(),
        startup_prompt,
        &expected_token_ids[..expected_token_ids.len().min(startup_prompt.len())],
    );
}

fn expected_timestamped_segments(
    fixture: &ExpectedTokens,
    policy: WhisperArtifactDecodePolicy,
) -> Vec<WhisperTimestampedSegment> {
    validate_expected_timestamped_segments(fixture, policy)
        .unwrap_or_else(|err| panic!("invalid timestamped expected_tokens contract - {err}"))
}

fn validate_expected_timestamped_segments(
    fixture: &ExpectedTokens,
    policy: WhisperArtifactDecodePolicy,
) -> Result<Vec<WhisperTimestampedSegment>, String> {
    expected_timestamped_segments_for_token_ids(&expected_token_ids(fixture), fixture, policy)
}

fn expected_timestamped_segments_for_tokens(
    token_ids: &[TokenId],
    fixture: &ExpectedTokens,
    policy: WhisperArtifactDecodePolicy,
) -> Vec<WhisperTimestampedSegment> {
    expected_timestamped_segments_for_token_ids(token_ids, fixture, policy)
        .unwrap_or_else(|err| panic!("invalid generated timestamped segments - {err}"))
}

fn expected_timestamped_segments_for_token_ids(
    token_ids: &[TokenId],
    fixture: &ExpectedTokens,
    policy: WhisperArtifactDecodePolicy,
) -> Result<Vec<WhisperTimestampedSegment>, String> {
    let startup_len = policy
        .startup_prompt(WhisperTimestampMode::Timestamps)
        .len();
    let generated_suffix = token_ids.get(startup_len..).ok_or_else(|| {
        "expected_token_ids are shorter than timestamp startup prompt".to_string()
    })?;
    let parsed = parse_whisper_timestamped_segments(policy.tokens, generated_suffix)
        .map_err(|err| format!("failed to parse timestamped expected_token_ids - {err}"))?;

    if parsed.len() != fixture.expected_segments.len() {
        return Err(format!(
            "expected_segments length must be {}, got {}",
            parsed.len(),
            fixture.expected_segments.len()
        ));
    }

    for (idx, (actual, expected)) in parsed.iter().zip(&fixture.expected_segments).enumerate() {
        assert_segment_matches(idx, actual, expected)?;
    }

    Ok(parsed)
}

fn assert_segment_matches(
    idx: usize,
    actual: &WhisperTimestampedSegment,
    expected: &ExpectedTimestampedSegment,
) -> Result<(), String> {
    if (actual.start_seconds - expected.start_seconds).abs() > f32::EPSILON {
        return Err(format!(
            "expected_segments[{idx}].start_seconds must be {:.2}, got {:.2}",
            actual.start_seconds, expected.start_seconds
        ));
    }
    if (actual.end_seconds - expected.end_seconds).abs() > f32::EPSILON {
        return Err(format!(
            "expected_segments[{idx}].end_seconds must be {:.2}, got {:.2}",
            actual.end_seconds, expected.end_seconds
        ));
    }
    let expected_text_tokens: Vec<TokenId> = expected
        .text_token_ids
        .iter()
        .copied()
        .map(TokenId)
        .collect();
    if actual.text_tokens != expected_text_tokens {
        return Err(format!(
            "expected_segments[{idx}].text_token_ids must be {:?}, got {:?}",
            actual.text_tokens, expected_text_tokens
        ));
    }
    Ok(())
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

fn generate_tokens_to_expected_length(
    model: &WhisperModel,
    mel: &LogMelSpectrogram,
    expected_len: usize,
    policy: WhisperArtifactDecodePolicy,
    mode: WhisperTimestampMode,
) -> Vec<TokenId> {
    let mut generated = policy.startup_prompt(mode);
    let mask = WhisperDecodeMask::transcribe(policy.tokens, mode);
    let encoded_audio = model
        .encode_audio_features(&mel.values, mel.frames)
        .unwrap_or_else(|err| {
            panic!(
                "Whisper encode_audio_features failed before decode at generated length {} - {err:?}",
                generated.len()
            )
        });

    while generated.len() < expected_len {
        let logits = model
            .forward_next_token_logits_from_audio(&encoded_audio, &generated)
            .unwrap_or_else(|err| {
                panic!(
                    "Whisper forward_next_token_logits_from_audio failed at generated length {} - {err:?}",
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

fn read_wav_metadata(path: &Path) -> WavMetadata {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|err| panic!("failed to read WAV at {} - {err}", path.display()));
    parse_wav_metadata(&bytes)
        .unwrap_or_else(|err| panic!("invalid WAV metadata at {} - {err}", path.display()))
}

fn parse_wav_metadata(bytes: &[u8]) -> Result<WavMetadata, String> {
    parse_wav_layout(bytes).map(|layout| layout.metadata)
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

fn build_test_wav(audio_format: u16, bits_per_sample: u16, payload: &[u8]) -> Vec<u8> {
    let channels = 1_u16;
    let sample_rate = 16_000_u32;
    let block_align = channels * (bits_per_sample / 8);
    let byte_rate = sample_rate * u32::from(block_align);
    let riff_size = 4 + (8 + 16) + (8 + payload.len() as u32);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16_u32.to_le_bytes());
    bytes.extend_from_slice(&audio_format.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&bits_per_sample.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes
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

fn assert_seconds(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= f32::EPSILON,
        "expected {expected:.2}s, got {actual:.2}s"
    );
}
