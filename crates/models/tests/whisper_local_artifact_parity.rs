//! W-ASR.5 opt-in local-artifact harness for Whisper tiny.en.
//!
//! The default-on tests in this file validate the harness schema without
//! touching local artifacts. The ignored test checks the first local bundle
//! contract:
//!
//! ```text
//! local-artifacts/whisper_tiny_en/
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

use ocelotl_loader::inspect_safetensors;
use serde::Deserialize;

const LOCAL_ARTIFACT_DIR: &str = "local-artifacts/whisper_tiny_en";
const CONFIG_JSON: &str = "config.json";
const TOKENIZER_JSON: &str = "tokenizer.json";
const MODEL_SAFETENSORS: &str = "model.safetensors";
const REFERENCE_AUDIO: &str = "reference/sample_16khz_mono.wav";
const EXPECTED_TOKENS_JSON: &str = "reference/expected_tokens.json";
const EXPECTED_AUDIO_FIELD: &str = "reference/sample_16khz_mono.wav";

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WavMetadata {
    audio_format: u16,
    channels: u16,
    sample_rate_hz: u32,
    bits_per_sample: u16,
    data_bytes: u32,
}

fn repo_artifact_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(LOCAL_ARTIFACT_DIR)
        .join(relative)
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

#[test]
fn whisper_local_artifact_contract_lists_exact_required_paths() {
    assert_eq!(LOCAL_ARTIFACT_DIR, "local-artifacts/whisper_tiny_en");
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
fn expected_tokens_schema_accepts_documented_shape() {
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
#[ignore = "requires local-artifacts/whisper_tiny_en/{config.json,tokenizer.json,model.safetensors,reference/sample_16khz_mono.wav,reference/expected_tokens.json} - see docs/artifact-preparation.md"]
fn local_whisper_tiny_en_artifact_contract_is_well_formed() {
    for relative in required_artifacts() {
        let path = repo_artifact_path(relative);
        assert!(
            path.exists(),
            "missing artifact at {} - expected {} under {}; see docs/artifact-preparation.md",
            path.display(),
            relative,
            LOCAL_ARTIFACT_DIR,
        );
    }

    let config_path = repo_artifact_path(CONFIG_JSON);
    assert_json_object(&config_path, "config.json");

    let tokenizer_path = repo_artifact_path(TOKENIZER_JSON);
    assert_json_object(&tokenizer_path, "tokenizer.json");

    let model_path = repo_artifact_path(MODEL_SAFETENSORS);
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

    let audio_path = repo_artifact_path(REFERENCE_AUDIO);
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

    let expected_path = repo_artifact_path(EXPECTED_TOKENS_JSON);
    let expected = parse_expected_tokens_file(&expected_path);
    validate_expected_tokens(&expected);
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
    if fixture.timestamps {
        return Err("timestamps must be false for the first W-ASR.5 fixture".to_string());
    }
    if fixture.expected_token_ids.is_empty() {
        return Err("expected_token_ids must be non-empty".to_string());
    }
    if let Some(text) = &fixture.expected_text {
        if text.trim().is_empty() {
            return Err("expected_text must be non-empty when present".to_string());
        }
    }
    Ok(())
}

fn read_wav_metadata(path: &Path) -> WavMetadata {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|err| panic!("failed to read WAV at {} - {err}", path.display()));
    parse_wav_metadata(&bytes)
        .unwrap_or_else(|err| panic!("invalid WAV metadata at {} - {err}", path.display()))
}

fn parse_wav_metadata(bytes: &[u8]) -> Result<WavMetadata, String> {
    if bytes.len() < 12 {
        return Err("file is shorter than a RIFF/WAVE header".to_string());
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("file must start with RIFF/WAVE".to_string());
    }

    let mut offset = 12usize;
    let mut fmt = None;
    let mut data_bytes = None;

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
                data_bytes = Some(chunk_size as u32);
            }
            _ => {}
        }

        offset = chunk_end + (chunk_size % 2);
    }

    let (audio_format, channels, sample_rate_hz, bits_per_sample) =
        fmt.ok_or_else(|| "missing fmt chunk".to_string())?;
    let data_bytes = data_bytes.ok_or_else(|| "missing data chunk".to_string())?;

    Ok(WavMetadata {
        audio_format,
        channels,
        sample_rate_hz,
        bits_per_sample,
        data_bytes,
    })
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
