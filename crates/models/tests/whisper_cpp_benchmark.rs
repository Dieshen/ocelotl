//! W-ASR.13 default-on benchmark harness contract tests.
//!
//! These tests validate the manifest and result record shapes without running
//! Ocelotl, whisper.cpp, local model artifacts, or network-dependent tooling.

use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct BenchmarkManifest {
    fixture_version: u32,
    name: String,
    model_path: String,
    audio_path: String,
    threads: u32,
    ocelotl: BenchmarkCommand,
    whisper_cpp: WhisperCppCommand,
}

#[derive(Debug, Deserialize)]
struct BenchmarkCommand {
    command: Vec<String>,
    model_path: String,
    audio_path: String,
    #[serde(default)]
    required_inputs: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct WhisperCppCommand {
    binary: String,
    command: Vec<String>,
    model_path: String,
    audio_path: String,
}

#[derive(Debug, Deserialize)]
struct BenchmarkRecord {
    fixture_version: u32,
    manifest_name: String,
    status: String,
    ocelotl: BenchmarkRun,
    whisper_cpp: BenchmarkRun,
}

#[derive(Debug, Deserialize)]
struct BenchmarkRun {
    command: Vec<String>,
    model_path: String,
    audio_path: String,
    threads: u32,
    status: String,
    wall_time_ms: Option<u64>,
    exit_code: Option<i32>,
    output: BenchmarkOutput,
    skip_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkOutput {
    token_count: Option<u64>,
    text: Option<String>,
    stdout_excerpt: Option<String>,
}

#[test]
fn whisper_cpp_benchmark_manifest_names_commands_inputs_and_threads() {
    let manifest = parse_manifest("whisper_cpp_manifest.example.json");

    assert_eq!(manifest.fixture_version, 1);
    assert_eq!(manifest.name, "whisper_tiny_en_sample_16khz_mono");
    assert_eq!(
        manifest.model_path,
        "local-artifacts/whisper_tiny_en/model.safetensors"
    );
    assert_eq!(
        manifest.audio_path,
        "local-artifacts/whisper_tiny_en/reference/sample_16khz_mono.wav"
    );
    assert_eq!(manifest.threads, 4);

    assert_command_shape(
        &manifest.ocelotl,
        &manifest.model_path,
        &manifest.audio_path,
    );
    assert_eq!(
        manifest.ocelotl.command[0], "target/release/ocelotl.exe",
        "Ocelotl benchmark command must name the dedicated timing hook binary, not cargo test"
    );
    assert!(
        manifest
            .ocelotl
            .command
            .iter()
            .any(|arg| arg == "bench-whisper-transcribe"),
        "Ocelotl benchmark command must use the dedicated transcription timing hook"
    );
    assert!(
        manifest
            .ocelotl
            .command
            .windows(2)
            .any(|args| args == ["--cpu-kernel-mode", "optimized"]),
        "W-ASR.24 benchmark command must opt into the W-ASR.23 optimized CPU backend"
    );
    assert!(
        !manifest
            .ocelotl
            .command
            .windows(2)
            .any(|args| args == ["cargo", "test"]),
        "Ocelotl benchmark command must not time the Rust test harness"
    );
    assert!(
        manifest
            .ocelotl
            .required_inputs
            .iter()
            .any(|path| path == "local-artifacts/whisper_tiny_en/config.json"),
        "Ocelotl benchmark command must declare the local config artifact it needs"
    );
    assert!(
        manifest
            .ocelotl
            .required_inputs
            .iter()
            .any(|path| path == "local-artifacts/whisper_tiny_en/reference/expected_tokens.json"),
        "Ocelotl benchmark command must declare the local expected-token artifact it needs"
    );
    assert_command_shape(
        &BenchmarkCommand {
            command: manifest.whisper_cpp.command.clone(),
            model_path: manifest.whisper_cpp.model_path.clone(),
            audio_path: manifest.whisper_cpp.audio_path.clone(),
            required_inputs: Vec::new(),
        },
        "local-artifacts/whisper_cpp/ggml-tiny.en.bin",
        &manifest.audio_path,
    );
    assert_eq!(
        manifest.whisper_cpp.binary,
        "local-artifacts/whisper_cpp/whisper-cli.exe"
    );
    assert!(
        manifest
            .whisper_cpp
            .command
            .iter()
            .any(|arg| arg == &manifest.threads.to_string()),
        "whisper.cpp command should include the manifest thread count"
    );
}

#[test]
fn whisper_cpp_benchmark_record_names_timing_and_output_fields() {
    let record = parse_record("whisper_cpp_record.example.json");

    assert_eq!(record.fixture_version, 1);
    assert_eq!(record.manifest_name, "whisper_tiny_en_sample_16khz_mono");
    assert_eq!(record.status, "completed");

    assert_completed_run(&record.ocelotl);
    assert_completed_run(&record.whisper_cpp);
    assert_eq!(record.ocelotl.audio_path, record.whisper_cpp.audio_path);
}

#[test]
fn missing_whisper_cpp_binary_record_has_clear_remediation() {
    let record = parse_record("whisper_cpp_missing_binary_record.example.json");

    assert_eq!(record.fixture_version, 1);
    assert_eq!(record.status, "skipped");
    assert_eq!(record.whisper_cpp.status, "skipped");
    assert_eq!(record.whisper_cpp.wall_time_ms, None);
    assert_eq!(record.whisper_cpp.exit_code, None);

    let skip_reason = record
        .whisper_cpp
        .skip_reason
        .as_ref()
        .expect("missing whisper.cpp binary skip should include a reason");
    assert!(skip_reason.contains("whisper.cpp"));
    assert!(skip_reason.contains("local-artifacts/whisper_cpp/whisper-cli.exe"));
    assert!(skip_reason.contains("docs/benchmarks/whisper-cpp.md"));
}

fn assert_command_shape(command: &BenchmarkCommand, model_path: &str, audio_path: &str) {
    assert!(
        !command.command.is_empty(),
        "benchmark command must name an executable"
    );
    assert_eq!(command.model_path, model_path);
    assert_eq!(command.audio_path, audio_path);
}

fn assert_completed_run(run: &BenchmarkRun) {
    assert_eq!(run.status, "completed");
    assert!(
        !run.command.is_empty(),
        "completed benchmark run must record its command"
    );
    assert!(
        !run.model_path.trim().is_empty(),
        "completed benchmark run must record its model path"
    );
    assert!(
        !run.audio_path.trim().is_empty(),
        "completed benchmark run must record its audio path"
    );
    assert!(run.threads > 0, "thread count must be positive");
    assert!(
        run.wall_time_ms.is_some(),
        "completed benchmark run must record wall_time_ms"
    );
    assert_eq!(run.exit_code, Some(0));
    assert!(
        run.output.token_count.is_some()
            || run
                .output
                .text
                .as_ref()
                .is_some_and(|text| !text.trim().is_empty())
            || run
                .output
                .stdout_excerpt
                .as_ref()
                .is_some_and(|stdout| !stdout.trim().is_empty()),
        "completed benchmark run must record an output token/text/stdout summary"
    );
}

fn parse_manifest(file_name: &str) -> BenchmarkManifest {
    let raw = read_fixture(file_name);
    serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("failed to parse benchmark manifest fixture - {err}"))
}

fn parse_record(file_name: &str) -> BenchmarkRecord {
    let raw = read_fixture(file_name);
    serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("failed to parse benchmark record fixture - {err}"))
}

fn read_fixture(file_name: &str) -> String {
    let path = fixture_path(file_name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed to read {} - {err}", path.display()))
}

fn fixture_path(file_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("benchmarks")
        .join(file_name)
}
