//! W-ASR.13 default-on benchmark harness contract tests.
//!
//! These tests validate manifest/result shapes and exercise the runner with
//! synthetic PowerShell targets. They never run real model binaries, read local
//! model artifacts, or require network-dependent tooling.

use std::{
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct BenchmarkManifest {
    fixture_version: u32,
    name: String,
    model_path: String,
    audio_path: String,
    model_family: String,
    language: String,
    decoding_mode: String,
    threads: u32,
    warmup_iterations: u32,
    measured_iterations: u32,
    execution_order: String,
    output_equivalence: String,
    ocelotl: BenchmarkCommand,
    whisper_cpp: WhisperCppCommand,
}

#[derive(Debug, Deserialize)]
struct BenchmarkCommand {
    backend: String,
    cpu_kernel_mode: String,
    command: Vec<String>,
    model_path: String,
    audio_path: String,
    #[serde(default)]
    required_inputs: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct WhisperCppCommand {
    binary: String,
    transcript_path: String,
    backend: String,
    comparison_mode: String,
    command: Vec<String>,
    model_path: String,
    audio_path: String,
}

#[derive(Debug, Deserialize)]
struct BenchmarkRecord {
    fixture_version: u32,
    manifest_name: String,
    status: String,
    benchmark_plan: BenchmarkPlan,
    environment: BenchmarkEnvironment,
    comparison: BenchmarkComparison,
    samples: Vec<BenchmarkSample>,
    ocelotl: BenchmarkRun,
    whisper_cpp: BenchmarkRun,
}

#[derive(Debug, Deserialize)]
struct BenchmarkRun {
    command: Vec<String>,
    model_path: String,
    audio_path: String,
    effective_config: EffectiveConfig,
    status: String,
    aggregate: Option<BenchmarkAggregate>,
    latest_output: BenchmarkOutput,
    skip_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkPlan {
    warmup_iterations: u32,
    measured_iterations: u32,
    execution_order: String,
}

#[derive(Debug, Deserialize)]
struct BenchmarkEnvironment {
    captured_at_utc: String,
    repository_revision: String,
    repository_dirty: bool,
    os: String,
    architecture: String,
    powershell_version: String,
    logical_processors: u32,
}

#[derive(Debug, Deserialize)]
struct BenchmarkComparison {
    status: String,
    workload_match: bool,
    model_family: String,
    language: String,
    decoding_mode: String,
    output_equivalence: String,
    output_match: Option<bool>,
    audio_sha256: Option<String>,
    reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkSample {
    sequence: u32,
    phase: String,
    iteration: u32,
    engine: String,
    wall_time_ms: u64,
    exit_code: i32,
    output: BenchmarkOutput,
}

#[derive(Debug, Deserialize)]
struct EffectiveConfig {
    backend: String,
    threads: u32,
    cpu_kernel_mode: Option<String>,
    comparison_mode: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkAggregate {
    sample_count: u32,
    min_wall_time_ms: u64,
    median_wall_time_ms: f64,
    p95_wall_time_ms: f64,
    max_wall_time_ms: u64,
}

#[derive(Debug, Deserialize)]
struct BenchmarkOutput {
    token_count: Option<u64>,
    text: Option<String>,
    stdout_excerpt: Option<String>,
    parsed_json: Option<serde_json::Value>,
}

#[test]
fn whisper_cpp_benchmark_manifest_names_commands_inputs_and_threads() {
    let manifest = parse_manifest("whisper_cpp_manifest.example.json");

    assert_eq!(manifest.fixture_version, 2);
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
    assert_eq!(manifest.model_family, "whisper-tiny.en");
    assert_eq!(manifest.language, "en");
    assert_eq!(manifest.decoding_mode, "greedy_no_fallback");
    assert_eq!(manifest.warmup_iterations, 1);
    assert_eq!(manifest.measured_iterations, 10);
    assert_eq!(manifest.execution_order, "alternating");
    assert_eq!(manifest.output_equivalence, "normalized_transcript");
    assert_eq!(manifest.ocelotl.backend, "cpu");
    assert_eq!(manifest.ocelotl.cpu_kernel_mode, "scalar");

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
            .any(|args| args == ["--cpu-kernel-mode", "scalar"]),
        "W-ASR.36 benchmark command must use the current scalar Whisper CPU path"
    );
    let thread_count = manifest.threads.to_string();
    assert!(
        manifest
            .ocelotl
            .command
            .windows(2)
            .any(|args| args == ["--cpu-threads", thread_count.as_str()]),
        "Ocelotl command must explicitly name the manifest thread count"
    );
    assert!(
        manifest
            .ocelotl
            .command
            .windows(2)
            .any(|args| args == ["--backend", manifest.ocelotl.backend.as_str()]),
        "Ocelotl command must explicitly name the declared backend"
    );
    assert!(
        manifest.ocelotl.command.windows(2).any(|args| args
            == [
                "--tokenizer-path",
                "local-artifacts/whisper_tiny_en/tokenizer.json"
            ]),
        "W-ASR.37 benchmark command must request tokenizer-backed text output"
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
    assert!(
        manifest
            .ocelotl
            .required_inputs
            .iter()
            .any(|path| path == "local-artifacts/whisper_tiny_en/tokenizer.json"),
        "Ocelotl benchmark command must declare the local tokenizer artifact it needs"
    );
    assert_command_shape(
        &BenchmarkCommand {
            backend: manifest.whisper_cpp.backend.clone(),
            cpu_kernel_mode: String::new(),
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
    assert_eq!(
        manifest.whisper_cpp.transcript_path,
        "local-artifacts/whisper_tiny_en/reference/sample_16khz_mono.wav.txt"
    );
    assert_eq!(manifest.whisper_cpp.backend, "cpu");
    assert_eq!(manifest.whisper_cpp.comparison_mode, "greedy_no_fallback");
    assert!(
        manifest
            .whisper_cpp
            .command
            .iter()
            .any(|arg| arg == &manifest.threads.to_string()),
        "whisper.cpp command should include the manifest thread count"
    );
    assert!(
        manifest.whisper_cpp.command.iter().any(|arg| arg == "-ng"),
        "CPU comparison command must disable whisper.cpp GPU offload"
    );
    assert!(
        manifest
            .whisper_cpp
            .command
            .windows(2)
            .any(|args| args == ["-bs", "1"])
            && manifest
                .whisper_cpp
                .command
                .windows(2)
                .any(|args| args == ["-bo", "1"])
            && manifest.whisper_cpp.command.iter().any(|arg| arg == "-nf"),
        "W-ASR.36 whisper.cpp command must pin greedy no-fallback comparison flags"
    );
}

#[test]
fn whisper_cpp_benchmark_record_names_timing_and_output_fields() {
    let record = parse_record("whisper_cpp_record.example.json");

    assert_eq!(record.fixture_version, 2);
    assert_eq!(record.manifest_name, "whisper_tiny_en_sample_16khz_mono");
    assert_eq!(record.status, "completed");
    assert_plan_and_environment(&record);
    assert_eq!(record.comparison.status, "comparable");
    assert!(record.comparison.workload_match);
    assert_eq!(record.comparison.model_family, "whisper-tiny.en");
    assert_eq!(record.comparison.language, "en");
    assert_eq!(record.comparison.decoding_mode, "greedy_no_fallback");
    assert_eq!(
        record.comparison.output_equivalence,
        "normalized_transcript"
    );
    assert_eq!(record.comparison.output_match, Some(true));
    assert!(record.comparison.audio_sha256.is_some());
    assert_eq!(record.comparison.reason, None);
    assert_eq!(record.samples.len(), 8);
    assert_alternating_samples(&record.samples, 1, 3);

    assert_completed_run(&record.ocelotl);
    assert_completed_run(&record.whisper_cpp);
    assert_eq!(record.ocelotl.audio_path, record.whisper_cpp.audio_path);
}

#[test]
fn missing_whisper_cpp_binary_record_has_clear_remediation() {
    let record = parse_record("whisper_cpp_missing_binary_record.example.json");

    assert_eq!(record.fixture_version, 2);
    assert_eq!(record.status, "skipped");
    assert_eq!(record.whisper_cpp.status, "skipped");
    assert!(record.whisper_cpp.aggregate.is_none());
    assert!(record.samples.is_empty());
    assert_eq!(record.comparison.status, "skipped");

    let skip_reason = record
        .whisper_cpp
        .skip_reason
        .as_ref()
        .expect("missing whisper.cpp binary skip should include a reason");
    assert!(skip_reason.contains("whisper.cpp"));
    assert!(skip_reason.contains("local-artifacts/whisper_cpp/whisper-cli.exe"));
    assert!(skip_reason.contains("docs/benchmarks/whisper-cpp.md"));
}

#[test]
fn dry_run_emits_plan_environment_and_truthful_effective_configuration() {
    let repo_root = repo_root();
    let output = Command::new("pwsh")
        .args([
            "-NoProfile",
            "-File",
            "tools/whisper-cpp-bench.ps1",
            "-ManifestPath",
            "fixtures/benchmarks/whisper_cpp_manifest.example.json",
            "-DryRun",
            "-WarmupIterations",
            "2",
            "-MeasuredIterations",
            "3",
        ])
        .current_dir(&repo_root)
        .output()
        .expect("PowerShell benchmark dry run must launch");

    assert!(
        output.status.success(),
        "dry run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let record: BenchmarkRecord = serde_json::from_slice(&output.stdout)
        .expect("dry run stdout must be exactly one benchmark JSON record");
    assert_eq!(record.status, "planned");
    assert_eq!(record.benchmark_plan.warmup_iterations, 2);
    assert_eq!(record.benchmark_plan.measured_iterations, 3);
    assert_eq!(record.ocelotl.effective_config.threads, 4);
    assert_eq!(record.ocelotl.effective_config.backend, "cpu");
    assert_eq!(
        record.ocelotl.effective_config.cpu_kernel_mode.as_deref(),
        Some("scalar")
    );
    assert_eq!(record.whisper_cpp.effective_config.threads, 4);
    assert_eq!(record.whisper_cpp.effective_config.backend, "cpu");
    assert_eq!(
        record
            .whisper_cpp
            .effective_config
            .comparison_mode
            .as_deref(),
        Some("greedy_no_fallback")
    );
}

#[test]
fn dry_run_rejects_command_thread_configuration_mismatch() {
    let repo_root = repo_root();
    let mut manifest: serde_json::Value =
        serde_json::from_str(&read_fixture("whisper_cpp_manifest.example.json"))
            .expect("manifest JSON must parse");
    let command = manifest["ocelotl"]["command"]
        .as_array_mut()
        .expect("Ocelotl command must be an array");
    let threads_position = command
        .iter()
        .position(|arg| arg == "--cpu-threads")
        .expect("fixture command must name --cpu-threads");
    command[threads_position + 1] = serde_json::Value::String("1".to_string());

    let path = std::env::temp_dir().join(format!(
        "ocelotl-whisper-benchmark-mismatch-{}.json",
        std::process::id()
    ));
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&manifest).expect("serialize mismatch manifest"),
    )
    .expect("write mismatch manifest");

    let output = Command::new("pwsh")
        .args([
            "-NoProfile",
            "-File",
            "tools/whisper-cpp-bench.ps1",
            "-ManifestPath",
        ])
        .arg(&path)
        .arg("-DryRun")
        .current_dir(&repo_root)
        .output()
        .expect("PowerShell benchmark mismatch dry run must launch");
    let _ = std::fs::remove_file(path);

    assert!(!output.status.success(), "configuration mismatch must fail");
    let diagnostics = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        diagnostics.contains("Ocelotl command --cpu-threads must equal manifest threads"),
        "unexpected mismatch diagnostic: {diagnostics}"
    );
}

#[test]
fn runner_executes_alternating_samples_and_preserves_ocelotl_timing_json() {
    let repo_root = repo_root();
    let pwsh = powershell_executable();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after Unix epoch")
        .as_nanos();
    let temp = std::env::temp_dir().join(format!(
        "ocelotl-whisper-benchmark-runner-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&temp).expect("create runner fixture directory");

    let ocelotl_script = temp.join("fake-ocelotl.ps1");
    let whisper_script = temp.join("fake-whisper.ps1");
    let ocelotl_model = temp.join("model.safetensors");
    let whisper_model = temp.join("ggml-model.bin");
    let audio = temp.join("sample.wav");
    let transcript = temp.join("sample.wav.txt");
    std::fs::write(&ocelotl_model, b"synthetic ocelotl model")
        .expect("write synthetic Ocelotl model");
    std::fs::write(&whisper_model, b"synthetic whisper.cpp model")
        .expect("write synthetic whisper.cpp model");
    std::fs::write(&audio, b"synthetic shared audio").expect("write synthetic audio");
    std::fs::write(
        &ocelotl_script,
        r#"[ordered]@{
    status = "completed"
    token_count = 2
    text = "Stable, transcript!"
    matches_expected = $true
    backend = "cpu"
    cpu_kernel_mode = "scalar"
    cpu_threads = 2
    timings_ms = [ordered]@{ total = 7; log_mel = 2; decode_total = 3 }
} | ConvertTo-Json -Compress -Depth 8
"#,
    )
    .expect("write fake Ocelotl command");
    std::fs::write(
        &whisper_script,
        format!(
            "Set-Content -LiteralPath '{}' -Value 'stable transcript' -NoNewline\nWrite-Output 'fake whisper.cpp timing output'\n",
            powershell_single_quoted_path(&transcript)
        ),
    )
    .expect("write fake whisper.cpp command");

    let manifest = serde_json::json!({
        "fixture_version": 2,
        "name": "synthetic_runner_contract",
        "model_path": ocelotl_model,
        "audio_path": audio,
        "model_family": "whisper-tiny.en",
        "language": "en",
        "decoding_mode": "greedy_no_fallback",
        "threads": 2,
        "warmup_iterations": 1,
        "measured_iterations": 2,
        "execution_order": "alternating",
        "output_equivalence": "normalized_transcript",
        "ocelotl": {
            "backend": "cpu",
            "cpu_kernel_mode": "scalar",
            "command": [
                pwsh, "-NoProfile", "-File", ocelotl_script,
                "--cpu-threads", "2", "--backend", "cpu",
                "--cpu-kernel-mode", "scalar"
            ],
            "model_path": ocelotl_model,
            "audio_path": audio,
            "required_inputs": []
        },
        "whisper_cpp": {
            "binary": pwsh,
            "transcript_path": transcript,
            "backend": "cpu",
            "comparison_mode": "greedy_no_fallback",
            "command": [
                pwsh, "-NoProfile", "-File", whisper_script,
                "-t", "2", "-ng", "-otxt"
            ],
            "model_path": whisper_model,
            "audio_path": audio
        }
    });
    let manifest_path = temp.join("manifest.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("serialize runner manifest"),
    )
    .expect("write runner manifest");

    let output = Command::new("pwsh")
        .args([
            "-NoProfile",
            "-File",
            "tools/whisper-cpp-bench.ps1",
            "-ManifestPath",
        ])
        .arg(&manifest_path)
        .current_dir(&repo_root)
        .output()
        .expect("PowerShell benchmark runner must launch");

    assert!(
        output.status.success(),
        "synthetic benchmark run failed: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let record: BenchmarkRecord = serde_json::from_slice(&output.stdout)
        .expect("synthetic runner stdout must be a benchmark record");
    assert_eq!(record.status, "completed");
    assert_eq!(record.comparison.status, "comparable");
    assert_eq!(record.comparison.output_match, Some(true));
    assert_eq!(record.samples.len(), 6);
    assert_alternating_samples(&record.samples, 1, 2);
    assert_eq!(
        record
            .samples
            .iter()
            .find(|sample| sample.phase == "measured" && sample.engine == "ocelotl")
            .and_then(|sample| sample.output.parsed_json.as_ref())
            .and_then(|json| json["timings_ms"]["total"].as_u64()),
        Some(7)
    );
    assert_eq!(
        record
            .ocelotl
            .aggregate
            .as_ref()
            .expect("Ocelotl aggregate must exist")
            .sample_count,
        2
    );
    assert_eq!(
        record
            .whisper_cpp
            .aggregate
            .as_ref()
            .expect("whisper.cpp aggregate must exist")
            .sample_count,
        2
    );

    let _ = std::fs::remove_dir_all(temp);
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
    assert!(
        run.effective_config.threads > 0,
        "thread count must be positive"
    );
    let aggregate = run
        .aggregate
        .as_ref()
        .expect("completed benchmark run must record aggregate statistics");
    assert_eq!(aggregate.sample_count, 3);
    assert!(aggregate.min_wall_time_ms <= aggregate.median_wall_time_ms as u64);
    assert!(aggregate.median_wall_time_ms <= aggregate.p95_wall_time_ms);
    assert!(aggregate.p95_wall_time_ms <= aggregate.max_wall_time_ms as f64);
    assert!(
        run.latest_output.token_count.is_some()
            || run
                .latest_output
                .text
                .as_ref()
                .is_some_and(|text| !text.trim().is_empty())
            || run
                .latest_output
                .stdout_excerpt
                .as_ref()
                .is_some_and(|stdout| !stdout.trim().is_empty()),
        "completed benchmark run must record an output token/text/stdout summary"
    );
    if run.effective_config.cpu_kernel_mode.is_some() {
        assert!(run.latest_output.parsed_json.is_some());
    }
}

fn assert_plan_and_environment(record: &BenchmarkRecord) {
    assert_eq!(record.benchmark_plan.warmup_iterations, 1);
    assert_eq!(record.benchmark_plan.measured_iterations, 3);
    assert_eq!(record.benchmark_plan.execution_order, "alternating");
    assert!(!record.environment.captured_at_utc.trim().is_empty());
    assert!(!record.environment.repository_revision.trim().is_empty());
    let _ = record.environment.repository_dirty;
    assert!(!record.environment.os.trim().is_empty());
    assert!(!record.environment.architecture.trim().is_empty());
    assert!(!record.environment.powershell_version.trim().is_empty());
    assert!(record.environment.logical_processors > 0);
}

fn assert_alternating_samples(
    samples: &[BenchmarkSample],
    warmup_iterations: u32,
    measured_iterations: u32,
) {
    let mut expected_sequence = 0;
    for (phase, count) in [
        ("warmup", warmup_iterations),
        ("measured", measured_iterations),
    ] {
        for iteration in 0..count {
            let expected_engines = if iteration % 2 == 0 {
                ["ocelotl", "whisper_cpp"]
            } else {
                ["whisper_cpp", "ocelotl"]
            };
            for engine in expected_engines {
                let sample = &samples[expected_sequence as usize];
                assert_eq!(sample.sequence, expected_sequence);
                assert_eq!(sample.phase, phase);
                assert_eq!(sample.iteration, iteration);
                assert_eq!(sample.engine, engine);
                assert!(sample.wall_time_ms > 0);
                assert_eq!(sample.exit_code, 0);
                if engine == "ocelotl" {
                    assert!(sample.output.parsed_json.is_some());
                }
                expected_sequence += 1;
            }
        }
    }
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
    repo_root()
        .join("fixtures")
        .join("benchmarks")
        .join(file_name)
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn powershell_executable() -> String {
    let output = Command::new("pwsh")
        .args(["-NoProfile", "-Command", "(Get-Command pwsh).Source"])
        .output()
        .expect("PowerShell executable discovery must launch");
    assert!(
        output.status.success(),
        "PowerShell executable discovery failed"
    );
    let path = String::from_utf8(output.stdout).expect("PowerShell path must be UTF-8");
    path.trim().to_string()
}

fn powershell_single_quoted_path(path: &std::path::Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}
