# whisper.cpp Benchmark Harness

W-ASR.13 defines the local benchmark harness shape for comparing Ocelotl's
current Whisper path with whisper.cpp on the same local audio input. This is a
performance baseline only. whisper.cpp output is not the canonical correctness
oracle for Ocelotl; correctness still comes from the default synthetic tests and
the opt-in exact-token local-artifact parity fixture.

The committed coverage for this harness is intentionally default-on and local
artifact free:

```powershell
cargo test -p ocelotl-models --test whisper_cpp_benchmark
```

That test validates the JSON manifest and record shapes under
`fixtures/benchmarks/`. It does not run whisper.cpp, read local model artifacts,
or require network access.

## Local Prerequisites

The opt-in benchmark runner expects these local-only files:

```text
local-artifacts/
  whisper_tiny_en/
    config.json
    tokenizer.json
    model.safetensors
    reference/
      sample_16khz_mono.wav
      expected_tokens.json
  whisper_cpp/
    whisper-cli.exe
    ggml-tiny.en.bin
target/
  release/
    ocelotl.exe
```

Prepare `local-artifacts/whisper_tiny_en/` as described in
`docs/artifact-preparation.md`. Build whisper.cpp locally and copy the CLI
binary plus the equivalent `ggml` tiny.en model into `local-artifacts/whisper_cpp/`,
or edit the benchmark manifest to point at your local whisper.cpp paths.

The upstream whisper.cpp CLI currently documents the `whisper-cli` executable
with `-m` for model path, `-f` for audio path, `-t` for thread count, and `-otxt`
for text output; see the official whisper.cpp
`examples/cli/README.md` at
`https://github.com/ggml-org/whisper.cpp/blob/master/examples/cli/README.md`.
Ocelotl keeps those paths local and ignored instead of adding download steps to
default validation.

Build the dedicated Ocelotl timing hook before running a real benchmark:

```powershell
cargo build --release
```

## Manifest

The default manifest fixture is:

```text
fixtures/benchmarks/whisper_cpp_manifest.example.json
```

It names:

- `model_path`: the Ocelotl safetensors model path.
- `audio_path`: the shared WAV input path.
- `threads`: the thread count passed to whisper.cpp.
- `ocelotl.command`: the dedicated Ocelotl transcription timing hook.
- `ocelotl.required_inputs`: the local Ocelotl artifact files checked before
  invoking the hook.
- `whisper_cpp.binary`: the whisper.cpp executable to check before running.
- `whisper_cpp.command`: the exact whisper.cpp invocation.

The Ocelotl side now runs a dedicated binary hook outside the Rust test harness:

```powershell
target/release/ocelotl.exe bench-whisper-transcribe `
  --config-path local-artifacts/whisper_tiny_en/config.json `
  --model-path local-artifacts/whisper_tiny_en/model.safetensors `
  --audio-path local-artifacts/whisper_tiny_en/reference/sample_16khz_mono.wav `
  --expected-tokens-path local-artifacts/whisper_tiny_en/reference/expected_tokens.json
```

The hook loads the Ocelotl safetensors bundle, decodes the shared WAV input,
runs the current no-timestamps transcription path to the expected token length,
and prints a JSON summary. It is a timing hook for local comparison, not a
throughput-optimized public transcription CLI.

The whisper.cpp side of the example manifest runs:

```powershell
local-artifacts/whisper_cpp/whisper-cli.exe -m local-artifacts/whisper_cpp/ggml-tiny.en.bin -f local-artifacts/whisper_tiny_en/reference/sample_16khz_mono.wav -t 4 -otxt -nt
```

## Running Locally

Validate the manifest without checking local files or running commands:

```powershell
pwsh -NoProfile -File tools/whisper-cpp-bench.ps1 `
  -ManifestPath fixtures/benchmarks/whisper_cpp_manifest.example.json `
  -DryRun
```

Run the opt-in benchmark and write a local record:

```powershell
pwsh -NoProfile -File tools/whisper-cpp-bench.ps1 `
  -ManifestPath fixtures/benchmarks/whisper_cpp_manifest.example.json `
  -OutputPath local-artifacts/benchmarks/whisper_cpp_tiny_en.json
```

`local-artifacts/benchmarks/` is ignored and should not be committed.

## Skip Behavior

If the whisper.cpp binary named by `whisper_cpp.binary` is missing, the runner
does not fail the default workflow. It emits a JSON record with:

- top-level `status = "skipped"`;
- `whisper_cpp.status = "skipped"`;
- null `wall_time_ms` and `exit_code` for the skipped target;
- a `skip_reason` that names the missing binary and this document.

The committed fixture
`fixtures/benchmarks/whisper_cpp_missing_binary_record.example.json` pins that
shape so missing local whisper.cpp installs produce an actionable remediation
instead of a vague command failure.

If whisper.cpp is present but the Ocelotl release binary or Ocelotl local
artifacts are missing, the runner also emits a skipped record with a
`skip_reason` naming the missing path and the remediation. Missing local inputs
are expected on machines that have not prepared `local-artifacts/`.

## Record Shape

A completed benchmark record names both command lines, both model paths, the
shared audio path, thread count, exit code, wall-clock time in milliseconds, and
an output summary. The output summary is intentionally loose at W-ASR.13: it may
carry token count, text, or a stdout excerpt depending on the target. Do not
compare transcripts here as a correctness gate.

## Current Limits

- Ocelotl's side is a narrow benchmark hook, not a production transcription
  CLI.
- The Ocelotl and whisper.cpp model files use different on-disk formats, so the
  manifest must name both paths even when they represent the same tiny.en model.
- The harness records wall-clock command time only. It does not separate model
  load, preprocessing, encoder, decoder, or text decode timings yet.
- No performance parity claim exists until real local records are captured and
  reviewed.
