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
  --expected-tokens-path local-artifacts/whisper_tiny_en/reference/expected_tokens.json `
  --cpu-kernel-mode optimized
```

The hook loads the Ocelotl safetensors bundle, decodes the shared WAV input,
encodes audio once, runs the current no-timestamps transcription path to the
expected token length using the encoded-audio state, and prints a JSON summary.
It is a timing hook for local comparison, not a throughput-optimized public
transcription CLI.

`--cpu-kernel-mode` defaults to `scalar`. The benchmark manifest currently
passes `optimized` to measure the W-ASR.23 optimized CPU backend explicitly.

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
an output summary. The Ocelotl stdout JSON includes `cpu_kernel_mode` plus
`resident_model_ms` and `timings_ms`.

`resident_model_ms` separates the loaded-model product path from benchmark
setup:

- `audio_to_tokens`: log-mel + audio encoder + token decode after model load.
- `mel_to_tokens`: audio encoder + token decode after model load and log-mel.

`timings_ms` includes:

- `config_parse`
- `manifest_validate`
- `expected_tokens_read`
- `tensor_load_model`
- `wav_read`
- `log_mel`
- `audio_encode`
- `decode_total`
- `decode_token`

The output summary is intentionally loose at W-ASR.13/W-ASR.20: it may carry
token count, text, or a stdout excerpt depending on the target. Do not compare
transcripts here as a correctness gate.

## Current Limits

- Ocelotl's side is a narrow benchmark hook, not a production transcription
  CLI.
- The Ocelotl and whisper.cpp model files use different on-disk formats, so the
  manifest must name both paths even when they represent the same tiny.en model.
- The runner records wall-clock command time for both sides. Only the Ocelotl
  side currently emits stage-level timings.
- No CPU performance parity claim exists yet. The 2026-05-12 tiny.en local run
  after encoded-audio reuse measured Ocelotl at 14,190 ms and whisper.cpp at
  482 ms (about 29.4x slower). Before encoder reuse, the same local comparison
  measured about 65,591 ms versus 550 ms (about 119.3x slower).

## W-ASR.24 Refresh - 2026-05-12

Fresh local record after W-ASR.23:

```text
local-artifacts/benchmarks/whisper_cpp_tiny_en_after_optimized_cpu.json
```

Result on this machine:

| Target | Mode | Wall time | Notes |
| ------ | ---- | --------- | ----- |
| Ocelotl | `optimized` | 16,648 ms | `matches_expected = true`; `decode_total = 9,437 ms`; `audio_encode = 2,397 ms`. |
| whisper.cpp | n/a | 564 ms | Same tiny.en GGML model, same WAV, `-t 4 -otxt -nt`. |
| Ocelotl direct hook | `scalar` | 14,179 ms | Same release binary and artifacts; `decode_total = 7,109 ms`; not run through the two-target wrapper. |

The optimized backend is selectable and parity-clean, but it is **not** a
Whisper performance win yet. In this run, optimized mode was about 17% slower
than scalar and about 29.5x slower than whisper.cpp. Keep scalar as the default
Whisper product path until optimized mode beats scalar on the gates below.

The likely reason is layout-specific: W-ASR.23's optimized matmul improves the
pre-transposed `[in, out]` path used by Qwen, but Whisper still stores many
projection weights as `[out, in]`; the current optimized `linear_out_by_in`
loop walks those weights with a large stride. Fixing that should be a targeted
Whisper weight-layout or kernel-layout task, not a default-mode flip.

## CPU Gates

These are local performance gates, not default CI gates. A run only counts if
`matches_expected = true` and the record uses the same audio, model family, and
thread count as the comparison baseline.

| Gate | Requirement | Current W-ASR.24 result |
| ---- | ----------- | ----------------------- |
| Correctness gate | Ocelotl generated token IDs exactly match the expected local fixture. | Passes for scalar and optimized tiny.en runs. |
| Scalar regression gate | Scalar release hook should not regress more than 10% from the latest same-machine scalar baseline (`14,179 ms` total, `7,109 ms` decode). | Passes; current scalar is the refreshed baseline. |
| Optimized-default gate | Optimized mode must be at least 10% faster than scalar total time and decode time on the same machine before becoming the Whisper default. | Fails; optimized is slower (`16,648 ms` vs `14,179 ms`). |
| First CPU competitiveness gate | Ocelotl tiny.en should reach <=10x whisper.cpp wall time before claiming meaningful CPU progress. | Fails; current optimized comparison is ~29.5x slower. |
| CPU-competitive claim gate | Ocelotl tiny.en should reach <=3x whisper.cpp wall time, with exact token parity, before calling the CPU path competitive. | Fails; still far outside the target. |

The next CPU task should target the actual remaining hot path: decoder
projection/layout work and attention/MLP buffer reuse, measured by stage-level
timings before and after the change.

## W-ASR.25 Resident-Model Timing

The benchmark hook now emits `resident_model_ms` in addition to raw wall-clock
stage timings. This does not make the CPU path faster, but it prevents model
loading and local fixture I/O from hiding the actual embedded-app latency
surface.

For the W-ASR.25 scalar direct hook run, the resident-model view is:

| Metric | Formula | W-ASR.25 scalar value |
| ------ | ------- | --------------------- |
| Loaded model, audio to tokens | `log_mel + audio_encode + decode_total` | `9,925 ms` |
| Loaded model, mel to tokens | `audio_encode + decode_total` | `9,172 ms` |

For tiny.en, the first optimization gate should focus on this resident path.
Model load remains important for startup and memory footprint, but it is the
wrong denominator for steady-state embedded transcription latency.
