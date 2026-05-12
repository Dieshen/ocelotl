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

## W-ASR.26 Cross-Attention K/V Cache

The decoder now precomputes cross-attention key/value projections once inside
`WhisperEncodedAudio` instead of recomputing them for every generated token.
This is the first CPU speedup that changes model internals while preserving the
same public transcription path and exact token parity.

Fresh scalar release run on the same tiny.en fixture:

| Metric | W-ASR.25 scalar | W-ASR.26 scalar | Change |
| ------ | --------------- | --------------- | ------ |
| Total hook wall time | `13,869 ms` | `8,582 ms` | ~38% faster |
| Resident audio to tokens | `9,925 ms` | `4,635 ms` | ~53% faster |
| Resident mel to tokens | `9,172 ms` | `3,884 ms` | ~58% faster |
| Decode total | `7,062 ms` | `1,519 ms` | ~78% faster |

Optimized mode remains opt-in and is still slower for Whisper after this change:
`12,468 ms` total, `7,015 ms` resident audio-to-tokens, and `3,277 ms`
decode total. Scalar remains the correct Whisper default.

Current scalar hot spots after W-ASR.26:

- `tensor_load_model`: `3,944 ms` startup cost.
- `audio_encode`: `2,365 ms`, now including one-time cross-attention K/V
  precompute.
- `decode_total`: `1,519 ms`.
- `log_mel`: `751 ms`.

The next CPU speed task should target decoder self-attention KV reuse /
one-token incremental decode. After that, the obvious remaining CPU work is
FFT-based log-mel and Whisper-specific projection weight packing.

## W-ASR.27 Decoder Self-Attention K/V Cache

The decoder now prepares a `WhisperDecoderState` for the startup prompt and
then appends generated tokens one at a time. Each decoder layer keeps
self-attention key/value projections for the current prefix, so generation no
longer reruns the full decoder prefix for every new token. The older stateless
`forward_next_token_logits_from_audio` API remains available and delegates
through the same state preparation path; runtime transcription, local parity
helpers, WER corpus helpers, and the benchmark hook use the cache-aware loop.

Fresh scalar release run on the same tiny.en fixture:

| Metric | W-ASR.26 scalar | W-ASR.27 scalar | Change |
| ------ | --------------- | --------------- | ------ |
| Total hook wall time | `8,582 ms` | `7,409 ms` | ~14% faster |
| Resident audio to tokens | `4,635 ms` | `3,506 ms` | ~24% faster |
| Resident mel to tokens | `3,884 ms` | `2,729 ms` | ~30% faster |
| Decode total | `1,519 ms` | `319 ms` | ~79% faster |

Optimized mode still passes exact token parity but remains slower than scalar
for Whisper: `8,849 ms` total, `5,023 ms` resident audio-to-tokens, and
`1,591 ms` decode total. Scalar remains the correct Whisper default.

Current scalar hot spots after W-ASR.27:

- `tensor_load_model`: `3,901 ms` startup cost.
- `audio_encode`: `2,410 ms`, including encoder forward and one-time
  cross-attention K/V precompute.
- `log_mel`: `777 ms`.
- `decode_total`: `319 ms`.

Against the W-ASR.24 whisper.cpp wall-time baseline (`564 ms`), the scalar
Ocelotl hook is now about `13.1x` slower by total wall time. It is materially
closer, but still fails the documented `<=10x` first CPU competitiveness gate.
The loaded-model resident audio-to-tokens view is about `6.2x` that baseline,
which is useful product signal but not a replacement for the wall-time gate.

The next CPU task should break down `audio_encode` internally before more broad
optimization. Likely follow-ups are encoder/projection weight packing and
FFT-based log-mel; decoder work is no longer the dominant resident bottleneck
for tiny.en.

## W-ASR.28 Bulk Safetensors Value Loading

The benchmark hook and local proof helpers now load all required Whisper
safetensors values from one parsed archive instead of calling
`load_safetensors_tensor_f32` once per tensor. This is a startup-path
optimization, not a resident-model optimization: it removes repeated file reads
and safetensors header parses before model construction.

Fresh scalar release run on the same tiny.en fixture:

| Metric | W-ASR.27 scalar | W-ASR.28 scalar | Change |
| ------ | --------------- | --------------- | ------ |
| Total hook wall time | `7,409 ms` | `3,620 ms` | ~51% faster |
| Tensor load + model construction | `3,901 ms` | `61 ms` | ~98% faster |
| Resident audio to tokens | `3,506 ms` | `3,557 ms` | ~1% slower/noise |
| Decode total | `319 ms` | `344 ms` | noise |

Optimized mode remains parity-clean but slower for Whisper: `5,196 ms` total,
`5,130 ms` resident audio-to-tokens, and `1,612 ms` decode total.

This is the first result that clears the documented first CPU competitiveness
gate by wall time: `3,620 ms / 564 ms ~= 6.4x`, under the `<=10x` threshold.
It does **not** clear the competitive claim gate (`<=3x`). The remaining
resident scalar hot spots are now:

- `audio_encode`: `2,455 ms`.
- `log_mel`: `758 ms`.
- `decode_total`: `344 ms`.
- `tensor_load_model`: `61 ms`.

The next task should target resident audio processing, starting with an
internal timing split for `audio_encode` so encoder forward and cross-attention
K/V precompute are measured separately before optimizing either path.

## W-ASR.29 Precomputed STFT Fourier Basis

`log_mel_spectrogram` now precomputes the fixed 400-point STFT Fourier basis
once with `OnceLock` and reuses it across frames. This keeps the same reference
DFT math but removes repeated `sin`/`cos` calls from the per-frame hot loop.

Fresh scalar release run on the same tiny.en fixture:

| Metric | W-ASR.28 scalar | W-ASR.29 scalar | Change |
| ------ | --------------- | --------------- | ------ |
| Total hook wall time | `3,620 ms` | `2,859 ms` | ~21% faster |
| Resident audio to tokens | `3,557 ms` | `2,795 ms` | ~21% faster |
| Log-mel | `758 ms` | `45 ms` | ~94% faster |
| Audio encode | `2,455 ms` | `2,402 ms` | noise |
| Decode total | `344 ms` | `348 ms` | noise |

Optimized mode remains parity-clean but slower for Whisper: `4,489 ms` total,
`4,427 ms` resident audio-to-tokens, `2,746 ms` audio encode, and `1,636 ms`
decode total.

Against the W-ASR.24 whisper.cpp wall-time baseline (`564 ms`), scalar Ocelotl
is now about `5.1x` slower (`2,859 / 564`). That is closer but still does not
clear the `<=3x` competitive claim gate. The remaining scalar hot spots are now:

- `audio_encode`: `2,402 ms`.
- `decode_total`: `348 ms`.
- `tensor_load_model`: `60 ms`.
- `log_mel`: `45 ms`.

The next CPU task should split `audio_encode` into encoder forward vs
cross-attention K/V precompute and then optimize the dominant half.

## W-ASR.30 Audio Encode Detail Timing

The benchmark hook now reports `timings_ms.audio_encode_detail` with:

- `encoder`: convolution + encoder transformer stack + final encoder layer norm.
- `cross_attention_precompute`: one-time decoder cross-attention K/V projection
  from the encoded audio.

Fresh scalar release run on the same tiny.en fixture:

| Metric | W-ASR.30 scalar |
| ------ | --------------- |
| Total hook wall time | `2,817 ms` |
| Resident audio to tokens | `2,753 ms` |
| Log-mel | `46 ms` |
| Audio encode total | `2,387 ms` |
| Encoder | `2,153 ms` |
| Cross-attention K/V precompute | `234 ms` |
| Decode total | `320 ms` |

This is a measurement seam, not a claimed optimization over W-ASR.29; the
small total-time difference is run-to-run noise. The useful result is the split:
about 90% of `audio_encode` is now known to be encoder forward, so the next CPU
optimization should target the encoder transformer path rather than
cross-attention K/V setup.

## W-ASR.31 Whisper Attention Context Accumulation Locality

Whisper attention now accumulates each V row contiguously into the context row
instead of walking V with a strided per-output-dimension loop. The per-dimension
sum order stays key-position order, so token parity is unchanged, but the hot
context accumulation loop is friendlier to CPU caches.

Fresh scalar release run on the same tiny.en fixture:

| Metric | W-ASR.30 scalar | W-ASR.31 scalar | Change |
| ------ | --------------- | --------------- | ------ |
| Total hook wall time | `2,817 ms` | `2,559 ms` | ~9% faster |
| Resident audio to tokens | `2,753 ms` | `2,496 ms` | ~9% faster |
| Audio encode total | `2,387 ms` | `2,132 ms` | ~11% faster |
| Encoder | `2,153 ms` | `1,896 ms` | ~12% faster |
| Cross-attention K/V precompute | `234 ms` | `235 ms` | noise |
| Decode total | `320 ms` | `317 ms` | noise |
| Log-mel | `46 ms` | `47 ms` | noise |

Against the W-ASR.24 whisper.cpp wall-time baseline (`564 ms`), scalar Ocelotl
is now about `4.5x` slower (`2,559 / 564`). The remaining dominant bottleneck
is still encoder forward.
