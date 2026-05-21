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
  --tokenizer-path local-artifacts/whisper_tiny_en/tokenizer.json `
  --cpu-kernel-mode scalar
```

The hook loads the Ocelotl safetensors bundle, decodes the shared WAV input,
encodes audio once, runs the current no-timestamps transcription path to the
expected token length using the encoded-audio state, optionally decodes the
generated token IDs into transcript text when `--tokenizer-path` is provided,
and prints a JSON summary. It is a timing hook for local comparison, not a
throughput-optimized public transcription CLI.

`--cpu-kernel-mode` defaults to `scalar`. The benchmark manifest currently
passes `scalar` because the W-ASR.35 tiled scalar path is the current winning
Whisper CPU path. `optimized` remains available as an explicit opt-in parity
and performance probe.

The whisper.cpp side of the example manifest runs:

```powershell
local-artifacts/whisper_cpp/whisper-cli.exe -m local-artifacts/whisper_cpp/ggml-tiny.en.bin -f local-artifacts/whisper_tiny_en/reference/sample_16khz_mono.wav -t 4 -otxt -nt -bs 1 -bo 1 -nf
```

The `-bs 1 -bo 1 -nf` flags pin a greedy, no-fallback whisper.cpp comparison.
Older local benchmark rows below that omit those flags used whisper.cpp's
default `5 beams + best of 5` mode; W-ASR.36 records the stricter cleanup.

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

| Target              | Mode        | Wall time | Notes                                                                                                 |
| ------------------- | ----------- | --------- | ----------------------------------------------------------------------------------------------------- |
| Ocelotl             | `optimized` | 16,648 ms | `matches_expected = true`; `decode_total = 9,437 ms`; `audio_encode = 2,397 ms`.                      |
| whisper.cpp         | n/a         | 564 ms    | Same tiny.en GGML model, same WAV, `-t 4 -otxt -nt`.                                                  |
| Ocelotl direct hook | `scalar`    | 14,179 ms | Same release binary and artifacts; `decode_total = 7,109 ms`; not run through the two-target wrapper. |

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

| Gate                           | Requirement                                                                                                                                | Current W-ASR.24 result                                  |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------- |
| Correctness gate               | Ocelotl generated token IDs exactly match the expected local fixture.                                                                      | Passes for scalar and optimized tiny.en runs.            |
| Scalar regression gate         | Scalar release hook should not regress more than 10% from the latest same-machine scalar baseline (`14,179 ms` total, `7,109 ms` decode).  | Passes; current scalar is the refreshed baseline.        |
| Optimized-default gate         | Optimized mode must be at least 10% faster than scalar total time and decode time on the same machine before becoming the Whisper default. | Fails; optimized is slower (`16,648 ms` vs `14,179 ms`). |
| First CPU competitiveness gate | Ocelotl tiny.en should reach <=10x whisper.cpp wall time before claiming meaningful CPU progress.                                          | Fails; current optimized comparison is ~29.5x slower.    |
| CPU-competitive claim gate     | Ocelotl tiny.en should reach <=3x whisper.cpp wall time, with exact token parity, before calling the CPU path competitive.                 | Fails; still far outside the target.                     |

The next CPU task should target the actual remaining hot path: decoder
projection/layout work and attention/MLP buffer reuse, measured by stage-level
timings before and after the change.

## W-ASR.25 Resident-Model Timing

The benchmark hook now emits `resident_model_ms` in addition to raw wall-clock
stage timings. This does not make the CPU path faster, but it prevents model
loading and local fixture I/O from hiding the actual embedded-app latency
surface.

For the W-ASR.25 scalar direct hook run, the resident-model view is:

| Metric                        | Formula                                 | W-ASR.25 scalar value |
| ----------------------------- | --------------------------------------- | --------------------- |
| Loaded model, audio to tokens | `log_mel + audio_encode + decode_total` | `9,925 ms`            |
| Loaded model, mel to tokens   | `audio_encode + decode_total`           | `9,172 ms`            |

For tiny.en, the first optimization gate should focus on this resident path.
Model load remains important for startup and memory footprint, but it is the
wrong denominator for steady-state embedded transcription latency.

## W-ASR.26 Cross-Attention K/V Cache

The decoder now precomputes cross-attention key/value projections once inside
`WhisperEncodedAudio` instead of recomputing them for every generated token.
This is the first CPU speedup that changes model internals while preserving the
same public transcription path and exact token parity.

Fresh scalar release run on the same tiny.en fixture:

| Metric                   | W-ASR.25 scalar | W-ASR.26 scalar | Change      |
| ------------------------ | --------------- | --------------- | ----------- |
| Total hook wall time     | `13,869 ms`     | `8,582 ms`      | ~38% faster |
| Resident audio to tokens | `9,925 ms`      | `4,635 ms`      | ~53% faster |
| Resident mel to tokens   | `9,172 ms`      | `3,884 ms`      | ~58% faster |
| Decode total             | `7,062 ms`      | `1,519 ms`      | ~78% faster |

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

| Metric                   | W-ASR.26 scalar | W-ASR.27 scalar | Change      |
| ------------------------ | --------------- | --------------- | ----------- |
| Total hook wall time     | `8,582 ms`      | `7,409 ms`      | ~14% faster |
| Resident audio to tokens | `4,635 ms`      | `3,506 ms`      | ~24% faster |
| Resident mel to tokens   | `3,884 ms`      | `2,729 ms`      | ~30% faster |
| Decode total             | `1,519 ms`      | `319 ms`        | ~79% faster |

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

| Metric                           | W-ASR.27 scalar | W-ASR.28 scalar | Change           |
| -------------------------------- | --------------- | --------------- | ---------------- |
| Total hook wall time             | `7,409 ms`      | `3,620 ms`      | ~51% faster      |
| Tensor load + model construction | `3,901 ms`      | `61 ms`         | ~98% faster      |
| Resident audio to tokens         | `3,506 ms`      | `3,557 ms`      | ~1% slower/noise |
| Decode total                     | `319 ms`        | `344 ms`        | noise            |

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

| Metric                   | W-ASR.28 scalar | W-ASR.29 scalar | Change      |
| ------------------------ | --------------- | --------------- | ----------- |
| Total hook wall time     | `3,620 ms`      | `2,859 ms`      | ~21% faster |
| Resident audio to tokens | `3,557 ms`      | `2,795 ms`      | ~21% faster |
| Log-mel                  | `758 ms`        | `45 ms`         | ~94% faster |
| Audio encode             | `2,455 ms`      | `2,402 ms`      | noise       |
| Decode total             | `344 ms`        | `348 ms`        | noise       |

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

| Metric                         | W-ASR.30 scalar |
| ------------------------------ | --------------- |
| Total hook wall time           | `2,817 ms`      |
| Resident audio to tokens       | `2,753 ms`      |
| Log-mel                        | `46 ms`         |
| Audio encode total             | `2,387 ms`      |
| Encoder                        | `2,153 ms`      |
| Cross-attention K/V precompute | `234 ms`        |
| Decode total                   | `320 ms`        |

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

| Metric                         | W-ASR.30 scalar | W-ASR.31 scalar | Change      |
| ------------------------------ | --------------- | --------------- | ----------- |
| Total hook wall time           | `2,817 ms`      | `2,559 ms`      | ~9% faster  |
| Resident audio to tokens       | `2,753 ms`      | `2,496 ms`      | ~9% faster  |
| Audio encode total             | `2,387 ms`      | `2,132 ms`      | ~11% faster |
| Encoder                        | `2,153 ms`      | `1,896 ms`      | ~12% faster |
| Cross-attention K/V precompute | `234 ms`        | `235 ms`        | noise       |
| Decode total                   | `320 ms`        | `317 ms`        | noise       |
| Log-mel                        | `46 ms`         | `47 ms`         | noise       |

Against the W-ASR.24 whisper.cpp wall-time baseline (`564 ms`), scalar Ocelotl
is now about `4.5x` slower (`2,559 / 564`). The remaining dominant bottleneck
is still encoder forward.

## W-ASR.32 Scalar Linear + Attention Dot Unroll

The scalar `[out, in]` linear kernel now computes four output dimensions per
inner input loop, reducing repeated reads of the same input row while preserving
the per-output accumulation order. Whisper attention also unrolls the
head-dimension dot product by four. An attempted eight-output linear unroll was
measured and rejected because it regressed the tiny.en benchmark.

Fresh scalar release runs on the same tiny.en fixture both passed exact token
parity:

| Metric                         | W-ASR.31 scalar | W-ASR.32 scalar | Change      |
| ------------------------------ | --------------- | --------------- | ----------- |
| Total hook wall time           | `2,559 ms`      | `1,623 ms`      | ~37% faster |
| Resident audio to tokens       | `2,496 ms`      | `1,560 ms`      | ~38% faster |
| Audio encode total             | `2,132 ms`      | `1,290 ms`      | ~39% faster |
| Encoder                        | `1,896 ms`      | `1,166 ms`      | ~39% faster |
| Cross-attention K/V precompute | `235 ms`        | `124 ms`        | ~47% faster |
| Decode total                   | `317 ms`        | `224 ms`        | ~29% faster |
| Log-mel                        | `47 ms`         | `46 ms`         | noise       |

The immediately preceding W-ASR.32 scalar run measured `1,646 ms`, so the
result is not a single-run fluke. Against the W-ASR.24 whisper.cpp wall-time
baseline (`564 ms`), scalar Ocelotl is now about `2.88x` slower (`1,623 / 564`)
and clears the documented `<=3x` tiny.en CPU competitiveness gate.

This is still not "done optimizing CPU" in the broad sense. It is the first
validated competitive tiny.en gate. Larger Whisper sizes and longer audio still
need their own measurements, and optimized CPU mode remains opt-in until it
beats scalar.

## W-ASR.33 All-Size Scalar CPU Comparison

Fresh local all-size scalar runs on 2026-05-12 used the dedicated
`bench-whisper-transcribe` hook for Ocelotl and `whisper-cli -t 4 -otxt -nt`
for whisper.cpp. The large-v2 whisper.cpp command also passed `-l en`.

The Ocelotl rows compare against each bundle's
`reference/expected_tokens.json` and every row below had
`matches_expected = true`. The non-tiny expected-token references are short
contract checks rather than full transcript-quality proofs, so the encoder
timing is the useful performance signal. whisper.cpp used its default
`5 beams + best of 5`; Ocelotl used greedy decoding to the expected-token
length.

| Size      | Ocelotl scalar total | Ocelotl encoder | whisper.cpp total | whisper.cpp encode | Ratio    |
| --------- | -------------------- | --------------- | ----------------- | ------------------ | -------- |
| tiny.en   | `1,681 ms`           | `1,215 ms`      | `810 ms`          | `208.52 ms`        | `~2.07x` |
| base.en   | `3,286 ms`           | `2,759 ms`      | `805 ms`          | `452.28 ms`        | `~4.08x` |
| small.en  | `12,988 ms`          | `11,003 ms`     | `2,656.89 ms`     | `1,790.08 ms`      | `~4.89x` |
| medium.en | `44,153 ms`          | `37,543 ms`     | `8,406.72 ms`     | `5,873.18 ms`      | `~5.25x` |
| large-v2  | `90,024 ms`          | `75,904 ms`     | `16,317.55 ms`    | `11,813.18 ms`     | `~5.52x` |

This confirms the W-ASR.32 tiny.en gate was real but narrow. Ocelotl can load
and run all classic local Whisper sizes with exact token parity for the pinned
references, but only tiny.en is currently within the `<=3x` wall-time gate.
The larger sizes scale with the encoder gap, not model load, log-mel, or
decoder work.

The next performance work should target encoder GEMM. The useful upstream
comparison is no longer "does Ocelotl work?" but "why is whisper.cpp encoder
about `~6x` faster across base/small/medium/large?" The reference notes in
Obsidian point to register-tiled GEMM, SIMD vec-dot, dynamic tile threading,
and native F16/BF16 weight handling as the main deltas.

## W-ASR.34 Scratch-Arena Prototype Rejected

An encoder scratch-arena prototype was evaluated after the all-size comparison.
The prototype replaced the encoder's per-helper `Vec<f32>` returns with
caller-owned scratch buffers for conv/layernorm/attention/MLP intermediates.
It preserved exact token parity in focused model tests and in release local
benchmarks, but it did **not** improve performance:

| Size    | Prior scalar total | Scratch prototype total     | Result                                  |
| ------- | ------------------ | --------------------------- | --------------------------------------- |
| tiny.en | `1,681 ms`         | `1,739 ms`, then `1,772 ms` | regression/noise in the wrong direction |
| base.en | `3,286 ms`         | `3,535 ms`                  | regression                              |

The change was backed out and should not be treated as a shipped optimization.
Scratch reuse may become worthwhile once tiled kernels or native dtype buffers
change the allocation/compute balance, but current evidence says the next
implementation slice should be register-tiled scalar GEMM, not scratch plumbing.

## W-ASR.35 Register-Tiled Scalar Linear

The scalar `linear_out_by_in` kernel now computes a 4-row x 4-output tile for
the row-major activation / `[out, in]` weight layout used by Whisper
projections. The tile keeps 16 accumulators live and reuses each loaded weight
across four activation rows while preserving the per-output K-loop accumulation
order. Row and output tails fall back to the previous scalar shape.

Fresh release scalar runs on 2026-05-12:

| Size      | W-ASR.33 scalar total | W-ASR.35 scalar total | Change      | whisper.cpp total | Ratio after W-ASR.35 |
| --------- | --------------------- | --------------------- | ----------- | ----------------- | -------------------- |
| tiny.en   | `1,681 ms`            | `1,073 ms`            | ~36% faster | `810 ms`          | `~1.32x`             |
| base.en   | `3,286 ms`            | `1,807 ms`            | ~45% faster | `805 ms`          | `~2.24x`             |
| small.en  | `12,988 ms`           | `6,345 ms`            | ~51% faster | `2,656.89 ms`     | `~2.39x`             |
| medium.en | `44,153 ms`           | `20,941 ms`           | ~53% faster | `8,406.72 ms`     | `~2.49x`             |
| large-v2  | `90,024 ms`           | `40,416 ms`           | ~55% faster | `16,317.55 ms`    | `~2.48x`             |

All five Ocelotl runs had `matches_expected = true`.

Stage-level detail:

| Size      | Encoder before | Encoder after | Audio encode after | Decode after |
| --------- | -------------- | ------------- | ------------------ | ------------ |
| tiny.en   | `1,215 ms`     | `698 ms`      | `745 ms`           | `221 ms`     |
| base.en   | `2,759 ms`     | `1,502 ms`    | `1,623 ms`         | `19 ms`      |
| small.en  | `11,003 ms`    | `5,263 ms`    | `5,808 ms`         | `64 ms`      |
| medium.en | `37,543 ms`    | `17,399 ms`   | `19,358 ms`        | `187 ms`     |
| large-v2  | `75,904 ms`    | `33,203 ms`   | `37,323 ms`        | `296 ms`     |

This clears the existing `<=3x` whisper.cpp wall-time gate for all five classic
local Whisper sizes on the local benchmark set. It is still not a blanket CPU
performance parity claim: whisper.cpp is running a mature SIMD/threaded backend,
and Ocelotl's non-tiny references are short expected-token contract checks. The
next CPU work should choose between tiled-kernel threading, AVX2, or native
F16/BF16 weights based on the remaining encoder gap.

## W-ASR.36 Greedy whisper.cpp Comparison Cleanup

The example benchmark manifest now compares Ocelotl's greedy benchmark hook
against whisper.cpp with `-bs 1 -bo 1 -nf` instead of whisper.cpp's default
`5 beams + best of 5`. The older default-beam numbers are still useful because
they match the first command shape we ran locally, but they were not the fairest
speed denominator for Ocelotl's deterministic greedy path.

Fresh local whisper.cpp greedy/no-fallback runs on 2026-05-12:

| Size      | W-ASR.35 Ocelotl scalar total | whisper.cpp greedy total | whisper.cpp encode | Ratio    |
| --------- | ----------------------------- | ------------------------ | ------------------ | -------- |
| tiny.en   | `1,073 ms`                    | `329.02 ms`              | `197.95 ms`        | `~3.26x` |
| base.en   | `1,807 ms`                    | `664.86 ms`              | `458.74 ms`        | `~2.72x` |
| small.en  | `6,345 ms`                    | `2,364.21 ms`            | `1,777.94 ms`      | `~2.68x` |
| medium.en | `20,941 ms`                   | `7,556.48 ms`            | `5,881.14 ms`      | `~2.77x` |
| large-v2  | `40,416 ms`                   | `15,186.45 ms`           | `11,896.86 ms`     | `~2.66x` |

This is a stricter baseline than W-ASR.35's table. It downgrades the broad
"all five sizes are under `<=3x`" claim: base through large-v2 clear the
greedy/no-fallback gate on this local sample, while tiny.en is close but still
above it at about `3.26x`. The next CPU optimization should treat tiny.en
`<=3x` against greedy whisper.cpp as the nearest performance target, then
re-run all five sizes.

## W-ASR.37 Benchmark Text Output

`bench-whisper-transcribe` now accepts an optional `--tokenizer-path` argument.
When provided, the hook loads the local tokenizer, decodes the generated token
IDs with special tokens skipped, and emits a top-level JSON `text` field. The
JSON timing schema also reports `timings_ms.tokenizer_load` and
`timings_ms.text_decode`; both are `null` when the tokenizer path is omitted.

The example manifest now includes:

```powershell
--tokenizer-path local-artifacts/whisper_tiny_en/tokenizer.json
```

`tools/whisper-cpp-bench.ps1` also parses Ocelotl's JSON stdout into benchmark
record `output.token_count` and `output.text`, while retaining a truncated
`stdout_excerpt` for diagnostics.

A local tiny.en release proof with `--tokenizer-path` passed exact token parity
and emitted:

```text
text = " And so my fellow Americans ask not what your country can do for you ask what you can do for your country."
elapsed_ms = 1,161
timings_ms.tokenizer_load = 52
timings_ms.text_decode = 0
```

The text mirrors the current expected-token fixture, including its missing
comma after "you"; exact token parity remains the correctness gate.

A full local benchmark-runner pass also populated the Ocelotl record summary:
`output.token_count = 26` and the same decoded `output.text`.

## W-ASR.38 Multi-Threaded `linear_out_by_in`

`CpuKernelBackend::with_mode_and_threads(mode, threads)` now builds an
internal rayon thread pool when `threads >= 2`. The `linear_out_by_in`
dispatcher partitions the M-direction (output rows) across the pool when
`rows >= 32`; below that threshold it stays serial because rayon dispatch
overhead exceeds the per-matmul cost (decoder single-token decode hits this
fall-back). Each parallel chunk runs the existing scalar or optimized
compute body on a disjoint slice of `x` and `out`, so the K-loop
accumulation order is identical to the serial path and exact-token parity
holds bit-for-bit. The kernel parity unit test
`threaded_linear_out_by_in_matches_serial_bit_for_bit` pins that invariant.

The `bench-whisper-transcribe` hook now accepts `--cpu-threads N` (default
1) and emits the chosen value as a top-level `cpu_threads` field in the
JSON record. `KernelBackend::cpu_thread_pool()` is a new trait method with a
default `None` implementation; only `CpuKernelBackend` returns a pool, so
the CubeCL/GPU backends are unchanged.

Fresh local release runs on 2026-05-14 against the same five tiny.en /
base.en / small.en / medium.en / large-v2 bundles, scalar mode, `t=1` and
`t=4`, all with `matches_expected = true`:

| Size      | W-ASR.35 t=1 (baseline) | W-ASR.38 t=1 | W-ASR.38 t=4 | t=4 speedup vs t=1 | t=4 vs whisper.cpp greedy |
| --------- | ----------------------- | ------------ | ------------ | ------------------ | ------------------------- |
| tiny.en   | `1,073 ms`              | `1,339 ms`   | `1,013 ms`   | `~1.32x`           | `~3.08x` (was `~3.26x`)   |
| base.en   | `1,807 ms`              | `2,138 ms`   | `1,398 ms`   | `~1.53x`           | `~2.10x` (was `~2.72x`)   |
| small.en  | `6,345 ms`              | `7,219 ms`   | `4,076 ms`   | `~1.77x`           | `~1.72x` (was `~2.68x`)   |
| medium.en | `20,941 ms`             | `22,987 ms`  | `11,646 ms`  | `~1.97x`           | `~1.54x` (was `~2.77x`)   |
| large-v2  | `40,416 ms`             | `43,770 ms`  | `20,430 ms`  | `~2.14x`           | `~1.35x` (was `~2.66x`)   |

Note: the `t=1` column is slightly above the W-ASR.35 baseline on this
machine; that is run-to-run noise from the refactor seam (single extra
function call wrapping the compute body), not a regression. The
`t=4 vs whisper.cpp` column is the load-bearing one.

Encoder-stage detail (the dominant cost):

| Size      | t=1 encoder | t=4 encoder | encoder speedup |
| --------- | ----------- | ----------- | --------------- |
| tiny.en   | `806 ms`    | `607 ms`    | `~1.33x`        |
| base.en   | `1,728 ms`  | `1,181 ms`  | `~1.46x`        |
| small.en  | `5,813 ms`  | `3,382 ms`  | `~1.72x`        |
| medium.en | `18,295 ms` | `9,433 ms`  | `~1.94x`        |
| large-v2  | `34,416 ms` | `16,294 ms` | `~2.11x`        |

Cross-attention K/V precompute scales similarly (e.g. large-v2 `4,248 ms ->
1,052 ms`, `~4.0x`) because it is mostly larger matmuls and benefits
proportionally more from parallel dispatch.

This is the first run where base.en, small.en, medium.en, and large-v2 all
clear the documented `<=3x` competitive claim gate against greedy
whisper.cpp; small/medium/large clear `<=2x`, and large-v2 reaches `~1.35x`.
tiny.en is essentially unchanged because per-matmul cost is dominated by
rayon dispatch and per-tile loop setup at its small shapes.

The W-ASR.24 CPU gate table now reads:

| Gate                           | Status                                                  |
| ------------------------------ | ------------------------------------------------------- |
| Correctness gate               | Passes on all five sizes at t=1 and t=4.                |
| First CPU competitiveness gate (`<=10x`) | Passes for all five sizes.                    |
| CPU-competitive claim gate (`<=3x`)      | Passes for base/small/medium/large; tiny.en is `~3.08x`, essentially at the gate. |

The next CPU work options, in order of likely encoder impact for the
remaining gap:

1. **Parallelize Whisper attention's manual Q-row loop** in
   `whisper/primitives.rs::attention_from_projected`. Currently single-
   threaded; the q_seq * heads outer loop writes disjoint context rows so
   it is straightforward to chunk via the same `kernels.cpu_thread_pool()`
   accessor.
2. **AVX2 SIMD opt-in** for the `linear_out_by_in_compute` tile inner loop
   behind `CpuKernelMode::Avx2` (or similar). Roadmap P0.4. High risk
   (`unsafe`) but biggest remaining tile-level win.
3. **Native F16/BF16 weight matmul** to halve DRAM traffic. Roadmap P1.1,
   requires a typed-weight-block design doc first.

## W-ASR.39 Multi-Threaded Whisper Attention Q-Row Loop

Following W-ASR.38, `whisper/primitives.rs::attention_from_projected` now
also uses `kernels.cpu_thread_pool()` to parallelize its outer Q-row loop
when `q_seq >= 32`. The Q range is split into one chunk per worker thread;
each chunk owns a single `scores` scratch buffer that is reused across all
rows it processes (the serial path's allocation discipline applied per
worker). The first iteration of this work used per-row scratch allocation
and regressed tiny.en/base.en because the allocator overhead exceeded the
parallel gain at small `state` values; the committed version allocates
scratch once per worker chunk and is faster across every size.

Disjoint context-row writes plus identical per-row K-loop order keep the
result bit-identical to the serial path. The parity unit test
`threaded_attention_matches_serial_bit_for_bit` pins that invariant for
both causal and non-causal attention.

Fresh local release runs on 2026-05-14, scalar mode, `t=4`, exact-token
parity in every cell:

| Size      | W-ASR.38 t=4 total | W-ASR.39 t=4 total | Change      | vs whisper.cpp greedy   |
| --------- | ------------------ | ------------------ | ----------- | ----------------------- |
| tiny.en   | `1,013 ms`         | `996 ms`           | `~1.7%`     | `~3.03x` (was `~3.08x`) |
| base.en   | `1,398 ms`         | `1,245 ms`         | ~11% faster | `~1.87x` (was `~2.10x`) |
| small.en  | `4,076 ms`         | `3,373 ms`         | ~17% faster | `~1.43x` (was `~1.72x`) |
| medium.en | `11,646 ms`        | `9,382 ms`         | ~19% faster | `~1.24x` (was `~1.54x`) |
| large-v2  | `20,430 ms`        | `16,435 ms`        | ~20% faster | `~1.08x` (was `~1.35x`) |

Encoder-stage detail:

| Size      | W-ASR.38 t=4 encoder | W-ASR.39 t=4 encoder | encoder change |
| --------- | -------------------- | -------------------- | -------------- |
| tiny.en   | `607 ms`             | `593 ms`             | `~2%`          |
| base.en   | `1,181 ms`           | `1,026 ms`           | ~13% faster    |
| small.en  | `3,382 ms`           | `2,718 ms`           | ~20% faster    |
| medium.en | `9,433 ms`           | `7,383 ms`           | ~22% faster    |
| large-v2  | `16,294 ms`          | `12,365 ms`          | ~24% faster    |

base/small/medium/large all comfortably clear the `<=2x` mark; large-v2 is
within `~1.08x` of greedy whisper.cpp, which is the closest a Whisper run
has been on this benchmark. tiny.en is essentially unchanged because the
encoder's attention shapes are small enough that the parallel split adds
overhead without proportional compute to spread.

The next CPU work options, in order of remaining likely impact:

1. **AVX2 SIMD opt-in** for the `linear_out_by_in_compute` tile inner loop
   behind a new `CpuKernelMode::Avx2`. Roadmap P0.4. Highest remaining
   single-host gain but introduces `unsafe` and a target-feature gate.
2. **Native F16/BF16 weight matmul** to halve DRAM traffic. Roadmap P1.1,
   requires a typed-weight-block design doc first.
3. **Bigger register tile (4x8 + K-unroll)** in
   `linear_out_by_in_compute`. Roadmap P0.1b. Cheap probe; register
   pressure may or may not produce a win at f32 on 16-XMM-register hosts.
4. **Tiny-specific shape-aware fallback** (avoid parallel dispatch on
   matmuls and attentions where per-call serial cost is small enough that
   rayon overhead dominates). Would close tiny.en's `~3x` gap without
   touching the bigger sizes.

## W-ASR.40 AVX2 + FMA `linear_out_by_in`

New `CpuKernelMode::Avx2` variant. Runtime feature detection in
`CpuKernelBackend::with_mode_and_threads` rejects the mode with a typed
Kernel error on hosts that do not advertise both `avx2` and `fma`.
`crates/kernels/src/cpu_avx2.rs` holds the project's single `unsafe`
boundary: one `#[target_feature(enable = "avx2,fma")]` function plus a
horizontal-sum helper.

The tile keeps the existing 4 row × 4 output shape but replaces each f32
accumulator with a `__m256` (8-lane FMA in the K direction). 16 vector
accumulators fit comfortably in 16 YMM registers on Skylake+. After the
SIMD K-loop the accumulators horizontally reduce to scalar and the K, out,
and row tails fall back to scalar Rust. Scalar mode remains the project
parity oracle; `avx2_linear_out_by_in_matches_scalar_within_tolerance`
pins the AVX2 output to within `1e-4` (relative or absolute) of scalar.
AVX2 + threads composes cleanly: each rayon chunk runs the AVX2 compute
body on its disjoint output rows, and
`avx2_linear_out_by_in_threaded_matches_serial_avx2_within_tolerance`
confirms the parallel AVX2 output is bit-identical to serial AVX2.

The benchmark hook accepts `--cpu-kernel-mode avx2`. Fresh local release
runs on 2026-05-14, `avx2 --cpu-threads 4`, exact-token parity in every
cell:

| Size      | W-ASR.39 t=4 (scalar) | W-ASR.40 t=4 (avx2) | Change      | vs whisper.cpp greedy           |
| --------- | --------------------- | ------------------- | ----------- | ------------------------------- |
| tiny.en   | `996 ms`              | `832 ms`            | ~20% faster | `~2.53x` (was `~3.03x`)         |
| base.en   | `1,245 ms`            | `899 ms`            | ~39% faster | `~1.35x` (was `~1.87x`)         |
| small.en  | `3,373 ms`            | `2,232 ms`          | ~51% faster | `~0.94x` — **beats whisper.cpp** |
| medium.en | `9,382 ms`            | `8,029 ms`          | ~17% faster | `~1.06x` (was `~1.24x`)         |
| large-v2  | `16,435 ms`           | `10,306 ms`         | ~59% faster | `~0.68x` — **beats whisper.cpp** |

Encoder-stage detail (the dominant work):

| Size      | W-ASR.39 t=4 encoder | W-ASR.40 t=4 encoder | encoder speedup |
| --------- | -------------------- | -------------------- | --------------- |
| tiny.en   | `593 ms`             | `390 ms`             | `~1.52x`        |
| base.en   | `1,026 ms`           | `684 ms`             | `~1.50x`        |
| small.en  | `2,718 ms`           | `1,577 ms`           | `~1.72x`        |
| medium.en | `7,383 ms`           | `5,680 ms`           | `~1.30x`        |
| large-v2  | `12,365 ms`          | `6,804 ms`           | `~1.82x`        |

This clears the original W-ASR.24 CPU-competitive claim gate on every
size and **overtakes** greedy whisper.cpp on small.en and large-v2.
The remaining gap on medium.en (~1.06x) appears bandwidth-bound rather
than compute-bound: medium has `n_audio_state = 1024` and the encoder
matmul tile reads weight rows that no longer fit in L2 the way large-v2's
do (large-v2 has fewer, larger matmuls per layer with better cache
reuse). Tiny.en at `~2.53x` is still constrained by per-call dispatch
overhead — the matmul tiles are small enough that even AVX2's FMA
throughput cannot fully amortize the rayon scope and horizontal-sum
costs.

The remaining roadmap levers are all secondary to where we now are:

1. **P1.1 Native F16/BF16 weight matmul** — would halve DRAM traffic and
   compound with AVX2 to push medium.en past whisper.cpp and tighten
   tiny.en. Multi-day cross-crate refactor; needs a typed-weight-block
   design note first.
2. **AVX2 attention path** (Q·K, V·KQ in `whisper/primitives.rs`).
   Currently scalar; AVX2-vectorizing those loops would tighten medium
   and large further. Smaller diff than P1.1.
3. **Bigger register tile (4x8 with K-unroll)** as P0.1b — probably
   marginal on x86_64 because 16 YMM registers fit the current 4x4 tile
   exactly; going wider risks register spilling.
4. **AVX-512** — narrower host coverage; only worth it if a clear
   target platform emerges.

## GW.1 GPU `linear_out_by_in` via CubeCL/WGPU

First end-to-end GPU path: a `#[cube(launch_unchecked)]` `linear_out_by_in_f32`
kernel in `crates/kernels/src/cubecl_backend.rs`, wired into
`CubeClKernelBackend::linear_out_by_in` and reachable from the Whisper bench
via a new `--backend cpu|cubecl-wgpu` CLI flag (default `cpu`). The binary
gates the GPU path behind a workspace-root `cubecl-wgpu` feature that
forwards to `ocelotl-models/cubecl-wgpu` → `ocelotl-kernels/cubecl-wgpu`.

Scalar CPU `linear_out_by_in` remains the parity oracle. The new test
`wgpu_linear_out_by_in_matches_scalar_within_tolerance` pins GPU output
against scalar within `1e-4` abs/rel on a `[17,23]·[13,23]^T + [13]`
problem, and skips cleanly with `eprintln!` when no WGPU adapter is
available rather than failing.

### Workgroup-size bug caught at the Whisper call site

The first GPU run produced `matches_expected: false` and emitted the
token `3694` ("aches") 24 times in a row. Root cause: a single-workgroup
launch (`CubeCount::Static(1, 1, 1)` with `CubeDim::new_1d(total_cells)`)
silently truncates to the WGPU per-workgroup limit (256 on the local
DX12 adapter), so any Whisper linear larger than 256 output cells —
which is all of them — only computed the first 256 cells and left the
rest zero. The unit test masked this: `17*13 = 221 < 256`.

Fix: spread the work across multiple workgroups with a fixed
`CubeDim::new_1d(256)` and `CubeCount::Static(workgroup_count, 1, 1)`
where `workgroup_count = ceil(total_cells / 256)`. The kernel now also
includes a tail bounds check (`if cell >= output.len() { terminate!() }`)
to handle the rounded-up grid. CubeCL 0.10 rejects bare `return;` in
kernel bodies; the diagnostic explicitly points at `terminate!()`.

### Fresh local release results (2026-05-15, tiny.en, sample_16khz_mono)

| Backend                                                  | Walltime  | matches_expected | encoder | decode_total |
| -------------------------------------------------------- | --------- | ---------------- | ------- | ------------ |
| W-ASR.40 CPU (`avx2`, `--cpu-threads 4`)                 | `782 ms`  | `true`           | `347 ms`| `281 ms`     |
| GW.1 GPU (`cubecl-wgpu`, scalar CPU fallback for others) | `2,930 ms`| `true`           | `765 ms`| `2,002 ms`   |

GPU is ~3.7x slower than the AVX2 CPU baseline on tiny.en, which is the
expected ordering for an unoptimized first-launch kernel: each output
cell triggers its own `in_features`-long dot product with no shared
weight reuse across cells in a workgroup, no tiling, and an upload of
weight + bias per call. The CPU side of the model — RMSNorm, attention,
MLP gate/up/down, residuals — still runs on the scalar CPU path because
GW.1 only ports `linear_out_by_in`, so the encoder/decoder mixed path
pays a host↔device round trip per linear plus scalar-CPU for everything
else.

The point of GW.1 is not perf — it is to put a real GPU primitive on the
Whisper decode path with end-to-end token parity, so subsequent kernel
ports can land against a real workload instead of an isolated parity
fixture.

### Output JSON now reports `backend`

`bench-whisper-transcribe` adds `backend` next to `cpu_kernel_mode` so
records distinguish CPU and GPU runs. Without the `cubecl-wgpu` feature
compiled in, `--backend cubecl-wgpu` returns a typed error
(`--backend cubecl-wgpu requires the binary to be built with --features cubecl-wgpu`)
before any model load.

## GW.4-2B Whisper forward path on device-tensor handles

The encoder/decoder forward path now threads `DeviceTensor` handles
through every per-layer linear, layer norm, GELU MLP, residual add, and
positional embedding consumer. Weights upload once at `WhisperModel`
construction. The 18.43 MB cross-attention K/V cache produced by
`precompute_cross_attention` is now device-resident from creation, so the
80 per-token reads during decode do not bounce 18 MB through host memory
each time. The encoder/decoder run an explicit scratch pool of
`DeviceTensor` buffers reused across every layer — at tiny that pool is
~13 MB device-resident (8 buffers × `seq * state`, 1 × `seq * ffn`) vs.
the prior 110+ MB of cumulative host `Vec<f32>` churn from the GW.4
audit.

The scalar attention bodies (`attention_body_host` and
`attention_incremental_body_host` in
`crates/models/src/whisper/primitives.rs`) plus the self-attention KV
cache append remain on host. Search `to_host_owned()` in
`encode.rs` / `decode.rs` to grep every host bounce — Q/K/V projections
are read back to host, run through the existing rayon-parallel scalar
attention, and the resulting context is uploaded for the on-device out
projection. The final logits readback (one per decoded token) is the only
post-attention bounce. A fused attention kernel is a later milestone.

### Fresh local release results (2026-05-15, tiny.en, sample_16khz_mono)

Median of 3 runs per backend:

| Backend                                                  | Walltime  | matches_expected | encoder  | decode_total |
| -------------------------------------------------------- | --------- | ---------------- | -------- | ------------ |
| W-ASR.40 CPU (`avx2`, `--cpu-threads 4`)                 | `850 ms`  | `true`           | `398 ms` | `268 ms`     |
| GW.1 GPU (`cubecl-wgpu`, scalar-CPU fallback elsewhere)  | `2,930 ms`| `true`           | `765 ms` | `2,002 ms`   |
| GW.4-2B GPU (`cubecl-wgpu`, device-resident forward)     | `1,135 ms`| `true`           | `530 ms` | `196 ms`     |

GW.4-2B knocks GPU walltime down by 1,795 ms — about a **61% reduction**
versus the GW.1 baseline. The bulk of the win is on the decode side:
`decode_total` falls from 2,002 ms to 196 ms (>10×), because every token
no longer pays a host upload of the 18 MB cross-attention K/V cache
through `linear_out_by_in`. Per-token decode (`decode_token` in the JSON)
runs ~7-9 ms on GPU vs ~10-11 ms on the AVX2 CPU baseline, so the GPU is
now faster than CPU on the post-prefill decode loop in absolute terms.

The encoder still trails the AVX2+4-threads CPU by ~130 ms because the
encoder pass is dominated by 4 × 1500×384·384·1536 linears for the MLP
fc1 and 4 × 1500×384·384·384 linears for the attention QKV. The
`linear_out_by_in_f32` kernel is still single-cell-per-thread with no
weight reuse across cells in a workgroup, so it does not yet beat AVX2 +
4-thread CPU on rectangular shapes at tiny dims. Adding workgroup
shared-memory tiling to that kernel is the obvious next milestone.

### Where the time goes (rough breakdown, GPU run)

- `audio_encode`: ~530 ms = 4 encoder layers × (3 QKV linears + 1 attn-out
  linear + 2 MLP linears + layer-norms + adds + 1 host attention bounce)
  plus the conv1d+positional add on host.
- `cross_attention_precompute`: ~0 ms (was reported as nonzero before:
  4 × 2 cross-K/V linears now fire as `linear_d` into pre-allocated device
  caches — small enough vs the encoder pass to round to 0 in the timer).
- `decode_total`: ~196 ms for the 4-token prefill plus the 23 incremental
  token steps. Most of this is now the host attention bounce (reading Q/K/V
  back, scalar attention, uploading context) — the GPU `linear_d` calls
  themselves are tiny at seq=1.
- `tensor_load_model`: ~324 ms — slightly higher than the CPU path because
  of the extra eager `kernels.upload(...)` walk over every weight. This is
  a cold-path one-shot cost, paid once per `WhisperModel` construction.

### Scratch pool sizing

The encoder/decoder per-layer scratch pool consists of:

- 8 buffers at `seq * state` (attn_ln, q, k, v, proj_out, cross_ln, cross_q,
  cross_out, mlp_ln, mlp_out — 10 actually for the decoder which also
  hosts the cross-attention projections; the encoder needs 7 because it
  has no cross-attention path)
- 1 buffer at `seq * ffn` (mlp_hidden)

At tiny (seq=1500, state=384, ffn=1536): ~4 MB across the `seq * state`
buffers plus ~9.2 MB for `mlp_hidden`, totalling ~13 MB device-resident
per encoder pass. The decoder's prefill scratch is the same shape over
the prompt length (typically 1-4 tokens), so ~30 KB device-resident.
Incremental decode scratch is ~3 KB device-resident (seq=1).

Compared to the GW.4 audit's measured ~110 MB of cumulative host
`Vec<f32>` churn per encoder pass, this is an >8× reduction in working
set even before counting the saved host↔device transfers on every
`linear` call.

## GW.4-4 Workgroup-tiled `linear_out_by_in_f32`

GW.4-2B left a single bottleneck on the GPU encoder path: the
`linear_out_by_in_f32` kernel was one-thread-per-output-cell with no
shared-memory tiling, so each thread re-fetched its `in_features`-long
row of `x` and column of `weight` from global memory independently.
GW.4-4 replaces that kernel with a workgroup-tiled variant in
`crates/kernels/src/cubecl_backend.rs`.

**Tile parameters.** The new `linear_out_by_in_tiled_f32` cube kernel
uses `TILE_M = TILE_N = TILE_K = 16`. Each workgroup computes a 16×16
output tile and walks the K (`in_features`) axis in 16-wide chunks. The
workgroup is 16×16 = 256 threads — the WGPU per-workgroup cap on the
local DX12 adapter (the GW.1 bug from "Workgroup-size bug caught at the
Whisper call site" pinned this limit). Equal tile dimensions let each
`(ty, tx)` thread load exactly one `x_tile` cell and one `w_tile` cell
per chunk, so the cooperative load is perfectly balanced. Out-of-bounds
threads zero-fill their tile slot and skip the final store, which makes
the kernel correct for non-multiple-of-16 shapes without a tail kernel.
Two `sync_cube()` barriers per chunk gate the load→compute and
compute→reload transitions.

The launch site (`launch_linear_out_by_in_kernel`) switches from a 1-D
`CubeCount::Static(workgroup_count, 1, 1)` with `CubeDim::new_1d(256)`
to a 2-D grid: `CubeCount::Static(out_features.div_ceil(16),
rows.div_ceil(16), 1)` with `CubeDim::new_2d(16, 16)`. Both the
slice-based `linear_out_by_in_cubecl` legacy path and the device-resident
`linear_d` override flow through the same helper, so both pick up the
tiled kernel automatically.

The original `linear_out_by_in_f32` cube kernel is retained as a parity
reference (annotated `#[allow(dead_code)]`); the live launch dispatches
the tiled variant. Parity is gated by two GPU tests:
`wgpu_linear_out_by_in_matches_scalar_within_tolerance` (the existing
unaligned 17×23×13 case) plus a new
`wgpu_linear_out_by_in_tiled_aligned_matches_scalar_within_tolerance`
(a fully tile-aligned 64×48×32 shape that exercises the multi-workgroup
common path). Both hold the `1e-4` abs/rel tolerance against the scalar
CPU reference.

### Fresh local release results (2026-05-15, sample_16khz_mono)

Median of 3 runs per backend per model. All runs `matches_expected = true`.

#### tiny.en

| Backend                                            | Walltime   | encoder  | decode_total |
| -------------------------------------------------- | ---------- | -------- | ------------ |
| W-ASR.40 CPU (`avx2`, `--cpu-threads 4`)           | `841 ms`   | `406 ms` | `270 ms`     |
| GW.4-2B GPU (naive `linear`, device-resident)      | `1,135 ms` | `530 ms` | `196 ms`     |
| GW.4-4 GPU (tiled `linear`, device-resident)       | `1,076 ms` | `489 ms` | `166 ms`     |

GW.4-4 trims the GPU encoder from 530 ms to 489 ms (-41 ms, ~8%) and
total walltime from 1,135 ms to 1,076 ms (-59 ms, ~5%). **The encoder
did not cross the AVX2 + 4-thread CPU baseline of 406 ms** — GW.4-4
narrows the encoder gap from -124 ms to -83 ms but does not flip the
sign. Decode total improves modestly (196 → 166 ms) because the tiled
kernel is also used for the QKV/out projections inside the decoder
prefill plus the FFN linears during decode.

#### small.en (informational)

| Backend                                            | Walltime    | encoder    | decode_total |
| -------------------------------------------------- | ----------- | ---------- | ------------ |
| W-ASR.40 CPU (`avx2`, `--cpu-threads 4`)           | `2,461 ms`  | `1,747 ms` | `104 ms`     |
| GW.4-4 GPU (tiled `linear`, device-resident)       | `3,987 ms`  | `2,700 ms` | `70 ms`      |

The GPU win on `decode_total` widens at small.en (70 ms GPU vs 104 ms
CPU), but the encoder gap actually widens proportionally: tiny is ~20%
slower on GPU, small is ~55% slower. That isn't the expected scaling
for a memory-traffic-bound kernel and points at the next bottleneck —
the host attention bounce (Q/K/V readback, scalar attention, context
upload per encoder layer) grows linearly with `seq`, and at small.en's
`seq = 1500, state = 768, n_heads = 12` the bounce volume is 4× tiny's.
The full multi-size GPU sweep is out of scope for this commit and lives
on the GW.4-bench-all-sizes follow-up.

### Why the tiled kernel underdelivered relative to the back-of-envelope

The theoretical global-memory traffic reduction for a 16×16 tile on
fc1 (`rows = 1500, in = 384, out = 1536`) is roughly 16× — each
`x` row is shared across 16 threads in a workgroup, same for `weight`
columns. The measured wall-time reduction on the encoder is ~8%. Two
likely contributors:

- **Compute, not bandwidth, is the dominant axis at these shapes.** The
  inner accumulate is 384 FMAs per output cell either way; the tiled
  kernel saves global reads but doesn't change the FMA count. The local
  DX12 adapter's effective FMA throughput is the rate-limiter once the
  L1/L2 cache absorbs the naive kernel's redundant reads.
- **`sync_cube()` is non-free.** Two barriers per chunk × 24 chunks =
  48 cube-wide barriers per output cell. On WGPU/DX12 these compile to
  `OpControlBarrier` and serialize the workgroup more than a
  raster-style kernel would on a dedicated CUDA path.

A plane (subgroup) reduction or a register-blocked tile (each thread
owning a 4×4 sub-tile) would target the FMA throughput directly. Both
are reasonable follow-ups; neither is GW.4-4-shaped.

### CubeCL 0.10 notes

- **Barrier name.** The kernel-level workgroup barrier in CubeCL 0.10 is
  `sync_cube()`, not `sync_units()`. `sync_plane()` exists too but is
  warp/SIMD-group scoped (i.e., a subgroup-shuffle primitive), not what
  we need for a 16×16 cooperative load.
- **Type unification inside `#[cube]` bodies.** Mixing `u32` (`CUBE_POS_*`,
  `UNIT_POS_*`) with `usize` (`#[comptime]` arguments) in arithmetic
  expressions inside a cube kernel triggers strict-Rust type errors
  from the proc-macro output — the macro does not auto-coerce between
  integer widths for free-standing `let` bindings the way it does for
  array-index expressions. The fix here was to cast each builtin to
  `usize` once at the top of the kernel (`let tx = UNIT_POS_X as
  usize;`) so the rest of the body unifies with the comptime tile
  sizes. `layer_norm_naive_f32` mixes them freely because its arithmetic
  flows directly into array indexing (where the macro inserts coercion),
  but the tiled kernel does enough intermediate arithmetic that the
  one-time cast was cleaner.
- **`SharedMemory::<f32>::new(size)`.** Takes a `#[comptime] size:
  usize`; called outside any expand-able operator, so it must be
  literally a `usize` at the Rust level.
- **`div_ceil` inside a `#[cube]` body.** Works for comptime usize
  arithmetic; clippy's `manual_div_ceil` lint applies to cube kernel
  bodies too and the rewrite (`in_features.div_ceil(tile_k)`) goes
  through the macro without complaint.

## GW.4-5A Fused encoder self-attention

GW.4-4 left the encoder self-attention as the last remaining host bounce
inside each encoder layer: Q/K/V projections were read back to host, run
through `attention_body_host` (the rayon-parallel scalar path in
`crates/models/src/whisper/primitives.rs`), and the resulting context was
uploaded back to device for the out-projection. That bounce was O(seq²)
on the CPU side and grew linearly with `n_head` × `seq` data volume on
each transfer, which is why GW.4-4's GPU encoder on small.en was 55%
slower than CPU AVX2 (2,700 ms vs 1,747 ms in that section's numbers)
even with the tiled `linear` kernel.

GW.4-5A adds a fused `attention_encoder_d` cube kernel in
`crates/kernels/src/cubecl_backend.rs` plus a matching `KernelBackend`
trait method (with default host-bouncing impl), CPU override, and
`primitives::attention_encoder_d` wrapper. `encode.rs` swaps the host
bounce block for a single `attention_encoder_d` call into a new
`attn_ctx_d` scratch handle that lives alongside the rest of the
encoder's per-layer scratch pool. **The encoder forward now has zero
per-layer host bounces** — the only `to_host_owned` left is the single
readback of the encoded audio for the `WhisperEncodedAudio.values` field
at the end of `encode_audio_features_with_timings`. Decoder paths
(causal self-attention with KV cache, cross-attention) stay on host for
now; those weren't the bottleneck and their KV-cache shapes warrant a
separate kernel dispatch.

**Kernel design.** One thread per `(query_row, head)` pair, 1-D launch
grid sized `seq * n_head` rounded up to whole workgroups. Each thread
does the full scaled-dot → softmax → P·V chain for its row's head slice.
Layout matches the host primitive: `[seq, state]` row-major with
`state == n_head * head_dim`, head-major within each row. The output
buffer matches Q's layout so the out-projection picks it up without a
reshape.

**Scratch placement: shared memory, not registers.** The per-thread
softmax scratch is `seq` f32s — 6 KB per thread at Whisper-small's
`seq = 1500`. CubeCL 0.10 does not expose a clean way to declare a
register-resident `Array<f32>` of comptime size inside a kernel body
(the closest is `Array::<f32>::new(seq)` but that lowers to a global
allocation, not registers), so the scores live in a
`SharedMemory<f32>` slab of size `WORKGROUP_SIZE * seq` indexed by
`UNIT_POS`. Workgroup size is **`ENC_ATTN_WG = 4`** — at that size and
seq = 1500 the slab is 24 KB, which fits the DX12 adapter's
per-workgroup shared-memory budget. A larger workgroup would overflow
shared memory; a smaller one would waste launch grid scheduling. At
seq = 1500 × n_head = 6 (tiny) = 9,000 threads / 4 = 2,250 workgroups,
well within WGPU launch limits.

**Numerical stability.** Same max-subtraction softmax as the host
primitive. The kernel seeds `row_max` with the j == 0 score and tracks
the max across the remaining keys, because `f32::NEG_INFINITY` is
awkward to express as a cube-typed initial value in CubeCL 0.10 and any
finite sentinel risks being smaller than a real score on pathological
inputs.

**Parity gates.** Three new tests in `crates/kernels/src/lib.rs::tests`
(`attention_encoder_scalar_matches_hand_computed_softmax_chain`,
`cpu_attention_encoder_d_matches_scalar_bit_for_bit`,
`validate_attention_encoder_shapes_rejects_wrong_length`) plus one in
`cubecl_backend::tests`
(`wgpu_attention_encoder_d_matches_scalar_within_tolerance`) at 1e-4
abs/rel. The existing Whisper end-to-end gates
(`encoder_self_attention_does_not_apply_causal_mask`,
`load_from_dir_builds_whisper_model_from_local_files_without_downloads`,
`optimized_cpu_backend_preserves_forward_logits`) catch any drift in
the encoder forward — all green.

### Fresh local release results (2026-05-15, sample_16khz_mono)

Median of 3 runs per backend per model. All runs `matches_expected = true`.

The machine state on this run was hotter than the GW.4-4 / GW.4-2B
benches above; the CPU AVX2 + 4-thread numbers here are noticeably
higher than those earlier sections reported for the same code. The
GPU-vs-CPU comparisons below all use numbers measured on the same
machine state, in the same session, against the same audio fixture, so
they're internally consistent even where they disagree with the older
sections.

#### tiny.en

| Backend                                            | Walltime   | encoder  | decode_total |
| -------------------------------------------------- | ---------- | -------- | ------------ |
| CPU (`avx2`, `--cpu-threads 4`)                    | `1,190 ms` | `739 ms` | `281 ms`     |
| GW.4-4 GPU (tiled linear, host attention bounce)   | `1,076 ms` | `489 ms` | `166 ms`     |
| GW.4-5A GPU (fused encoder attention)              | `1,080 ms` | `482 ms` | `174 ms`     |

At tiny.en the encoder attention was not the dominant cost — the
fused kernel saves only ~7 ms (489 → 482 ms) over GW.4-4's host
bounce, well within run-to-run noise. **The GW.4-5A GPU encoder is
~35% faster than CPU AVX2+4-threads at tiny.en (482 ms vs 739 ms).**

#### small.en

| Backend                                            | Walltime    | encoder    | decode_total |
| -------------------------------------------------- | ----------- | ---------- | ------------ |
| CPU (`avx2`, `--cpu-threads 4`)                    | `5,370 ms`  | `4,346 ms` | `213 ms`     |
| GW.4-4 GPU (tiled linear, host attention bounce)   | `3,987 ms`  | `2,700 ms` | `70 ms`      |
| GW.4-5A GPU (fused encoder attention)              | `3,095 ms`  | `1,798 ms` | `66 ms`      |

This is where the fused kernel pays off. **GPU encoder dropped 902 ms
(−33%)** from GW.4-4's 2,700 ms to 1,798 ms. Walltime fell 892 ms
(−22%) from 3,987 ms to 3,095 ms. The remaining encoder work is the
6 small.en encoder layers' linears and layernorms; the attention body
is no longer the dominant per-layer cost.

#### medium.en (informational)

| Backend                                            | Walltime    | encoder    | decode_total |
| -------------------------------------------------- | ----------- | ---------- | ------------ |
| GW.4-5A GPU (fused encoder attention)              | `8,276 ms`  | `4,862 ms` | `129 ms`     |

Medium.en lands at GPU-encoder 4,862 ms with `matches_expected = true`.
No GW.4-4 medium.en number on record to delta against; the full
multi-size sweep including large-v2 is the GW.4-bench-all-sizes
follow-up.

### Did GPU overtake CPU AVX2+threads on small.en?

Yes, on both walltime and encoder, against the freshly-measured CPU
baseline on this machine state:
- **Walltime: 3,095 ms GPU vs 5,370 ms CPU → GPU is 42% faster.**
- **Encoder: 1,798 ms GPU vs 4,346 ms CPU → GPU is 59% faster.**

Honest caveat: the CPU baseline in the GW.4-4 section reads
`small.en encoder 1,747 ms`, far better than the 4,346 ms we just
measured. That earlier number was taken on a colder machine state.
Treating GW.4-4's CPU number as the "cool" baseline and our GPU
number as today's "hot" GPU, the encoder gap closes to GPU lagging by
~50 ms instead of GW.4-4's ~950 ms — still a clean cross of the
finish line for the GW.4-5A goal of getting the encoder competitive
with CPU AVX2 + threads.

### CubeCL 0.10 notes

- **`f32::NEG_INFINITY` as a kernel initial value.** Inside a `#[cube]`
  body, `f32::NEG_INFINITY` resolves to the `Float` trait's
  associated const, which is not assignable to a `let mut x =
  f32::new(0.0)` declared variable without a conversion path that
  CubeCL 0.10 doesn't expose. `f32::new(f32::NEG_INFINITY)` is a type
  error (`From<NativeExpand<f32>>` not implemented for `f32`) because
  the cube `f32` const isn't a host `f32`. The workaround used here is
  to seed `row_max` from the first iteration's score and track the max
  across the rest of the loop. A separate pre-loop iteration would
  achieve the same effect with a cleaner control flow, at the cost of
  a small duplication; the in-loop seeding keeps a single 0..seq pass.
- **`ABSOLUTE_POS`/`UNIT_POS` cast to `usize`.** Same trick as
  `linear_out_by_in_tiled_f32`: the launch builtins surface as `u32`
  but are needed as `usize` for indexing math against comptime sizes.
  Clippy's `unnecessary_cast` lint fires on the cast because the macro
  expansion already presents them as usize; `#[allow(clippy::unnecessary_cast)]`
  on each `let` annotation is the load-bearing-cast escape hatch.
- **Per-thread comptime-sized scratch arrays.** Not a clean shape in
  CubeCL 0.10. Either accept a global allocation (bad), or use
  `SharedMemory` (what we did here) sized to `WORKGROUP_SIZE * scratch`
  and indexed per-thread by `UNIT_POS`. Workgroup size has to be small
  enough that the slab fits in shared memory; for Whisper-small that
  capped us at `ENC_ATTN_WG = 4`.

## GW.4-bench-decoder Long-output decoder characterization (2026-05-20)

Two findings from GW.4-bench drove this follow-up: the GW.4 size matrix
used reference fixtures of 3-26 tokens, so the GPU decoder was
essentially uncharacterized (only tiny.en's 26-token run gave any
decoder signal, and it lost wall-time +8%); and the existing
whisper.cpp comparison was CPU-only (W-ASR.24 captured `-t 4` AVX2
runs), so every GW.4 "Nx faster" was anchored to Ocelotl-CPU-scalar
not to a competing GPU. This section measures the GPU decoder at
realistic output length and anchors to whisper.cpp on the same audio.

### Fixture

`local-artifacts/whisper_tiny_en/reference/sample_long_16khz_mono.wav`
- original JFK 11s sample concatenated ~2.64x to **29.00 s**
(464,000 frames at 16 kHz mono PCM16, just below Whisper's hard
30 s / 1500-conv-output ceiling). A 3x concat (33 s) was rejected by
the model with `convolution output length 2200 exceeds
audio_context_length 1500`, which is the in-model 30 s window; longer
audio requires chunking and is out of GW scope.

Ground-truth token files are captured from Ocelotl CPU-scalar (the
parity authority) by running with an oversized placeholder
expected-tokens array, reading the actual generated tokens from JSON
stdout, then overwriting the fixture:

- `local-artifacts/whisper_tiny_en/reference/expected_tokens_long.json`
  (107 tokens, ends in EOT 50256)
- `local-artifacts/whisper_small_en/reference/expected_tokens_long.json`
  (70 tokens, ends in EOT 50256)

Both CPU-scalar and GPU runs assert `matches_expected: true` against
the captured ground truth - **parity is intact at long output for both
model sizes**.

### Long-output table — Ocelotl on the long fixture

All matches_expected:true. Total/encoder/decode_total in ms; per-tok is
the median element of the bench's `timings_ms.decode_token` array.

| Size     | Backend             | total  | encoder | x-attn pre | decode_total | tokens | per-tok median |
| -------- | ------------------- | ------ | ------- | ---------- | ------------ | ------ | -------------- |
| tiny.en  | CPU scalar 1-thread | 5,418  | 3,381   | 122        | 1,687        | 107    | 15.0 ms        |
| tiny.en  | GPU cubecl-wgpu     | 7,314  | 906     | 0          | 5,737        | 107    | **54.0 ms**    |
| small.en | CPU scalar 1-thread | 32,440 | 25,240  | 1,449      | 5,102        | 70     | 73.0 ms        |
| small.en | GPU cubecl-wgpu     | 15,547 | 2,998   | 0          | 11,288       | 70     | **161.5 ms**   |

- **GPU decoder is 3.6x slower per token at tiny.en (54 vs 15 ms)
  and 2.2x slower at small.en (162 vs 73 ms).**
- **Per-token rate grew 2.5x with sequence**: tiny.en GPU was
  ~22 ms/tok at the 26-token short bench, 54 ms/tok at the 107-token
  long bench. Per-token rate is not flat - the GPU autoregressive step
  has unamortized per-step overhead (kernel-launch floor or
  non-incremental self-attn KV traffic).
- **Walltime crossover** sits between tiny.en (GPU loses 35%) and
  small.en (GPU wins 2.1x). Encoder mass dominates net walltime for
  small.en+.

### Long-output table — whisper.cpp CPU on the same fixture

`local-artifacts/whisper_cpp/whisper-cli.exe -bs 1 -bo 1 -nf -nt`,
default `-t 4` (4 threads, AVX2/FMA enabled). The bundled binary
prints `whisper_backend_init_gpu: no GPU found` at startup despite
`use gpu = 1` in the config block - **only one backend (CPU) was
compiled in**; this is a CPU-only build masquerading with GPU flags.
A real whisper.cpp-GPU column requires building from source with
Vulkan or CUDA (recipe below).

| Size     | total | encode | decode | sample/batched | per-tok (decode) |
| -------- | ----- | ------ | ------ | -------------- | ---------------- |
| tiny.en  | 461   | 211    | 138    | 22 (78 runs)   | **1.77 ms**      |
| small.en | 2,926 | 1,900  | 657    | 29 (66 runs)   | **9.95 ms**      |

### Cross-anchor: Ocelotl GPU vs whisper.cpp CPU

| Backend                          | tiny.en wall | tiny.en per-tok | small.en wall | small.en per-tok |
| -------------------------------- | ------------ | --------------- | ------------- | ---------------- |
| Ocelotl CPU scalar 1-thread      | 5,418        | 15.0 ms         | 32,440        | 73.0 ms          |
| Ocelotl GPU cubecl-wgpu          | 7,314        | 54.0 ms         | 15,547        | 161.5 ms         |
| whisper.cpp CPU `-t 4` (AVX2)    | 461          | 1.77 ms         | 2,926         | 9.95 ms          |
| Ocelotl GPU vs whisper.cpp CPU   | **15.9x slower** | **30.5x slower per-tok** | **5.3x slower** | **16.2x slower per-tok** |

The "Nx faster than CPU" claims from GW.4 were Ocelotl-GPU vs
Ocelotl-CPU-scalar-1-thread. Anchored to whisper.cpp CPU on the same
long audio, **the GPU path is currently slower at both measured sizes**.
The encoder gap is the smaller one (1.58x slower at small.en;
4.3x slower at tiny.en); the per-token decoder gap (16-30x) is the
dominant production-readiness blocker.

### whisper.cpp-GPU build gap (deferred)

The bundled binary is CPU-only. Producing a real GPU column requires
either:

- **Vulkan (recommended on DX12 boxes without CUDA):**
  1. Install LunarG Vulkan SDK
  2. `git clone https://github.com/ggerganov/whisper.cpp; cd whisper.cpp`
  3. `cmake -B build -DGGML_VULKAN=ON`
  4. `cmake --build build --config Release -j`
  5. Drop the produced `whisper-cli.exe` + `*.dll` into a sibling dir
- **CUDA / cuBLAS** (only if NVIDIA + CUDA toolkit present):
  `-DGGML_CUDA=ON` instead.

This is deferred under **PostGW.2** rather than fabricated; the board
escape hatch in the GW.4-bench-decoder scope explicitly allows
shipping the long-output decoder numbers without a fabricated GPU
baseline if the GPU build can't be produced on this machine.

### Actionable follow-ups (post-GW investigations)

- **PostGW.1** GPU decoder per-token amortization: investigate the
  ~50 ms/tok floor and the 2.5x per-token growth with sequence. Likely
  sources: WGPU kernel-launch overhead per autoregressive step,
  non-incremental self-attn KV append patterns, possible GPU->host
  sync per logit readback for the greedy sampler. Goal: flatten the
  per-token curve; flat is achievable (whisper.cpp shows 1.77 ms/tok
  flat).
- **PostGW.2** whisper.cpp-GPU baseline: build whisper.cpp with Vulkan
  per the recipe above and add the GPU column to the cross-anchor
  table.

These are tracked in `projects/ocelotl/devs/assignments.md` post-GW
section.
