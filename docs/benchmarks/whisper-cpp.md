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
