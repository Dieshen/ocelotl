# Post-M3 Whisper ASR Tasks

Working shorthand: "M3.3 Whisper". This is a post-M3 expansion track and does
not modify the closed M3.3 RMSNorm task.

## Kickoff Status

Kicked off 2026-05-08 as a post-M3 expansion track. W-ASR.1 is the senior-owned
boundary/docs task. W-ASR.2 and W-ASR.3 are the first implementation slices once
junior agents are explicitly dispatched.

Phase split:

- Phase 0: W-ASR.1 boundary and artifact contract.
- Phase 1: W-ASR.2 audio/log-mel fixture and W-ASR.3 tokenizer startup rules.
- Phase 2: W-ASR.4 tiny synthetic Whisper path.
- Phase 3: W-ASR.5 local-artifact parity and W-ASR.6 runtime API shape.
- Phase 4: W-ASR.7 safetensors value loading and W-ASR.8 Whisper real-artifact
  config/tensor contract.
- Phase 5: W-ASR.9 real Whisper model adapter and W-ASR.10 output-token
  parity.
- Phase 6: W-ASR.11 timestamp decode policy, W-ASR.12 WER corpus harness,
  W-ASR.13 whisper.cpp benchmark harness, and W-ASR.14 model-size
  compatibility audit. Multilingual remains deferred until after the English
  ASR path has timestamps, corpus quality checks, chunking, and baseline
  performance coverage.
- Phase 7: W-ASR.15 timestamped local-artifact parity, W-ASR.16 local WER
  corpus runner, W-ASR.17 streaming/chunked transcription contract, W-ASR.18
  dedicated Ocelotl timing hook for benchmark runs, and W-ASR.19 all-size
  local-artifact parity harnesses.

## W-ASR.1 Define The Whisper Boundary

- `Crates`: docs first; likely `ocelotl-models`, `ocelotl-loader`,
  `ocelotl-tokenizer`, and `ocelotl-runtime` later.
- `Test first`: no code test; add this milestone/task pair and link the
  boundary from `docs/roadmap.md`.
- `Done when`: docs name the first artifact format, API boundary, non-goals,
  and the rule that Burn types do not escape public Ocelotl APIs.

## W-ASR.2 Add Audio Fixture And Log-Mel Reference

- `Crates`: likely future `ocelotl-audio` or `ocelotl-models::whisper`.
- `Test first`: a tiny 16 kHz mono waveform fixture maps to pinned log-mel
  values.
- `Done when`: unsupported sample rates and non-mono input fail before compute,
  and the deterministic log-mel fixture passes offline.
- `Status note`: first slice landed in `ocelotl-models::whisper::audio` with a
  Whisper-style scalar CPU reference and no Burn dependency. If the audio
  surface grows beyond Whisper model-family semantics, split a dedicated audio
  crate before exposing it through runtime APIs.

## W-ASR.3 Pin Whisper Tokenizer Startup Rules

- `Crates`: `ocelotl-tokenizer`.
- `Test first`: a fixture asserts start-of-transcript, language, task,
  no-timestamp, timestamp, and end-of-text token handling.
- `Done when`: no-timestamp masking and special-token treatment are explicit and
  tested for the whole decode loop, not just the prompt prefix.

## W-ASR.4 Build A Tiny Synthetic Whisper Path

- `Crates`: `ocelotl-models`, `ocelotl-kernels` if needed.
- `Test first`: construct a tiny synthetic Whisper-shaped encoder/decoder and
  assert output shape plus one pinned token/logit fixture.
- `Done when`: the model path proves encoder, cross-attention, decoder, and
  token projection wiring without loading a real model.

## W-ASR.5 Add Opt-In Local-Artifact Parity

- `Crates`: `ocelotl-loader`, `ocelotl-models`, `ocelotl-tokenizer`.
- `Test first`: an ignored test checks for a local converted Whisper tiny model
  and skips with a clear message when missing.
- `Done when`: the test documents exact artifact paths and compares one short
  audio clip against a pinned reference transcript or token sequence.
- `Status note`: W-ASR.5 now has a default-on schema/path contract and an
  ignored local-artifact harness at
  `crates/models/tests/whisper_local_artifact_parity.rs`. The ignored test
  checks `local-artifacts/whisper_tiny_en/`, parses the expected-token
  reference shape, validates 16 kHz mono WAV metadata, and inspects the
  safetensors header. Real output-token comparison remains blocked on a
  converted Whisper tiny.en weight loader/model adapter.

## W-ASR.6 Add Runtime Transcription API Shape

- `Crates`: `ocelotl-runtime`.
- `Test first`: runtime rejects empty audio and unsupported audio metadata
  before model compute.
- `Done when`: runtime exposes Ocelotl-owned transcription request/result types
  and reaches the Whisper model through the same public lifecycle discipline as
  Qwen prefill/decode.
- `Status note`: W-ASR.6 now exposes `TranscriptionRequest`,
  `TranscriptionResponse`, and `transcribe` in `ocelotl-runtime`. The default
  runtime tests reject empty audio and unsupported audio metadata before model
  compute, then run the synthetic path through
  `log_mel_spectrogram -> WhisperTinyModel::forward -> greedy_sample`. The
  response is token/logit shaped; token-to-text decoding, multi-token ASR
  decode, timestamp policy, and real Whisper weights remain follow-up work.

## Track Closure

The track closes when a default-on offline fixture proves audio preprocessing and
a tiny synthetic Whisper-shaped decode path, and an ignored local-artifact test
documents real-model parity.

Current status: W-ASR.1 through W-ASR.6 have shipped the synthetic/default-on
track and the ignored local-artifact harness. Real output-token parity remains
blocked on a converted Whisper tiny.en weight loader/model adapter; W-ASR.5
documents the exact bundle and reference-token contract but does not claim that
Ocelotl can run real Whisper weights yet.

## W-ASR.7 Add Safetensors Value Loading

- `Crates`: `ocelotl-loader`.
- `Test first`: construct tiny safetensors fixtures with F32, F16, and BF16
  payloads and assert loading a named tensor returns Ocelotl-owned `Vec<f32>`
  values plus shape metadata.
- `Done when`: loader can read a single named tensor from safetensors without
  exposing the foreign `safetensors` crate outside `ocelotl-loader`; missing
  tensors, dtype mismatch, data-length mismatch, and malformed payloads return
  typed `OcelotlError`s. This is generic loader groundwork for Whisper and
  later Qwen real-weight parity.
- `Status note`: W-ASR.7 landed `LoadedTensor` and
  `load_safetensors_tensor_f32` in `ocelotl-loader`. The API loads one named
  safetensors tensor into Ocelotl-owned shape/dtype/value data and converts
  F32, F16, and BF16 payloads into `Vec<f32>` while preserving typed error
  behavior for missing files, missing tensors, unsupported dtypes, and malformed
  payloads.

## W-ASR.8 Add Whisper Config And Tensor Contract

- `Crates`: `ocelotl-models`.
- `Test first`: parse a tiny Whisper config fixture and validate a synthetic
  safetensors manifest against the real Whisper tensor-name and shape contract.
- `Done when`: `ocelotl_models::whisper` has Ocelotl-owned real-config and
  tensor-contract types covering tiny.en-shaped dimensions, encoder convs,
  encoder blocks, decoder token/position embeddings, decoder self/cross
  attention, GELU MLP projections, and layer norms. The contract should reject
  missing tensors, wrong shapes, unsupported dtypes, and inconsistent head
  dimensions before any model compute.
- `Status note`: W-ASR.8 landed `WhisperConfig`,
  `parse_whisper_config_json`, `required_whisper_tensor_names`, and
  `validate_whisper_tensors`. The ignored local-artifact harness now parses the
  real config and validates the safetensors manifest against the canonical
  OpenAI-style Whisper tensor contract, including encoder convs, encoder
  `ln_post`, decoder embeddings, decoder self-attention, decoder
  cross-attention, MLP projections, and final decoder layer norm. Alias support
  for HF/Burn-converted tensor names remains intentionally deferred until an
  actual local manifest proves the needed alternate names.

## W-ASR.9 Build Real Whisper Model Adapter

- `Crates`: `ocelotl-models`, `ocelotl-kernels` if new CPU primitives are
  needed.
- `Test first`: use tiny hand-checked layer fixtures for the operations the
  synthetic path does not cover: conv1d, LayerNorm, GELU MLP, encoder
  self-attention, decoder causal self-attention, and decoder cross-attention.
- `Done when`: a real Whisper-shaped model struct can be constructed from a
  `WhisperConfig` plus loaded weights and can produce next-token logits for one
  decoder prompt against log-mel input. This task still may use committed tiny
  synthetic weights for default tests; real weights remain `#[ignore]`.

## W-ASR.10 Extend Local-Artifact Parity To Output Tokens

- `Crates`: `ocelotl-models`, `ocelotl-runtime`, `ocelotl-tokenizer` if text
  decode is included.
- `Test first`: extend the existing ignored
  `whisper_local_artifact_parity.rs` harness so it runs
  `local-artifacts/whisper_tiny_en/reference/sample_16khz_mono.wav` through the
  real adapter and compares exact generated token IDs against
  `reference/expected_tokens.json`.
- `Done when`: the opt-in local-artifact test proves exact token parity for the
  pinned tiny.en bundle. Text output can remain optional unless the fixture
  includes `expected_text`.

## W-ASR.11 Add English Timestamp Decode Policy

- `Crates`: `ocelotl-tokenizer`, `ocelotl-models`, `ocelotl-runtime` if the
  runtime result shape needs timestamp fields.
- `Test first`: add timestamp-token fixtures that prove timestamp tokens are
  allowed when no-timestamps mode is disabled, text tokens and timestamp tokens
  obey distinct masking rules, and decoded segments have deterministic start/end
  boundaries for a tiny fixture.
- `Done when`: English transcription can run with timestamp tokens enabled
  through Ocelotl-owned policy types, default tests pin the mask/segment rules,
  and the opt-in local-artifact harness can validate a timestamped reference
  without changing the no-timestamps parity contract.
- `Out of scope`: multilingual language prompts, streaming chunk stitching, and
  performance tuning.

## W-ASR.12 Add WER Corpus Harness

- `Crates`: likely `ocelotl-models` tests first; `ocelotl-runtime` only if the
  harness uses the public transcription API.
- `Test first`: add a tiny committed transcript-normalization fixture that
  proves lowercasing, punctuation handling, whitespace folding, and insertion /
  deletion / substitution counting before any real-audio corpus is wired in.
- `Done when`: an ignored corpus harness can read a manifest of local WAV files
  plus expected transcripts, run deterministic transcription, compute WER, and
  report per-sample plus aggregate results. The first pass may be
  transcript-only; timestamp-aware scoring can be added after W-ASR.11 lands.
- `Out of scope`: claiming model quality from one sample, downloading corpora in
  default tests, or making WER thresholds block CI before a real corpus policy is
  approved.

## W-ASR.13 Add whisper.cpp Benchmark Harness

- `Crates`: docs and test tooling first; avoid runtime changes unless benchmark
  instrumentation needs an Ocelotl-owned timing hook.
- `Test first`: add a manifest/parser test or command-shape test that proves the
  benchmark harness records model path, audio path, thread count, command, wall
  time, and output token/text summary without requiring whisper.cpp to be
  installed.
- `Done when`: contributors can run an ignored/local benchmark comparing Ocelotl
  and whisper.cpp on the same audio/model inputs, with clear skip behavior when
  whisper.cpp is absent and no performance parity claim until numbers are
  captured.
- `Out of scope`: optimization work, GPU comparison, and treating whisper.cpp
  output as the canonical correctness oracle.
- `Status note`: W-ASR.13 now has default-on manifest/record contract tests in
  `crates/models/tests/whisper_cpp_benchmark.rs`, schema fixtures under
  `fixtures/benchmarks/`, an opt-in local runner at
  `tools/whisper-cpp-bench.ps1`, and the local workflow documented in
  `docs/benchmarks/whisper-cpp.md`. The current Ocelotl command is the ignored
  W-ASR.10 local-artifact parity harness, so captured wall-clock numbers are a
  baseline shape only until a dedicated transcription CLI or timing hook exists.

## W-ASR.14 Audit Whisper Model-Size Compatibility

- `Crates`: `ocelotl-models`, `ocelotl-loader`; docs if artifact paths are added.
- `Test first`: add config/tensor-contract fixtures for at least one non-tiny
  Whisper size, or table-driven tests that cover the known OpenAI Whisper size
  dimensions without loading large weights.
- `Done when`: the config and tensor contract are proven not to be tiny-only,
  local-artifact paths for additional sizes are documented, and any unsupported
  size-specific assumption fails with a typed error before compute.
- `Out of scope`: full output-token parity for every size; that comes after
  timestamps, WER, streaming/chunking, and baseline performance measurement.

## W-ASR.15 Add Timestamped Local-Artifact Parity

- `Crates`: `ocelotl-tokenizer`, `ocelotl-models`, `ocelotl-runtime` only if
  runtime result types must carry timestamp segments.
- `Test first`: extend the local-artifact reference schema with a timestamped
  fixture shape that pins `timestamps: true`, expected token IDs containing
  timestamp boundary tokens, and expected segment boundaries.
- `Done when`: the default tests validate timestamped reference-schema shape
  and segment parsing, and an ignored local-artifact test can run a timestamped
  reference against the same `local-artifacts/whisper_tiny_en/` bundle without
  weakening the W-ASR.10 no-timestamps exact-token proof.
- `Out of scope`: multilingual prompts, streaming chunk stitching, and WER
  scoring of timestamped segments.

## W-ASR.16 Add Local WER Corpus Runner

- `Crates`: `ocelotl-models` tests first; `ocelotl-runtime` if the runner uses
  runtime transcription APIs rather than the model test harness.
- `Test first`: add a committed corpus manifest fixture that names local WAV
  paths, expected transcripts, optional expected token paths, and skip behavior
  when corpus artifacts are absent.
- `Done when`: an ignored local corpus test can load a manifest, run
  deterministic Whisper transcription for every case present on disk, compute
  per-sample and aggregate WER via `whisper::wer`, and emit a readable report
  without making WER thresholds block CI.
- `Out of scope`: downloading corpora, deciding a product-quality WER threshold,
  or treating one tiny sample as a corpus-quality claim.

## W-ASR.17 Add Streaming/Chunked Transcription Contract

- `Crates`: `ocelotl-runtime` first; model code only if a default-on synthetic
  streaming fixture needs it.
- `Test first`: add chunk-planning tests for 16 kHz mono audio that pin window
  size, overlap, last-partial-chunk behavior, and monotonic chunk time ranges.
- `Done when`: runtime exposes Ocelotl-owned chunk metadata/request types and a
  deterministic chunk planner. The contract must make it clear whether model
  state is reused or each chunk is decoded independently.
- `Out of scope`: KV/cache reuse, real-time microphone capture, diarization,
  and final transcript stitching heuristics.

## W-ASR.18 Add Dedicated Ocelotl Benchmark Timing Hook

- `Crates`: likely root CLI or `tools/`; `ocelotl-runtime` only if timing needs
  a public helper.
- `Test first`: update the benchmark manifest/record tests so the Ocelotl side
  names a dedicated transcription timing command rather than `cargo test`.
- `Done when`: `tools/whisper-cpp-bench.ps1` can run whisper.cpp and an Ocelotl
  timing command on the same local model/audio inputs, record wall-clock timing
  and output summary for both, and skip cleanly when either side's local
  prerequisites are missing.
- `Out of scope`: optimizing Ocelotl to match whisper.cpp; this task only makes
  the measurement meaningful enough to compare later.

## W-ASR.19 Add All-Size Local-Artifact Parity Harnesses

- `Crates`: `ocelotl-models`, `ocelotl-loader`, docs.
- `Test first`: add default-on path/schema tests for classic Whisper
  `tiny.en`, `base.en`, `small.en`, `medium.en`, and `large` local bundles.
- `Done when`: ignored local-artifact parity tests can run exact token checks
  for every size whose bundle exists, skip absent bundles with explicit
  remediation, and report each size independently. The default tests must not
  load large weights.
- `Out of scope`: `large-v3`, `turbo`, multilingual quality claims, and making
  every size mandatory for CI.

## W-ASR.20 Add Stage-Level Whisper CPU Timing

- `Crates`: root CLI first; `ocelotl-models` only if timing requires an API
  seam that is already part of the production shape.
- `Test first`: add a default-on test that pins the Ocelotl benchmark output
  schema includes `timings_ms` for config parsing, manifest validation,
  expected-token read, tensor load/model construction, WAV read, log-mel,
  audio encode, decode total, and per-generated-token decode timing.
- `Done when`: `ocelotl bench-whisper-transcribe` emits stage-level timing
  fields in its JSON output, the whisper.cpp benchmark record preserves those
  fields inside the Ocelotl stdout excerpt, and local benchmark runs can identify
  whether model load, encoder, or decoder work dominates.
- `Out of scope`: making the CPU backend faster; this task adds measurement and
  the minimum production-shaped API seam needed to measure encoder reuse.

## W-ASR.21 Add Whisper Encoded-Audio Session State

- `Crates`: `ocelotl-models`, then `ocelotl-runtime` once the model API is
  proven.
- `Test first`: prove that logits computed from cached encoded audio exactly
  match the legacy `forward_next_token_logits(log_mel, tokens)` wrapper on a
  synthetic Whisper fixture, and that malformed encoded-audio shapes fail before
  compute.
- `Done when`: Whisper exposes an Ocelotl-owned encoded-audio/session type,
  public runtime transcription can hold that state while decoding, and the
  legacy no-cache wrapper remains available for reference parity.
- `Status note`: landed on 2026-05-12. `WhisperEncodedAudio` is the model-level
  encoded-audio state, and `WhisperTranscriptionState` is the runtime-held state
  produced by `prepare_whisper_transcription`. Decode controls live separately in
  `WhisperDecodeRequest` so a prepared state can be reused without passing audio
  samples back into the decode call.
- `Out of scope`: KV cache reuse for decoder self-attention; this task only
  separates invariant audio encoder output from token decode.

## W-ASR.22 Reuse Encoder Output During Decode

- `Crates`: root CLI benchmark hook, `ocelotl-runtime`, and any tests that run
  autoregressive local-artifact parity.
- `Test first`: add or update a local/synthetic decode test proving a
  cached-audio decode loop returns the same token IDs as the legacy loop.
- `Done when`: the benchmark hook and runtime transcription path encode audio
  once per audio chunk and reuse the encoded audio for every generated token.
  Stage timings should show encoder time paid once, not once per token.
- `Status note`: landed on 2026-05-12. The runtime path now decodes from
  `WhisperTranscriptionState`, and the ignored local-artifact parity loop encodes
  audio once before the token loop. The exact tiny.en ignored proof passed
  locally in the debug test harness (`122.92s`).
- `Out of scope`: optimizing decoder kernels or introducing decoder KV cache.

## W-ASR.23 Add Optimized CPU Kernel Selection

- `Crates`: `ocelotl-kernels`, then model callers.
- `Test first`: add backend-selection tests that prove the default scalar path
  remains available and an optimized CPU path can be selected without changing
  public model outputs.
- `Done when`: CPU kernel selection is explicit and Ocelotl can route hot
  matmul/attention work to a faster CPU implementation while preserving scalar
  fallback for correctness and portability.
- `Status note`: landed on 2026-05-12. `CpuKernelBackend` now has explicit
  `Scalar` and `Optimized` modes. The scalar backend remains default; optimized
  mode routes Qwen prefill matmul/attention and real Whisper linear projection
  work through cache-friendlier safe-Rust CPU kernels while preserving public
  model outputs within pinned tolerances.
- `Out of scope`: GPU/CubeCL and quantization-specific kernels.

## W-ASR.24 Refresh whisper.cpp Baseline and Set CPU Gates

- `Crates`: docs, benchmark fixtures, and local tooling.
- `Test first`: update benchmark record-shape tests if the record starts
  carrying parsed stage timing fields rather than only stdout excerpts.
- `Done when`: a fresh local tiny.en whisper.cpp comparison is captured after
  W-ASR.20-W-ASR.23, the performance delta is documented, and follow-up CPU
  gates are stated as measurable targets rather than generic optimization work.
- `Out of scope`: claiming parity across all Whisper sizes or across GPU.
- `Status note`: landed on 2026-05-12. The benchmark hook now accepts
  `--cpu-kernel-mode scalar|optimized`, and the example manifest measures the
  W-ASR.23 optimized backend explicitly. The fresh local optimized run was
  parity-clean but slower than scalar: Ocelotl optimized `16,648 ms` vs
  whisper.cpp `564 ms` (~29.5x slower), while the same release hook in scalar
  mode measured `14,179 ms`. `docs/benchmarks/whisper-cpp.md` now records the
  result and the measurable correctness, regression, optimized-default, and
  CPU-competitiveness gates. Optimized mode must not become the Whisper default
  until it beats scalar on the documented gate.
