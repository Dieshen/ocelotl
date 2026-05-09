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
