# Project Status

Last reconciled: 2026-07-10.

This is the canonical status page for Ocelotl. Milestone task checkboxes record
historical execution; this page states the current release posture. Acceptance
evidence lives in `docs/validation/test-matrix.md`, and current test results win
if any prose has drifted.

## Release Posture

Ocelotl is in **alpha-production hardening**. It is not alpha-ready yet.

The first alpha target is a local, single-process Rust library and CLI that
loads trusted local artifacts. The minimum model-family scope is:

- Whisper for offline 16 kHz mono batch transcription.
- Gemma4 for text-only inference from the selected Q4_K_M GGUF family.

This target does not include an internet-facing service. M8 server work has not
started, so Ocelotl must not be presented as a hardened remote or multi-tenant
service.

## Milestone State

| Surface | State | Proven scope and limit |
| --- | --- | --- |
| M0-M3 | closed | Workspace, CPU/reference behavior, local loading/tokenization, and the first Qwen2.5-shaped public forward path are covered by default offline tests. |
| M4 | closed, narrow | CubeCL/WGPU can be selected and RoPE executes on GPU with opt-in CPU/GPU parity. Other operations retain CPU fallback; this is not full-model GPU residency. |
| M5 | closed for CPU/reference | Request-scoped contiguous Qwen KV behavior and cached-decode parity are covered. GPU cache residency remains deferred. |
| M6 | closed for CPU/reference | Paged allocation, multi-page behavior, cleanup, and contiguous/paged parity are covered. GPU paged-attention kernels remain deferred. |
| M7 | closed as correctness plumbing | Bounded admission, state transitions, cancellation, fairness, and deterministic batch parity are covered. The scheduler is not yet a throughput-optimized production scheduler. |
| M8 | not started | There is no supported network server endpoint, streaming transport, authentication, or external error contract. |
| Whisper ASR | alpha candidate; opt-in evidence green | Real tiny.en exact-token parity and a repeated equal-resource whisper.cpp comparison passed locally. The proof remains opt-in because weights and reference binaries are not committed. |
| Gemma4 text | executes; real final-logit parity blocked | The selected Q4_K_M text path loads and executes all 42 layers with native Q4_K/Q5_K/Q6_K projections. Layer 0 is green, but the final real-artifact logit distribution is still outside tolerance. |

## Whisper

### Implemented

- 16 kHz mono validation and deterministic Whisper-style log-mel preprocessing.
- Local safetensors configuration, tensor validation, value loading, tokenizer,
  and real Whisper model construction behind Ocelotl-owned APIs.
- Reusable encoded-audio state, decoder self-attention cache, cross-attention
  precomputation, masked greedy decode, and token/text benchmark output.
- Scalar and AVX2/threaded CPU paths, with feature-gated WGPU experiments kept
  behind the same kernel boundary.

### Validated

- Default offline fixtures cover preprocessing, tensor contracts, model
  semantics, runtime lifecycle, cache parity, and CPU optimization parity.
- The ignored tiny.en local-artifact proof has passed exact expected-token
  comparison through the real adapter on 2026-07-10.
- The reproducible comparison runner now records warmups, repeated alternating
  samples, raw timings, revision/dirty state, effective commands, and output
  equivalence. A 2026-07-10 release run used one warmup plus ten measured
  samples per engine and matched both expected Ocelotl tokens and normalized
  transcript text. Ocelotl averaged `620 ms` (`629 ms` median, `645 ms` p95)
  versus whisper.cpp `428 ms` (`425 ms` median, `497 ms` p95), or `1.449x`
  whisper.cpp full-process wall time on that machine.

### Alpha Blockers And Limits

- Local artifact parity is opt-in because weights and reference binaries are
  not committed. Release evidence must still be refreshed at the candidate
  commit and name the exact model, reference revision, commands, and hardware.
- Timestamped segments, streaming/chunk stitching, multilingual quality claims,
  and an approved WER threshold are outside the first alpha support claim.

## Gemma4

### Implemented

- Bounded GGUF v3 metadata/tensor inspection, embedded tokenizer extraction,
  GGUF BPE tokenization, and Q4_K/Q5_K/Q6_K value handling.
- Text-only synthetic prefill/decode through the public runtime, including
  mixed sliding-window/global attention, shared KV, logit softcap, Gemma4 RoPE,
  per-layer embeddings, and post-branch normalization semantics.
- Real Q4_K_M-origin text weights through a dequantized F32 fallback plus native
  Q4_K/Q5_K/Q6_K Q8_K projection sidecars for every text attention, FFN,
  per-layer input/output, and tied output projection.

### Validated

- Default offline fixtures cover the supported synthetic text path, malformed
  GGUF rejection, tokenizer contracts, quantization arithmetic, and public
  runtime behavior.
- The opt-in tokenizer proof matched the pinned llama.cpp reference.
- With llama.cpp pinned to non-flash attention, F32 K/V cache, and no repacking,
  the complete real-artifact layer-0 discriminator matches through `l_out-0`;
  the largest sampled difference was approximately `6.2e-5`, well inside the
  `0.05` contract.
- The full selected 42-layer artifact now loads and reaches final logits through
  the public text prefill path. The latest run selected the same top-1 token as
  llama.cpp and had 18/20 overlap in the top-20 token set.

### Alpha Blockers And Limits

- Full final-logit parity remains red. Against the deterministic non-repacked
  llama.cpp reference, the latest 262,144-logit comparison had max absolute
  error `2.1698594`, mean absolute error `0.38767775`, RMS error `0.48177935`,
  and 241,146 logits above `0.05`. The top-1 agreement is useful smoke evidence,
  but it does not satisfy the numeric contract.
- Layer tracing shows smooth accumulated drift rather than a new semantic cliff:
  the first sampled layer above `0.05` was layer 7, the worst sampled edge was
  about `0.2625` at layer 29, and the shared-KV transition at layer 24 was not a
  discontinuity. The next parity work must narrow that accumulated numerical
  drift without weakening the tolerance.
- The selected artifact is multimodal, but the first alpha target is explicitly
  text-only. Gemma4 image/audio/video input remains unsupported.
- The current loader retains eagerly dequantized dense fallback weights beside
  native quantized sidecars. That is intentionally correctness-first, but its
  peak memory and load-time cost must be removed or bounded before the selected
  Gemma artifact is production-alpha ready.

## Alpha Release Gates

An alpha tag requires all of the following at the same commit:

1. `tools/verify.ps1 -Mode Full` passes with the committed lockfile on the
   pinned development toolchain.
2. The Rust 1.85 MSRV job and the RustSec dependency audit pass, or an advisory
   has an explicit, time-bounded policy exception.
3. Public request/artifact limits and target-feature dispatch cannot be bypassed
   through safe APIs.
4. Whisper exact-token local parity and its repeatable equal-resource benchmark
   record pass against pinned artifacts.
5. Gemma4 text-only real-artifact tokenizer, layer, full-logit, and generated
   token parity pass through the public runtime against a pinned reference.
6. Default tests remain offline; local weights, network access, real GPU launch,
   and external reference binaries remain opt-in and documented.
7. Release notes state that M8 server, Gemma4 multimodal input, Whisper
   streaming/timestamps, and unproved hardware backends are unsupported.

## Evidence Index

- `docs/validation/test-matrix.md` — acceptance-to-test traceability.
- `docs/validation/parity.md` — numeric and exact-token parity policy.
- `docs/benchmarks/whisper-cpp.md` — local Whisper benchmark methodology and
  historical records.
- `docs/ci.md` — deterministic build, MSRV, offline, and CI policy.
- `docs/tasks/post-m3-whisper-asr.md` — detailed Whisper task history.
- `docs/tasks/post-m3-model-family-expansion.md` — detailed Gemma4 task history.
