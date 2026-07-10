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
| Whisper ASR | implemented; alpha hardening active | Real local inference exists and exact-token local parity has passed, but local-artifact proof and performance evidence are not default CI gates. |
| Gemma4 text | implemented subset; real parity blocked | Synthetic/public-path behavior and substantial real-artifact execution are present. Full real-artifact logits parity is still red. |

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
  comparison through the real adapter.
- Stage-level timing and whisper.cpp comparison tooling exists. Its records are
  diagnostic evidence, not a correctness oracle or a CI performance gate.

### Alpha Blockers And Limits

- The benchmark runner must prove equal effective resources, warmups, repeated
  samples, raw-sample retention, and output equivalence before results can gate
  releases.
- Public request and artifact budgets must reject excessive audio, token, and
  allocation requests before expensive preprocessing or model work.
- Local artifact parity is opt-in because weights and reference binaries are
  not committed. Alpha evidence must name the exact model and reference
  revisions used.
- Timestamped segments, streaming/chunk stitching, multilingual quality claims,
  and an approved WER threshold are outside the first alpha support claim.

## Gemma4

### Implemented

- Bounded GGUF v3 metadata/tensor inspection, embedded tokenizer extraction,
  GGUF BPE tokenization, and Q4_K/Q5_K/Q6_K value handling.
- Text-only synthetic prefill/decode through the public runtime, including
  mixed sliding-window/global attention, shared KV, logit softcap, Gemma4 RoPE,
  per-layer embeddings, and post-branch normalization semantics.
- Real Q4_K_M-origin text weights through dequantized F32 fallback plus native
  Q5_K/Q6_K attention projection sidecars.

### Validated

- Default offline fixtures cover the supported synthetic text path, malformed
  GGUF rejection, tokenizer contracts, quantization arithmetic, and public
  runtime behavior.
- The opt-in tokenizer proof matched the pinned llama.cpp reference.
- The real-artifact layer-0 discriminator matches through `kqv_out-0`.

### Alpha Blockers And Limits

- Real-artifact execution diverges at the native Q5_K attention output
  projection: the latest recorded discriminator has Ocelotl
  `attn_output_proj-0 = -3.062695` versus llama.cpp `node_33 = -3.404522`, a
  `0.34182692` difference at `0.05` tolerance.
- After fixing the first divergent output, layer outputs and the full final-logit
  vector must be refreshed through the same public text path before Gemma4 can
  enter the alpha support claim.
- The selected artifact is multimodal, but the first alpha target is explicitly
  text-only. Gemma4 image/audio/video input remains unsupported.
- Quantized performance and memory use need repeatable release-mode evidence;
  synthetic green tests alone are not production proof.

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
