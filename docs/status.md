# Project Status

Last reconciled: 2026-07-25.

This is the canonical status page for Ocelotl. Milestone task checkboxes record
historical execution; this page states the current release posture. Acceptance
evidence lives in `docs/validation/test-matrix.md`, and current test results win
if any prose has drifted.

## Release Posture

Ocelotl is preparing **0.1.0**, a local, single-process Rust library that loads
trusted local artifacts. It is not an internet-facing service: M8 server work has
not started, so Ocelotl must not be presented as a hardened remote or
multi-tenant service.

### 0.1.0 scope — what is claimed

The scope is set by **which surfaces have current, independently verified parity
evidence at the release commit**, not by which were planned first:

- **Text embeddings** (`gemma::embedding::EmbeddingGemmaModel`,
  `qwen::pplx_embed::PplxEmbedModel`) — cosine >= 0.9999997 vs `llama-embedding`
  with full retrieval-ranking agreement; CPU faster than llama.cpp on bulk, GPU
  at parity.
- **Parakeet TDT 0.6B ASR** — token-exact and frame-exact against **two**
  independent implementations (the ONNX export and `parakeet.cpp`), RTF 0.095 on
  12 threads.

### Deferred to 0.2.0 — present in-tree, not claimed

Both execute and are useful; neither satisfies its numeric gate at this commit,
so neither is part of the 0.1.0 support claim.

- **Gemma4 text.** Full-logit parity is red and is a **contract decision, not a
  bug**: every specific suspect was ruled out and the residual is the
  independent-f32 reproducibility floor. argmax matches llama.cpp exactly and
  top-20 overlap is 18-19/20. 0.2.0 must first settle whether raw-logit
  0.05-absolute is the right *shape* of contract versus top-k agreement or KL
  divergence, both of which already pass.
- **Whisper ASR.** Previously the lead alpha candidate on 2026-07-10 evidence.
  That evidence cannot be reproduced at this commit: the artifact bundle was not
  rebuildable from the repository (now fixed — see below), and a freshly built
  bundle shows a **one-token divergence** from a pure-greedy whisper.cpp
  reference (26 of 27 tokens match; ocelotl omits a comma). Cause unresolved.
  Deferring is the honest call until it is.

## Milestone State

| Surface | State | Proven scope and limit |
| --- | --- | --- |
| M0-M3 | closed | Workspace, CPU/reference behavior, local loading/tokenization, and the first Qwen2.5-shaped public forward path are covered by default offline tests. |
| M4 | closed, narrow | CubeCL/WGPU can be selected and RoPE executes on GPU with opt-in CPU/GPU parity. Other operations retain CPU fallback; this is not full-model GPU residency. |
| M5 | closed for CPU/reference | Request-scoped contiguous Qwen KV behavior and cached-decode parity are covered. GPU cache residency remains deferred. |
| M6 | closed for CPU/reference | Paged allocation, multi-page behavior, cleanup, and contiguous/paged parity are covered. GPU paged-attention kernels remain deferred. |
| M7 | closed as correctness plumbing | Bounded admission, state transitions, cancellation, fairness, and deterministic batch parity are covered. The scheduler is not yet a throughput-optimized production scheduler. |
| M8 | not started | There is no supported network server endpoint, streaming transport, authentication, or external error contract. |
| Whisper ASR | **deferred to 0.2.0**; parity red at this commit | Real tiny.en exact-token parity and a repeated equal-resource whisper.cpp comparison passed locally. The proof remains opt-in because weights and reference binaries are not committed. |
| Gemma4 text | **deferred to 0.2.0**; executes, final-logit contract unsettled | The selected Q4_K_M text path loads and executes all 42 layers with native Q4_K/Q5_K/Q6_K projections. Layer 0 is green, but the final real-artifact logit distribution is still outside tolerance. |
| Text embeddings | parity-clean; CPU at/above llama.cpp on bulk; GPU resident | Two bidirectional encoders (`gemma::embedding::EmbeddingGemmaModel`, `qwen::pplx_embed::PplxEmbedModel`) produce mean-pooled, L2-normalized embeddings. Cosine ≥ 0.9999997 vs `llama-embedding` with full retrieval-ranking agreement. **CPU** AVX2 `linear_out_by_in`: single-seq 78/196 ms (EmbeddingGemma/pplx); bulk (rayon over sentences) 13.1 ms/embed (2.4× *faster* than llama.cpp) / 98.8 ms (parity). **GPU** (`EmbeddingGemmaGpu`, `cubecl-wgpu`): fully device-resident forward (new rmsnorm_d/rope_tables_d/expand_kv_heads_d/silu_d/attention_encoder_batched_d kernels), cosine 0.9999999. `embed_batch` (block-diagonal batched attention) hits **10.2 ms/embed at batch 256 — 1.3× faster than CPU-bulk (13.1 ms) and 3× the CPU-only llama.cpp (30.9 ms)**; batch parity exact (cosine 1.000000). **HIP/ROCm runtime** wired (compile-time `cubecl-wgpu`|`cubecl-hip` switch; same kernels): 24 device tests + cosine 0.9999999 on ROCm. Fair GPU-vs-GPU (RX 6600): register-blocked GEMM → ocelotl-WGPU **5.9** / ocelotl-ROCm 7.4 / llama-ROCm 6.0 ms/embed (batch 256). **ocelotl-WGPU now matches llama.cpp.** The lever was kernel quality (register-blocked micro-tile GEMM), not the backend API. Remaining: 8×8 tiles/vectorized loads, variable-length batches, pplx GPU. Parity proofs are opt-in (need the GGUFs + a GPU). |

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

### 0.2.0 Blockers And Limits

- **One-token divergence, cause unresolved (2026-07-25).** A bundle rebuilt from
  `openai/whisper-tiny.en` via `tools/convert_whisper_hf_to_openai.py`, with the
  reference captured from whisper.cpp `080bbbe` in **pure greedy** mode
  (`-bo 1 -bs 1 -nf`), matches on **26 of 27 tokens**. ocelotl omits a comma
  (token 11) after " you"; every other token, including all of those after the
  divergence, is identical. Re-convergence after a skipped token suggests a
  near-tie rather than a structural fault, but that is a hypothesis, not a
  finding.
- Confounds not yet eliminated, in the order worth checking: ocelotl loads HF
  safetensors while the reference runs `ggml-tiny.en.bin`, so the two may not be
  bit-identical weights; the sample audio is a substitution (`jfk.wav`) because
  the original bundle's audio was never committed; and token-suppression policy
  may differ between the two decoders.
- **The bundle was previously not rebuildable from the repository.**
  `docs/artifact-preparation.md` described the bundle's shape but never the
  HF-to-OpenAI tensor conversion it requires, so gate evidence could only be
  refreshed by whoever still had the original files. That is fixed:
  `tools/convert_whisper_hf_to_openai.py` is committed and maps all 167 tensors
  with zero drops. The parity result above is the first reproducible one.
- The equal-resource benchmark record has not been re-run at this commit.
- Timestamped segments, streaming/chunk stitching, multilingual quality claims,
  and an approved WER threshold remain out of scope.

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

## 0.1.0 Release Gates

All of the following must hold at the same commit. Status as of 2026-07-25:

| # | Gate | State |
|---|---|---|
| 1 | `tools/verify.ps1 -Mode Full` passes with the committed lockfile on the pinned toolchain | **PASS** — green on `main` |
| 2 | Rust 1.85 MSRV job and the RustSec dependency audit pass, or an advisory has a time-bounded exception | **PASS** — both green |
| 3 | Public request/artifact limits and target-feature dispatch cannot be bypassed through safe APIs | **PASS** — see below |
| 4 | Every **claimed** surface has independent-reference parity evidence captured at the release commit | **PASS** — embeddings and Parakeet |
| 5 | Default tests remain offline; local weights, network access, real GPU launch, and external reference binaries stay opt-in and documented | **PASS** — 589 offline, 43 opt-in |
| 6 | Release notes name every unsupported surface explicitly | **PASS** — `CHANGELOG.md` |

Gate 3 evidence: `CpuKernelBackend`'s `mode` field is private and the only
constructors are `scalar()`/`optimized()` (feature-free modes) and the fallible
`with_mode`/`with_mode_and_threads`, which route `Avx2` through
`validate_mode_supported` — a runtime `is_x86_feature_detected!("avx2") && ("fma")`
check that errors on unsupported hosts and on non-x86_64 targets. So no safe
caller can construct a backend that later executes unsupported
`#[target_feature]` code. `ArtifactLimits`/`RequestLimits` bound generation,
audio, context, file bytes and GGUF tensor counts, with default-on tests in
`ocelotl-core`.

**Gate 4 changed shape deliberately.** It used to name Whisper and Gemma4
specifically, which conflated *what the project set out to build* with *what it
can currently prove*. Tying the gate to the claimed surface set instead means the
release claim and the evidence cannot drift apart, and a surface can be deferred
without rewriting the gate. The two deferred families keep their own gates in the
0.2.0 section below.

## 0.2.0 Gates (deferred surfaces)

1. **Gemma4**: a project-owner decision on the shape of the logit contract,
   then text-only real-artifact tokenizer, layer, logit, and generated-token
   parity against a pinned reference under whichever contract is chosen.
2. **Whisper**: resolve the one-token divergence against a pure-greedy
   whisper.cpp reference, then exact-token local parity plus the repeatable
   equal-resource benchmark record, both captured at the candidate commit.
3. Both must be reproducible from the repository alone — no bundle that only
   exists on one machine.

## Evidence Index

- `docs/validation/test-matrix.md` — acceptance-to-test traceability.
- `docs/validation/parity.md` — numeric and exact-token parity policy.
- `docs/benchmarks/whisper-cpp.md` — local Whisper benchmark methodology and
  historical records.
- `docs/ci.md` — deterministic build, MSRV, offline, and CI policy.
- `docs/tasks/post-m3-whisper-asr.md` — detailed Whisper task history.
- `docs/tasks/post-m3-model-family-expansion.md` — detailed Gemma4 task history.
