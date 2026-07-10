# Roadmap

This roadmap is ordered to reduce correctness risk. Each milestone should land
with tests, acceptance criteria, and validation commands before broadening scope.
Ocelotl is developed test-first; see `docs/validation/tdd.md` and
`docs/validation/test-matrix.md` for the project-wide testing policy. The
provisional first model target is documented in `docs/model-target.md`.
Executable task backlogs for each milestone live in `docs/tasks/README.md`.
Current implemented, validated, and blocked state is canonical in
`docs/status.md`; the roadmap describes ordering rather than live release
readiness.

## Milestones

| Milestone | State | Spec | Tasks | Summary |
| --- | --- | --- | --- | --- |
| M0 | closed | `docs/milestones/m0-skeleton.md` | `docs/tasks/m0-skeleton.md` | Workspace, crate boundaries, publishing metadata, and control docs. |
| M1 | closed | `docs/milestones/m1-cpu-reference.md` | `docs/tasks/m1-cpu-reference.md` | Deterministic CPU reference path for a tiny Qwen2.5-shaped model. |
| M2 | closed | `docs/milestones/m2-loader-tokenizer.md` | `docs/tasks/m2-loader-tokenizer.md` | Local metadata loading, tokenizer fixtures, and chat-template contracts. |
| M3 | closed | `docs/milestones/m3-single-model-forward.md` | `docs/tasks/m3-single-model-forward.md` | Qwen2.5-style prefill and one-token decode through runtime APIs. |
| M4 | closed, narrow | `docs/milestones/m4-gpu-kernel-path.md` | `docs/tasks/m4-gpu-kernel-path.md` | First GPU-backed kernel boundary and opt-in parity; most operations retain CPU fallback. |
| M5 | closed, CPU/reference | `docs/milestones/m5-contiguous-kv-cache.md` | `docs/tasks/m5-contiguous-kv-cache.md` | Request-scoped contiguous KV cache used by decode. |
| M6 | closed, CPU/reference | `docs/milestones/m6-paged-kv-cache.md` | `docs/tasks/m6-paged-kv-cache.md` | Paged KV with multi-page tests and contiguous/paged parity. |
| M7 | closed, correctness | `docs/milestones/m7-continuous-batching.md` | `docs/tasks/m7-continuous-batching.md` | Deterministic scheduler plumbing; throughput-oriented batching remains deferred. |
| M8 | not started | `docs/milestones/m8-server-api.md` | `docs/tasks/m8-server-api.md` | Server layer around runtime APIs with intentional error and streaming semantics. |

## Active Post-M3 Expansion Tracks

These tracks are intentionally not inserted into the closed M3 task numbering.
They now contain implemented behavior, but their alpha claims remain governed by
`docs/status.md` and current parity evidence.

| Track | Spec | Tasks | Summary |
| --- | --- | --- | --- |
| Whisper ASR | `docs/milestones/post-m3-whisper-asr.md` | `docs/tasks/post-m3-whisper-asr.md` | Speech-to-text track using `whisper-burn` as reference material and Burn behind Ocelotl-owned APIs. |
| Qwen3.5 + Gemma4 | `docs/milestones/post-m3-model-family-expansion.md` | `docs/tasks/post-m3-model-family-expansion.md` | Compatibility discovery and model-family expansion for Qwen3.5 and Gemma4/GGUF without treating either as a small Qwen2.5 extension. |

## Active Alpha Hardening

Before starting M8, the current track hardens the trusted-local runtime and CLI
for an alpha whose minimum supported families are Whisper and text-only Gemma4.
The release floor includes bounded inputs, safe target-feature dispatch,
deterministic builds, trustworthy benchmark records, Whisper exact-token local
parity, and Gemma4 full real-artifact text parity. See `docs/status.md` for the
current blockers and exclusions.

## CI Baseline

The CI policy is documented in `docs/ci.md`. CI runs the repository-owned full
gate with a committed lockfile, an MSRV job, an all-feature no-launch check, and
a dependency audit. Default tests remain offline and hardware/local-artifact
proofs remain opt-in.

## Development Rule

Do not implement a milestone by adding code first. Start with the smallest test
that captures the next behavior, confirm it fails for the expected reason, then
implement the minimal change. Benchmarks follow correctness, not the reverse.

## Deferred

- Multi-GPU execution.
- Broad model-family support.
- Broad quantization support.
- Whisper streaming, timestamped-segment, multilingual-quality, and diarization
  claims beyond the first offline batch-transcription alpha.
- Gemma4 image/audio/video input and broad multimodal model support.
- Speculative decoding.
- Distributed serving.
- Tool/function calling semantics.
