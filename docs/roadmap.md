# Roadmap

This roadmap is ordered to reduce correctness risk. Each milestone should land
with tests, acceptance criteria, and validation commands before broadening scope.
Ocelotl is developed test-first; see `docs/validation/tdd.md` and
`docs/validation/test-matrix.md` for the project-wide testing policy. The
provisional first model target is documented in `docs/model-target.md`.
Executable task backlogs for each milestone live in `docs/tasks/README.md`.

## Milestones

| Milestone | Spec | Tasks | Summary |
| --- | --- | --- | --- |
| M0 | `docs/milestones/m0-skeleton.md` | `docs/tasks/m0-skeleton.md` | Workspace, crate boundaries, publishing metadata, and control docs. |
| M1 | `docs/milestones/m1-cpu-reference.md` | `docs/tasks/m1-cpu-reference.md` | Deterministic CPU reference path for a tiny Qwen2.5-shaped model. |
| M2 | `docs/milestones/m2-loader-tokenizer.md` | `docs/tasks/m2-loader-tokenizer.md` | Local metadata loading, tokenizer fixtures, and chat-template contracts. |
| M3 | `docs/milestones/m3-single-model-forward.md` | `docs/tasks/m3-single-model-forward.md` | Qwen2.5-style prefill and one-token decode through runtime APIs. |
| M4 | `docs/milestones/m4-gpu-kernel-path.md` | `docs/tasks/m4-gpu-kernel-path.md` | First GPU-backed kernel path with CPU/GPU parity. |
| M5 | `docs/milestones/m5-contiguous-kv-cache.md` | `docs/tasks/m5-contiguous-kv-cache.md` | Request-scoped contiguous KV cache used by decode. |
| M6 | `docs/milestones/m6-paged-kv-cache.md` | `docs/tasks/m6-paged-kv-cache.md` | Paged KV with multi-page tests and contiguous/paged parity. |
| M7 | `docs/milestones/m7-continuous-batching.md` | `docs/tasks/m7-continuous-batching.md` | Scheduler and continuous batching without changing deterministic outputs. |
| M8 | `docs/milestones/m8-server-api.md` | `docs/tasks/m8-server-api.md` | Server layer around runtime APIs with intentional error and streaming semantics. |

## Post-M3 Expansion Candidates

These are intentionally not inserted into the closed M3 task numbering. They are
candidate tracks for broadening Ocelotl after the Qwen2.5 M3 path, and should be
scheduled explicitly before implementation starts.

| Track | Spec | Tasks | Summary |
| --- | --- | --- | --- |
| Whisper ASR | `docs/milestones/post-m3-whisper-asr.md` | `docs/tasks/post-m3-whisper-asr.md` | Speech-to-text track using `whisper-burn` as reference material and Burn behind Ocelotl-owned APIs. |
| Qwen3.5 + Gemma4 | `docs/milestones/post-m3-model-family-expansion.md` | `docs/tasks/post-m3-model-family-expansion.md` | Compatibility discovery and model-family expansion for Qwen3.5 and Gemma4/GGUF without treating either as a small Qwen2.5 extension. |

## CI Baseline

The baseline CI policy is documented in `docs/ci.md`. Default CI must stay
offline and run formatting, workspace check, and workspace tests.

## Development Rule

Do not implement a milestone by adding code first. Start with the smallest test
that captures the next behavior, confirm it fails for the expected reason, then
implement the minimal change. Benchmarks follow correctness, not the reverse.

## Deferred

- Multi-GPU execution.
- Broad model-family support.
- Broad quantization support.
- Speech-to-text and multimodal model support.
- Speculative decoding.
- Distributed serving.
- Tool/function calling semantics.
