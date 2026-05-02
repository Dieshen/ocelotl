# Roadmap

This roadmap is ordered to reduce correctness risk. Each milestone should land
with tests, acceptance criteria, and validation commands before broadening scope.
Ocelotl is developed test-first; see `docs/validation/tdd.md` and
`docs/validation/test-matrix.md` for the project-wide testing policy.

## Milestones

| Milestone | Spec | Summary |
| --- | --- | --- |
| M0 | `docs/milestones/m0-skeleton.md` | Workspace, crate boundaries, publishing metadata, and control docs. |
| M1 | `docs/milestones/m1-cpu-reference.md` | Deterministic CPU reference path for one narrow model shape. |
| M2 | `docs/milestones/m2-loader-tokenizer.md` | Local metadata loading, tokenizer fixtures, and chat-template contracts. |
| M3 | `docs/milestones/m3-single-model-forward.md` | Prefill and one-token decode for one model family through runtime APIs. |
| M4 | `docs/milestones/m4-gpu-kernel-path.md` | First GPU-backed kernel path with CPU/GPU parity. |
| M5 | `docs/milestones/m5-contiguous-kv-cache.md` | Request-scoped contiguous KV cache used by decode. |
| M6 | `docs/milestones/m6-paged-kv-cache.md` | Paged KV with multi-page tests and contiguous/paged parity. |
| M7 | `docs/milestones/m7-continuous-batching.md` | Scheduler and continuous batching without changing deterministic outputs. |
| M8 | `docs/milestones/m8-server-api.md` | Server layer around runtime APIs with intentional error and streaming semantics. |

## Development Rule

Do not implement a milestone by adding code first. Start with the smallest test
that captures the next behavior, confirm it fails for the expected reason, then
implement the minimal change. Benchmarks follow correctness, not the reverse.

## Deferred

- Multi-GPU execution.
- Broad model-family support.
- Broad quantization support.
- Speculative decoding.
- Distributed serving.
- Tool/function calling semantics.
