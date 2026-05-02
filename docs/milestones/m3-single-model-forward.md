# M3 Single Model Forward

## Goal

Run prefill and one-token decode for a single supported model family through the
runtime API and compare outputs against reference fixtures.

## Target Family

M3 targets the Qwen2.5-style dense decoder-only forward path. Any generalization
beyond that family should wait until the Qwen2.5 path has reference fixtures and
parity tests.

## Non-Goals

- GPU execution.
- Multiple model families.
- Quantized weights.
- Paged KV.
- Continuous batching.
- Server API compatibility.

## TDD Plan

Write tests before implementation for:

- Model construction rejects incompatible metadata.
- Prefill produces expected logits or committed reference values.
- One-token decode produces expected logits or next token.
- Context-length overflow fails before compute.
- Shape and dtype mismatches fail explicitly.

## Design

M3 should use the runtime API, not a model-only shortcut. This keeps the path
useful for later GPU and KV work.

The first implementation can be slow and direct. Avoid optimizing kernels until
there is a stable parity baseline.

## Acceptance Criteria

- Qwen2.5-style dense decoder-only inference has explicit support.
- Prefill and decode are separate operations in the runtime path.
- Reference fixtures cover at least one short prompt.
- Greedy output is deterministic.
- Unsupported model metadata fails before execution.
- Tests document the reference source and tolerance.

## Validation Commands

```powershell
cargo test -p ocelotl-models
cargo test -p ocelotl-runtime
cargo test --workspace
```

## Known Risks

- Matching generated text without checking logits can hide numerical bugs.
- A model-only forward path that bypasses runtime will not validate request and
  cache lifecycle.
- The first supported model family may bias abstractions too early; keep generic
  abstractions minimal until a second family is implemented.
