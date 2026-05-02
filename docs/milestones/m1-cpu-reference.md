# M1 CPU Reference

## Goal

Build the first deterministic CPU reference path for one narrow decoder-only
model shape. The purpose is correctness, not speed.

## Target Model Shape

M1 targets a tiny synthetic Qwen2.5-shaped dense decoder-only model. It does not
load the full real model, but it should use metadata categories compatible with
the first real artifact candidate in `docs/model-target.md`.

## Non-Goals

- GPU execution.
- Quantized weights.
- Paged KV.
- Continuous batching.
- Multiple model families.
- OpenAI-compatible serving.

## Design

M1 should establish a minimal path through the same high-level runtime API that
future GPU execution will use:

1. Load or construct Qwen2.5-shaped synthetic metadata.
2. Tokenize or inject a fixed prompt fixture.
3. Run prefill.
4. Run one decode step.
5. Produce logits or next-token output that can be compared against a reference.

The CPU reference may be slow and straightforward. It should favor readable code,
explicit shape checks, and deterministic tests over kernel-level performance.

## Acceptance Criteria

- One supported model shape is declared explicitly: tiny synthetic Qwen2.5-shaped decoder-only metadata.
- Unsupported features fail before execution begins.
- Prefill and one-token decode run through `ocelotl-runtime`.
- A fixture test validates deterministic output without network access.
- Output is compared against a documented reference or committed fixture.
- Shape, dtype, and context-length errors are explicit.

## Validation Commands

```powershell
cargo test --workspace
cargo check --workspace
```

## Known Risks

- A CPU path that diverges from the intended runtime API will not help later GPU
  work.
- Reference fixtures can become misleading if tokenizer or chat-template behavior
  is not pinned.
- Model-specific assumptions should be visible in code and docs, not hidden in
  generic helpers.
