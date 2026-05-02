# Initial Model Target

Ocelotl needs a named first model target before M1 starts. The provisional target
is the Qwen2.5 dense decoder-only family, using `Qwen/Qwen2.5-0.5B-Instruct` as
the first real artifact candidate for M2/M3 fixtures.

## Decision

- Family: Qwen2.5-style dense decoder-only transformer.
- First artifact candidate: `Qwen/Qwen2.5-0.5B-Instruct`.
- License: Apache-2.0 according to the Hugging Face model metadata surfaced in
  search results on 2026-05-02.
- Format fit: Hugging Face repository exposes safetensors and `tokenizer.json`.
- Scope: text-only causal language modeling.

## Why This Target

Qwen2.5-0.5B-Instruct is small enough to be practical for early local fixtures
and CPU/reference work, while still forcing Ocelotl to handle real LLM concerns:

- tokenizer JSON behavior,
- chat-template behavior,
- RoPE metadata,
- grouped-query style attention metadata,
- safetensors weight naming,
- context-length validation,
- decoder-only prefill/decode semantics.

It is also permissively licensed, which reduces friction for examples and local
validation compared with gated or more restrictive model families.

## What This Does Not Mean

This does not mean Ocelotl becomes Qwen-specific. It means the first vertical
slice has a concrete target. Abstractions should stay honest: implement the
Qwen2.5 path directly where needed, then generalize only when a second model
family proves the abstraction.

## M1 Synthetic Target

M1 does not need to load the full model. It should use a tiny Qwen2.5-shaped
synthetic fixture that exercises the same metadata categories:

- architecture = `qwen2`
- decoder-only
- one or two layers
- small hidden size
- explicit attention and KV-head metadata
- explicit RoPE metadata
- tokenizer fixture with exact token IDs

## M2/M3 Real Artifact Candidate

M2 should add local metadata/tokenizer fixtures derived from the real artifact.
M3 can add a small reference-output fixture if licensing and size constraints are
acceptable. Large model weights should not be committed to the repository.

## Revalidation Rule

Before implementation depends on this target, re-check the live Hugging Face
repository and pin an exact revision in the fixture metadata. This document is a
planning decision, not a permanent guarantee that upstream files will not move.
