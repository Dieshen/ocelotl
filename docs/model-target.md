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

## Pinned Revision

M2 implementation must reference an exact artifact identity, not a moving branch
name. The pin below is the source of truth that other docs and fixtures point at;
the same fields appear in machine-readable form at
`fixtures/manifest/qwen2_5_0_5b_instruct.json`.

- Repository: `Qwen/Qwen2.5-0.5B-Instruct`
- Revision (commit SHA on Hugging Face `main`): `7ae557604adf67be50417f59c2c2f167def9a775`
- Repository URL at pinned revision: <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775>
- License: Apache-2.0 (per Hugging Face repo metadata at pinned revision)
- Date pinned: 2026-05-03
- Pinned by: M2.1 (see `docs/tasks/m2-loader-tokenizer.md`)
- Scope of pin: tokenizer fixtures (M2.2, M2.3), chat-template fixtures (M2.4),
  safetensors metadata fixtures (M2.5). No model weights are committed; see
  `docs/validation/fixtures.md` for the storage policy.

If upstream re-tags or rewrites history under this SHA, treat that as a
breaking-change event and bump this section in a single dedicated commit so
`git log -- docs/model-target.md` records every revision change.

The exact local files contributors must place under their working copy to
exercise tests against this revision will be documented by M2.9 (see
`docs/tasks/m2-loader-tokenizer.md`).
