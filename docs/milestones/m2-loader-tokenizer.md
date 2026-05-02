# M2 Loader And Tokenizer

## Goal

Load model metadata and tokenizer behavior from local fixtures or artifacts, then
normalize both into Ocelotl-owned contracts.

## Target Artifact

M2 should use fixtures derived from a pinned `Qwen/Qwen2.5-0.5B-Instruct`
revision when real tokenizer or metadata behavior is needed. Large model weights
must not be committed.

## Non-Goals

- Full model execution.
- GPU kernels.
- Quantized execution.
- Paged KV.
- Continuous batching.
- Network-dependent model downloads in default tests.

## TDD Plan

Write tests before implementation for:

- Known-good Qwen2.5-shaped metadata fixture parses into exact normalized fields.
- Missing required metadata fails with an explicit error.
- Unsupported architecture fails with an explicit error.
- Exact token IDs for a short prompt.
- Exact chat-template rendering for one structured message fixture.

## Design

The loader and tokenizer should remain separate. The loader can discover or point
to tokenizer assets, but tokenizer behavior belongs in `ocelotl-tokenizer`.

M2 should prefer small fixtures over real large models. If real model artifacts
are introduced, default tests should still run offline without downloading them.

## Acceptance Criteria

- `ocelotl-loader` exposes normalized metadata for at least one supported Qwen2.5-shaped model fixture.
- `ocelotl-loader` rejects malformed metadata fixtures.
- `ocelotl-tokenizer` encodes and decodes known fixtures exactly.
- Chat-template behavior is covered by a deterministic test.
- Runtime-facing metadata types contain enough fields for M3 model construction.
- Default tests do not require network access.

## Validation Commands

```powershell
cargo test -p ocelotl-loader
cargo test -p ocelotl-tokenizer
cargo test --workspace
cargo check --workspace
```

## Known Risks

- Loader metadata that hides model-family details will force runtime rewrites.
- Tokenizer tests that only round-trip text can miss BOS/EOS and template bugs.
- Large fixtures can make the normal test suite too slow or legally awkward.
