# M1 Tasks

M1 builds the deterministic CPU reference path for a tiny Qwen2.5-shaped model.
The purpose is correctness and contracts, not speed.

## Entry Criteria

- M0 exit criteria are met.
- `docs/model-target.md` names the provisional first model family.
- Synthetic metadata fixtures exist under `fixtures/metadata/`.

## Task List

- [x] M1.1 Define the shared error taxonomy.
  - Crates: `ocelotl-core`
  - Test first: add tests that construct each core error category and verify stable `Display` output and source preservation where applicable.
  - Done when: `ocelotl_core::Error` and `ocelotl_core::Result<T>` cover invalid requests, unsupported configs, artifact errors, tokenizer errors, kernel errors, runtime errors, and internal invariants.

- [x] M1.2 Define typed scalar and ID newtypes.
  - Crates: `ocelotl-core`
  - Test first: add compile-time and runtime tests for `TokenId`, sequence length, batch size, and context length bounds.
  - Done when: public APIs avoid raw `u32` or `usize` for token and request-domain values where a domain type is clearer.

- [x] M1.3 Define typed model metadata structs.
  - Crates: `ocelotl-core`
  - Test first: add a test that deserializes `fixtures/metadata/qwen2_5_tiny_synthetic.json` into a typed metadata object.
  - Done when: metadata includes architecture, vocab size, layer count, hidden size, head counts, KV head counts, intermediate size, max position embeddings, RoPE settings, dtype, and tokenizer model hint.

- [x] M1.4 Reject unsupported metadata explicitly.
  - Crates: `ocelotl-core`, `ocelotl-loader`
  - Test first: load `fixtures/metadata/unsupported_unknown_architecture.json` and assert an unsupported-architecture error, not a generic parse failure.
  - Done when: unsupported architecture, unsupported dtype, and missing required fields produce typed errors.

- [x] M1.5 Add metadata fixture loading for tests.
  - Crates: `ocelotl-loader`, integration tests
  - Test first: add a fixture-path helper test that fails if the metadata fixture cannot be located from the workspace root.
  - Done when: tests use shared fixture helpers instead of duplicating relative path logic.

- [x] M1.6 Sketch the minimal generation request contract.
  - Crates: `ocelotl-core`, `ocelotl-runtime`
  - Test first: construct a request with prompt token IDs and deterministic generation options, then validate it without running a model.
  - Done when: runtime input validation checks empty prompts, context overflow, unsupported sampling options, and max-new-token bounds.

- [x] M1.7 Add CPU reference tensor primitives for tiny fixtures.
  - Crates: `ocelotl-kernels`
  - Test first: add small hand-checked tests for vector add, dot product, softmax, and simple matrix multiplication using fixed arrays.
  - Done when: CPU reference kernels are deterministic, documented as reference-only, and do not expose GPU-specific assumptions.

- [x] M1.8 Add deterministic sampling for greedy output.
  - Crates: `ocelotl-runtime`
  - Test first: feed fixed logits and assert the selected token ID under greedy settings.
  - Done when: greedy sampling is implemented with deterministic tie behavior and explicit rejection of unsupported sampling modes.

- [x] M1.9 Wire a tiny CPU reference generation path.
  - Crates: `ocelotl-models`, `ocelotl-runtime`, `ocelotl-kernels`
  - Test first: add a smoke test that sends a fixed token prompt through a tiny synthetic model and expects a fixed next token.
  - Done when: request validation, model metadata, CPU kernels, sampling, and runtime response types are exercised through public APIs.

- [x] M1.10 Document and enforce M1 validation.
  - Crates: docs and workspace
  - Test first: add or update validation docs before broadening implementation.
  - Done when: M1 acceptance commands are listed and `cargo test --workspace` covers the CPU reference path offline.

## Closure (2026-05-03)

All M1 tasks landed and exit criteria are met. Acceptance is provable from
`cargo test --workspace` against main through commit
`b7cf755 test(runtime): add M1 smoke integration test through public API`
(58 tests passing: 17 core + 15 kernels + 5 loader + 4 models + 12 runtime
unit + 1 runtime smoke + 4 kernel doctests). The per-criterion mapping lives
in `docs/validation/test-matrix.md` (M1 Acceptance Traceability) — that table
is the authoritative status surface for M1; this checklist mirrors it.

## Exit Criteria

- A tiny Qwen2.5-shaped metadata fixture loads into typed structs.
- Unsupported metadata fails with typed errors.
- A deterministic CPU reference path can produce one expected token from a synthetic model fixture.
- Runtime request validation happens before compute.
- Default workspace tests stay offline.

## Deferred

- Real safetensors weight loading.
- Real tokenizer execution.
- GPU kernels.
- KV cache optimization.
- Multi-request scheduling.
