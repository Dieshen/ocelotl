# Test Matrix

This matrix maps milestones to required tests. Each milestone spec can add more,
but it should not provide less than this baseline.

| Milestone | Required Tests |
| --- | --- |
| M0 Skeleton | Workspace check, formatting, publish dry runs. |
| M1 CPU Reference | Unit tests for request validation, fixture test for deterministic CPU output, unsupported-config tests. |
| M2 Loader And Tokenizer | Loader good/bad fixtures, exact tokenizer ID fixtures, chat-template fixtures. |
| M3 Single Model Forward | Prefill logits parity, one-token decode parity, shape/dtype failure tests. |
| M4 GPU Kernel Path | CPU/GPU parity for each GPU kernel, unsupported device/dtype tests. |
| M5 Contiguous KV Cache | KV read/write position tests, prefill/decode cache parity, request isolation tests. |
| M6 Paged KV Cache | Page allocation tests, multi-page decode test, contiguous/paged parity. |
| M7 Continuous Batching | Scheduler ordering tests, cancellation tests, batched/unbatched parity. |
| M8 Server API | API request validation, runtime error mapping, streaming lifecycle tests. |

## Validation Tiers

Focused commands should be used while developing. Workspace commands are required
before merging.

Focused examples:

```powershell
cargo test -p ocelotl-loader
cargo test -p ocelotl-tokenizer
cargo test -p ocelotl-runtime
```

Workspace gate:

```powershell
cargo fmt --all
cargo test --workspace
cargo check --workspace
```

## Offline Rule

Default tests must not require network access. Any network-dependent test should
be ignored by default and documented with the exact command to run it.

## M1 Acceptance Traceability

The table below maps each acceptance criterion in
`docs/milestones/m1-cpu-reference.md` to the test (or test set) that proves it.
Reviewers should be able to read this table and confirm every M1 acceptance
bullet has a green test in `cargo test --workspace`. As tests land, fill in the
`status` column; placeholders cite the task that will land the test.

| # | Acceptance criterion | Test(s) proving it | Status |
| - | -------------------- | ------------------ | ------ |
| 1 | One supported model shape declared explicitly: tiny synthetic Qwen2.5-shaped decoder-only metadata loads into typed structs. | `ocelotl_core::tests::qwen2_5_tiny_synthetic_fixture_deserializes_correctly` (in `crates/core/src/lib.rs`). | green |
| 2 | Unsupported features fail before execution begins. | `ocelotl_loader::tests::load_metadata_rejects_unknown_architecture_with_typed_unsupported_error`, `ocelotl_loader::tests::load_metadata_rejects_unknown_dtype_with_typed_unsupported_error` (in `crates/loader/src/lib.rs`); plus M1.4 missing-field test (TBD), and M1.6 runtime request-validation tests in `crates/runtime/src/lib.rs` (TBD: empty prompt, context overflow, unsupported sampling option, max-new-token bounds). | partial |
| 3 | Prefill and one-token decode run through `ocelotl-runtime`. | M1.9 smoke test wiring tiny synthetic model through `Runtime::generate` (TBD: lands in `crates/runtime` integration tests, refs M1.9). | not yet landed |
| 4 | A fixture test validates deterministic output without network access. | Same M1.9 smoke test as criterion 3 — runs offline by construction (no `--ignored`, no network calls, no model downloads). | not yet landed |
| 5 | Output is compared against a documented reference or committed fixture. | M1.9 smoke test asserts the next token against a committed fixture under `fixtures/logits/` (currently a `README.md` placeholder; populated by M1.9). | not yet landed |
| 6 | Shape, dtype, and context-length errors are explicit. | dtype: `ocelotl_loader::tests::load_metadata_rejects_unknown_dtype_with_typed_unsupported_error`; context-length: M1.6 runtime validation tests (TBD); shape: M1.7 kernel shape-mismatch tests in `crates/kernels` (TBD); core display tests for `OcelotlError::InvalidModel` and `OcelotlError::Unsupported` (in `crates/core/src/lib.rs`) cover the error-rendering contract. | partial |

### Note — offline by construction

M1 default tests are offline by construction; the offline-by-default principle
and its forward-looking implications for milestones that introduce network
access live in `docs/ci.md`.
