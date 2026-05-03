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
| 2 | Unsupported features fail before execution begins. | Loader: `ocelotl_loader::tests::{load_metadata_rejects_unknown_architecture_with_typed_unsupported_error, load_metadata_rejects_unknown_dtype_with_typed_unsupported_error, load_metadata_rejects_missing_required_field_with_invalid_model_error}`. Runtime: `ocelotl_runtime::tests::{validate_request_rejects_empty_prompt, validate_request_rejects_zero_max_new_tokens, validate_request_rejects_temperature_with_unsupported_sampling_mode, validate_request_rejects_context_overflow, validate_request_temperature_check_fires_before_other_violations}`. | green |
| 3 | Prefill and one-token decode run through `ocelotl-runtime`. | `ocelotl_runtime::m1_cpu_reference_smoke_produces_expected_token` (integration test at `crates/runtime/tests/m1_smoke.rs`) — wires `validate_request` → `tiny_synthetic_forward` → `greedy_sample` and asserts a pinned next-token. | green |
| 4 | A fixture test validates deterministic output without network access. | Same M1.9 smoke test as criterion 3 — runs offline by construction (no `--ignored`, no network calls, no model downloads). Determinism also pinned by `ocelotl_models::tests::tiny_synthetic_forward_is_deterministic_for_identical_inputs`. | green |
| 5 | Output is compared against a documented reference or committed fixture. | M1.9 smoke test asserts `TokenId(5)` for prompt `[TokenId(7)]` against `fixtures/logits/m1_smoke_expected.json`. The fixture's `expected_next_token` field is the pinned reference; updates require regeneration when `tiny_synthetic_forward` intentionally changes. | green |
| 6 | Shape, dtype, and context-length errors are explicit. | Dtype: `ocelotl_loader::tests::load_metadata_rejects_unknown_dtype_with_typed_unsupported_error`. Context-length: `ocelotl_runtime::tests::{validate_request_rejects_context_overflow, validate_request_accepts_request_exactly_filling_context}` (boundary pinned). Shape: kernels rejection tests `ocelotl_kernels::tests::{vec_add_rejects_mismatched_input_lengths, vec_add_rejects_mismatched_output_length, dot_rejects_mismatched_lengths, matmul_rejects_inner_dimension_mismatch, matmul_rejects_wrong_a_slice_length, matmul_rejects_wrong_output_length}`. Display: `ocelotl_core::tests::{invalid_model_error_display_includes_path_field_and_message, unsupported_error_display_mentions_feature_requested_and_supported, kernel_error_display_includes_backend_and_message}`. | green |

### Closure note (2026-05-03)

All six rows are green. M1 acceptance is provable from `cargo test --workspace` against the commit set on main through `b7cf755 test(runtime): add M1 smoke integration test through public API`. Total at M1 close: 17 core + 15 kernels + 5 loader + 4 models + 12 runtime unit + 1 runtime smoke + 4 kernel doctests = 58 tests passing.

### Note — offline by construction

M1 default tests are offline by construction; the offline-by-default principle
and its forward-looking implications for milestones that introduce network
access live in `docs/ci.md`.
