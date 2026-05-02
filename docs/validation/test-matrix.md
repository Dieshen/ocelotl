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
