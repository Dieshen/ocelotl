# Correctness Validation

Ocelotl should treat correctness as a release gate. A model path is not correct
because it runs; it is correct when it matches a reference within documented
tolerance and fails clearly outside its supported contract.

## Validation Layers

1. Shape and metadata validation.
2. Tokenizer and chat-template validation.
3. CPU reference execution.
4. CPU versus reference logits or tokens.
5. GPU versus CPU parity.
6. Cache layout equivalence.
7. End-to-end generation fixtures.

## Fixture Policy

Fixtures should be small, deterministic, and committed when licensing permits.
Network access should not be required for normal correctness tests.

Each fixture should document:

- Model family.
- Artifact source.
- Tokenizer source.
- Prompt.
- Expected tokens or logits.
- Tolerance.
- Reason the fixture exists.

## Tolerances

Tolerances must be explicit per test category. Examples:

- Exact token match for deterministic greedy generation.
- Absolute or relative tolerance for logits.
- Wider tolerance for lower-precision GPU kernels when justified.

A tolerance increase is a behavior change and should be reviewed like one.

## Unsupported Configs

Unsupported features should produce explicit errors, not fallback behavior. Tests
should cover common unsupported cases:

- Unknown architecture.
- Unsupported quantization.
- Unsupported RoPE scaling.
- Unsupported attention head layout.
- Context length beyond model limit.
- Missing tokenizer or chat template.

## GPU Parity

Every GPU kernel path needs a CPU parity test before it becomes part of the
runtime default. GPU tests should compare both simple first-page behavior and
multi-page/cache-boundary behavior when KV paging exists.

## Release Gate

A release that adds a model family, kernel path, quantization format, or cache
layout should include a validation note listing the exact tests and fixtures that
prove the path is safe to expose.

## Malformed Artifact Coverage (M2.7)

This inventory maps each malformed-artifact failure mode named in the M2.7 task
spec to the test that pins the typed-error contract for it. Use this as the
discovery surface when adding a new failure mode (extend the table) or when an
error category changes (find every test that asserts the old shape).

All listed tests are unit tests that build their fixture programmatically — no
binary fixtures are committed under `fixtures/safetensors/`. The header-only
inspection contract lets these stay byte-exact and deterministic without a
real model artifact.

| Failure mode                          | Test                                                                                                          | Crate     | Typed error                  | Commit    |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------- | --------- | ---------------------------- | --------- |
| Missing tokenizer file                | `tokenizer::tests::json_tokenizer_missing_file_returns_typed_tokenizer_error_with_path`                       | tokenizer | `OcelotlError::Tokenizer`    | `2761ed9` |
| Malformed tokenizer JSON              | `tokenizer::tests::json_tokenizer_malformed_json_returns_typed_tokenizer_error_with_path`                     | tokenizer | `OcelotlError::Tokenizer`    | `8180173` |
| Missing required tensor               | `safetensors_inspect::tests::require_tensors_returns_invalid_model_error_when_a_required_tensor_is_missing`   | loader    | `OcelotlError::InvalidModel` | `06a9c3e` |
| Unsupported safetensors dtype         | `safetensors_inspect::tests::inspect_safetensors_rejects_unsupported_dtype_with_typed_unsupported_error`      | loader    | `OcelotlError::Unsupported`  | `79481c0` |
| Truncated safetensors header          | `safetensors_inspect::tests::inspect_safetensors_rejects_truncated_header_with_invalid_model_error`           | loader    | `OcelotlError::InvalidModel` | `e9c02ca` |
| Safetensors shape vs offsets mismatch | `safetensors_inspect::tests::inspect_safetensors_rejects_shape_offsets_mismatch_with_invalid_model_error`     | loader    | `OcelotlError::InvalidModel` | `19897f7` |

### Notes

- **Truncated header** is *not* the same case as **shape vs offsets mismatch**:
  the truncated case is a header-vs-file-size disagreement (file ends before
  the claimed header does), while the shape-mismatch case is a
  header-vs-header inconsistency (declared shape times dtype size disagrees
  with declared `data_offsets` byte length, but the file itself matches what
  the header claims).
- Both safetensors header-level inconsistencies are detected inside the
  third-party `safetensors` crate's `read_metadata` and surface through the
  loader wrapper as `OcelotlError::InvalidModel`. The shape-mismatch test is
  a *behavioral pin*, not a feature add — it ensures a future safetensors
  upgrade or wrapper refactor cannot silently regress the contract.
- The dtype case maps to `Unsupported` (not `InvalidModel`) because the
  artifact is well-formed but uses a dtype outside Ocelotl's declared
  supported subset (`F32`, `F16`, `BF16`); see
  `crates/loader/src/safetensors_inspect.rs::SupportedDtype`.
