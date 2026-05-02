# Unsupported Configurations

Unsupported configurations must fail explicitly. Silent fallback is dangerous for
LLM runtimes because wrong output can look plausible.

## Required Failure Cases

The validation suite should cover unsupported cases for:

- Unknown architecture.
- Missing required tensors.
- Mismatched tensor shapes.
- Unsupported dtype.
- Unsupported quantization.
- Unsupported RoPE scaling.
- Unsupported attention layout.
- Context length beyond model limit.
- GPU requested but unavailable.
- Cache layout incompatible with model metadata.

## Error Quality

Errors should include enough information to act:

- What was requested.
- What is supported.
- Which artifact or model field caused the problem.
- Which crate produced the error where useful.

Errors should not include secrets, huge tensor dumps, or unnecessary internal
backtraces in normal user-facing output.

## Runtime Policy

The runtime should validate before launching compute. Kernel-level failures
should be reserved for bugs or hardware/runtime failures, not predictable model
compatibility issues.

## Regression Tests

Every time a new model feature is intentionally unsupported, add a regression
test that proves it fails clearly. When support is later added, update the test
into a positive coverage case.
