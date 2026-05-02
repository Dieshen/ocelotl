# CI Policy

CI enforces Ocelotl's offline-by-default validation rule. It should start small
and become stricter as milestones add behavior.

## Required PR Checks

Every pull request should run:

```powershell
cargo fmt --all --check
cargo check --workspace
cargo test --workspace
```

These checks must not require network access, model downloads, GPU hardware, or
large local artifacts.

## Test Classes

Default CI:

- formatting,
- workspace check,
- unit tests,
- fixture tests,
- unsupported-config tests,
- CPU/reference tests that use committed small fixtures.

Ignored or separate CI:

- GPU tests,
- benchmark tests,
- network-dependent model download tests,
- tests requiring large local artifacts.

## GPU CI

GPU CI should be added only after M4 introduces the first GPU kernel path. GPU
jobs should report hardware, driver, backend, and feature flags. GPU failures
should block GPU-default changes but should not be required for M1-M3 CPU-only
work.

## Offline Rule

If a test needs network access, mark it ignored by default and document the exact
command to run it. Do not let default `cargo test --workspace` fetch model files
or hit external APIs.

## Initial GitHub Actions Shape

The initial workflow is intentionally minimal:

- checkout,
- install stable Rust,
- run fmt check,
- run workspace check,
- run workspace tests.

As fixtures and crates grow, add focused jobs only when they reduce feedback time
or isolate hardware requirements.
