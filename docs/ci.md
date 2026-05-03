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

## Offline By Default Across Milestones

The offline rule is enforced differently at different milestones. The principle
is: **the milestone that introduces network access owns the enforcement**.

- **M1 (CPU reference)**: offline by construction. No M1 task introduces
  network access — fixtures are committed under `fixtures/`, no model
  downloads happen, no HTTP clients are called. `cargo test --workspace`
  therefore proves the offline contract automatically. M1 does not add
  `--offline` flags to CI because there is nothing for them to enforce.
- **M2 (loader and tokenizer)**: first milestone that *can* reach the
  network. The `tokenizers` crate supports loading from HuggingFace Hub;
  loader tests must use committed fixtures. The M2 tasks are the right
  place to add `cargo test --workspace --offline` (or equivalent network
  blocking) to CI as enforcement. Until then, CI relies on the test
  authors to keep network access out of default tests.
- **Later milestones (M3+)**: should not regress the M2 offline
  enforcement. If a milestone needs network access (e.g. for an
  integration test against a real artifact), the test must be
  `#[ignore]` by default and runnable on demand with a documented
  command, per the Offline Rule above.

This split avoids two failure modes: enforcing `--offline` before any test
needs it (a hypothetical solution), and discovering after-the-fact that a
milestone silently introduced a network dependency (no enforcement).

## Initial GitHub Actions Shape

The initial workflow is intentionally minimal:

- checkout,
- install stable Rust,
- run fmt check,
- run workspace check,
- run workspace tests.

As fixtures and crates grow, add focused jobs only when they reduce feedback time
or isolate hardware requirements.
