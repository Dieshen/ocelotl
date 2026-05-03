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
  loader tests must use committed fixtures. M2.8 enforces the offline
  contract via the **offline gate** (`ci/check-offline.ps1`, run as a
  step in the CI workflow before `cargo test --workspace`). The gate
  scans `crates/**/*.rs` and `crates/**/Cargo.toml` for known network-
  fetching APIs (`reqwest::`, `ureq::`, `hf_hub::`, `HfApi`,
  `.from_pretrained`, literal `huggingface.co` / `hf.co` URLs, and a
  list of forbidden network-client deps). Matches in production code
  fail the gate; matches inside `#[test]` functions fail unless the
  enclosing test is `#[ignore]`'d. See § Offline Gate below.
- **Later milestones (M3+)**: should not regress the M2 offline
  enforcement. If a milestone needs network access (e.g. for an
  integration test against a real artifact), the test must be
  `#[ignore]` by default and runnable on demand with a documented
  command, per the Offline Rule above.

This split avoids two failure modes: enforcing `--offline` before any test
needs it (a hypothetical solution), and discovering after-the-fact that a
milestone silently introduced a network dependency (no enforcement).

## Offline Gate

The offline gate is a static check that scans the workspace for code paths
that would let `cargo test --workspace` (without `--ignored`) reach the
network. It runs as a CI step *before* `cargo test --workspace` so a
violation is reported before tests execute.

**Where it lives:** `ci/check-offline.ps1` (a PowerShell 7+ script,
chosen because the CI runner is `windows-latest`).

**What it scans for:**

- HTTP-client crates: `reqwest::`, `ureq::`, `isahc::`, `surf::`,
  `attohttpc::`, `hyper::Client` (in `crates/**/*.rs`).
- HuggingFace Hub fetchers: `hf_hub::`, `HfApi`, and the
  `.from_pretrained` helper that the `tokenizers` crate exposes.
- Literal URLs that point at the model host: `https://huggingface.co/...`,
  `https://hf.co/...`.
- The same set of crate names as `[dependencies]` entries in any
  `crates/**/Cargo.toml`.

**What it permits:**

- Any of the above patterns inside a function annotated with `#[test]`
  AND `#[ignore = "..."]`. The `#[ignore]` attribute may sit either
  immediately above `#[test]` or immediately below it; the gate accepts
  both idiomatic positions.
- Doc comments (`///` and `//!` lines) — they describe behavior, they
  don't invoke it.

**How to add a legitimate exception** (the only supported way): mark the
test `#[ignore = "<remediation message>"]` per
`docs/artifact-preparation.md` § 5. Default `cargo test --workspace`
will skip it; a contributor who has fetched the artifacts can opt in
with `cargo test --workspace -- --ignored`. The canonical example is
`crates/tokenizer/tests/qwen2_5_basic_prompt.rs`
(`json_tokenizer_round_trips_qwen2_5_basic_prompt`, M2.3).

**Honest limitations:**

- The gate is greppable. A determined contributor could rename a network
  call or use a transitive dep to bypass it. The gate's job is catching
  *accidents*, not adversaries — paired with `#[ignore]` discipline and
  reviewer attention, that's enough for now.
- A future tightening would add a sandbox CI step that runs `cargo test
  --workspace` inside a network-disabled container (e.g.
  `--network none`); that is the authoritative check. The gate here
  gives fast local feedback (runs in milliseconds) while the sandbox
  step would be the slower belt to its suspenders.

**Running the gate locally:**

```powershell
./ci/check-offline.ps1
```

Exit 0 = clean; exit 1 = violations printed to stderr with file/line
references and a remediation hint.

## Initial GitHub Actions Shape

The initial workflow is intentionally minimal:

- checkout,
- install stable Rust,
- run fmt check,
- run workspace check,
- run the offline gate (M2.8),
- run workspace tests.

As fixtures and crates grow, add focused jobs only when they reduce feedback time
or isolate hardware requirements.
