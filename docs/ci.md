# CI Policy

CI enforces Ocelotl's deterministic, offline-by-default validation contract.
The current release posture is tracked in `docs/status.md`.

## Local Verification Entry Point

Use the repository-owned PowerShell entry point instead of maintaining a
personal copy of the gate commands:

```powershell
# Short edit loop: format, default all-target check, offline policy.
pwsh -NoProfile -File tools/verify.ps1 -Mode Fast

# Pre-merge gate: Fast plus default tests, all-feature no-launch check, clippy.
pwsh -NoProfile -File tools/verify.ps1 -Mode Full
```

`Full` is the required pull-request gate. It runs:

```powershell
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked
pwsh -NoProfile -File ci/check-offline.ps1
cargo test --workspace --locked
cargo check --workspace --all-targets --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
```

The all-feature commands compile the CubeCL/WGPU surface without launching a
real GPU proof. Hardware and local-artifact tests remain explicit opt-in work.

## Reproducible Toolchain And Dependencies

- `rust-toolchain.toml` pins the development toolchain, rustfmt, and clippy.
- `Cargo.toml` declares Rust 1.85 as the minimum supported Rust version (MSRV).
- `Cargo.lock` is committed because the workspace ships the `ocelotl` binary.
- Build, test, clippy, benchmark, and release commands use `--locked`.

Update the development toolchain and lockfile intentionally. A toolchain update
must run `Full`; a dependency update must run `Full`, the Rust 1.85 checks, and
`cargo audit`. Do not make a benchmark comparison across different lockfiles or
toolchains without labeling it as a separate environment.

## Required GitHub Checks

The workflow separates failures by contract and runs independent jobs in
parallel:

| Job | Contract |
| --- | --- |
| Full validation | Runs `tools/verify.ps1 -Mode Full` on Windows with the pinned development toolchain. |
| MSRV | Runs default all-target check and workspace tests with Rust 1.85.0. `RUSTUP_TOOLCHAIN` overrides the repository development pin for this job only. |
| Dependency audit | Audits the committed lockfile against RustSec advisories. Vulnerability findings fail the job; informational warnings must still be reviewed. |

Top-level workflow permissions are `contents: read`. The audit job alone adds
`checks: write` so its result can be reported as a GitHub check. Checkout does
not persist credentials.

The workflow cancels superseded runs for the same branch or pull request. Rust
build outputs are restored through separate development/MSRV cache keys. A
cache hit is only an acceleration: no correctness or benchmark claim may depend
on a warm cache.

All third-party actions are pinned to exact upstream commits with the verified
release/version in an adjacent comment. When updating an action, resolve the
official upstream tag to a commit and change the SHA and comment together.

## Test Classes

Default CI includes:

- formatting,
- default-feature all-target compilation,
- unit, integration, fixture, and doctests,
- unsupported-configuration and malformed-artifact tests,
- CPU/reference tests using committed small fixtures,
- all-feature/all-target no-launch compilation,
- clippy with warnings denied,
- the offline policy gate,
- MSRV and dependency-audit jobs.

Separate or ignored validation includes:

- real GPU launch and CPU/GPU execution parity,
- performance benchmarks,
- network-dependent acquisition,
- tests requiring large or license-bearing local artifacts,
- tests requiring external reference binaries such as llama.cpp or
  whisper.cpp.

An ignored test must name its prerequisites and exact local command. Default
`cargo test --workspace --locked` must never download artifacts or call an
external API.

## Offline By Default Across Milestones

The offline rule is enforced differently at different milestones. The
principle is: the milestone that introduces network access owns its enforcement.

- **M1 (CPU reference):** offline by construction. Fixtures are committed and
  no model download or HTTP client is part of the runtime.
- **M2 (loader and tokenizer):** introduced APIs whose upstream libraries can
  fetch artifacts. M2.8 added `ci/check-offline.ps1` to reject accidental
  network clients or model-host calls in the default surface.
- **Later milestones:** real artifacts and reference binaries remain local and
  opt-in. Network-dependent tests are ignored by default and document how the
  contributor acquires a pinned artifact separately.

`--locked` and the offline policy solve different problems. `--locked` prevents
dependency resolution drift. The offline gate prevents Ocelotl code and default
tests from initiating model/network fetches. Cargo may still need registry
access on a clean machine to obtain the exact packages named by the committed
lockfile.

## Offline Gate

`ci/check-offline.ps1` is a static check that runs before default tests.

It scans:

- the root `Cargo.toml` and every manifest under `crates/`,
- root `src/`, `tests/`, `benches/`, `examples/`, and `build.rs` when present,
- every Rust source under `crates/`, including crate tests, examples, benches,
  and build scripts.

The root package is a workspace member, so it receives the same policy as the
named crates. Adding a forbidden dependency or call to the root CLI/tests must
fail the gate.

The gate rejects:

- HTTP clients: `reqwest`, `ureq`, `isahc`, `surf`, `attohttpc`, and
  `hyper::Client`,
- Hugging Face clients: `hf_hub`, `huggingface_hub`, `HfApi`, and
  `.from_pretrained`,
- literal `huggingface.co` and `hf.co` URLs in executable code,
- the matching network-client dependencies in workspace manifests.

It permits those source patterns only inside a `#[test]` with an adjacent
`#[ignore = "..."]` attribute. Doc comments are permitted because they do not
execute.

Run it directly with:

```powershell
pwsh -NoProfile -File ci/check-offline.ps1
```

Exit 0 means no known network-fetching pattern was found in the default
surface. Exit 1 prints file/line violations. Exit 2 means repository discovery
or script invocation failed.

The gate is intentionally greppable and catches accidents, not adversarial
evasion. A future stronger layer may run tests inside a network-disabled
container. Until then, reviewers must still inspect new dependencies, build
scripts, proc-macro dependencies, and indirect process launches.

## GPU And Local-Artifact Validation

Default/all-feature CI proves feature compilation without requiring a GPU.
Ignored GPU execution tests remain local until controlled hardware runners are
available. A future GPU job must record runner identity, adapter, driver,
backend, feature flags, and parity tolerance before it can block a release.

Whisper and Gemma4 local-artifact proofs remain ignored because the repository
does not commit model weights or external reference executables. An alpha
release record must capture their exact artifact and reference revisions, but
the normal public-runner CI workflow does not claim to rerun them.

## Benchmark Jobs

Performance is not a normal GitHub-hosted-runner gate. Hosted-runner variance,
unknown CPU placement, absent GPU identity, and cold/warm cache differences make
those results unsuitable for regression thresholds.

When controlled benchmark jobs are added, they must:

- run on named, stable hardware,
- use the committed lockfile and pinned toolchain,
- preserve raw warmup and measured samples,
- record effective threads/backend/model/artifact hashes,
- validate output equivalence before comparing speed,
- report robust summary statistics and the commit's dirty state,
- keep external-baseline comparisons separate from internal regression gates.

Today the repository documents benchmark commands and local records, but it
does not claim that a controlled benchmark CI gate exists.
