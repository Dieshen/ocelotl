# Start Here

This guide is the contributor entry point for Ocelotl.

## 1. Understand The Project Shape

Read these first:

1. `docs/status.md`
2. `docs/overview.md`
3. `docs/architecture.md`
4. `docs/crate-boundaries.md`
5. `docs/roadmap.md`
6. `docs/tasks/README.md`
7. `docs/model-target.md`
8. `docs/validation/tdd.md`
9. `docs/ci.md`
10. `docs/artifact-preparation.md` (only when a task needs real local model files; default tests are offline)

The short version: Ocelotl is a Rust-first LLM inference runtime. The project is
correctness-first and test-driven. CPU/reference behavior comes before GPU;
contiguous KV comes before paged KV; one request comes before scheduling.
The active release track is local alpha hardening for Whisper and Gemma4 text;
M8 network-server work has not started.

## 2. Validate The Workspace

From the repository root:

```powershell
pwsh -NoProfile -File tools/verify.ps1 -Mode Fast
pwsh -NoProfile -File tools/verify.ps1 -Mode Full
```

Use `Fast` while editing and `Full` before merge. Both use the committed
lockfile; default tests do not require model downloads or network access. CI
runs `Full` plus separate MSRV and dependency-audit jobs; see `docs/ci.md`.

## 3. Pick Work From A Milestone

Read `docs/status.md` first to distinguish closed, active, and blocked work.
Then start with the relevant milestone/track spec under `docs/milestones/` and
use the matching execution backlog under `docs/tasks/`. Each milestone spec has:

- Goal.
- Non-goals.
- TDD plan.
- Design notes.
- Acceptance criteria.
- Validation commands.
- Known risks.

Do not start implementation from the roadmap summary alone. Use the milestone
spec for design intent and the task backlog for the next test-first slice.

## 4. Follow The TDD Loop

For non-trivial changes:

1. Write or update the relevant design/milestone doc.
2. Add the smallest failing test.
3. Confirm it fails for the expected reason.
4. Implement the smallest correct change.
5. Re-run the focused test.
6. Run the relevant crate tests.
7. Run workspace validation.

## 5. Respect Crate Boundaries

If code feels convenient but crosses a `must not` rule in
`docs/crate-boundaries.md`, stop and update the design first. Boundary drift is a
bug in this project.

## 6. Use Library Docs Before Adding Dependencies

Before adding a dependency, check `docs/libraries/` and current upstream docs.
The project rule is: do not add a library until a failing test needs it and the
owning crate boundary is clear.

## 7. Error Behavior Matters

Read `docs/design/errors.md` before adding new errors. Ocelotl should fail early
and explicitly for unsupported model features, invalid requests, invalid model
artifacts, and unsupported kernel layouts.

## Useful Commands

```powershell
# Workspace health
pwsh -NoProfile -File tools/verify.ps1 -Mode Fast
pwsh -NoProfile -File tools/verify.ps1 -Mode Full

# Focused crates
cargo test -p ocelotl-core --locked
cargo test -p ocelotl-loader --locked
cargo test -p ocelotl-tokenizer --locked
cargo test -p ocelotl-kernels --locked
cargo test -p ocelotl-models --locked
cargo test -p ocelotl-runtime --locked
cargo test -p ocelotl-server --locked
```

## First Good Contribution

For the current alpha-hardening track, a good contribution is usually one of:

- A failing boundary or malformed-input fixture tied to an alpha release gate.
- A focused Whisper or Gemma4 parity discriminator with a pinned reference.
- A typed error or resource-budget improvement with regression tests.
- A build/benchmark reproducibility improvement that keeps default tests
  offline.

M8 server tasks are a separate milestone. Do not add an ad hoc transport layer
while alpha work still targets the trusted-local runtime/CLI surface.
