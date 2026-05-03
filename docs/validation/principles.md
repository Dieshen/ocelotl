# Validation Principles

Cross-cutting principles that govern how Ocelotl validates correctness. These
are stable across milestones; specific test lists live in
`docs/validation/test-matrix.md` and per-milestone specs.

## Correctness Gates vs. Hygiene Gates

Validation commands fall into two categories. They are kept separate on
purpose: conflating them produces false alarms and wastes review attention.

### Correctness gates (block merges, define "milestone done")

- `cargo check --workspace` — proves the workspace compiles.
- `cargo test --workspace` — proves all default tests pass.

These gates answer: **does the code do what the spec says?** They are part of
every milestone's acceptance criteria. A failure here means the contract is
broken.

### Hygiene gates (run in pre-commit and CI, do not block correctness validation)

- `cargo fmt --all --check` — proves the code is formatted to project style.
- `cargo clippy` — proves the code passes lint policy.

These gates answer: **does the code conform to project style?** They are
enforced in pre-commit hooks and CI, but they are not part of any milestone's
acceptance criteria. A failure here is a style nit, not a broken contract.

### Why this matters

1. **Onboarding gates.** A new contributor running the day-one workspace-green
   check should see a green workspace whenever the merged code is correct,
   regardless of whether a teammate has unformatted in-flight edits on disk.
   Folding `cargo fmt --check` into the day-one gate creates a false alarm
   the first time another dev is mid-loop on a shared crate. Day-one runs
   correctness gates only.

2. **Milestone acceptance.** "M1 is done" should mean "the M1 contract holds,"
   not "the M1 contract holds and every line is also formatted." Hygiene
   problems are caught in CI on every PR; they do not need to be re-asserted
   at milestone-acceptance time.

3. **Future milestones.** When a milestone introduces a new hygiene gate (a
   stricter clippy lint, a new fmt rule), document it as hygiene. Do not fold
   it into the milestone's acceptance criteria. Acceptance criteria describe
   the *behavioral* contract; hygiene gates police *style*.

### What this is not

- Hygiene gates are not optional. Pre-commit and CI must enforce them.
- The split is not "correctness matters, hygiene doesn't." Both matter; they
  fail in different ways and live in different places.
- This is not an excuse to skip `cargo fmt` before a PR. Run it on a clean
  tree before pushing.

## Offline By Default

Default tests must not require network access. The full cross-milestone
policy and per-milestone enforcement responsibilities live in
`docs/ci.md` under "Offline By Default Across Milestones". The short version:
M1 is offline by construction; M2 owns the `--offline` CI enforcement when
it introduces network-capable code paths.

## Test-First

Every non-trivial change starts with a failing test. Policy and the seven-
step loop are documented in `docs/validation/tdd.md`.
