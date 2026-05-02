# Start Here

This guide is the contributor entry point for Ocelotl.

## 1. Understand The Project Shape

Read these first:

1. `docs/overview.md`
2. `docs/architecture.md`
3. `docs/crate-boundaries.md`
4. `docs/roadmap.md`
5. `docs/tasks/README.md`
6. `docs/model-target.md`
7. `docs/validation/tdd.md`
8. `docs/ci.md`

The short version: Ocelotl is a Rust-first LLM inference runtime. The project is
correctness-first and test-driven. CPU/reference behavior comes before GPU;
contiguous KV comes before paged KV; one request comes before scheduling.

## 2. Validate The Workspace

From the repository root:

```powershell
cargo fmt --all
cargo check --workspace
cargo test --workspace
```

Default tests should not require network access. CI runs the same baseline
commands; see `docs/ci.md`.

## 3. Pick Work From A Milestone

Start with the current milestone spec under `docs/milestones/`, then use the
matching execution backlog under `docs/tasks/`. Each milestone spec has:

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
cargo fmt --all
cargo check --workspace
cargo test --workspace

# Focused crates
cargo test -p ocelotl-core
cargo test -p ocelotl-loader
cargo test -p ocelotl-tokenizer
cargo test -p ocelotl-kernels
cargo test -p ocelotl-models
cargo test -p ocelotl-runtime
cargo test -p ocelotl-server
```

## First Good Contribution

For early work, a good contribution is usually one of:

- A fixture and failing test for M1 or M2.
- A typed error improvement with tests.
- A small interface refinement that makes crate boundaries clearer.
- A doc update that removes ambiguity before implementation.
