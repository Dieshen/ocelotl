# M0 Tasks

M0 establishes the repository, publishing shape, documentation map, fixtures,
and default validation path. Most M0 work is already complete; this document
keeps the bootstrap criteria explicit for future cleanup.

## Entry Criteria

- A new repository exists for Ocelotl.
- The first public crate namespace has been chosen.

## Task List

- [x] M0.1 Scaffold the Rust workspace.
  - Crates: workspace, `ocelotl`, `ocelotl-core`, `ocelotl-loader`, `ocelotl-tokenizer`, `ocelotl-kernels`, `ocelotl-models`, `ocelotl-runtime`, `ocelotl-server`
  - Test first: `cargo check --workspace` fails until all packages are wired correctly.
  - Done when: every crate builds from the workspace root.

- [x] M0.2 Use short crate folders with publishable package names.
  - Crates: workspace metadata
  - Test first: `cargo metadata` shows packages named `ocelotl-*` while paths remain `crates/core`, `crates/runtime`, and so on.
  - Done when: package names and folder names are intentionally decoupled.

- [x] M0.3 Publish or reserve the crate namespace.
  - Crates: all public crates
  - Test first: `cargo publish --dry-run -p <crate>` succeeds for each package before publish.
  - Done when: the intended `ocelotl-*` crates and root `ocelotl` crate exist on crates.io.

- [x] M0.4 Establish the documentation tree.
  - Crates: docs only
  - Test first: a contributor can find roadmap, architecture, crate boundaries, design docs, validation docs, and publishing docs from the README.
  - Done when: README and `docs/start-here.md` link the core orientation documents.

- [x] M0.5 Add the fixture scaffold.
  - Crates: docs and fixtures
  - Test first: M1 and M2 docs can point to concrete fixture paths.
  - Done when: `fixtures/metadata`, `fixtures/tokenizer`, and `fixtures/logits` exist with placeholder or synthetic fixtures.

- [x] M0.6 Add baseline CI documentation and workflow.
  - Crates: workspace
  - Test first: CI commands match the local validation commands.
  - Done when: `.github/workflows/ci.yml` and `docs/ci.md` describe the offline default validation path.

- [ ] M0.7 Add package-level README stubs before expanding public APIs.
  - Crates: all public crates
  - Test first: `cargo package --list -p <crate>` includes useful crate README content or points to the root README.
  - Done when: crates.io package pages do not rely only on skeleton descriptions.

## Exit Criteria

- `cargo fmt --all --check` passes.
- `cargo check --workspace` passes.
- `cargo test --workspace` passes.
- README links the start-here guide, roadmap, task backlog, and validation policy.
- Crate names and folder names match the intended publishing strategy.

## Deferred

- Real model execution.
- GPU execution.
- Large fixture artifacts.
- API compatibility claims.
