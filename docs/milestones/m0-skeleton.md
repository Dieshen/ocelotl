# M0 Skeleton

## Goal

Create a publishable, compileable workspace that establishes the project name,
crate boundaries, initial public API placeholders, and documentation structure.

## Non-Goals

- Real model loading.
- Real tokenization.
- Real generation.
- GPU kernel integration.
- Server endpoints.

## Design

The workspace is split into focused crates with short local folder names and
published `ocelotl-*` package names. The root crate exposes a small versioned
entrypoint while implementation crates remain intentionally minimal.

## Acceptance Criteria

- Workspace compiles with `cargo check --workspace`.
- Workspace formats with `cargo fmt --all`.
- Crate names and repository metadata are publish-ready.
- Docs explain the initial crate boundaries and milestone sequence.
- Placeholder APIs fail explicitly for unimplemented behavior.

## Validation Commands

```powershell
cargo fmt --all
cargo check --workspace
```

## Known Risks

- Placeholder crates can look like name reservations if development stalls.
- Public API names may need revision once M1 and M2 reveal the real runtime
  shape.
