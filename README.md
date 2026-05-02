# Ocelotl

Rust-first LLM inference runtime.

Ocelotl is an early-stage workspace for a local LLM runtime with explicit model,
loader, tokenizer, kernel, runtime, and serving boundaries. The first milestone
is a narrow, correct single-process runtime before adding broad model coverage or
high-scale serving features.

## Start Here

New contributors should start with [docs/start-here.md](docs/start-here.md).

Core orientation docs:

- [Overview](docs/overview.md)
- [Architecture](docs/architecture.md)
- [Crate Boundaries](docs/crate-boundaries.md)
- [Interface Sketches](docs/design/interfaces.md)
- [Error Design](docs/design/errors.md)
- [Roadmap](docs/roadmap.md)
- [Milestone Task Backlog](docs/tasks/README.md)
- [Model Target](docs/model-target.md)
- [CI Policy](docs/ci.md)
- [TDD Policy](docs/validation/tdd.md)

## Crates

- `ocelotl-core`: shared types, errors, model metadata, and device contracts.
- `ocelotl-loader`: model artifact loading and validation.
- `ocelotl-tokenizer`: tokenizer and chat-template boundary.
- `ocelotl-kernels`: portable kernel dispatch boundary.
- `ocelotl-models`: model-family implementations.
- `ocelotl-runtime`: request lifecycle, KV cache, scheduling, and generation.
- `ocelotl-server`: API/server integration layer.
- `ocelotl`: root crate and CLI entrypoint.

## Validation

```powershell
cargo fmt --all
cargo check --workspace
cargo test --workspace
```

## Current Status

This is a project skeleton. Public APIs are intentionally small while the runtime
shape is established.
