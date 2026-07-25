# Ocelotl

Rust-first local LLM and speech inference runtime.

Ocelotl is a correctness-first runtime with explicit model, loader, tokenizer,
kernel, runtime, and serving boundaries — with **no FFI**: every kernel, loader,
and tokenizer path is Rust.

**0.1.0 supports two surfaces**, chosen because they have independent-reference
parity evidence at the release commit:

- **Text embeddings** (EmbeddingGemma, pplx-embed) — cosine ≥ 0.9999997 vs
  `llama-embedding`; 2.4× faster than llama.cpp on bulk CPU.
- **Parakeet TDT 0.6B ASR** — token-exact against *two* independent
  implementations; RTF 0.095 on 12 threads.

Whisper ASR and Gemma4 text generation are present in-tree but **deferred to
0.2.0** — they execute, but neither meets its numeric parity contract at this
commit. See [CHANGELOG.md](CHANGELOG.md) for the full support boundary.

It is not an internet-facing server: M8 server work has not started.

## Start Here

New contributors should start with [docs/start-here.md](docs/start-here.md).

Core orientation docs:

- [Current Status And Alpha Gates](docs/status.md)
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
# Short edit loop.
pwsh -NoProfile -File tools/verify.ps1 -Mode Fast

# Required pre-merge gate.
pwsh -NoProfile -File tools/verify.ps1 -Mode Full
```

The workspace commits `Cargo.lock`, pins its development toolchain, and validates
Rust 1.85 separately as the minimum supported Rust version. See
[docs/ci.md](docs/ci.md) for the exact commands and opt-in hardware/artifact
policy.

## Current Status

M0 through M7 are closed at their documented correctness scopes. Alpha
hardening is active, M8 server work has not started, Whisper local exact-token
parity is opt-in, and Gemma4 real-artifact text parity remains blocked at the
first divergent Q5_K attention-output projection. The canonical implemented /
validated / blocked split is maintained in [docs/status.md](docs/status.md).
