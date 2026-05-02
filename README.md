# Ocelotl

Rust-first LLM inference runtime.

Ocelotl is an early-stage workspace for a local LLM runtime with explicit model,
loader, kernel, and serving boundaries. The first milestone is a narrow,
correct single-process runtime before adding broad model coverage or high-scale
serving features.

## Crates

- `ocelotl-core`: shared types, errors, model metadata, and device contracts.
- `ocelotl-loader`: model artifact loading and validation.
- `ocelotl-tokenizer`: tokenizer and chat-template boundary.
- `ocelotl-kernels`: portable kernel dispatch boundary.
- `ocelotl-models`: model-family implementations.
- `ocelotl-runtime`: request lifecycle, KV cache, scheduling, and generation.
- `ocelotl-server`: API/server integration layer.
- `ocelotl`: root crate and CLI entrypoint.

## Current Status

This is a project skeleton. Public APIs are intentionally small while the runtime
shape is established.
