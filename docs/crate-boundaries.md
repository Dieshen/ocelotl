# Crate Boundaries

The workspace uses short folder names and published crate names under the
`ocelotl-*` namespace.

## `ocelotl-core`

Owns shared types, errors, model metadata, device descriptors, dtype enums, and
small contracts that multiple crates need.

Must not own model loading, tokenizer implementations, kernel launches, runtime
scheduling, or server APIs.

## `ocelotl-loader`

Owns artifact discovery, format detection, metadata normalization, and validation
of local model files.

Must not own model execution, tokenization, request scheduling, or kernel code.

## `ocelotl-tokenizer`

Owns tokenizer traits, token ID types, text encode/decode behavior, and
chat-template rendering boundaries.

Must not own generation policy, model forward passes, or server request types
beyond reusable text/token abstractions.

## `ocelotl-kernels`

Owns portable kernel dispatch and low-level compute backend contracts. This crate
is where CPU, CubeCL, CubeK, Burn, or future backend integrations should meet a
single Ocelotl-facing kernel interface.

Must not own request scheduling, model artifact loading, or high-level model
semantics.

## `ocelotl-models`

Owns model-family implementations and model-specific forward semantics. This is
where Llama, Qwen, Mistral, Gemma, and other architecture differences belong.

Must not own file format parsing, HTTP APIs, or scheduler policy.

## `ocelotl-runtime`

Owns request lifecycle, model/session state, prefill/decode flow, KV cache,
sampling orchestration, cancellation, and scheduling.

Must not parse model files directly or expose transport-specific API contracts.

## `ocelotl-server`

Owns transport and API adaptation around the runtime. This includes future HTTP,
gRPC, or OpenAI-compatible endpoints.

Must not contain model-family logic or kernel-specific implementation details.

## `ocelotl`

The root crate and CLI entrypoint. It should re-export stable top-level APIs and
provide developer commands without becoming a dumping ground for implementation.
