# Server API Design

The server adapts external protocols to the runtime. It must not own model
semantics or kernel behavior.

## Goals

- Keep transport concerns outside `ocelotl-runtime`.
- Provide a minimal local API before claiming compatibility with external APIs.
- Support streaming once runtime token emission is stable.
- Expose errors without leaking internal implementation details.

## Initial API

The first server milestone should support one local generation endpoint that
maps cleanly onto `GenerateRequest` and `GenerateResponse`.

OpenAI-compatible routes should wait until request, response, streaming, and
error semantics are intentionally mapped.

## Responsibilities

- Parse external requests.
- Validate transport-level inputs.
- Call runtime APIs.
- Stream tokens or return final responses.
- Expose health and version metadata.
- Emit structured logs and metrics.

## Non-Responsibilities

- Model loading policy beyond selecting a configured runtime.
- Tokenizer implementation.
- Scheduler policy.
- Kernel dispatch.

## Error Mapping

Server errors should map runtime errors into stable public responses. Internal
shape or kernel details can be logged, but user-facing errors should stay clear
and bounded.
