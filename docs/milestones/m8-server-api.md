# M8 Server API

## Goal

Expose Ocelotl through a server layer that maps external requests to runtime
requests and streams or returns generated output.

## Non-Goals

- Full OpenAI API compatibility on the first server milestone.
- Authentication and multi-tenant production hardening.
- Distributed serving.
- Browser UI.

## TDD Plan

Write tests before implementation for:

- Request validation maps invalid inputs to stable public errors.
- Runtime errors map to bounded server errors.
- Successful generation returns the expected response shape.
- Streaming lifecycle opens, emits, and closes in order.
- Cancellation or dropped clients release runtime resources.

## Design

The server should adapt transport to runtime. It should not contain model-family
logic, tokenizer internals, or kernel dispatch decisions.

Start with a minimal local API. Claim OpenAI compatibility only after request,
response, streaming, error, and model-list semantics are intentionally mapped.

## Acceptance Criteria

- Server crate exposes a minimal generation API.
- Runtime errors are mapped intentionally.
- Streaming behavior is tested if exposed.
- Server tests can use a mock runtime.
- Public API docs describe what is and is not compatible.

## Validation Commands

```powershell
cargo test -p ocelotl-server
cargo test -p ocelotl-runtime
cargo test --workspace
```

## Known Risks

- API compatibility claims create support obligations.
- Server tests that require real models are slow and brittle.
- Transport cancellation must be tied back to runtime resource cleanup.
