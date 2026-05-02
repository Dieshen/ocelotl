# Error Design

Many Ocelotl docs say to fail explicitly. This document defines what that means.
The goal is consistent errors across crates and clear behavior for runtime and
server callers.

## Principles

- Use typed error categories, not only free-form strings.
- Include the failing field, requested value, and supported values when useful.
- Reject unsupported configurations before compute starts.
- Keep user-facing errors bounded and actionable.
- Preserve source errors for diagnostics without leaking huge internals.

## Error Categories

`ocelotl-core` should own the top-level error enum:

```rust
pub enum OcelotlError {
    InvalidModel(InvalidModelError),
    InvalidRequest(InvalidRequestError),
    Unsupported(UnsupportedError),
    Tokenizer(TokenizerError),
    Kernel(KernelError),
    Runtime(RuntimeError),
    Io(IoError),
}
```

M1 can start smaller, but new crates should map errors toward these categories.

## Invalid Model

Use when a model artifact is malformed or inconsistent.

Examples:

- Missing required tensor.
- Mismatched tensor shape.
- Unknown architecture.
- Invalid metadata value.
- Tensor dtype does not match metadata.

Suggested fields:

```rust
pub struct InvalidModelError {
    pub path: Option<PathBuf>,
    pub field: Option<String>,
    pub message: String,
}
```

## Invalid Request

Use when the caller asks for something invalid independent of model support.

Examples:

- Empty prompt when disallowed.
- `max_new_tokens == 0` when generation requires at least one token.
- Context length exceeds model limit.
- Invalid sampling parameter.

Suggested fields:

```rust
pub struct InvalidRequestError {
    pub field: String,
    pub message: String,
}
```

## Unsupported

Use when a request or model is valid in general, but Ocelotl does not support it
yet.

Examples:

- Unsupported quantization format.
- Unsupported RoPE scaling.
- Unsupported attention layout.
- GPU requested but backend not compiled.
- Paged KV requested before M6.

Suggested fields:

```rust
pub struct UnsupportedError {
    pub feature: String,
    pub requested: Option<String>,
    pub supported: Vec<String>,
}
```

## Tokenizer

Use for tokenizer loading, encoding, decoding, or chat-template failures.

Examples:

- Missing tokenizer file.
- Invalid tokenizer JSON.
- Unknown special token.
- Chat template references unsupported message fields.

## Kernel

Use for backend dispatch and compute launch failures.

Examples:

- Unsupported dtype for selected backend.
- Unsupported stride/layout.
- Device allocation failure.
- Kernel launch failure.
- CPU/GPU parity test failure should be a test failure, not a runtime error.

## Runtime

Use for lifecycle failures after request/model validation has succeeded.

Examples:

- Cache allocation failure.
- Request cancelled.
- Scheduler closed.
- Internal invariant violation.

Internal invariant errors should be rare and treated as bugs.

## Server Error Mapping

Server-facing APIs should map core errors intentionally:

| Core Error | Server Class |
| --- | --- |
| `InvalidRequest` | client error |
| `InvalidModel` | configuration or server setup error |
| `Unsupported` | client error or configuration error depending on source |
| `Tokenizer` | client error for input/template issues, server error for setup issues |
| `Kernel` | server error unless caused by unsupported request |
| `Runtime` | cancellation or server error |
| `Io` | server setup or storage error |

Do not expose raw backtraces or full tensor data in public server responses.

## Test Requirements

Every new error category should have tests for:

- Display text contains actionable context.
- Source errors are preserved where applicable.
- Server mapping is stable if exposed through `ocelotl-server`.
- Unsupported configurations fail before compute starts.
