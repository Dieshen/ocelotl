# Scheduler Design

The scheduler decides which request work runs next. It should be introduced only
after the single-request runtime path is correct.

## Goals

- Preserve output ordering per request.
- Support cancellation.
- Keep admission control explicit.
- Avoid starving long prompts or decode-heavy requests.
- Make batching decisions observable in tests and traces.

## Deferred Until M7

Before M7, the runtime should run one request at a time. This keeps correctness
work focused on model execution, cache ownership, and sampling behavior.

## Work Types

The scheduler eventually needs to handle:

- Prefill work.
- Decode work.
- Cache allocation and release.
- Streaming token emission.
- Cancellation cleanup.

## Invariants

- A cancelled request must release runtime-owned resources.
- A request must not observe another request's KV cache.
- Decode steps must use the correct cache position.
- Batching must not change deterministic greedy output.

## Validation

Scheduler tests should use deterministic mock model outputs before they use real
model kernels. This makes fairness, cancellation, and ordering bugs easier to
isolate.
