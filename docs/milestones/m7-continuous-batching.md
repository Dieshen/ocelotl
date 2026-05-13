# M7 Continuous Batching

## Goal

Add scheduling that can batch work across requests while preserving correctness,
fairness, cancellation, and output ordering.

## Non-Goals

- Distributed serving.
- Multi-GPU scheduling.
- Speculative decoding.
- Prefix cache sharing unless explicitly added as a separate follow-up.

## TDD Plan

Write tests before implementation for:

- Batched greedy output matches unbatched greedy output.
- Request output order is preserved.
- Cancellation releases request resources.
- Long-prefill requests do not permanently starve decode requests.
- Scheduler decisions are testable with a mock model.

## Design

Start with a deterministic scheduler and mock model tests. Do not rely only on
real model integration tests; scheduler bugs are easier to isolate with
predictable fake outputs.

The scheduler should operate on runtime work items: prefill, decode, emit, and
cleanup.

## Acceptance Criteria

- Multiple requests can be active concurrently.
- Batching does not change deterministic greedy outputs.
- Cancellation releases KV and scheduler state.
- Scheduler behavior has unit tests with mock work items.
- Runtime integration tests cover at least two concurrent requests.

## Validation Commands

```powershell
cargo test -p ocelotl-runtime
cargo test --workspace
```

## Known Risks

- Scheduling can hide cache ownership bugs.
- Fairness and throughput goals can conflict; correctness wins first.
- Cancellation paths often miss GPU/cache cleanup unless tested directly.

## Closure Note (2026-05-13)

Closed as deterministic scheduler correctness plumbing. Runtime exposes
bounded admission, explicit request states, cancellation cleanup, round-robin
token emission, and a public Qwen2.5 batch helper whose outputs match
independent unbatched greedy decode. Throughput-oriented scheduling remains
deferred.
