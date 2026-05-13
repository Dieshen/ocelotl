# Scheduler Design

The scheduler decides which request work runs next. It should be introduced only
after the single-request runtime path is correct.

## Goals

- Preserve output ordering per request.
- Support cancellation.
- Keep admission control explicit.
- Avoid starving long prompts or decode-heavy requests.
- Make batching decisions observable in tests and traces.

## M7 Runtime Surface

M7 introduces a deterministic single-process scheduler surface in
`ocelotl-runtime`:

- `ScheduledGenerationRequest`
- `ScheduledGenerationResponse`
- `SchedulerRequestState`
- `SchedulerEvent`
- `ContinuousBatchScheduler`
- `generate_qwen_batch`

The first implementation is a correctness harness, not a performance scheduler.
It drives greedy Qwen2.5 decode through the public runtime path and records
observable events for tests.

## Work Types

The scheduler handles:

- Prefill work.
- Decode work.
- Streaming token emission.
- Cancellation cleanup.

Cache allocation and release are explicit runtime responsibilities. The M7
scheduler tests focus on state transitions, ordering, cancellation, queue
bounds, and batched/unbatched parity.

## State Machine

Allowed request transitions:

- `Queued -> Prefill`
- `Prefill -> Decode`
- `Decode -> Emit`
- `Emit -> Decode`
- `Emit -> Complete`
- `Complete -> Cleanup`
- `Queued | Prefill | Decode | Emit -> Canceled`
- `Canceled -> Cleanup`

Invalid transitions return a runtime error and are covered by unit tests.

## Admission And Fairness

`SchedulerConfig::max_queue_len` bounds pending plus active requests. Requests
with empty prompts, zero `max_new_tokens`, or duplicate `request_id` values are
rejected at submission.

The first fairness policy is round-robin at the token-emission step. A long
request emits one token, then moves behind other active requests. This keeps a
short request from waiting for the long request to finish all requested tokens.

## Invariants

- A cancelled request must release runtime-owned resources.
- A request must not observe another request's KV cache.
- Decode steps must use the correct cache position.
- Batching must not change deterministic greedy output.
- Queue growth is bounded by configuration.
- Request IDs are unique within pending, active, and completed scheduler state.
- Scheduler events must expose enough state for tests to prove ordering and
  cleanup.

## Validation

Scheduler tests should use deterministic mock model outputs before they use real
model kernels. This makes fairness, cancellation, and ordering bugs easier to
isolate.

Current tests prove:

- Mock-model round-robin emission order.
- Explicit state-transition validation.
- Batched Qwen2.5 greedy output matches independent unbatched decode.
- Cancellation of one request does not corrupt an active peer.
- Queue overflow returns a typed `InvalidRequest` error.
- Duplicate request IDs return a typed `InvalidRequest` error.
- Short requests make progress before a long request completes.
