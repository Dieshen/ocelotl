# Tokio

## Current Shape

Tokio is the async runtime Ocelotl should use for server and scheduler-facing
I/O. Crates.io currently shows `tokio = "1.52.1"`; Context7 returned detailed
Tokio 1.49 docs, which are still useful for stable runtime, task, channel, and
testing patterns. Re-check docs before depending on APIs introduced after 1.49.

The current workspace already depends on:

```toml
tokio = { version = "1", features = ["macros", "rt-multi-thread", "sync"] }
```

That is enough for async tests, runtime tasks, and channels. Networking, timers,
and signal handling will require additional Tokio features when the server crate
needs them.

## Best Use In Ocelotl

Use Tokio for:

- Server request handling.
- Async streaming responses.
- Runtime command channels.
- Scheduler coordination.
- Cancellation signals.
- Test harnesses with `#[tokio::test]`.

Do not use async tasks for CPU-heavy model execution directly. Heavy CPU work
should run on dedicated threads or `spawn_blocking`, and GPU execution should be
scheduled through explicit runtime/kernel boundaries.

## Runtime Setup

For binaries, use the multi-thread runtime unless a test or specialized local
executor requires current-thread behavior:

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Start server or runtime driver here.
    Ok(())
}
```

For tests:

```rust
#[tokio::test]
async fn generation_request_can_be_cancelled() {
    // Arrange runtime and cancellation handle.
}
```

## Blocking Work

Context7 Tokio docs show `spawn_blocking` for blocking work:

```rust
let handle = tokio::task::spawn_blocking(|| {
    // Blocking file I/O or CPU-heavy setup.
});
handle.await.expect("blocking task panicked");
```

Use this carefully. Long-lived model execution should not be hidden in ad hoc
`spawn_blocking` calls once Ocelotl has a real scheduler. Prefer a dedicated
runtime worker model where request ownership, cancellation, and backpressure are
explicit.

## Channels

Recommended channel use:

- `mpsc`: request queue or token stream from one runtime worker to one receiver.
- `oneshot`: single response or cancellation acknowledgement.
- `watch`: latest configuration/state notification.
- `broadcast`: fan-out events only when missed messages are acceptable.

Use bounded channels by default. Unbounded channels can hide backpressure bugs.

## Cancellation And Shutdown

Cancellation should be modeled explicitly. Dropping a future is not enough if GPU
buffers, KV pages, or scheduler slots need cleanup.

Server shutdown should:

1. Stop accepting new requests.
2. Signal active requests.
3. Drain or cancel runtime work.
4. Release KV/cache resources.
5. Stop worker tasks.

Tokio runtimes can be shut down with timeout in manually managed binaries, but
Ocelotl should prefer graceful resource release before runtime shutdown.

## TDD Requirements

- Scheduler and server tests should use `#[tokio::test]`.
- Cancellation tests must assert resource cleanup, not just task completion.
- Channel tests should use bounded capacities to expose backpressure behavior.
- Mock runtimes should be used before real model tests for server behavior.

## Risks

- Blocking executor threads with CPU model work.
- Spawning detached tasks with no cancellation path.
- Using unbounded channels for token streams or request queues.
- Holding async locks across `.await` points that can call back into the runtime.
- Assuming task abort releases non-Rust GPU resources correctly.
