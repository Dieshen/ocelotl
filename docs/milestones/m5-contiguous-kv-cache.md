# M5 Contiguous KV Cache

## Goal

Add request-scoped contiguous KV cache support and route decode through the cache
abstraction.

## Non-Goals

- Paged KV.
- Prefix sharing.
- Eviction.
- Continuous batching.
- Multi-request cache compaction.

## TDD Plan

Write tests before implementation for:

- KV allocation uses model metadata dimensions.
- KV write/read returns exact values for small tensors.
- Decode uses cached keys and values instead of recomputing full prompt state.
- Context overflow fails before writing beyond allocated cache.
- Request caches are isolated from each other.

## Design

Start with a simple contiguous layout. Prefer explicit indexes and shape checks
over memory tricks. The point of M5 is to make cache semantics correct and visible
before adding page tables.

## Acceptance Criteria

- Runtime owns request-scoped KV cache state.
- Prefill writes prompt KV into cache.
- Decode reads existing KV and appends new KV.
- Contiguous cache output matches the M3 reference path.
- Cache bounds and ownership errors are explicit.
- CPU and GPU paths, if GPU exists, agree within tolerance.

## Validation Commands

```powershell
cargo test -p ocelotl-runtime
cargo test -p ocelotl-models
cargo test --workspace
```

## Known Risks

- Cache position bugs often only appear after the first token.
- Request isolation must be tested before scheduler work begins.
- A contiguous layout should not be over-generalized into a fake paged design.

## Closure Note (2026-05-13)

Closed for CPU/reference execution. Runtime owns contiguous Qwen2.5 cache state,
prefill writes prompt K/V, cached decode reads existing K/V and appends the next
position, and cached greedy output matches the M3 no-cache path. GPU cache
residency remains deferred.
