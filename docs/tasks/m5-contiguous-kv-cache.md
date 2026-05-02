# M5 Tasks

M5 adds a request-scoped contiguous KV cache. This milestone prioritizes clear
layout and decode correctness before page allocation or batching.

## Entry Criteria

- M3 single-request prefill and decode behavior is correct.
- Kernel and model APIs can describe KV tensor shape explicitly.

## Task List

- [ ] M5.1 Define contiguous KV cache metadata.
  - Crates: `ocelotl-core`, `ocelotl-runtime`
  - Test first: construct a cache layout for the tiny model and assert layer, head, sequence, head-dim, dtype, and byte-size calculations.
  - Done when: layout metadata is typed and shared by runtime/model boundaries.

- [ ] M5.2 Allocate a per-request contiguous cache.
  - Crates: `ocelotl-runtime`
  - Test first: create two request caches and assert they do not share mutable storage.
  - Done when: allocation is request-scoped and cleanup is deterministic.

- [ ] M5.3 Add KV write tests for prefill.
  - Crates: `ocelotl-models`, `ocelotl-runtime`
  - Test first: run tiny prefill and assert expected KV positions are written.
  - Done when: prefill output and cache contents are both observable in tests.

- [ ] M5.4 Add KV read/append tests for decode.
  - Crates: `ocelotl-models`, `ocelotl-runtime`
  - Test first: decode one token after prefill and assert the new token appends at the next position.
  - Done when: decode uses cache reads instead of recomputing the full prompt path.

- [ ] M5.5 Enforce context-length and capacity errors.
  - Crates: `ocelotl-runtime`
  - Test first: request more tokens than the cache can hold and assert a typed capacity error.
  - Done when: overflow is rejected before any partial cache mutation.

- [ ] M5.6 Preserve M3 output parity.
  - Crates: `ocelotl-models`, `ocelotl-runtime`
  - Test first: compare no-cache or full-prefill behavior to contiguous-cache decode for the same tiny fixture.
  - Done when: contiguous cache does not change deterministic outputs.

- [ ] M5.7 Clean up on cancellation and failure.
  - Crates: `ocelotl-runtime`
  - Test first: inject a model error or cancellation after allocation and assert cache resources are released.
  - Done when: request lifecycle owns cache cleanup.

## Exit Criteria

- Prefill writes KV entries into a contiguous per-request cache.
- Decode reads prior KV entries and appends the next position.
- Cache capacity errors are explicit and tested.
- Contiguous-cache outputs match M3 deterministic expectations.

## Deferred

- Paged allocation.
- Cross-request cache sharing.
- Continuous batching.
- GPU cache residency.
