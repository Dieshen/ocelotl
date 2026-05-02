# M6 Tasks

M6 replaces or supplements contiguous KV with a paged layout. The first paged
implementation must prove correctness with multi-page tests before performance
work begins.

## Entry Criteria

- M5 contiguous KV cache behavior is correct and covered by parity tests.
- Runtime request lifecycle can allocate and release cache resources reliably.

## Task List

- [ ] M6.1 Define page size, page table, and allocator contracts.
  - Crates: `ocelotl-core`, `ocelotl-runtime`
  - Test first: calculate page counts and logical-to-physical positions for several sequence lengths.
  - Done when: paged layout metadata is explicit and independent from kernel launch details.

- [ ] M6.2 Implement page allocation and release.
  - Crates: `ocelotl-runtime`
  - Test first: allocate pages for multiple requests, release one request, and assert pages return to the free pool.
  - Done when: allocator state remains consistent across allocation, release, and failure paths.

- [ ] M6.3 Add multi-page read/write tests.
  - Crates: `ocelotl-runtime`, `ocelotl-models`
  - Test first: write a sequence spanning at least two pages and assert reads from page ID greater than zero.
  - Done when: tests would fail if only page zero were ever used.

- [ ] M6.4 Add contiguous/paged output parity.
  - Crates: `ocelotl-runtime`, `ocelotl-models`
  - Test first: run the same tiny prompt through contiguous and paged cache modes and compare logits or selected tokens.
  - Done when: paged cache preserves deterministic M5 behavior.

- [ ] M6.5 Reject invalid page layouts before kernel execution.
  - Crates: `ocelotl-kernels`, `ocelotl-runtime`
  - Test first: pass malformed page tables, duplicate pages, out-of-range pages, and dtype mismatches.
  - Done when: invalid layouts produce typed errors without launching kernels.

- [ ] M6.6 Clean up pages on cancellation.
  - Crates: `ocelotl-runtime`
  - Test first: cancel a request after partial page allocation and assert all pages are released.
  - Done when: allocator leak tests pass under failure and cancellation.

- [ ] M6.7 Document paged-cache invariants.
  - Crates: docs only
  - Test first: task completion requires updating `docs/design/kv-cache.md` with page-table invariants.
  - Done when: contributors can reason about logical positions, physical pages, and kernel-visible layout from docs.

## Exit Criteria

- Paged KV supports sequences spanning multiple pages.
- Paged and contiguous modes produce equivalent deterministic outputs for shared fixtures.
- Page allocation and release are tested under normal, error, and cancellation paths.
- Invalid page tables fail before kernel execution.

## Deferred

- Prefix caching.
- Cache compaction.
- Cross-request page sharing.
- Distributed cache management.
