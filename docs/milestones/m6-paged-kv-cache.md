# M6 Paged KV Cache

## Goal

Add paged KV cache allocation, page tables, and multi-page decode behavior while
preserving parity with contiguous KV.

## Non-Goals

- Full continuous batching scheduler.
- Prefix cache sharing.
- Eviction policy beyond explicit request release.
- Multi-GPU page placement.

## TDD Plan

Write tests before implementation for:

- Page allocation and release.
- Token positions that cross from page 0 to page 1.
- Page table lookup for decode.
- Contiguous and paged KV parity on the same prompt.
- Unsupported page sizes or model layouts fail at construction time.

## Design

Paged KV should be introduced as a new cache layout under the same runtime-owned
cache contract. It should not create a second runtime path.

Page metadata should make ownership, layer, token range, and device location
observable in debug builds.

## Acceptance Criteria

- Paged cache can represent requests longer than one page.
- Decode reaches `page_id > 0` in tests.
- Contiguous and paged KV outputs match within tolerance.
- Invalid page layouts fail before kernel launch.
- Page release happens on request completion and cancellation.

## Validation Commands

```powershell
cargo test -p ocelotl-runtime
cargo test -p ocelotl-kernels
cargo test --workspace
```

## Known Risks

- Tests that only hit page 0 do not validate paged KV.
- Stride mismatches between writer and reader kernels can pass simple tests.
- Model-family attention variation can break page sizing assumptions.

## Closure Note (2026-05-13)

Closed for CPU/reference execution. Runtime owns paged layout, allocator,
release, failure cleanup, multi-page read/write, and paged/no-cache Qwen2.5
greedy-token parity. Kernel-visible paged attention and GPU-resident pages
remain deferred.
