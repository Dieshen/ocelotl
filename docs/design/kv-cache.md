# KV Cache Design

The KV cache stores key and value tensors produced during prefill and reused
during decode. It is central to performance and correctness.

## Goals

- Keep request-scoped cache ownership explicit.
- Support a simple contiguous cache before paged KV.
- Keep cache layout visible to model and kernel code.
- Validate dimensions at construction time.
- Make unsupported cache layouts fail early.

## M5: Contiguous KV

The first KV implementation is contiguous and simple:

- One clearly indexed contiguous block for all key tensors, and one matching
  block for all value tensors.
- Fixed maximum sequence length per request.
- Explicit layer, head, token, and head-dimension indexing.
- No page tables.
- No eviction.
- No prefix sharing.

The shared metadata lives in `ocelotl_core::KvCacheLayout`:

- `num_layers`
- `num_key_value_heads`
- `capacity_tokens`
- `head_dim`
- `dtype`
- `device`

For M5-M7 runtime-owned storage is CPU-resident F32. The CPU reference path may
load BF16 model weights, but cache writes happen after F32 compute.

`ocelotl_runtime::ContiguousKvCache` implements `KvCacheStore`, so the model can
write and read K/V through an Ocelotl-owned trait without owning allocation or
cleanup. `Qwen2_5Model::prefill_with_cache` writes every layer/position and
advances cache length only after prefill succeeds. `decode_token_with_cache`
reads previous K/V, appends the new token at `len_tokens`, and advances length
after logits for the following token are computed.

Runtime owns request state through:

- `prepare_qwen2_5_contiguous_cache`
- `decode_one_token_with_contiguous_cache`
- `Qwen2_5ContiguousCacheState::release`

This provides the stable baseline for paged KV.

## M6: Paged KV

Paged KV adds page allocation, page tables, and multi-page decode behavior on
top of the same `KvCacheStore` trait.

The shared metadata lives in `ocelotl_core::PagedKvCacheLayout`:

- `base: KvCacheLayout`
- `page_size_tokens`
- `physical_pages`

Invariants:

- A page table maps logical pages to unique physical pages.
- Page IDs must be less than `physical_pages`.
- The table must cover the requested cache capacity.
- `logical_page = position / page_size_tokens`.
- `offset_in_page = position % page_size_tokens`.
- Invalid page size, duplicate pages, out-of-range pages, dtype mismatch, and
  device mismatch fail before model compute.

`ocelotl_runtime::PagedKvCacheAllocator` owns the free-page pool. Runtime
`prepare_qwen2_5_paged_cache` returns allocated pages if prefill fails after
allocation. `Qwen2_5PagedCacheState::release_into` returns pages explicitly for
cancellation and normal cleanup.

The first paged implementation still copies a visible layer's K/V into a
contiguous scratch buffer before calling the existing attention kernel. That is
intentional for M6: parity and ownership are proven before introducing a
kernel-visible page-table attention path.

## Model-Specific Risks

Some model families need per-layer or per-group variation in attention metadata.
The cache must not assume a single global head dimension unless the active model
contract guarantees it.

## Debuggability

The runtime should expose enough cache metadata in debug builds to answer:

- Which request owns this cache?
- Which model and layer shape created it?
- How many tokens are resident?
- Is the cache contiguous or paged?
- Which device owns the underlying storage?

## Validation

Current CPU/reference tests prove:

- Contiguous layout shape and byte calculations.
- Request-isolated contiguous allocations.
- Prefill writes K/V and cached decode appends the next position.
- Capacity overflow fails before cache length advances.
- Paged allocation, release, and failure cleanup.
- Multi-page read/write across page ID 1.
- Paged/no-cache greedy-token parity for a tiny Qwen2.5 model.
