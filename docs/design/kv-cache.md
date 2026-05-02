# KV Cache Design

The KV cache stores key and value tensors produced during prefill and reused
during decode. It is central to performance and correctness.

## Goals

- Keep request-scoped cache ownership explicit.
- Support a simple contiguous cache before paged KV.
- Keep cache layout visible to model and kernel code.
- Validate dimensions at construction time.
- Make unsupported cache layouts fail early.

## M1 To M5: Contiguous KV

The first KV implementation should be contiguous and simple:

- One allocation per layer per request, or one clearly indexed contiguous block.
- Fixed maximum sequence length per request.
- Explicit layer, head, token, and head-dimension indexing.
- No page tables.
- No eviction.
- No prefix sharing.

This is easier to test and provides a stable baseline for later paged KV work.

## M6: Paged KV

Paged KV adds page allocation, page tables, and multi-page decode behavior. It
must not be introduced until contiguous KV is correct.

Paged KV acceptance criteria should include:

- A test that decodes across `page_id > 0`.
- Shape checks for every cache dimension.
- Explicit ownership and release rules.
- Kernel tests that compare contiguous and paged outputs.
- Construction-time rejection of unsupported page layouts.

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
