# Runtime Design

The runtime owns request lifecycle and generation flow. It coordinates loaded
model state, tokenizer output, prefill, decode, KV cache, sampling, streaming,
and cancellation.

## Responsibilities

- Accept normalized generation requests.
- Validate model/runtime compatibility before execution.
- Own request-scoped state and KV cache handles.
- Route prefill and decode through model and kernel boundaries.
- Apply sampling policy after logits are produced.
- Preserve deterministic behavior for tests.
- Surface explicit errors for unsupported features.

## Non-Responsibilities

- Parsing model artifact files.
- Implementing tokenizer internals.
- Implementing transport-specific HTTP or gRPC APIs.
- Hiding model-family differences.
- Owning backend-specific kernel code.

## Request Lifecycle

1. Receive a normalized request.
2. Validate options and context length.
3. Tokenize or accept pre-tokenized input.
4. Allocate runtime request state.
5. Run prefill.
6. Enter decode loop.
7. Sample or select next token.
8. Update KV and request state.
9. Emit tokens or final text.
10. Release request resources.

## Prefill And Decode

Prefill and decode should use the same model/runtime contract but may dispatch to
different kernel strategies. Prefill favors throughput over full prompt length;
decode favors low-latency one-token steps and efficient KV reads.

## Error Handling

The runtime should reject unsupported configurations before launching compute.
Examples:

- Context length exceeds model limit.
- Unsupported dtype or quantization.
- Missing tokenizer or chat template.
- Unsupported attention layout.
- GPU requested but no GPU backend is available.

## First Implementation Rule

Start with a single request and no scheduler. Add scheduling only after the
single-request path has parity tests.
