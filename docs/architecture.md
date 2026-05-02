# Ocelotl Architecture Sketch

Ocelotl separates model semantics from runtime execution:

1. Loaders validate model artifacts and expose normalized metadata.
2. Tokenizers own text, token IDs, and chat-template behavior.
3. Models describe architecture-specific forward behavior.
4. Kernels own portable compute dispatch and hot operations.
5. Runtime owns request lifecycle, scheduling, KV cache, and generation.
6. Server adapts the runtime to external APIs.

The first serious milestone should stay narrow: one model family, one process,
one GPU, f16/bf16 weights, contiguous KV, and CPU/GPU parity tests. Paged KV,
continuous batching, quantized weights, and GGUF compatibility should be added
after the simple path is correct.
