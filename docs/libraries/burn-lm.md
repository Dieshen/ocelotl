# Burn LM

## Current Shape

Burn LM is an LLM framework built on Burn. Context7 docs describe a registry,
CLI shell, OpenAI-compatible HTTP server, first-party Llama implementations,
quantization helpers, KV cache management, RoPE positional encoding, and token
streaming.

It appears closer to a full runtime/framework than a small library dependency.
Use it as a research reference before adopting it directly.

## Lessons For Ocelotl

Reuse the ideas:

- Model implementations can be registered behind a common server/runtime trait.
- Transformer forward should take explicit cache and positional-encoding state.
- CLI and HTTP interfaces should sit above the model/runtime boundary.
- Quantization helpers should not be mixed into the first correctness milestone.

Avoid copying blindly:

- Ocelotl should own its runtime and cache contracts.
- Ocelotl should keep crate boundaries small and test-first.
- Ocelotl should not claim OpenAI compatibility until server semantics are tested.

## Reference Example Shape

Context7 shows Burn LM transformer forward taking input tokens, cache state,
positional encoding, and mask:

```rust
pub fn forward(
    input_tokens: TokenTensor,
    cache: &mut TransformerCache,
    pos_encoding: &PositionalEncodingState,
    mask: Option<AttentionMask>,
) -> LogitsTensor {
    // Model-family implementation owns this behavior.
    todo!()
}
```

For Ocelotl, this reinforces that KV cache and positional state should be
explicit parameters, not hidden globals.

## TDD Requirements

Before borrowing a Burn LM pattern, add an Ocelotl fixture that proves the
pattern works with Ocelotl metadata, tokenizer, cache, and runtime contracts.

## Risks

- Direct adoption can pull Ocelotl toward another framework's architecture.
- Burnpack-oriented model loading may not match Ocelotl's safetensors-first M2
  plan.
- Full framework features can obscure the smaller M1-M4 correctness path.
