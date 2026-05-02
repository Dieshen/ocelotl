# Model Forward Design

Model-family crates define forward semantics. The runtime owns request lifecycle;
models own architecture-specific math and tensor layout expectations.

## Responsibilities

- Implement model-family prefill and decode behavior.
- Validate model metadata against implementation assumptions.
- Apply model-specific operations such as RoPE, RMSNorm, gated MLPs, and logits
  transforms.
- Expose clear kernel requirements to `ocelotl-kernels`.

## Initial Shape

Start with one decoder-only architecture and one unquantized dtype. The first
implementation should prioritize readability and parity against a reference over
performance.

## Prefill And Decode

The model interface should distinguish prefill and decode because they stress
different kernels and cache access patterns.

Prefill:

- Processes prompt tokens.
- Writes all prompt KV entries.
- Produces logits for the final prompt position.

Decode:

- Processes one or a small number of new tokens.
- Reads existing KV.
- Appends new KV.
- Produces next-token logits.

## Model-Specific Behavior

Model-family behavior must be visible in implementation and tests. Examples:

- RoPE scaling.
- Sliding window attention.
- Grouped-query attention.
- MoE routing.
- Logit softcap.
- Per-layer head dimensions.

Unsupported behavior should fail at model construction time where possible.

## Non-Responsibilities

Model code should not parse files, own HTTP APIs, or make global scheduling
choices. It should not silently change tokenizer behavior.
