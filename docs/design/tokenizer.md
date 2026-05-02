# Tokenizer Design

The tokenizer boundary owns text-to-token and token-to-text behavior, including
chat-template rendering when a model expects structured messages.

## Responsibilities

- Encode user text or rendered chat prompts into token IDs.
- Decode token IDs into text.
- Define special-token handling.
- Own chat-template rendering boundaries.
- Provide deterministic fixture behavior.

## Token IDs

Use an Ocelotl-owned token ID type at crate boundaries. Avoid leaking a specific
tokenizer library type into the runtime or model crates.

## Chat Templates

Chat templates are model behavior. They should be explicit inputs to the
tokenizer layer or normalized by the loader when present in the artifact.

The runtime should not manually concatenate chat text. It should receive either:

- Plain prompt text for completion models.
- Structured messages plus a selected chat template.
- Already tokenized input for low-level tests.

## Special Tokens

Tokenizer implementations must document:

- BOS behavior.
- EOS behavior.
- PAD handling.
- Unknown-token handling.
- Whether decoding strips special tokens by default.

## Validation

Tokenizer fixtures should include exact token ID sequences for short prompts.
Do not rely only on round-trip text tests; many tokenizer bugs still round-trip
for simple ASCII.

## Non-Responsibilities

The tokenizer layer must not sample, schedule, execute model forward passes, or
interpret logits.
