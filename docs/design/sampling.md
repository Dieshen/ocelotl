# Sampling Design

Sampling turns logits into the next token. It is part of generation semantics and
must be deterministic when configured to be deterministic.

## Initial Policy

Start with greedy decoding. Add probabilistic sampling only after greedy output
is covered by fixtures.

## Supported Modes

Planned order:

1. Greedy.
2. Temperature.
3. Top-k.
4. Top-p.
5. Min-p.
6. Repetition penalties.
7. Grammar or JSON-constrained decoding.

## Determinism

Tests should pin random seeds for probabilistic sampling. Greedy tests should
require exact token sequences.

## Logit Processing

Logit processors should be explicit and ordered. The runtime should not hide
implicit penalties or template-specific token masking.

## Error Policy

Invalid sampling parameters should fail before generation starts. Examples:

- Negative temperature.
- Empty candidate set.
- Top-p outside valid range.
- Penalty settings that are unsupported by the active implementation.

## Non-Responsibilities

Sampling should not own tokenizer behavior, model forward passes, KV cache, or
scheduler policy.
