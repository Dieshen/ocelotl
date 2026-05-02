# Correctness Validation

Ocelotl should treat correctness as a release gate. A model path is not correct
because it runs; it is correct when it matches a reference within documented
tolerance and fails clearly outside its supported contract.

## Validation Layers

1. Shape and metadata validation.
2. Tokenizer and chat-template validation.
3. CPU reference execution.
4. CPU versus reference logits or tokens.
5. GPU versus CPU parity.
6. Cache layout equivalence.
7. End-to-end generation fixtures.

## Fixture Policy

Fixtures should be small, deterministic, and committed when licensing permits.
Network access should not be required for normal correctness tests.

Each fixture should document:

- Model family.
- Artifact source.
- Tokenizer source.
- Prompt.
- Expected tokens or logits.
- Tolerance.
- Reason the fixture exists.

## Tolerances

Tolerances must be explicit per test category. Examples:

- Exact token match for deterministic greedy generation.
- Absolute or relative tolerance for logits.
- Wider tolerance for lower-precision GPU kernels when justified.

A tolerance increase is a behavior change and should be reviewed like one.

## Unsupported Configs

Unsupported features should produce explicit errors, not fallback behavior. Tests
should cover common unsupported cases:

- Unknown architecture.
- Unsupported quantization.
- Unsupported RoPE scaling.
- Unsupported attention head layout.
- Context length beyond model limit.
- Missing tokenizer or chat template.

## GPU Parity

Every GPU kernel path needs a CPU parity test before it becomes part of the
runtime default. GPU tests should compare both simple first-page behavior and
multi-page/cache-boundary behavior when KV paging exists.

## Release Gate

A release that adds a model family, kernel path, quantization format, or cache
layout should include a validation note listing the exact tests and fixtures that
prove the path is safe to expose.
