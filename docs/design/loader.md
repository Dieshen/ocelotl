# Loader Design

The loader normalizes model artifacts into explicit metadata and typed weight
access. It should make unsupported files fail before runtime execution begins.

## Responsibilities

- Detect supported artifact formats.
- Validate required files exist.
- Parse model metadata into Ocelotl-owned types.
- Normalize architecture names and key model dimensions.
- Expose weight tensors or tensor handles without owning model execution.
- Report unsupported features with precise errors.

## Initial Formats

M1 can avoid full artifact loading if fixtures are synthetic. M2 should add one
real local format. Prefer safetensors first because it keeps metadata and weight
layout work explicit. GGUF should come later because it carries more format and
quantization policy.

## Metadata Contract

The loader should produce normalized metadata for:

- Architecture.
- Vocabulary size.
- Context length.
- Hidden size.
- Number of layers.
- Attention heads and KV heads.
- Head dimensions.
- RoPE settings.
- Dtype and quantization.
- Tokenizer and chat-template references, if discoverable.

Metadata must preserve model-family-specific details instead of flattening them
away. If a model has per-layer variation, the metadata type should expose that
variation explicitly.

## Error Policy

Unsupported metadata is an error, not a warning. Examples:

- Unknown architecture.
- Unknown quantization.
- Missing required tensor.
- Mismatched tensor shape.
- Unsupported RoPE scaling.
- Unsupported attention layout.

## Non-Responsibilities

The loader must not tokenize prompts, schedule requests, allocate KV cache, or
launch kernels. It prepares verified artifacts for model/runtime code.
