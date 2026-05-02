# Model Metadata Design

Model metadata is the contract between loaders, model-family implementations,
runtime allocation, and kernel dispatch.

## Goals

- Make model shape explicit before execution.
- Preserve model-family differences.
- Validate unsupported features early.
- Provide enough information to size KV cache and choose kernels.

## Required Fields

The initial metadata model should cover:

- Architecture name.
- Vocabulary size.
- Context length.
- Layer count.
- Hidden size.
- Intermediate size.
- Attention head count.
- KV head count.
- Head dimension.
- RoPE base and scaling.
- Normalization epsilon.
- Dtype and quantization.

## Per-Layer Metadata

Do not assume every model uses one global shape for all layers. The metadata
contract should be able to represent per-layer head dimension, attention type,
or other model-family differences even if M1 only uses uniform shapes.

## Compatibility Checks

Runtime construction should validate:

- Model family is implemented.
- Dtype is supported by selected backend.
- Context length is within runtime limits.
- KV layout supports the model's attention metadata.
- Quantization format has kernels or CPU fallback.

## Versioning

Metadata structs will evolve. Keep serialized fixtures versioned so old tests
can be understood after fields are added.
