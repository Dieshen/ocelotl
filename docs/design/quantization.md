# Quantization Design

Quantization support should be added after the unquantized path is correct.
Quantized execution changes both loader behavior and kernel behavior, so it needs
its own validation gates.

## Goals

- Keep quantization metadata explicit.
- Reject unsupported quantization formats early.
- Compare quantized output against unquantized or trusted reference output.
- Avoid mixing incompatible quantization layouts in one generic path.

## Deferred Until After M4

M1-M4 should use f16, bf16, or f32 weights. Quantization belongs after the basic
CPU and GPU execution paths have parity.

## Metadata

Quantized formats should describe:

- Quantization family.
- Block size.
- Scale dtype.
- Zero-point behavior.
- Packing layout.
- Per-tensor, per-channel, or per-block parameters.

## Loader Contract

The loader should not simply expose opaque bytes. It should validate that the
runtime and kernels understand the quantization format before model execution.

## Kernel Contract

Quantized kernels should document whether they dequantize eagerly, dequantize on
the fly, or use native quantized matmul. Each choice has different memory and
performance tradeoffs.

## Validation

Quantized tests should include:

- Shape validation.
- Known small tensors with exact expected dequantized values.
- End-to-end generation smoke tests.
- Parity against a trusted reference for representative prompts.
