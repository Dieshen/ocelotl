# Kernel Design

The kernel layer owns low-level compute dispatch. It should provide stable
Ocelotl-facing operations while allowing backend-specific implementations.

## Goals

- Keep hot operations behind explicit interfaces.
- Support a CPU reference path.
- Add GPU backends incrementally.
- Make shape and dtype requirements explicit.
- Validate CPU/GPU parity before defaulting to GPU paths.

## Candidate Backends

- Plain Rust CPU reference kernels.
- SIMD-optimized CPU kernels.
- CubeCL custom kernels.
- CubeK matmul, attention, reduction, and quantization kernels.
- Burn tensor ops where they are useful and do not obscure memory ownership.

## Initial Operations

The first useful kernel interface should cover only what M1-M4 need:

- Matrix multiply.
- RMSNorm.
- RoPE application.
- Attention for prefill.
- Attention for decode against KV.
- Gated MLP.
- Logits projection.

Sampling kernels can come later unless CPU sampling becomes a bottleneck.

## Shape Contracts

Every kernel call should receive explicit dimensions and strides. Do not rely on
implicit layout assumptions crossing crate boundaries.

## Error Policy

Unsupported dtype, quantization, shape, stride, or device combinations should
fail before launch. Kernel code should not reinterpret invalid inputs.

## Validation

Each GPU kernel needs a CPU parity test. Cache-related kernels need tests that
cover boundary cases, not only the first token or first page.
