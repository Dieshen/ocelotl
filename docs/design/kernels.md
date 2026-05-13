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

## M4 CubeCL Spike

The first non-CPU backend is a narrow CubeCL WGPU RoPE spike in
`ocelotl-kernels`, behind the optional `cubecl-wgpu` feature. CPU remains the
default backend and parity oracle.

Why RoPE first:

- It is layout-sensitive enough to test Ocelotl-owned indexing and validation.
- It avoids making the first GPU decision around GEMM, where CubeK or another
  tuned kernel library may be the right long-term choice.
- It has compact hand-checked CPU fixtures already in the kernel crate.

The spike deliberately copies CPU slices into CubeCL buffers, launches a simple
f32 kernel, and copies results back. That is not the performance model for M4;
it is a boundary proof that CubeCL can sit behind Ocelotl-owned kernel APIs
without leaking into `ocelotl-runtime` or model-family code.

Validation commands:

```powershell
cargo check -p ocelotl-kernels --features cubecl-wgpu --tests
cargo test -p ocelotl-kernels --features cubecl-wgpu cubecl_backend -- --nocapture
cargo test -p ocelotl-kernels --features cubecl-wgpu wgpu_rope_matches_cpu_reference_for_position_one -- --ignored --nocapture
```

The local execution proof uses `1e-5` tolerance against
`rope_apply_inplace`. Default workspace validation does not build CubeCL.

## M4 Model/Runtime Path

M4 wires the first GPU-backed kernel into the Qwen2.5 model path without
claiming full-model GPU execution. `Qwen2_5KernelBackend::CubeClWgpu` advertises
a CubeCL GPU execution backend and routes RoPE through CubeCL WGPU. Every other
operation still goes through the CPU fallback backend.

This split is deliberate:

- It proves the model/runtime dispatch seam can select a non-CPU backend.
- It preserves CPU as the parity oracle and default implementation.
- It avoids forcing matmul or attention through an immature GPU abstraction
  before Ocelotl has explicit device-buffer and dtype layout contracts.

Local M4 parity covers three levels:

- RoPE kernel output versus CPU RoPE at `1e-5`.
- Qwen tiny synthetic prefill logits versus CPU prefill at the existing M3
  `1e-4` fixture tolerance.
- Qwen tiny synthetic one-token decode versus CPU exact token equality.

Unsupported CubeCL RoPE layouts fail before launch. The currently supported
layout is contiguous f32 `[num_heads, head_dim]` rows with
`row_stride == head_dim`.

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
