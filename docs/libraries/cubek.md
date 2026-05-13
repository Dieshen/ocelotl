# CubeK

## Current Shape

CubeK is a CubeCL kernel library with optimized kernels for common operations.
Crates.io currently shows `cubek = "0.2.0-pre.4"` and subcrates including
`cubek-matmul`, `cubek-attention`, `cubek-reduce`, and `cubek-quant`.

Context7 docs show examples for:

- Matrix multiplication through `cubek-matmul` and `Strategy::Auto`.
- Scaled dot-product attention through `cubek-attention`.
- Reductions through `cubek-reduce`.
- Quantization and dequantization through `cubek-quant`.

## Best Use In Ocelotl

Use CubeK before writing custom CubeCL when the operation and layout fit:

- GEMM for projections and MLPs.
- Attention kernels for standard prefill attention.
- Reductions for norm and sampling helpers if suitable.
- Quant/dequant experiments after unquantized parity exists.

Use custom CubeCL instead when Ocelotl needs:

- Paged KV-specific attention.
- Non-standard strides or page tables.
- Fused model-family-specific behavior.
- Explicit layout control not expressible in CubeK APIs.

## Integration Boundary

CubeK should remain an implementation detail of `ocelotl-kernels`.

```rust
pub enum MatmulStrategy {
    Auto,
    Reference,
    BackendSpecific,
}

pub trait MatmulKernel {
    fn matmul(&self, problem: MatmulProblemRef<'_>) -> ocelotl_core::Result<()>;
}
```

Ocelotl should define its own problem structs with explicit shape, dtype, and
stride data, then adapt them to CubeK internally.

## TDD Requirements

- Compare CubeK matmul against CPU reference on small tensors first.
- Test batched and non-batched shapes separately.
- Test attention causal and non-causal behavior explicitly.
- Do not use CubeK quant/dequant in runtime defaults until quantized fixtures
  exist.

## Risks

- Pre-release API churn.
- CubeK standard attention may not match Ocelotl paged KV layout.
- Strategy auto-selection can change performance by hardware; correctness tests
  must be independent of selected strategy.
- Quantization APIs may not map directly to GGUF or other model-specific
  quantization formats.

## M4 Decision

Defer CubeK out of M4. The first backend proof uses custom CubeCL RoPE because
it is small, layout-sensitive, and already has compact CPU parity fixtures.
CubeK is still the preferred next evaluation target for GPU GEMM or attention,
but Ocelotl should not adapt it until the kernel boundary has explicit
device-buffer, dtype, stride, and ownership contracts for matmul/attention-size
data.

The smallest blocked case is Qwen/Whisper projection matmul: CPU currently
passes contiguous f32 slices and pre-transposed weights, while CubeK integration
will need an Ocelotl-owned problem struct that can describe device residency,
row/column stride, dtype, and fallback behavior without leaking CubeK types
outside `ocelotl-kernels`.
