# Burn

## Current Shape

Burn is a Rust deep-learning framework with backend-generic tensors, modules,
record loading, WGPU support, and extension points for custom backend operations.
Crates.io currently shows `burn = "0.21.0-pre.4"`, so the current public line is
pre-release. Treat API stability carefully.

Context7 docs show:

- Backend-generic inference through `B: burn::tensor::backend::Backend`.
- WGPU backend usage through `burn::backend::Wgpu`.
- Custom backend extension by defining an Ocelotl-specific trait that extends
  Burn's `Backend` trait.
- Model loading via Burn Store from PyTorch or safetensors, and saving to
  Burnpack.

## Best Use In Ocelotl

Use Burn for:

- Early CPU/GPU tensor prototypes where generic backend code is useful.
- Model module experiments before hot paths are moved behind `ocelotl-kernels`.
- Small reference implementations of transformer layers.
- Possible checkpoint conversion workflows if Burnpack becomes useful.

Do not use Burn as:

- The scheduler.
- The request lifecycle owner.
- The KV cache owner unless Ocelotl wraps the memory layout explicitly.
- A substitute for Ocelotl-owned kernel contracts.
- A reason to skip CPU/reference parity tests.

## Recommended Boundary

Burn should live behind `ocelotl-models` or `ocelotl-kernels` experiments. The
runtime should not expose Burn tensor types in public request APIs.

If Burn is used for model code, define an internal backend alias or trait at the
model boundary:

```rust
use burn::tensor::{Tensor, backend::Backend};

pub trait ModelBackend: Backend {}
impl<B: Backend> ModelBackend for B {}

pub fn rmsnorm_reference<B: ModelBackend>(input: Tensor<B, 2>) -> Tensor<B, 2> {
    // Placeholder: real implementation should include weights, epsilon, and
    // fixture parity tests before landing.
    input
}
```

## Custom Operation Pattern

Burn supports extending backend traits for custom operations. For Ocelotl, this
is useful only if the custom operation remains behind an Ocelotl kernel trait:

```rust
use burn::tensor::backend::Backend;

pub trait OcelotlBurnBackend: Backend {
    // Example shape only. Real signatures should include explicit shape and
    // stride contracts at the Ocelotl boundary.
    fn fused_rmsnorm_silu_gate(/* tensors */);
}
```

Prefer implementing Ocelotl-facing traits in `ocelotl-kernels` rather than
letting model/runtime code call backend-specific operations directly.

## TDD Requirements

Before introducing Burn to a crate:

- Add a CPU/reference fixture test for the operation or layer.
- Add unsupported-backend tests if a feature requires WGPU or another backend.
- Keep outputs compared against Ocelotl-owned fixtures, not just Burn examples.

## Risks

- Pre-release API churn can force refactors.
- Burn tensors can hide memory layout details that matter for KV cache and paged
  attention.
- Model-level code can become coupled to Burn-specific records if the loader
  contract is not kept separate.

## Follow-Up Questions

- Does Burn Store support the exact loader path Ocelotl needs, or should
  `ocelotl-loader` parse safetensors directly first?
- Can Burn's WGPU backend share buffers cleanly with CubeCL/CubeK kernels, or
  should Ocelotl own separate kernel buffers?
- Which operations are cheaper to prototype in Burn and later replace with
  CubeCL/CubeK?
