# CubeCL

## Current Shape

CubeCL is a Rust GPU compute framework for portable kernels. Crates.io currently
shows `cubecl = "0.10.0-pre.4"`, so it is also on a pre-release line.

Context7 docs show:

- Kernels are written with the `#[cube]` macro.
- Launchable kernels use `#[cube(launch)]` or `#[cube(launch_unchecked)]`.
- Runtimes include WGPU, CUDA, ROCm/HIP, Metal, Vulkan, and CPU/LLVM depending
  on enabled features and platform support.
- `Vector<F, N>` supports vectorized operations and comptime parameters.
- CubeCL has autotuning and persistent kernel cache concepts.

## Best Use In Ocelotl

Use CubeCL for Ocelotl-owned kernels where memory layout matters:

- RMSNorm.
- RoPE.
- KV write/read transforms.
- Decode attention over KV.
- Small fused elementwise operations.
- Layout conversion and packing/unpacking.

Use CubeK first for large standard kernels when available, especially matmul and
attention. Write raw CubeCL when Ocelotl needs custom layout, paging, or fusion.

## Recommended Boundary

`ocelotl-kernels` should own CubeCL integration. Other crates should call an
Ocelotl kernel trait, not CubeCL launch functions.

```rust
pub trait KernelBackend {
    fn name(&self) -> &'static str;
    fn supports_gpu(&self) -> bool;
}

pub trait RopeKernel: KernelBackend {
    fn apply_rope(&self /* explicit tensor handles, shape, strides */);
}
```

## Kernel Example Shape

Context7 shows elementwise kernels using `#[cube]` and vectorization:

```rust
use cubecl::prelude::*;

#[cube]
fn scale<F: Float, N: Size>(x: Vector<F, N>, #[comptime] factor: f32) -> Vector<F, N> {
    x * F::new(factor)
}

#[cube(launch_unchecked)]
fn scale_array<F: Float, N: Size>(
    input: &Array<Vector<F, N>>,
    output: &mut Array<Vector<F, N>>,
    #[comptime] factor: f32,
) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = scale(input[ABSOLUTE_POS], factor);
    }
}
```

Do not copy this shape directly into runtime code. Wrap it behind tested kernel
functions and explicit buffer contracts.

## TDD Requirements

For every CubeCL kernel:

- Start with a CPU reference test using small hand-checked tensors.
- Add a GPU parity test with the same shape.
- Add boundary tests for odd lengths, stride differences, and unsupported dtype.
- Mark hardware-dependent tests as ignored until CI hardware exists.

## Risks

- Pre-release version churn.
- Unsafe launch APIs can bypass validation if wrappers are thin.
- Autotuned behavior must not change numerical correctness.
- Kernel APIs that omit strides or layout will cause subtle KV bugs.

## Follow-Up Questions

- Which CubeCL runtime should be first: WGPU for portability or CUDA for fastest
  NVIDIA path?
- How should Ocelotl represent device buffers without leaking CubeCL types into
  `ocelotl-runtime`?
- Which kernels should be handwritten versus delegated to CubeK?
