# M4 GPU Kernel Path

## Goal

Introduce the first GPU-backed kernel path for the smallest useful set of model
operations while preserving CPU/GPU parity.

## Non-Goals

- Broad kernel coverage.
- Quantized kernels.
- Paged KV.
- Continuous batching.
- Multi-GPU execution.
- Replacing the CPU reference path.

## TDD Plan

Write tests before implementation for:

- Each GPU kernel matches CPU output on small hand-checked tensors.
- Unsupported dtype and shape combinations fail before launch.
- GPU runtime construction fails clearly when the backend is unavailable.
- M3 prefill/decode fixtures pass on GPU within documented tolerance.

## Design

`ocelotl-kernels` should define the Ocelotl-facing operation boundary. Backend
choices such as CubeCL, CubeK, Burn, or CPU SIMD should sit behind that boundary.

Do not move scheduler or runtime ownership into the kernel crate. Kernels compute;
runtime owns requests and cache lifecycle.

## Acceptance Criteria

- At least one hot operation has a GPU implementation and CPU parity test.
- Runtime can select CPU or GPU backend explicitly.
- GPU path fails clearly when unavailable.
- GPU prefill/decode parity exists for the M3 fixture path.
- CPU reference remains available and tested.

## Validation Commands

```powershell
cargo test -p ocelotl-kernels
cargo test -p ocelotl-runtime
cargo test --workspace
```

GPU-specific tests may be ignored by default until CI hardware exists, but the
exact command to run them locally must be documented.

First CubeCL WGPU RoPE spike commands:

```powershell
cargo check -p ocelotl-kernels --features cubecl-wgpu --tests
cargo test -p ocelotl-kernels --features cubecl-wgpu cubecl_backend -- --nocapture
cargo test -p ocelotl-kernels --features cubecl-wgpu wgpu_rope_matches_cpu_reference_for_position_one -- --ignored --nocapture
```

## Known Risks

- Kernel APIs that hide strides or layout will make cache bugs hard to diagnose.
- GPU output can look plausible while drifting numerically.
- Backend-specific tensors must not leak into unrelated crates unless that is an
  intentional public contract.
