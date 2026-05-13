# M4 Tasks

M4 adds the first GPU-backed kernel path with CPU/GPU parity. GPU support must be
feature-gated and must not weaken the CPU reference path.

## Entry Criteria

- M3 CPU forward path passes with deterministic fixtures.
- Kernel interfaces include explicit shape, stride, dtype, and device contracts.
- Library notes for Burn, CubeCL, and CubeK are current enough to guide the first backend decision.

## Task List

- [x] M4.1 Choose and document the first GPU execution backend.
  - Crates: docs, `ocelotl-kernels`
  - Test first: add a design note describing why the chosen backend can support the first parity kernel.
  - Done when: backend choice, feature flag name, and fallback behavior are documented.
  - Status (2026-05-13): CubeCL WGPU is the first execution backend, behind
    `cubecl-wgpu`. CPU remains the default fallback and parity oracle. CubeK is
    deferred to M4.6 for matmul/attention-size operations; Burn stays out of the
    kernel boundary for this spike.

- [x] M4.2 Add device and kernel dispatch contracts.
  - Crates: `ocelotl-core`, `ocelotl-kernels`
  - Test first: construct CPU and GPU device requests and assert unsupported device errors when the feature is absent.
  - Done when: callers cannot accidentally request GPU execution without an explicit supported device path.
  - Status (2026-05-13): `KernelBackend::context()` exposes `Device`, CPU
    requests fail `require_gpu` with a typed `UnsupportedError`, and
    `CubeClKernelBackend::new_gpu` advertises an explicit GPU context when the
    CubeCL feature is enabled.

- [x] M4.3 Implement the first GPU parity kernel.
  - Crates: `ocelotl-kernels`
  - Test first: reuse an existing CPU RMSNorm or RoPE fixture and assert CPU/GPU parity under the GPU feature.
  - Done when: one real GPU kernel executes through the kernel boundary with documented tolerance.
  - Status (2026-05-13): RoPE is implemented as the first CubeCL WGPU spike.
    The ignored local parity test compares against `rope_apply_inplace` at
    `1e-5` tolerance and passed locally.

- [x] M4.4 Add GPU test gating.
  - Crates: workspace, CI docs
  - Test first: default CI runs without GPU hardware and GPU tests are discoverable but skipped or feature-gated.
  - Done when: local GPU validation commands are documented and default tests remain portable.
  - Status (2026-05-13): CubeCL is optional. Default workspace checks do not
    build CubeCL. WGPU tests require `--features cubecl-wgpu`; the execution
    parity test is `#[ignore]` and has an explicit local command.

- [x] M4.5 Add failure tests for unsupported layouts.
  - Crates: `ocelotl-kernels`
  - Test first: pass unsupported shape, dtype, stride, or non-contiguous layout to the GPU path.
  - Done when: kernel dispatch rejects invalid layouts before launch.
  - Status (2026-05-13): CubeCL RoPE now carries an explicit
    `CubeClRopeLayout` with dtype, head_dim, and row_stride. Invalid shape,
    non-F32 dtype, and non-contiguous stride are rejected before the WGPU
    runtime client is requested.

- [x] M4.6 Evaluate CubeK or higher-level matmul integration.
  - Crates: docs, `ocelotl-kernels`
  - Test first: document the smallest blocked matmul or attention case that requires a library decision.
  - Done when: the repo has a clear decision to use, defer, or avoid CubeK for the next kernel family.
  - Status (2026-05-13): CubeK remains the preferred evaluation path for the
    next GPU matmul/attention-size operation, but is deferred out of M4 because
    RoPE is a better first boundary proof and the current Qwen model path still
    needs a device-buffer/tensor layout contract before CubeK can be adapted
    cleanly.

- [x] M4.7 Preserve CPU reference authority.
  - Crates: `ocelotl-kernels`, `ocelotl-runtime`
  - Test first: every GPU parity test has a CPU expected-output source.
  - Done when: GPU tests compare against CPU/reference fixtures, not GPU output from a previous run.
  - Status (2026-05-13): CubeCL RoPE, Qwen prefill, and Qwen one-token decode
    parity tests all compare against CPU output, not captured GPU output. The
    model/runtime path selects CubeCL only for RoPE and uses CPU fallback for
    matmul, attention, RMSNorm, MLP, residuals, and logits.

## Exit Criteria

- [x] At least one GPU-backed kernel executes through Ocelotl's kernel boundary.
- [x] CPU/GPU parity is tested with documented tolerance.
- [x] GPU tests are optional or feature-gated and do not break default CI.
- [x] Unsupported GPU layouts fail explicitly before kernel launch.

## Closure

M4 closed on 2026-05-13. The shipped GPU path is intentionally narrow:
Qwen2.5 prefill/decode can select a CubeCL WGPU execution backend, and that
backend executes RoPE on CubeCL while all non-M4 kernels run through the CPU
fallback. This satisfies the first-backend boundary proof without claiming full
model GPU execution.

## Deferred

- Full model GPU execution.
- Multi-GPU execution.
- GPU scheduler integration.
- GPU memory paging.
