# M4 Tasks

M4 adds the first GPU-backed kernel path with CPU/GPU parity. GPU support must be
feature-gated and must not weaken the CPU reference path.

## Entry Criteria

- M3 CPU forward path passes with deterministic fixtures.
- Kernel interfaces include explicit shape, stride, dtype, and device contracts.
- Library notes for Burn, CubeCL, and CubeK are current enough to guide the first backend decision.

## Task List

- [ ] M4.1 Choose and document the first GPU execution backend.
  - Crates: docs, `ocelotl-kernels`
  - Test first: add a design note describing why the chosen backend can support the first parity kernel.
  - Done when: backend choice, feature flag name, and fallback behavior are documented.

- [ ] M4.2 Add device and kernel dispatch contracts.
  - Crates: `ocelotl-core`, `ocelotl-kernels`
  - Test first: construct CPU and GPU device requests and assert unsupported device errors when the feature is absent.
  - Done when: callers cannot accidentally request GPU execution without an explicit supported device path.

- [ ] M4.3 Implement the first GPU parity kernel.
  - Crates: `ocelotl-kernels`
  - Test first: reuse an existing CPU RMSNorm or RoPE fixture and assert CPU/GPU parity under the GPU feature.
  - Done when: one real GPU kernel executes through the kernel boundary with documented tolerance.

- [ ] M4.4 Add GPU test gating.
  - Crates: workspace, CI docs
  - Test first: default CI runs without GPU hardware and GPU tests are discoverable but skipped or feature-gated.
  - Done when: local GPU validation commands are documented and default tests remain portable.

- [ ] M4.5 Add failure tests for unsupported layouts.
  - Crates: `ocelotl-kernels`
  - Test first: pass unsupported shape, dtype, stride, or non-contiguous layout to the GPU path.
  - Done when: kernel dispatch rejects invalid layouts before launch.

- [ ] M4.6 Evaluate CubeK or higher-level matmul integration.
  - Crates: docs, `ocelotl-kernels`
  - Test first: document the smallest blocked matmul or attention case that requires a library decision.
  - Done when: the repo has a clear decision to use, defer, or avoid CubeK for the next kernel family.

- [ ] M4.7 Preserve CPU reference authority.
  - Crates: `ocelotl-kernels`, `ocelotl-runtime`
  - Test first: every GPU parity test has a CPU expected-output source.
  - Done when: GPU tests compare against CPU/reference fixtures, not GPU output from a previous run.

## Exit Criteria

- At least one GPU-backed kernel executes through Ocelotl's kernel boundary.
- CPU/GPU parity is tested with documented tolerance.
- GPU tests are optional or feature-gated and do not break default CI.
- Unsupported GPU layouts fail explicitly before kernel launch.

## Deferred

- Full model GPU execution.
- Multi-GPU execution.
- GPU scheduler integration.
- GPU memory paging.
