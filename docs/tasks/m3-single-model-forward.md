# M3 Tasks

M3 runs a Qwen2.5-style single-model forward path through Ocelotl APIs. The goal
is one narrow, correct prefill and one-token decode path before optimizing or
serving multiple requests.

## Entry Criteria

- M1 CPU reference behavior is deterministic.
- M2 loader and tokenizer contracts work with local artifacts.
- Required tensor names and model metadata for the first target family are known.

## Task List

- [ ] M3.1 Define the Qwen2.5 model-family metadata contract.
  - Crates: `ocelotl-models`, `ocelotl-core`
  - Test first: create a valid Qwen2.5 metadata fixture and assert conversion into a model-family config.
  - Done when: model-specific validation lives in `ocelotl-models` and shared metadata remains in `ocelotl-core`.

- [ ] M3.2 Validate required tensor names and shapes.
  - Crates: `ocelotl-loader`, `ocelotl-models`
  - Test first: add a manifest fixture with one missing tensor and assert the exact missing-tensor error.
  - Done when: embedding, attention, MLP, norm, and output tensors are checked before execution.

- [ ] M3.3 Implement CPU RMSNorm for the model path.
  - Crates: `ocelotl-kernels`, `ocelotl-models`
  - Test first: compare a tiny RMSNorm input against a hand-computed expected vector.
  - Done when: the model forward path calls the kernel boundary rather than inline math in `ocelotl-models`.

- [ ] M3.4 Implement CPU RoPE application for the target shape.
  - Crates: `ocelotl-kernels`, `ocelotl-models`
  - Test first: verify position 0 identity behavior and one non-zero-position hand-checked vector.
  - Done when: RoPE configuration is derived from metadata and invalid head dimensions fail before compute.

- [ ] M3.5 Implement CPU attention for a single request.
  - Crates: `ocelotl-kernels`, `ocelotl-models`
  - Test first: run a tiny one-head attention fixture with expected probabilities and output.
  - Done when: causal masking and KV head mapping are explicit and tested.

- [ ] M3.6 Implement CPU MLP for the target activation.
  - Crates: `ocelotl-kernels`, `ocelotl-models`
  - Test first: add a tiny gated-MLP fixture with expected output.
  - Done when: activation, gate, up, and down projections are wired in the target model block.

- [ ] M3.7 Produce prefill logits for a tiny model fixture.
  - Crates: `ocelotl-models`, `ocelotl-runtime`
  - Test first: add or generate a tiny logits fixture and assert the final-token logits within documented tolerance.
  - Done when: prefill runs through embedding, transformer block, final norm, and output projection.

- [ ] M3.8 Produce one-token decode logits through the same public path.
  - Crates: `ocelotl-models`, `ocelotl-runtime`
  - Test first: prefill a tiny prompt, decode one token, and assert deterministic logits or selected token.
  - Done when: decode does not bypass runtime or model APIs used by prefill.

- [ ] M3.9 Add parity rules and tolerances for M3 fixtures.
  - Crates: docs, tests
  - Test first: a tolerance change causes a test or doc update requirement.
  - Done when: `docs/validation/parity.md` names the M3 comparison source and acceptable tolerances.

- [ ] M3.10 Reject unsupported model shapes before execution.
  - Crates: `ocelotl-models`, `ocelotl-runtime`
  - Test first: invalid hidden size, head count, KV head count, dtype, and RoPE config fixtures fail at construction time.
  - Done when: runtime does not launch a forward pass for unsupported shape combinations.

## Exit Criteria

- A Qwen2.5-style tiny model can run prefill through public runtime APIs.
- One-token decode uses the same model and runtime contracts.
- Required tensors, shapes, and dtypes are validated before execution.
- CPU outputs are compared against documented fixture expectations.
- Unsupported target-shape variants fail explicitly.

## Deferred

- GPU acceleration.
- Paged KV cache.
- Continuous batching.
- Sampling beyond deterministic greedy tests.
- Large model performance.
