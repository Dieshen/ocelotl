# Test-Driven Development

Ocelotl should be built test-first. For this project, TDD is not a style choice;
it is how we prevent plausible but wrong model output from becoming accepted
behavior.

## Policy

Every non-trivial change should start with one of these:

- A failing unit test for a pure contract.
- A failing fixture test for model, tokenizer, loader, or runtime behavior.
- A failing parity test against a reference implementation or committed fixture.
- A failing unsupported-config test that proves an invalid path is rejected.
- A benchmark harness only after correctness tests exist.

Implementation follows the smallest path that makes the test pass. Refactoring
comes after the test passes and must keep the test green.

## TDD Loop

1. Write or update the spec for the behavior.
2. Add the smallest failing test that captures the behavior.
3. Run the focused test and confirm it fails for the expected reason.
4. Implement the smallest correct change.
5. Re-run the focused test.
6. Run the relevant crate tests.
7. Run workspace validation before merging.
8. Update docs if the contract changed.

## Test Pyramid

- Unit tests: shape checks, metadata validation, token processing, sampling math.
- Fixture tests: tokenizer IDs, metadata normalization, known logits, known tokens.
- Parity tests: CPU/reference, GPU/CPU, contiguous/paged KV, quantized/reference.
- Integration tests: full request lifecycle through runtime APIs.
- Benchmarks: performance only after correctness is pinned.

## Required Test Categories By Area

Loader:

- Accepts known-good fixture metadata.
- Rejects missing tensors and mismatched shapes.
- Rejects unsupported architecture, dtype, quantization, and RoPE settings.

Tokenizer:

- Encodes known prompts to exact token IDs.
- Decodes known token IDs to exact text.
- Documents BOS/EOS behavior with tests.
- Renders chat templates deterministically.

Runtime:

- Rejects invalid requests before compute.
- Runs prefill and decode through the same public API.
- Cleans up request state on error and cancellation.
- Keeps deterministic greedy output stable.

Kernels:

- Match CPU reference for small hand-checked tensors.
- Match CPU reference across layout and boundary cases.
- Reject unsupported dtype, stride, and shape combinations.

KV cache:

- Writes and reads exact positions.
- Rejects out-of-range positions.
- Proves multi-page behavior when paged KV exists.
- Keeps request ownership isolated.

Scheduler:

- Preserves output order per request.
- Handles cancellation and cleanup.
- Does not change deterministic greedy outputs under batching.

## Merge Gate

A change is not ready if it only passes `cargo check`. It should include the
smallest meaningful test for the behavior it introduces. If a test cannot be
written yet, the PR should explain which missing harness blocks it and add that
harness as follow-up work.

## Anti-Patterns

- Adding a model path without an unsupported-config test.
- Adding a GPU kernel without CPU parity.
- Adding a loader feature without a malformed fixture.
- Adding a scheduler behavior with only real-model integration tests.
- Treating generated text quality as a substitute for logit or token parity.
