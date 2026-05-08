# Post-M3 Model-Family Expansion Tasks

Working shorthand: "M3.6 Qwen3.5 and Gemma4". This is a post-M3 expansion track
and does not modify the closed M3.6 MLP task.

## MF.1 Pin Candidate Artifacts

- `Crates`: docs/fixtures only.
- `Test first`: a fixture manifest names candidate model repos, revisions, file
  formats, license, and expected local paths.
- `Done when`: Qwen3.5 and Gemma4 each have one candidate artifact with a pinned
  source and an explicit note on whether it is safetensors, GGUF, quantized, or
  multimodal.

## MF.2 Add GGUF Header-Only Inspection

- `Crates`: `ocelotl-loader`.
- `Test first`: a tiny synthetic GGUF header fixture parses into an
  Ocelotl-owned manifest without reading tensor payloads.
- `Done when`: truncated headers, unsupported versions, bad tensor offsets, and
  oversized metadata fail with typed errors.

## MF.3 Add Gemma4 Metadata Contract

- `Crates`: `ocelotl-models`.
- `Test first`: a Gemma4 metadata fixture converts into a `Gemma4Config` or
  fails with a typed unsupported error.
- `Done when`: Gemma4-specific context length, sliding window, shared KV,
  softcapping, tokenizer metadata presence, and quantization status are
  preserved or explicitly rejected.

## MF.4 Add Qwen3.5 Metadata Contract

- `Crates`: `ocelotl-models`.
- `Test first`: a Qwen3.5 metadata fixture proves Ocelotl recognizes the family
  separately from Qwen2.5.
- `Done when`: unsupported hybrid/MoE/multimodal features fail before compute,
  and Qwen2.5 tests prove the existing path still accepts only its intended
  dense decoder contract.

## MF.5 Validate Required Tensor Inventories

- `Crates`: `ocelotl-models`, `ocelotl-loader`.
- `Test first`: required tensor names and shapes are enumerated for one selected
  Gemma4 artifact and one selected Qwen3.5 artifact.
- `Done when`: missing tensors, wrong shapes, unsupported dtypes, and quantized
  tensors without a dequant policy fail with typed errors.

## MF.6 Pin Tokenizer And Chat Template Behavior

- `Crates`: `ocelotl-tokenizer`.
- `Test first`: default-on shape fixtures plus ignored real-artifact tests for
  tokenization and chat-template behavior.
- `Done when`: each family has deterministic tokenizer/template fixtures without
  adding network access to default tests.

## MF.7 Add Tiny Synthetic Forward Per Supported Subset

- `Crates`: `ocelotl-models`, `ocelotl-runtime`, `ocelotl-kernels`.
- `Test first`: a tiny synthetic model for the explicitly supported subset
  produces pinned logits through the public runtime path.
- `Done when`: Qwen3.5 or Gemma4 has a minimal forward path only for the subset
  whose metadata/tensors are already validated.

## MF.8 Add Opt-In Real-Artifact Parity

- `Crates`: `ocelotl-loader`, `ocelotl-models`, `ocelotl-runtime`.
- `Test first`: ignored local-artifact tests compare one short prompt against a
  pinned reference output or token/logit fixture.
- `Done when`: the test explains exact artifact paths and tolerance, and default
  CI remains offline.

## Track Closure

This track closes when Qwen3.5 and Gemma4 can both be inspected and rejected
correctly for unsupported features, and at least one explicitly supported subset
has a tiny synthetic runtime fixture without regressing Qwen2.5 M3 parity.

