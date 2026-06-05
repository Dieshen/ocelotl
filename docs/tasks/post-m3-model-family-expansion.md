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
- `Status`: done 2026-05-13. `fixtures/manifest/post_m3_model_family_targets.json`
  pins Qwen3.5 FP8 safetensors and Gemma4 Q4_K_M GGUF candidates.

## MF.2 Add GGUF Header-Only Inspection

- `Crates`: `ocelotl-loader`.
- `Test first`: a tiny synthetic GGUF header fixture parses into an
  Ocelotl-owned manifest without reading tensor payloads.
- `Done when`: truncated headers, unsupported versions, bad tensor offsets, and
  oversized metadata fail with typed errors.
- `Status`: done 2026-05-13. `ocelotl_loader::inspect_gguf` parses GGUF v3
  metadata/tensor descriptors, summarizes array metadata without storing large
  tokenizer arrays, validates tensor offsets, and exposes an ignored local
  Gemma4 Q4_K_M header-contract proof.

## MF.3 Add Gemma4 Metadata Contract

- `Crates`: `ocelotl-models`.
- `Test first`: a Gemma4 metadata fixture converts into a `Gemma4Config` or
  fails with a typed unsupported error.
- `Done when`: Gemma4-specific context length, sliding window, shared KV,
  softcapping, tokenizer metadata presence, and quantization status are
  preserved or explicitly rejected.
- `Status`: done 2026-05-13. `Gemma4Config` converts the distilled GGUF
  metadata fixture, preserves context/sliding-window/shared-KV/softcap/tokenizer
  and Q4_K_M status, and rejects Gemma4 execution features before compute.

## MF.4 Add Qwen3.5 Metadata Contract

- `Crates`: `ocelotl-models`.
- `Test first`: a Qwen3.5 metadata fixture proves Ocelotl recognizes the family
  separately from Qwen2.5.
- `Done when`: unsupported hybrid/MoE/multimodal features fail before compute,
  and Qwen2.5 tests prove the existing path still accepts only its intended
  dense decoder contract.
- `Status`: done 2026-05-13. `parse_qwen3_5_config_json` recognizes the
  Qwen3.5 MoE config separately from Qwen2.5 and `Qwen3_5Config` rejects
  hybrid attention, MoE, multimodal, and FP8 execution features before compute.

## MF.5 Validate Required Tensor Inventories

- `Crates`: `ocelotl-models`, `ocelotl-loader`.
- `Test first`: required tensor names and shapes are enumerated for one selected
  Gemma4 artifact and one selected Qwen3.5 artifact.
- `Done when`: missing tensors, wrong shapes, unsupported dtypes, and quantized
  tensors without a dequant policy fail with typed errors.
- `Status`: Gemma4 side done 2026-06-05 for
  `google_gemma-4-E4B-it-Q4_K_M.gguf`. `Gemma4Config` now preserves the
  per-layer input width, SWA/global attention key/value widths, SWA RoPE facts,
  and sliding-window pattern length needed to derive the selected artifact's
  720 required GGUF tensor descriptors. Inventory validation accepts the real
  local header and rejects execution until a Gemma4 dequant policy exists.
  Qwen3.5 tensor inventory remains pending.
- `Follow-up`: Gemma4 dense GGUF value loading landed 2026-06-05.
  `ocelotl-loader` can load dense GGUF F32/F16/BF16 tensors into
  `LoadedTensor`, and `ocelotl-models` can load/validate the selected Gemma4
  artifact's 340 dense tensors, including BF16 PLE projection weights, while
  keeping Q4K/Q5K/Q6K matrices behind the dequant policy gate.
- `Follow-up`: GGUF Q4K/Q5K/Q6K layout policy landed 2026-06-05.
  `ocelotl-loader` now records ggml's K-quant block contract
  (256 elements/block; Q4K 144 bytes, Q5K 176 bytes, Q6K 210 bytes), validates
  block-aligned element counts and byte ranges during header inspection, and
  still returns typed `Unsupported` for quantized value loading.
- `Follow-up`: GGUF Q4K/Q5K/Q6K dequantized value loading landed 2026-06-05.
  The dense `load_gguf_tensor_f32` API remains dense-only, while
  `load_gguf_tensor_dequantized_f32` and
  `load_gguf_tensors_dequantized_f32` explicitly materialize Q4K/Q5K/Q6K
  tensors into F32 values. Exact small-vector tests pin ggml scale/min packing,
  Q5 high-bit lanes, and Q6 signed scales. `ocelotl-models` can now load and
  validate all required tensors from a tiny synthetic Gemma4 GGUF through the
  dequantized path, while Gemma4 execution remains rejected until MF.7.

## MF.6 Pin Tokenizer And Chat Template Behavior

- `Crates`: `ocelotl-tokenizer`.
- `Test first`: default-on shape fixtures plus ignored real-artifact tests for
  tokenization and chat-template behavior.
- `Done when`: each family has deterministic tokenizer/template fixtures without
  adding network access to default tests.
- `Status`: Gemma4 chat-template compatibility slice landed 2026-06-05.
  `ChatTemplate::apply_with_options` accepts serializable structured
  messages/tools plus `bos_token` and `enable_thinking`, MiniJinja macros are
  enabled, and default-on tests pin llama.cpp-style Gemma4 BOS, thinking,
  assistant-as-model, multimodal placeholder, tool-call/tool-response, and
  upstream filter-surface behavior. Exact tokenizer ID fixtures and ignored
  real-artifact drift checks remain pending for Gemma4 and Qwen3.5.

## MF.7 Add Tiny Synthetic Forward Per Supported Subset

- `Crates`: `ocelotl-models`, `ocelotl-runtime`, `ocelotl-kernels`.
- `Test first`: a tiny synthetic model for the explicitly supported subset
  produces pinned logits through the public runtime path.
- `Done when`: Qwen3.5 or Gemma4 has a minimal forward path only for the subset
  whose metadata/tensors are already validated.
- `Status`: Gemma4 text-only synthetic subset landed 2026-06-05.
  `Gemma4TextModel` and `Gemma4TextWeights` execute a dense F32 CPU/reference
  decoder-core path through `ocelotl_runtime::gemma::{prefill,
  decode_one_token}`, including embedding lookup, RMSNorm, q/k/v projection,
  Gemma q/k RMSNorm, RoPE, full attention, gated SiLU MLP, final norm, tied
  embedding logits, and greedy decode. The pinned runtime fixture is
  `fixtures/logits/gemma4_tiny_synthetic_text_prefill.json`, including
  `expected_decode_token = 7`. Runtime builder tests prove CPU selection and
  the no-launch CubeCL/WGPU backend contract for the supported subset. Real
  Gemma4 Q4_K_M GGUF execution remains rejected because multimodal,
  sliding-window, shared-KV, softcap, mixed width, and quantized-origin
  execution policies are not complete.

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
