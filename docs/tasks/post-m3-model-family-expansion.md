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
  upstream filter-surface behavior.
- `Follow-up`: Gemma4 GGUF tokenizer metadata extraction landed 2026-06-05.
  `ocelotl-loader` exposes `inspect_gguf_tokenizer` so the tokenizer track can
  read embedded GGUF tokens, scores, token types, merges, special-token IDs,
  chat-template text, and BOS/space-prefix flags without making the default
  `inspect_gguf` manifest retain 262k-token arrays. Default tests cover a tiny
  synthetic GGUF metadata fixture and malformed array typing; the ignored local
  Gemma4 proof passed against
  `D:\Dev\ideas\04-granola-ai-clone\models\google_gemma-4-E4B-it-Q4_K_M.gguf`
  on 2026-06-05.
- `Follow-up`: Gemma4 GGUF BPE tokenizer backend landed 2026-06-05.
  `ocelotl-tokenizer` constructs a byte-level BPE tokenizer from tokenizer-owned
  GGUF metadata parts, honors `add_space_prefix`, keeps plain `encode` BOS-free,
  exposes `encode_with_configured_bos`, registers control/unknown token types as
  skipped specials, and registers user-defined token types as literal added
  tokens. The root crate composes loader metadata into that tokenizer without
  adding a tokenizer-to-loader dependency. `fixtures/tokenizer/gemma4_gguf_basic_prompt.json`
  pins the selected GGUF backend's local `Hello` IDs as `[9259]` and configured
  BOS IDs as `[2, 9259]`; the ignored local proof passed against
  `D:\Dev\ideas\04-granola-ai-clone\models\google_gemma-4-E4B-it-Q4_K_M.gguf`
  on 2026-06-05.
- `Follow-up`: Gemma4 llama.cpp tokenizer-reference harness landed 2026-06-05.
  The root crate now has an ignored opt-in test that runs a local
  `llama-tokenize` binary with `--ids --no-escape --log-disable`, compares
  BOS-free output with `--no-bos`, compares configured-BOS output without
  `--no-bos`, and checks both against Ocelotl plus
  `fixtures/tokenizer/gemma4_gguf_basic_prompt.json`. Default tests cover the
  fixture's command schema and stdout parser. The ignored external proof passed
  locally on 2026-06-05 against llama.cpp commit
  `856c3adac1709be15e1ea2529a0e89f742d25fe0`
  (`b9127-1-g856c3adac`), matching `[9259]` and `[2, 9259]`.

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
  At initial landing, Gemma4 Q4_K_M GGUF execution remained rejected because
  multimodal, sliding-window, shared-KV, softcap, mixed width, and
  quantized-origin execution policies were not complete.
- `Follow-up`: Gemma4 real-shaped dequantized text weight adapter landed
  2026-06-05. `Gemma4TextWeights::from_loaded_tensors` now separates
  tensor-to-weight mapping from the executable text-forward feature gate, so a
  dequantized real-shaped bundle can map SWA and global attention layers with
  layer-specific q/k/v/norm widths. `Gemma4TextModel::new` still rejects that
  config before compute for real-artifact execution features. Later follow-up
  slices lifted the mixed-width and final-logit softcap blockers only for the
  synthetic text-forward subset.
- `Follow-up`: Gemma4 mixed SWA/global width text forward landed 2026-06-05.
  The synthetic text-only forward path now computes each decoder layer with its
  own SWA or global q/k/v widths and RoPE base. This lifts the mixed-width
  blocker only for unquantized dense F32, full-attention synthetic configs;
  multimodal real-artifact execution remains rejected before compute.
- `Follow-up`: Gemma4 final logit softcap landed 2026-06-05. The supported
  synthetic text-forward path now applies `cap * tanh(logit / cap)` after the
  tied embedding logits projection and rejects invalid non-positive or
  non-finite softcap values before compute. Real Q4_K_M execution remains
  blocked on multimodal handling and reference parity work.
- `Follow-up`: Gemma4 sliding-window masking landed 2026-06-09. The kernel
  boundary now has a windowed causal GQA attention variant, and
  `Gemma4TextModel::prefill` routes SWA layers through it when
  `attention_sliding_window` is present while global layers keep full causal
  attention. Real Q4_K_M execution remains blocked on multimodal handling and
  reference parity.
- `Follow-up`: Gemma4 sliding-window pattern metadata landed 2026-06-09. GGUF
  header inspection now preserves bool array values, `Gemma4Config` stores and
  validates one `gemma4.attention.sliding_window_pattern` entry per layer, and
  the text-forward path uses that pattern for SWA/global layer widths, RoPE
  base selection, and full/windowed attention dispatch. Real Q4_K_M execution
  remains blocked on multimodal handling and reference parity.
- `Follow-up`: Gemma4 shared-KV reuse landed 2026-06-09. The supported
  synthetic text-forward path now validates llama.cpp-style source mappings
  with `kv_from_start = block_count - shared_kv_layers`, reuses
  post-K-RMSNorm/post-RoPE K activations plus V activations from
  `kv_from_start - 2` for shared SWA/windowed layers and `kv_from_start - 1`
  for shared dense/global layers, and keeps current-layer Q, output projection,
  and MLP behavior. Real Q4_K_M execution remains blocked on multimodal
  handling and reference parity.
- `Follow-up`: Gemma4 dequantized Q4_K_M-origin text forward landed
  2026-06-09. The supported text path now accepts configs whose tensors came
  from Q4_K_M GGUF artifacts after the values have been explicitly
  materialized into Ocelotl-owned F32 weights through the dequantized loader
  path. Raw quantized loaded tensors still reject before compute. The selected
  real Q4_K_M artifact remains blocked on multimodal handling and reference
  parity.
- `Follow-up`: Gemma4 token embedding scaling landed 2026-06-10.
  `Gemma4TextModel::prefill` now applies llama.cpp-style `sqrt(hidden)`
  scaling to token embeddings before the first block, and
  `gemma4_text_prefill_scales_token_embeddings_before_first_block` pins that
  activation boundary before attention or MLP compute can hide it.
- `Follow-up`: Gemma4 attention/FFN semantics landed 2026-06-10.
  `scaled_dot_product_attention_with_scale` and its windowed variant preserve
  existing default attention behavior while allowing Gemma4 to use llama.cpp's
  `1.0` score scale. `Gemma4TextModel::prefill` now RMS-normalizes V with an
  all-ones weight before KV storage/reuse and runs FFN through tanh-approx
  GEGLU instead of SiLU. Kernel tests pin explicit-scale attention and ggml
  GEGLU values; `gemma4_text_prefill_normalizes_v_and_uses_explicit_attention_scale`
  pins the model boundary.
- `Follow-up`: Gemma4 post-branch/per-layer semantics landed 2026-06-10.
  `Gemma4TextWeights::from_loaded_tensors` now maps
  `blk.N.post_attention_norm.weight`, `blk.N.post_ffw_norm.weight`,
  `per_layer_token_embd.weight`, `per_layer_model_proj.weight`,
  `per_layer_proj_norm.weight`, per-layer PLE gate/projection/post-norm
  tensors, `blk.N.layer_output_scale.weight`, and `rope_freqs.weight`.
  `Gemma4TextModel::prefill` RMS-normalizes the attention output projection and
  FFN output before their residual adds, runs the per-layer embedding tail,
  applies layer output scales, and uses Gemma4 RoPE frequency factors for
  global attention while keeping SWA attention on plain RoPE.
  `gemma4_text_prefill_uses_post_attention_and_ffn_norm_weights`,
  `gemma4_text_prefill_uses_per_layer_embedding_block`,
  `gemma4_text_prefill_uses_layer_output_scale_weight`,
  `gemma4_text_prefill_uses_rope_frequency_factors_for_global_attention`, and
  `gemma4_text_prefill_ignores_rope_frequency_factors_for_swa_attention` pin
  those semantics. Remaining known parity blockers are no longer those text
  operations.
- `Follow-up`: Gemma4 GGUF matrix-layout and tensor-summary diagnostics landed
  2026-06-11. `Gemma4TextWeights::from_loaded_tensors` now treats GGUF
  embedding tensors as row-major token lookup tables while transposing GGUF
  `{input, output}` matrices into Ocelotl's row-major matmul layout. Hidden
  diagnostic tracing and ignored llama.cpp tensor-summary harnesses first
  isolated the real-artifact mismatch to `Qcur-0` after `inp_scaled` and
  `attn_norm-0` matched llama.cpp.
- `Follow-up`: native Gemma4 attention K-quant sidecars landed after that
  discriminator. `ocelotl-loader` exposes a bounded Q4_K/Q5_K/Q6_K raw-byte
  API, `ocelotl-kernels` executes Q5_K/Q6_K x Q8_K projections, and
  `Gemma4TextModel` can attach validated native Q/K/V/O sidecars while retaining
  dense fallback behavior for Q4_K. The selected layer-0 inventory is
  Q6_K/Q5_K/Q6_K/Q5_K. The local proof now matches llama.cpp through
  `kqv_out-0`; the first remaining mismatch is the Q5_K output projection
  (`attn_output_proj-0 = -3.062695` versus llama.cpp `node_33 = -3.404522`,
  diff `0.34182692`).

## MF.8 Add Opt-In Real-Artifact Parity

- `Crates`: `ocelotl-loader`, `ocelotl-models`, `ocelotl-runtime`.
- `Test first`: ignored local-artifact tests compare one short prompt against a
  pinned reference output or token/logit fixture.
- `Done when`: the test explains exact artifact paths and tolerance, and default
  CI remains offline.
- `Status`: Gemma4 logits harness slice landed 2026-06-10. The default suite
  validates `fixtures/logits/gemma4_q4_k_m_basic_prompt_logits_reference.json`
  for the selected Q4_K_M artifact, configured-BOS prompt token IDs, tolerance,
  selected probe IDs, and llama.cpp `llama-debug --save-logits` command shape.
  The ignored `local_gemma4_q4_k_m_prefill_logits_match_llama_cpp_debug` proof
  requires the selected GGUF plus `OCELOTL_LLAMA_DEBUG_PATH`, emits full
  final-token logits from llama.cpp, projects Ocelotl's loaded config to the
  text-only path by clearing `multimodal`, loads required tensors through
  `load_gemma4_dequantized_tensors_from_gguf`, and compares every
  final-position logit through `ocelotl_runtime::gemma::prefill`.
- `Status`: latest local tensor-summary proof on 2026-06-11 runs against
  llama.cpp `856c3adac1709be15e1ea2529a0e89f742d25fe0` and is still red.
  After the GGUF matrix-layout and native-attention changes, the layer-0 proof
  matches through `kqv_out-0`. The first mismatch is now the native Q5_K
  `attn_output.weight` projection (`attn_output_proj-0 = -3.062695` versus
  llama.cpp `node_33 = -3.404522`, diff `0.34182692`). The full-logit proof has
  not been refreshed after these changes. Remaining work should add a
  selected-row or selected-element Q5_K output discriminator, fix that numeric
  delta, then rerun layer outputs and the complete logits vector before claiming
  MF.8 parity.

## Track Closure

This track closes when Qwen3.5 and Gemma4 can both be inspected and rejected
correctly for unsupported features, and at least one explicitly supported subset
has a tiny synthetic runtime fixture without regressing Qwen2.5 M3 parity.
