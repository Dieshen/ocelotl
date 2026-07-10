# Post-M3 Model-Family Expansion

Working shorthand: "M3.6 Qwen3.5 and Gemma4". This is not the closed M3.6 task
in `docs/tasks/m3-single-model-forward.md`; that task is CPU MLP. This document
captures a later model-family expansion track without reopening M3 acceptance
history.

## Goal

Prepare Ocelotl to support additional local model families after the Qwen2.5 M3
path:

- Qwen3.5
- Gemma4

The first useful result is compatibility discovery and rejection correctness, not
full text/audio/video multimodal serving.

## Current Verified Facts

Qwen3.5:

- Official Hugging Face model cards exist under the `Qwen/` namespace, including
  `Qwen/Qwen3.5-35B-A3B`.
- MF.1 pins `Qwen/Qwen3.5-35B-A3B-FP8` at revision
  `9d1823d2dee688a6b25e77009dc727688c44936e` as the first Qwen3.5
  compatibility-discovery artifact. The base non-FP8 repo was observed at
  `59d61f3ce65a6d9863b86d2e96597125219dc754` at pin time.
- The Qwen3.5 model card describes a unified vision-language foundation and an
  efficient hybrid architecture using Gated Delta Networks plus sparse
  Mixture-of-Experts.
- That means Qwen3.5 is not a small Qwen2.5 dense-decoder extension.

Gemma4:

- Official Google Hugging Face model cards exist, including
  `google/gemma-4-E4B`.
- MF.1 pins `bartowski/google_gemma-4-E4B-it-GGUF` at revision
  `c04cb322fd63e347db759a08b6249b867488ccf8` for
  `google_gemma-4-E4B-it-Q4_K_M.gguf`. The base `google/gemma-4-E4B-it` repo
  was observed at `3555bddc93a623db8887dd2e52123facc45ade77` at pin time.
- Gemma4 models are multimodal. The small E2B/E4B models include native audio
  support; all generate text output.
- The local Gemma4 GGUF artifact inspected during post-M3 reconnaissance is
  quantized GGUF v3 with `general.architecture = gemma4`, embedded tokenizer
  metadata, sliding-window/shared-KV metadata, and Gemma-specific softcapping.
- MF.2 adds `ocelotl_loader::inspect_gguf`, a bounded header-only GGUF v3
  inspector that normalizes metadata and tensor descriptors into Ocelotl-owned
  structs without reading tensor payload bytes. The ignored local proof passed
  against `google_gemma-4-E4B-it-Q4_K_M.gguf` on 2026-05-13.
- MF.3 adds `Gemma4Config`, a model-layer projection of the GGUF manifest that
  preserves Gemma4 context length, sliding-window attention, shared-KV layers,
  final-logit softcapping, embedded tokenizer metadata, and Q4_K_M status while
  rejecting Gemma4 execution before compute.
- MF.5's Gemma4 slice adds the selected Q4_K_M artifact's required GGUF tensor
  inventory: 720 tensor descriptors covering global embeddings/projections and
  42 decoder blocks with SWA/global attention width differences. The inventory
  validator accepts F32 norms/scales, BF16 per-layer projection weights, and
  Q4K/Q5K/Q6K quantized matrices from the real local header, while execution
  remains rejected until a Gemma4 dequant policy exists.
- A follow-up Gemma4 value-loading slice adds dense GGUF payload reads for
  F32/F16/BF16 tensors. The selected Q4_K_M artifact's 340 dense tensors load
  into `LoadedTensor`, including BF16 PLE projection weights.
- A follow-up GGUF K-quant value-loading slice records the current ggml
  Q4_K/Q5_K/Q6_K block contract in the loader: 256 elements per block, with
  144, 176, and 210 bytes per block respectively. Parsed GGUF headers now
  validate K-quant tensor element counts and byte ranges. The explicit
  dequantized APIs load dense and Q4K/Q5K/Q6K tensors into F32 values, with
  exact vector tests for each format and a tiny synthetic Gemma4 all-tensor load;
  execution remains rejected until MF.7.
- A Gemma4 chat-template compatibility slice expands the tokenizer renderer
  context for current llama.cpp-style Gemma4 templates: macros are enabled in
  MiniJinja, and render options now include `bos_token`, `enable_thinking`,
  structured `tools`, and serializable message objects for tool/media fields.
- A follow-up GGUF tokenizer-metadata slice adds
  `ocelotl_loader::inspect_gguf_tokenizer`, a targeted extractor for embedded
  GGUF tokenizer tokens, scores, token types, merges, special-token IDs,
  chat-template text, and BOS/space-prefix flags. This keeps the default
  `inspect_gguf` manifest header-only and array-summarized, while giving the
  Gemma4 tokenizer track a tested bridge toward exact ID parity.
- A follow-up GGUF tokenizer backend slice adds `GgufBpeTokenizer` in
  `ocelotl-tokenizer` and root-crate Gemma4 composition helpers. The backend
  builds byte-level BPE tokenization from embedded GGUF tokens/merges, honors
  `add_space_prefix`, keeps plain `encode` BOS-free, provides an explicit
  configured-BOS path, and uses GGUF token types to register control/unknown and
  user-defined tokens. `fixtures/tokenizer/gemma4_gguf_basic_prompt.json` pins
  local backend IDs for `Hello`.
- A follow-up llama.cpp tokenizer-reference slice adds an ignored root-crate
  harness that runs a local `llama-tokenize` binary, parses `--ids` stdout, and
  compares both BOS-free and configured-BOS Gemma4 token IDs against Ocelotl.
  The default suite checks the harness parser and fixture command schema. The
  ignored external proof passed locally on 2026-06-05 against llama.cpp commit
  `856c3adac1709be15e1ea2529a0e89f742d25fe0`
  (`b9127-1-g856c3adac`) with IDs `[9259]` and `[2, 9259]`.
- MF.7 adds the first Gemma4 execution subset: `Gemma4TextModel` runs a
  text-only, unquantized, dense F32 synthetic decoder-core path through
  `ocelotl_runtime::gemma::{prefill, decode_one_token}`, with pinned logits and
  greedy decode token `TokenId(7)` in
  `fixtures/logits/gemma4_tiny_synthetic_text_prefill.json`. Runtime builder
  tests also prove CPU backend selection and the no-launch CubeCL/WGPU backend
  contract for the subset. At initial landing this did not enable the selected
  real Q4_K_M GGUF artifact; multimodal, sliding-window/shared-KV, softcap,
  mixed-width, and quantized-origin execution features remained rejected before
  compute.
- A follow-up real-shaped adapter slice separates Gemma4 text weight mapping
  from the executable feature gate. Dequantized tensors with SWA/global
  layer-specific q/k/v/norm widths can now map into `Gemma4TextWeights`, while
  `Gemma4TextModel::new` continues to reject those real-artifact features
  before compute.
- A follow-up text-forward slice makes the CPU/reference model loop
  layer-aware for SWA/global attention widths and RoPE bases. Mixed SWA/global
  widths are now executable only for the text-only unquantized dense F32
  full-attention subset; multimodal and reference real-artifact parity remain
  open.
- A follow-up final-logit softcap slice applies Gemma4's
  `cap * tanh(logit / cap)` transform in the same supported synthetic
  text-forward path and validates the cap before compute. This does not enable
  the selected multimodal real artifact yet.
- A follow-up sliding-window mask slice adds a windowed causal GQA kernel and
  routes Gemma4 SWA layers through it when `attention_sliding_window` is set.
  Global layers keep full causal attention. This still does not enable the real
  Q4_K_M artifact because multimodal handling and reference parity work remain
  open.
- A follow-up sliding-window pattern slice preserves GGUF bool array values for
  `gemma4.attention.sliding_window_pattern`, validates one pattern entry per
  layer, and uses the pattern as the authority for SWA/global width, RoPE base,
  and full/windowed attention dispatch. This matches llama.cpp's convention
  that `true` means SWA/windowed and `false` means dense/global, while keeping
  multimodal handling and real-artifact logits parity open.
- A follow-up shared-KV slice implements the llama.cpp source-layer mapping for
  the supported synthetic text path: `kv_from_start = block_count -
  shared_kv_layers`, shared SWA/windowed layers reuse `kv_from_start - 2`, and
  shared dense/global layers reuse `kv_from_start - 1`. `Gemma4TextModel`
  reuses cached post-K-RMSNorm, post-RoPE K activations plus V activations from
  the source layer while keeping current-layer Q, output projection, and MLP
  behavior. A later semantics slice normalizes V before it is cached. Real
  Q4_K_M execution remains blocked on multimodal handling and real-artifact
  logits parity.
- A follow-up dequantized-origin slice allows the same text-forward path to
  execute Q4_K_M-origin Gemma4 text tensors after they have been explicitly
  materialized through `load_gguf_tensors_dequantized_f32`. The gate now trusts
  the Ocelotl-owned F32 `Gemma4TextWeights` bundle rather than the artifact's
  quantization metadata. Raw quantized loaded tensors remain rejected by the
  dequantized tensor validator. The selected real artifact remains blocked on
  multimodal handling and real-artifact logits parity.
- A follow-up real-artifact logits harness slice adds an MF.8 opt-in
  llama.cpp reference proof for the selected Gemma4 Q4_K_M GGUF. The default
  suite validates the small logits-reference fixture and `llama-debug
  --save-logits` command schema. The ignored local test runs llama.cpp to emit
  the full final-token logits vector for `Hello`, tokenizes the same prompt with
  Ocelotl's configured-BOS GGUF path, clears `Gemma4Config.multimodal` only for
  this text-only harness, loads dequantized F32 text weights, and compares
  every final-position logit through `ocelotl_runtime::gemma::prefill`.
  The 2026-06-10 local run after the attention-scale, V-RMSNorm, tanh-GEGLU,
  post-branch RMSNorm, per-layer embedding, layer-output-scale, and RoPE
  frequency-factor fixes on 2026-06-10 proved the harness works but parity is
  still red: token 0 differed by `15.670528` against llama.cpp
  `856c3adac1709be15e1ea2529a0e89f742d25fe0`.
- A follow-up Gemma4 text semantics slice applies llama.cpp-style
  `sqrt(hidden)` token embedding scaling before the first block and pins it
  with a model-level activation-boundary test. This is the first closed drift
  item from the failed real-artifact logits proof, not the end of MF.8 parity.
- A follow-up Gemma4 attention/FFN semantics slice adds llama.cpp's text path
  score scale (`1.0` instead of `1/sqrt(head_dim)`), unweighted V RMSNorm
  before KV storage/reuse, and tanh-approx GEGLU FFN activation. Kernel tests
  pin explicit-scale full/windowed attention and ggml-style GEGLU values;
  `gemma4_text_prefill_normalizes_v_and_uses_explicit_attention_scale` pins the
  model boundary.
- A follow-up Gemma4 post-branch/per-layer semantics slice maps
  `blk.N.post_attention_norm.weight`, `blk.N.post_ffw_norm.weight`,
  `per_layer_token_embd.weight`, `per_layer_model_proj.weight`,
  `per_layer_proj_norm.weight`, per-layer PLE gate/projection/post-norm
  tensors, `blk.N.layer_output_scale.weight`, and `rope_freqs.weight`.
  `Gemma4TextModel::prefill` now applies weighted RMSNorm to the attention
  output projection and dense FFN output before their residual adds, runs the
  per-layer embedding tail, applies layer output scales, and routes global
  attention through Gemma4 RoPE frequency factors while keeping SWA attention
  on plain RoPE. Model tests pin each behavior. Real-artifact parity remains
  red after this slice; the selected artifact has `attn_v.weight` tensors and
  no detected `output.weight` or MoE names, so the remaining likely blocker is
  no longer those optional branches.
- A follow-up Gemma4 GGUF matrix-layout and tensor-summary diagnostics slice
  keeps token and per-layer-token embeddings in GGUF row order for lookup,
  transposes GGUF `{input, output}` matrices into Ocelotl row-major matmul
  layout, and adds ignored llama.cpp tensor-summary harnesses for late tensors,
  layer outputs, and layer-0 substeps. The initial 2026-06-11 local
  discriminator showed `inp_scaled` and `attn_norm-0` matching llama.cpp before
  `Qcur-0` diverged.
- A follow-up native attention K-quant slice adds bounded Q4_K/Q5_K/Q6_K raw
  payload loading, Q5_K/Q6_K x Q8_K kernels, and validated Gemma4 Q/K/V/O
  sidecars. The selected layer-0 inventory is Q6_K/Q5_K/Q6_K/Q5_K, and the
  public text parity path now matches llama.cpp through `kqv_out-0`. The active
  real-artifact blocker is the Q5_K output projection: Ocelotl
  `attn_output_proj-0 = -3.062695` versus llama.cpp
  `node_33 = -3.404522`, diff `0.34182692` at tolerance `0.05`. Q4_K native
  projection remains explicitly unsupported and uses the dequantized dense
  fallback.
- MF.4 adds `Qwen3_5Config`, a Qwen-family metadata contract for the
  `qwen3_5_moe` Hugging Face config shape. It recognizes Qwen3.5 separately
  from Qwen2.5 and rejects hybrid attention, sparse MoE, multimodal, and FP8
  execution features before compute.

## Boundary

Use family modules instead of flattening every architecture into the Qwen2.5
path:

- `crates/models/src/qwen/` keeps Qwen-family implementations.
- `crates/models/src/gemma/` should own Gemma-specific implementations.
- Qwen3.5 gets a separate config/validation contract from Qwen2.5 even if it
  lives under the same `qwen` family module.
- GGUF parsing belongs in `ocelotl-loader`, not `ocelotl-models`.

Public root exports may re-export stable types for ergonomics, but internal files
should stay family-scoped.

## Non-Goals

- Full Qwen3.5 multimodal support in the first slice.
- Full Gemma4 audio/image/video support in the first slice.
- Quantized GGUF execution before manifest and dequant policy are explicit.
- MoE routing before small metadata fixtures prove the contract.
- Replacing Whisper ASR with Gemma4 audio.
- GPU execution without CPU/reference parity.

## TDD Plan

Write tests before implementation for:

- Header-only or metadata-only artifact inspection.
- Model-family config conversion from pinned metadata fixtures.
- Explicit rejection of unsupported Qwen3.5/Gemma4 features.
- Required tensor-name/shape inventories for the selected first artifact.
- Tokenizer/chat-template fixture shape.
- Tiny synthetic forward only after metadata and tensor validation pass.

## Design Notes

### Qwen3.5

Do not assume the M3 Qwen2.5 dense path applies. The first Qwen3.5 task should
capture metadata and explicitly reject unsupported hybrid/MoE/multimodal features
until Ocelotl has a tested implementation for them.

Pick the smallest official artifact that matches the product need before writing
forward code. A 35B-A3B or larger model can be the compatibility target, but it
should not be the default test artifact.

### Gemma4

The local artifact is GGUF and quantized. Start with a GGUF inspector and a
Gemma4 manifest contract before any execution work.

Gemma4 E4B audio support may be useful for product workflows, but it is not a
drop-in Whisper replacement inside Ocelotl. Treat Gemma4 audio as multimodal
reasoning/text generation. Treat Whisper as the transcription-first path unless
real fixtures prove Gemma4 matches the ASR requirements.

### Loader format split

Safetensors remains the first supported real Qwen2.5 path. GGUF needs its own
bounded header/metadata inspector. Do not read multi-GB tensor payloads just to
decide whether a model is supported.

## Acceptance Criteria

- Qwen3.5 and Gemma4 each have a pinned candidate artifact and documented source.
- Ocelotl can inspect metadata for the selected artifacts without network access.
- Unsupported hybrid/MoE/multimodal/quantized features fail explicitly before
  compute.
- Gemma4 GGUF header metadata is normalized into an Ocelotl-owned manifest.
- At least one explicitly supported Gemma4 subset has a pinned tiny synthetic
  runtime prefill fixture without enabling unsupported real-artifact features.
- Family-specific code is isolated under `qwen` and `gemma` modules.
- No Qwen2.5 M3 parity fixture regresses.

## Validation Commands

```powershell
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
pwsh -NoProfile -File ci/check-offline.ps1
```

Any real-model tests must be ignored by default and documented with exact local
artifact paths.

## Known Risks

- Qwen3.5 and Gemma4 can both tempt the project into generic abstractions before
  the second real family has a passing fixture.
- GGUF quantization and embedded tokenizer metadata can blur loader/tokenizer
  boundaries.
- Gemma4's audio capability can be mistaken for a replacement for ASR-specific
  correctness requirements.
- Large artifacts can make default tests slow or non-portable if not kept behind
  local-artifact gates.
