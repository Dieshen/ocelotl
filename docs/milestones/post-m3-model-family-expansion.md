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
- The Qwen3.5 model card describes a unified vision-language foundation and an
  efficient hybrid architecture using Gated Delta Networks plus sparse
  Mixture-of-Experts.
- That means Qwen3.5 is not a small Qwen2.5 dense-decoder extension.

Gemma4:

- Official Google Hugging Face model cards exist, including
  `google/gemma-4-E4B`.
- Gemma4 models are multimodal. The small E2B/E4B models include native audio
  support; all generate text output.
- The local Gemma4 GGUF artifact inspected during post-M3 reconnaissance is
  quantized GGUF v3 with `general.architecture = gemma4`, embedded tokenizer
  metadata, sliding-window/shared-KV metadata, and Gemma-specific softcapping.

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

