# Model Forward Design

Model-family crates define forward semantics. The runtime owns request lifecycle;
models own architecture-specific math and tensor layout expectations.

## Responsibilities

- Implement model-family prefill and decode behavior.
- Validate model metadata against implementation assumptions.
- Apply model-specific operations such as RoPE, RMSNorm, gated MLPs, and logits
  transforms.
- Call low-level compute through the `ocelotl-kernels::KernelBackend` contract.

## Initial Shape

Start with one decoder-only architecture and one unquantized dtype. The first
implementation should prioritize readability and parity against a reference over
performance.

## Prefill And Decode

The model interface should distinguish prefill and decode because they stress
different kernels and cache access patterns.

Prefill:

- Processes prompt tokens.
- Writes all prompt KV entries.
- Produces logits for the final prompt position.

Decode:

- Processes one or a small number of new tokens.
- Reads existing KV.
- Appends new KV.
- Produces next-token logits.

## Model-Specific Behavior

Model-family behavior must be visible in implementation and tests. Examples:

- RoPE scaling.
- Sliding window attention.
- Grouped-query attention.
- MoE routing.
- Logit softcap.
- Per-layer head dimensions.

Unsupported behavior should fail at model construction time where possible.

## Non-Responsibilities

Model code should not parse file formats, own HTTP APIs, download artifacts, or
make global scheduling choices. It should not silently change tokenizer
behavior. It should not name concrete backend implementations such as CPU,
CubeCL, or future GPU providers in model-family APIs; runtime, CLI, or callers
select those backends and pass the kernel contract into the model.

Family-level local loaders such as `Qwen2_5Model::load_from_dir` and
`WhisperModel::load_from_dir` are allowed only as composition helpers: they call
`ocelotl-loader` for file inspection/value loading, then perform
model-family-specific tensor-name mapping, layout conversion, and construction.
They must stay local-only and must not hide network access behind model
construction.

App-facing ergonomics belong in the top-level `ocelotl` crate. The initial
facade stays deliberately small: `ocelotl::ChatModel` exposes local loading,
message accumulation, message inspection, and text generation by composing
loader, tokenizer, model, runtime, and kernel contracts. That facade must not
move chat/session behavior into `ocelotl-models`, and it must still use
explicit local artifacts.
