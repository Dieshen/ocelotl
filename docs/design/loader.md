# Loader Design

The loader normalizes model artifacts into explicit metadata and typed weight
access. It should make unsupported files fail before runtime execution begins.

## Responsibilities

- Detect supported artifact formats.
- Validate required files exist.
- Parse model metadata into Ocelotl-owned types.
- Normalize architecture names and key model dimensions.
- Expose weight tensors or tensor handles without owning model execution.
- Report unsupported features with precise errors.

## Initial Formats

M1 can avoid full artifact loading if fixtures are synthetic. M2 should add one
real local format. Prefer safetensors first because it keeps metadata and weight
layout work explicit. GGUF should come later because it carries more format and
quantization policy.

## Metadata Contract

The loader should produce normalized metadata for:

- Architecture.
- Vocabulary size.
- Context length.
- Hidden size.
- Number of layers.
- Attention heads and KV heads.
- Head dimensions.
- RoPE settings.
- Dtype and quantization.
- Tokenizer and chat-template references, if discoverable.

Metadata must preserve model-family-specific details instead of flattening them
away. If a model has per-layer variation, the metadata type should expose that
variation explicitly.

## Error Policy

Unsupported metadata is an error, not a warning. Examples:

- Unknown architecture.
- Unknown quantization.
- Missing required tensor.
- Mismatched tensor shape.
- Unsupported RoPE scaling.
- Unsupported attention layout.

## Non-Responsibilities

The loader must not tokenize prompts, schedule requests, allocate KV cache, or
launch kernels. It prepares verified artifacts for model/runtime code.

## Error Mapping

The loader follows the typed-error contract in `docs/design/errors.md`. The
two cases that came up in M2 are pinned here so future loader work doesn't
re-litigate them.

### Missing or unreadable artifact files map to `Io`

When `std::fs::read*` on an artifact file fails, the loader returns
`OcelotlError::Io` carrying the path and the underlying `std::io::Error` as
the source. Examples: file does not exist, permission denied, device error.

A file that exists but contains malformed bytes is a different category: the
loader returns `OcelotlError::InvalidModel` because the *artifact* is the
problem, not storage. This split lets server code map IO failures to a
storage-error response distinct from a malformed-artifact response.

Decided in M2.6 (paired). M2.5 originally collapsed both cases into
`InvalidModel`; that was incorrect and was swept in the M2.6 commit
`fix(loader): map missing-file errors to Io, not InvalidModel`.

### Loader `SupportedDtype` and `core::DType` stay separate

`crates/loader/src/safetensors_inspect.rs::SupportedDtype` and
`ocelotl_core::DType` describe two different surfaces:

- `SupportedDtype { F32, F16, BF16 }` is the **artifact-read** surface — the
  set of dtypes the loader accepts reading from a safetensors header today.
- `DType { F32, F16, BF16, Q4, Q8 }` is the **compute** surface — the set of
  dtypes kernels can dispatch to, including future quantized formats.

These will diverge further as quantization lands (the loader will eventually
accept Q4/Q8 weight files; a CPU-only build may not have Q4/Q8 kernels).
They also serve different rejection contracts: rejecting an artifact-read
dtype is `Unsupported(feature: "safetensors_dtype")`; rejecting a compute
dtype is `Unsupported(feature: "kernel_dtype")` or similar — same enum
variant, different feature key.

The bridge is `impl From<SupportedDtype> for core::DType` in the loader
crate. Total and lossless for the values that overlap. The impl lives in the
loader crate so `ocelotl-core` does not need to know loader exists; this
preserves the inward-only crate dependency direction documented in
the team workspace's `workflow/crossing-crate-boundaries.md`
(Obsidian: `projects/ocelotl/devs/workflow/crossing-crate-boundaries.md`).

Decided in M2.6 (paired) per the principle in
the team workspace's `concepts/external-crate-boundary.md` (Obsidian:
`projects/ocelotl/devs/concepts/external-crate-boundary.md`): "define your own subset enum
that captures only what you support, with explicit conversion."

## Metadata Entry Points

The loader exposes two parsing entry points for model metadata:

- `load_metadata(path)` consumes the Ocelotl-shaped fixture envelope used by
  M1 (`{ "model": { "architecture": "qwen2", "head_dim": 4, "dtype": "f32",
  ... } }`). Field names match `ocelotl_core::ModelMetadata` directly.
- `parse_hf_config(path)` consumes a real Hugging Face `config.json`
  (transformers format) and translates HF field names to the Ocelotl-shaped
  metadata. Renames: `model_type` -> `architecture`,
  `max_position_embeddings` -> `context_length`, `torch_dtype` -> `dtype`.
  Derives `head_dim = hidden_size / num_attention_heads` (HF doesn't carry
  it for Qwen2-family models).

Both functions return `OcelotlError::Unsupported` when the architecture or
dtype is outside the loader's allow-list, and `OcelotlError::InvalidModel`
when required fields are missing or internally inconsistent.
