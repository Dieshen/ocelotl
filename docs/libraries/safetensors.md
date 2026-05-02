# safetensors

## Current Shape

`safetensors` is a safe tensor storage format and Rust crate from Hugging Face.
Crates.io currently shows `safetensors = "0.7.0"`.

Context7 docs emphasize the file format:

- 8-byte little-endian header length.
- JSON header with tensor names, dtypes, shapes, and data offsets.
- Optional `__metadata__` string map.
- Little-endian row-major tensor data.
- No holes in the data buffer.
- Duplicate keys are disallowed.
- Tensor values are not checked for NaN or infinity.

## Best Use In Ocelotl

Use safetensors as the first real loader format for M2/M3 because it keeps weight
loading explicit and avoids GGUF quantization complexity at the start.

`ocelotl-loader` should:

- Parse safetensors metadata.
- List tensor names and shapes.
- Validate required tensor presence.
- Validate expected dtype and shape before exposing weights to model code.
- Preserve metadata needed for model-family construction.

## Loader Contract

Do not let model code ask for arbitrary tensor names directly. The loader should
normalize artifact data into model-family-specific verified structures.

```rust
pub struct VerifiedTensor<'a> {
    pub name: &'a str,
    pub dtype: OcelotlDType,
    pub shape: &'a [usize],
    pub data: &'a [u8],
}
```

The actual type can differ, but the contract should keep dtype, shape, and bytes
together so validation cannot be skipped accidentally.

## TDD Requirements

- Good fixture with expected tensor names, shapes, and dtype.
- Missing tensor fixture.
- Wrong shape fixture.
- Unsupported dtype fixture.
- Metadata parse fixture.

## Risks

- Safetensors validates format safety, not model semantic correctness.
- NaN and infinity values are possible and need Ocelotl-level policy.
- Some model repos shard weights across multiple safetensors files; M2 can defer
  sharding but should fail clearly if it sees unsupported sharding.
- Metadata values are strings, so model-specific metadata may still require
  separate config files.
