# Library Research Notes

These notes summarize current library shape and recommended use for Ocelotl.
They are not API guarantees. Before adding or upgrading a dependency, re-check
its current docs and crate version.

Research sources used for this pass:

- Context7 docs for Burn, CubeCL, CubeK, Burn LM, Tokio, safetensors, and
  Hugging Face tokenizers.
- `cargo search` against crates.io on 2026-05-02 for current package names and
  visible latest versions.

## Priority Order

1. `tokio`: server, scheduler, cancellation, channels, and async task runtime.
2. `safetensors`: first loader format for unquantized local model fixtures.
3. `tokenizers`: first tokenizer implementation boundary.
4. `burn`: prototype tensor/model execution and possible model module layer.
5. `cubecl`: custom portable kernels for Ocelotl-owned hot paths.
6. `cubek`: optimized matmul, attention, reduction, and quantization kernels.
7. `burn-lm`: reference architecture and cautionary comparison, not an immediate
   dependency default.

## Dependency Rule

Do not add a library just because it is promising. Add it when a milestone has a
failing test that needs it and the owning crate boundary is clear.
