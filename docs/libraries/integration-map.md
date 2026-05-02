# Library Integration Map

This map translates library research into Ocelotl crate ownership.

## `ocelotl-core`

Dependencies should stay minimal. Prefer `serde` and `thiserror`; avoid Burn,
CubeCL, Tokio, safetensors, or tokenizers here unless a public core type truly
needs them.

## `ocelotl-loader`

Likely dependencies:

- `safetensors` for M2.
- `serde` and `serde_json` for config and metadata fixtures.

Avoid Burn or runtime dependencies here.

## `ocelotl-tokenizer`

Likely dependencies:

- `tokenizers` for local tokenizer JSON support.
- `serde` for fixtures and chat-template inputs.

Keep tokenizer library types behind Ocelotl traits.

## `ocelotl-kernels`

Likely dependencies, introduced in stages:

- CPU reference only for M1-M3.
- `cubecl` for custom kernels in M4.
- `cubek-matmul`, `cubek-attention`, `cubek-reduce`, or `cubek-quant` only when
  a failing parity test needs them.
- Burn only if a Burn-backed kernel path proves useful and does not hide layout.

## `ocelotl-models`

Potential dependencies:

- Burn for prototype module/layer code if useful.
- No Tokio.
- No server dependencies.

Model code should consume normalized metadata and kernel traits.

## `ocelotl-runtime`

Potential dependencies:

- `tokio` for async coordination once request queues or cancellation are needed.
- `tracing` for runtime observability.

Avoid direct safetensors/tokenizers/CubeCL dependencies unless hidden behind
runtime-owned adapter traits.

## `ocelotl-server`

Potential dependencies:

- `tokio` with networking features once an HTTP server is introduced.
- A web framework later, selected when M8 starts.

The server should call runtime APIs and map errors. It should not load model
files or launch kernels.

## Version Caution

Current visible versions on 2026-05-02:

- `burn = "0.21.0-pre.4"`
- `cubecl = "0.10.0-pre.4"`
- `cubek = "0.2.0-pre.4"`
- `tokio = "1.52.1"`
- `safetensors = "0.7.0"`
- `tokenizers = "0.23.1"`

Pre-release dependencies should be isolated to implementation crates and covered
by focused tests so churn does not infect core contracts.
