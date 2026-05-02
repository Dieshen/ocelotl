# Publishing

Ocelotl uses short local folder names and published package names under the
`ocelotl-*` namespace.

## Crate Order

Publish crates in dependency order:

1. `ocelotl-core`
2. `ocelotl-kernels`
3. `ocelotl-loader`
4. `ocelotl-tokenizer`
5. `ocelotl-models`
6. `ocelotl-runtime`
7. `ocelotl-server`
8. `ocelotl`

## Pre-Publish Checks

Run these before publishing any crate:

```powershell
cargo fmt --all
cargo check --workspace
cargo test --workspace
```

For each crate:

```powershell
cargo publish --dry-run -p <crate-name>
```

## Publishing Commands

```powershell
cargo publish -p ocelotl-core
cargo publish -p ocelotl-kernels
cargo publish -p ocelotl-loader
cargo publish -p ocelotl-tokenizer
cargo publish -p ocelotl-models
cargo publish -p ocelotl-runtime
cargo publish -p ocelotl-server
cargo publish -p ocelotl
```

Crates.io can rate-limit new crate publication. If that happens, retry after the
time reported by Cargo.

## Versioning

Until APIs are useful, publish as `0.0.x`. Once the first usable CPU reference
path lands, move to `0.1.0` and document the supported surface.

## Name Reservation Caution

Do not leave placeholder crates idle. Each published crate should have a clear
purpose, a repository link, a README, and visible development activity.
