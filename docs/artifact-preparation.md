# Artifact Preparation

This guide tells contributors how to place real model artifacts on their local
machine so that tests and tooling that need real tokenizer, config, or weights
data can find them. **Large model weights are deliberately not committed to the
repository** — every contributor fetches them locally on demand.

If you only need to run the default workspace test suite (`cargo test
--workspace`), you do **not** need to follow this guide. Default tests are
offline and use the small fixtures committed under `fixtures/`. Follow this
guide when you start working on a task that explicitly opts into real artifact
behavior — currently the Qwen2.5 M2/M3 task families, the post-M3 Whisper ASR
local-parity harness, and anything downstream that re-uses the same paths.

## 1. Qwen2.5 Artifact

Ocelotl's first real LLM target is `Qwen/Qwen2.5-0.5B-Instruct`. The exact
revision SHA the project pins is recorded in `docs/model-target.md`
§ Pinned Revision (lands via M2.1) — always fetch the pinned SHA, not the
moving `main` branch on Hugging Face. If the doc has not been updated yet, ask
in the next retro before downloading; an unpinned fetch is a fixture-policy
violation and may produce token IDs that don't match the committed expected
fixtures.

License: Apache-2.0 (per `docs/model-target.md`). The weights and tokenizer
files carry license headers and `LICENSE` / `NOTICE` files in the upstream
repo — respect them even though we don't redistribute the files via this
repository.

## 2. Qwen2.5 Files To Fetch

The following files from the pinned Qwen2.5-0.5B-Instruct revision are needed
across M2 and M3. Approximate sizes are from the public repo as of
2026-05-03 and exist to set expectations, not as a contract.

| File                    | Purpose                                       | Approx. size | Used by                                |
| ----------------------- | --------------------------------------------- | ------------ | -------------------------------------- |
| `config.json`           | Model architecture metadata (qwen2 family)    | ~1 KB        | M2.5, M2.6 (loader)                    |
| `generation_config.json`| Default sampling / EOS configuration          | ~200 B       | M2.6 (loader, optional)                |
| `tokenizer.json`        | Full tokenizer (merges + vocab + pre-tok)     | ~7 MB        | M2.2, M2.3, M2.4 (tokenizer)           |
| `tokenizer_config.json` | Chat template + special-token configuration   | ~7 KB        | M2.4 (chat template)                   |
| `vocab.json`            | BPE vocabulary (redundant with tokenizer.json)| ~2.7 MB      | optional; tokenizer.json is sufficient |
| `merges.txt`            | BPE merges (redundant with tokenizer.json)    | ~1.6 MB      | optional; tokenizer.json is sufficient |
| `model.safetensors`     | Model weights (single shard for 0.5B)         | ~988 MB      | M2.5, M3                               |
| `LICENSE` / `NOTICE`    | License text from upstream repo               | ~12 KB       | reference only                         |

The four files load-bearing for M2 are `config.json`, `tokenizer.json`,
`tokenizer_config.json`, and `model.safetensors`. The others are nice to have.

## 3. Qwen2.5 Directory

Place all files under a single directory at the repository root:

```
local-artifacts/
  qwen2_5_0_5b_instruct/
    config.json
    generation_config.json
    tokenizer.json
    tokenizer_config.json
    model.safetensors
    LICENSE
```

Naming rationale:

- `local-artifacts/` makes the intent obvious to a reader skimming the repo
  root: these are local-only files, not committed sources.
- `qwen2_5_0_5b_instruct/` mirrors the `qwen2_5_*` underscore-separated
  convention already used by committed fixtures (e.g.
  `fixtures/metadata/qwen2_5_tiny_synthetic.json`,
  `fixtures/tokenizer/qwen2_5_basic_prompt.json`). Lowercase, underscores, no
  dots or slashes.
- One subdirectory per pinned model revision keeps future second-target
  artifacts (e.g. a different size or family) from colliding.

If you ever need a second revision of the same model side by side, append a
short SHA suffix: `qwen2_5_0_5b_instruct_<short-sha>/`. The plain name
without a suffix always means "the pinned revision from
`docs/model-target.md`".

## 4. Whisper tiny.en Artifact

The first post-M3 Whisper ASR local-artifact contract is a converted tiny.en
bundle under the repository root:

```text
local-artifacts/
  whisper_tiny_en/
    config.json
    tokenizer.json
    model.safetensors
    reference/
      sample_16khz_mono.wav
      expected_tokens.json
      expected_tokens_timestamped.json   # optional, W-ASR.15+
```

The default workspace tests do not require this bundle. The W-ASR.5/W-ASR.10
opt-in harness checks it only when explicitly run:

```powershell
cargo test -p ocelotl-models --test whisper_local_artifact_parity -- --ignored
```

The harness expects:

- `config.json`: a JSON object describing the converted Whisper tiny.en model.
  It must parse through `ocelotl_models::whisper::parse_whisper_config_json`;
  Hugging Face-style Whisper fields and OpenAI-style dimension fields are both
  accepted.
- `tokenizer.json`: a JSON object for the tokenizer artifact.
- `model.safetensors`: a safetensors file with a valid, non-empty header.
- `reference/sample_16khz_mono.wav`: RIFF/WAVE PCM or IEEE-float audio with 1
  channel, 16,000 Hz, a positive bits-per-sample value, and a non-empty `data`
  chunk.
- `reference/expected_tokens.json`: the pinned reference token sequence for
  the sample audio. The sequence must start with the no-timestamps startup
  prompt appropriate to the artifact's tokenizer/model family and include at
  least one generated token after that prompt.
- `reference/expected_tokens_timestamped.json`: optional pinned timestamped
  token sequence for the same sample audio. If present, it uses the same schema
  with `timestamps: true`, timestamp boundary token IDs in
  `expected_token_ids`, and deterministic `expected_segments` entries.

`expected_tokens.json` uses this no-timestamps schema:

```json
{
  "fixture_version": 1,
  "name": "whisper_tiny_en_sample_16khz_mono",
  "source": "describe the converter/reference command used to capture this",
  "audio": "reference/sample_16khz_mono.wav",
  "task": "transcribe",
  "language": "en",
  "timestamps": false,
  "expected_token_ids": [50257, 50362, 50256],
  "expected_text": "optional transcript text"
}
```

`expected_token_ids` must be non-empty and must include at least one generated
token after the startup prompt. `expected_text` is optional for W-ASR.5/W-ASR.10,
but if present it must be non-empty.

`expected_tokens_timestamped.json`, when present, uses the same base fields but
sets `timestamps` to `true`, omits the no-timestamps prompt token, and includes
`expected_segments`:

```json
{
  "fixture_version": 1,
  "name": "whisper_tiny_en_sample_16khz_mono_timestamped",
  "source": "describe the timestamp reference command used to capture this",
  "audio": "reference/sample_16khz_mono.wav",
  "task": "transcribe",
  "language": "en",
  "timestamps": true,
  "expected_token_ids": [50257, 50373, 42, 50388, 50256],
  "expected_segments": [
    {
      "start_seconds": 0.2,
      "end_seconds": 0.5,
      "text_token_ids": [42]
    }
  ],
  "expected_text": "optional transcript text"
}
```

The timestamped harness parses `expected_token_ids` with
`ocelotl-tokenizer`'s public timestamp segment policy and rejects
`expected_segments` that do not exactly match the boundary tokens.

Current supported startup variants are derived from `config.json`:

| Artifact family | Condition | Required no-timestamps startup prompt | EOT | First timestamp |
| --- | --- | --- | --- | --- |
| OpenAI English-only Whisper | `vocab_size < 51865` | `[50257, 50362]` | `50256` | `50363` |
| OpenAI multilingual Whisper, English transcribe | `vocab_size >= 51865` | `[50258, 50259, 50359, 50363]` | `50257` | `50364` |

For timestamped references, the same family detection applies, but
`<|notimestamps|>` is omitted from the startup prompt:

| Artifact family | Required timestamped startup prompt |
| --- | --- |
| OpenAI English-only Whisper | `[50257]` |
| OpenAI multilingual Whisper, English transcribe | `[50258, 50259, 50359]` |

The current ignored harness validates the bundle, parses the real config,
checks `model.safetensors` against Ocelotl's canonical OpenAI-style Whisper
tensor contract, loads every required tensor, preprocesses
`sample_16khz_mono.wav` through Ocelotl's log-mel path, runs
`WhisperModel::forward_next_token_logits` autoregressively with the
family-appropriate Whisper no-timestamps decode mask, and compares exact
generated token IDs against `expected_token_ids`. The W-ASR.15 timestamped
ignored test is opt-in on top of that bundle: it returns early with a clear
message when `reference/expected_tokens_timestamped.json` is absent, and when
present runs with timestamp tokens enabled and compares both exact token IDs
and parsed segment boundaries. If your converted artifact uses HF/Burn-style
tensor names instead of the canonical `encoder.*` / `decoder.*` names, keep the
manifest for a follow-up adapter task; alias support should be added from real
manifest evidence, not guessed in advance.

### Additional Whisper Size Directories

W-ASR.14 audits config and tensor-contract compatibility across OpenAI Whisper
size dimensions without loading large weights or claiming output-token parity
for every size. If later local-artifact tasks need size-specific bundles, use
one directory per upstream artifact name:

| Upstream artifact | Local directory |
| --- | --- |
| `tiny.en` | `local-artifacts/whisper_tiny_en/` |
| `base.en` | `local-artifacts/whisper_base_en/` |
| `small.en` | `local-artifacts/whisper_small_en/` |
| `medium.en` | `local-artifacts/whisper_medium_en/` |
| `tiny` | `local-artifacts/whisper_tiny/` |
| `base` | `local-artifacts/whisper_base/` |
| `small` | `local-artifacts/whisper_small/` |
| `medium` | `local-artifacts/whisper_medium/` |
| `large` | `local-artifacts/whisper_large/` |

The directory name only identifies the artifact family and size. It does not
expand the current English ASR decode surface, and it does not make non-tiny
parity part of the default test suite.

## 5. Keeping Artifacts Out Of Git

`local-artifacts/` is listed in `.gitignore`. This is intentional and
enforced by tooling, not by convention:

- The weights file alone is ~1 GB and would dominate the repository size.
- The fixture policy in `docs/validation/fixtures.md` § Storage Policy is
  unambiguous: **large model files should not be committed**.
- License terms are friendlier when contributors fetch upstream directly than
  when the project re-hosts.

If `git status` ever lists a file under `local-artifacts/`, do not force-add
it. Either the path is wrong or the `.gitignore` entry is broken — fix the
root cause before committing anything else.

## 6. How Tests Find The Artifacts

Tests that depend on real artifacts MUST be `#[ignore]` by default and run
explicitly with `cargo test -- --ignored` (or a focused
`cargo test -p <crate> -- --ignored <name>`). This preserves the
offline-by-default contract documented in `docs/ci.md` § Offline Rule and
§ Offline By Default Across Milestones, where M2 is the milestone that owns
network/local-artifact enforcement.

Concrete pattern for new tests:

```rust
// crates/<crate>/tests/<some_test>.rs

const LOCAL_ARTIFACT_DIR: &str = "local-artifacts/qwen2_5_0_5b_instruct";

fn artifact_path(file: &str) -> std::path::PathBuf {
    // tests run from the crate dir; walk up to repo root.
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(LOCAL_ARTIFACT_DIR)
        .join(file)
}

#[test]
#[ignore = "requires local-artifacts/qwen2_5_0_5b_instruct/tokenizer.json; see docs/artifact-preparation.md"]
fn tokenizer_encodes_known_prompt() {
    let path = artifact_path("tokenizer.json");
    assert!(
        path.exists(),
        "missing artifact at {} — see docs/artifact-preparation.md",
        path.display(),
    );
    // ... real assertions ...
}
```

Two non-negotiable parts of the pattern:

1. **`#[ignore]` with a reason string** that names both the missing file and
   this doc. The default `cargo test --workspace` then stays green for a
   contributor who has not fetched artifacts, and the ignore reason tells
   them exactly what to do if they want to opt in.
2. **An explicit `path.exists()` assertion with a remediation message**
   inside the test. If a contributor runs `--ignored` without having
   prepared artifacts, the failure is one line that points at this doc — not
   a panic from deep inside `tokenizers` or `safetensors`.

Tests that use only the small committed fixtures under `fixtures/` should
**not** be marked `#[ignore]`. The distinction is: committed fixtures =
default-on; local artifacts = opt-in.

## 7. How To Fetch

The project does not ship a fetch script. Each contributor runs the command
manually so that the pinned SHA is visible at the call site (matching the
regeneration discipline in `docs/validation/fixtures.md` § Regeneration).

Recommended command, using the official `huggingface-cli`:

```powershell
# Replace <SHA> with the value from docs/model-target.md § Pinned Revision.
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct `
    --revision <SHA> `
    --local-dir local-artifacts/qwen2_5_0_5b_instruct `
    --local-dir-use-symlinks False
```

`--local-dir-use-symlinks False` writes real files (not symlinks into the
HF cache), which is what test code reading by path expects on Windows.

If you only need a subset (e.g. just the tokenizer for M2.2 work and not the
1 GB weights file), pass `--include`:

```powershell
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct `
    --revision <SHA> `
    --local-dir local-artifacts/qwen2_5_0_5b_instruct `
    --local-dir-use-symlinks False `
    --include "tokenizer.json" "tokenizer_config.json" "config.json"
```

After download, sanity-check the layout:

```powershell
Get-ChildItem local-artifacts/qwen2_5_0_5b_instruct
```

You should see the files listed in section 2.

## 8. License Reminder

Apache-2.0 means redistribution is permitted with attribution. We avoid
checking the artifacts into this repository for size and review-friction
reasons, **not** because the license forbids it. If you build something
downstream of Ocelotl that does ship Qwen2.5 weights, copy the upstream
`LICENSE` and `NOTICE` files alongside the weights.

The pinned revision's `LICENSE`, `NOTICE`, and any model-card preamble
contain the authoritative terms — read them once when you first download.

## Related

- `docs/model-target.md` — the pinned model identity and SHA.
- `docs/validation/fixtures.md` — fixture policy, including the
  large-files-out-of-tree rule this guide implements.
- `docs/ci.md` — offline-by-default enforcement, especially the M2 paragraph.
- `docs/start-here.md` — contributor entry point that links here.
- `.gitignore` — where `local-artifacts/` is excluded.
