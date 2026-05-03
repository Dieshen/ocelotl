# Tokenizer fixtures

## `tiny_wordlevel.json` (M2.2)

Hand-built minimal `tokenizer.json` for testing the Ocelotl tokenizer
boundary in `crates/tokenizer/`. Three-token vocab:

| token   | id |
| ------- | -- |
| `<unk>` | 0  |
| `hello` | 1  |
| `world` | 2  |

Pre-tokenizer: `Whitespace`. Model type: `WordLevel`. Unknown token: `<unk>`.

The fixture exists because M2.2 needs to prove the Ocelotl `Tokenizer`
trait can load a real `tokenizer.json` file via the underlying
`tokenizers` crate without leaking that crate's types. A hand-built
fixture keeps the test fully deterministic and has no dependency on
external model artifacts (Qwen2.5 fixture work is M2.3).

### Pinned expectations

- `encode("hello world")` → `[TokenId(1), TokenId(2)]`
- `encode("hello unknown world")` → `[TokenId(1), TokenId(0), TokenId(2)]`
- `decode([TokenId(1), TokenId(2)])` → `"helloworld"` (no separator)

The decode result has no space between tokens because the configured
`WordPiece` decoder concatenates tokens without re-inserting whitespace.
That is the actual deterministic output of this fixture; restoring
whitespace is a function of the model-specific decoder, which the M2.3
Qwen2.5 fixture will exercise with a real BPE-style decoder.

### Regeneration

This file is hand-edited. There is no generation tool. If the schema of
the `tokenizers` crate changes in a future version (`0.23.x` is current
target), update the fields here and re-pin the IDs above. Do not
auto-format the JSON — keep the fields ordered for readability.

## `qwen2_5_basic_prompt.json` (M2.3)

Exact encode/decode round trip for the pinned Qwen2.5-0.5B-Instruct
tokenizer revision (`7ae557604adf67be50417f59c2c2f167def9a775`, see
`fixtures/manifest/qwen2_5_0_5b_instruct.json`). Small JSON record only —
the actual `tokenizer.json` (~7 MB) is **not** committed and lives under
`local-artifacts/qwen2_5_0_5b_instruct/tokenizer.json` per
`docs/artifact-preparation.md`.

### Pinned expectations

- `encode("Hello")` → `[TokenId(9707)]`
- `decode([TokenId(9707)])` → `"Hello"`

Single-token result because `Hello` is a common BPE token in the Qwen2.5
vocab; bare ASCII words at the start of input do not pre-pend a
leading-space prefix under this tokenizer's pre-tokenization. No special
tokens are added at the encode step (BOS/EOS belong to chat-template
rendering, M2.4).

### Round-trip semantics

The `decoded` field in the JSON is the source of truth for assertions —
not `input`. They happen to be equal for `"Hello"`, but pinning the
decoded form explicitly means future fixtures with non-trivial whitespace
or special-token semantics can pin a different decoded form without
changing the test code.

### Test surfaces

Two tests exercise this fixture, both in
`crates/tokenizer/tests/qwen2_5_basic_prompt.rs`:

1. `fixture_is_well_formed_and_populated` — runs by default, offline. Asserts
   the JSON parses, references the pinned SHA, and has non-empty
   `expected_token_ids` + a declared `decoded` form. Guards against
   regressing back to the placeholder shape.
2. `json_tokenizer_round_trips_qwen2_5_basic_prompt` — `#[ignore]`'d by
   default. Loads `local-artifacts/qwen2_5_0_5b_instruct/tokenizer.json`,
   encodes via `JsonTokenizer`, asserts IDs match the fixture, decodes,
   asserts decoded text matches the fixture's `decoded` field. Run with
   `cargo test -p ocelotl-tokenizer --test qwen2_5_basic_prompt -- --ignored`.

### Regeneration

If the upstream tokenizer.json changes under the pinned SHA (treated as
a breaking-change event per `docs/model-target.md`), re-run the
`#[ignore]`'d test — the failure message reports the new IDs. Update
`expected_token_ids` and `decoded` in the JSON, and bump the manifest +
`docs/model-target.md` in the same dedicated commit so
`git log -- docs/model-target.md` records every revision change.
