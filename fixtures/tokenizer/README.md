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

## `qwen2_5_basic_prompt.json`

Placeholder fixture for M2.3 (Qwen2.5 tokenizer expected token IDs).
`expected_token_ids` is intentionally empty until M2.3 lands.
