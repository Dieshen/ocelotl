# Hugging Face tokenizers

## Current Shape

The Hugging Face `tokenizers` crate provides fast tokenizer implementations and
local `tokenizer.json` loading. Crates.io currently shows `tokenizers = "0.23.1"`.

Context7 docs show:

- Loading from a local tokenizer JSON file.
- Encoding single strings and pairs.
- Batch encoding.
- Decoding token IDs with optional special-token skipping.
- Padding and truncation configuration.
- Added token behavior.
- Vocabulary lookups.

## Best Use In Ocelotl

Use `tokenizers` as the first tokenizer implementation backend, but keep it
behind Ocelotl-owned traits in `ocelotl-tokenizer`.

Recommended boundary:

```rust
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> ocelotl_core::Result<Vec<TokenId>>;
    fn decode(&self, tokens: &[TokenId]) -> ocelotl_core::Result<String>;
}
```

The runtime should never depend directly on `tokenizers::Tokenizer`.

## Chat Templates

The tokenizers crate handles tokenization. Chat-template behavior should be an
explicit Ocelotl contract. If a model artifact ships a tokenizer and a chat
template, loader/tokenizer code should preserve both and tests should pin the
rendered prompt.

## TDD Requirements

- Exact token ID fixture for a short prompt.
- Exact decode fixture for known token IDs.
- BOS/EOS handling test.
- Special-token skip/no-skip decode tests.
- Chat-template rendering fixture once chat messages exist.

## Risks

- Round-trip encode/decode tests can miss BOS/EOS and special-token bugs.
- Tokenizer versions can produce different behavior for edge cases.
- Chat-template formatting is model behavior and should not be guessed in the
  runtime crate.
- Padding/truncation should not be enabled implicitly for generation prompts.
