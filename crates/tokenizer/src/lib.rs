//! Tokenizer and chat-template boundary.

use std::path::Path;

pub use ocelotl_core::TokenId;
use ocelotl_core::{OcelotlError, Result, TokenizerError, UnsupportedError};

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>>;
    fn decode(&self, tokens: &[TokenId]) -> Result<String>;
}

/// Concrete `Tokenizer` backed by a local `tokenizer.json` file loaded via
/// the Hugging Face `tokenizers` crate. The `tokenizers` types are kept
/// strictly inside this struct — public methods only expose Ocelotl
/// `TokenId`s and `OcelotlError`.
pub struct JsonTokenizer {
    inner: tokenizers::Tokenizer,
}

impl std::fmt::Debug for JsonTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Avoid leaking the inner `tokenizers::Tokenizer` debug shape across
        // the boundary; callers only need to know the wrapper exists.
        f.debug_struct("JsonTokenizer").finish_non_exhaustive()
    }
}

impl JsonTokenizer {
    /// Load a tokenizer from a local `tokenizer.json` file. Errors map to
    /// `OcelotlError::Tokenizer` so callers never see `tokenizers` types.
    pub fn from_json_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let inner = tokenizers::Tokenizer::from_file(path_ref).map_err(|source| {
            OcelotlError::Tokenizer(TokenizerError {
                message: format!(
                    "failed to load tokenizer.json from `{}`",
                    path_ref.display()
                ),
                source: Some(source),
            })
        })?;
        Ok(Self { inner })
    }
}

impl Tokenizer for JsonTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        let encoding = self.inner.encode(text, false).map_err(|source| {
            OcelotlError::Tokenizer(TokenizerError {
                message: "encode failed".to_string(),
                source: Some(source),
            })
        })?;
        Ok(encoding.get_ids().iter().copied().map(TokenId).collect())
    }

    fn decode(&self, tokens: &[TokenId]) -> Result<String> {
        let raw: Vec<u32> = tokens.iter().map(|t| t.0).collect();
        self.inner.decode(&raw, true).map_err(|source| {
            OcelotlError::Tokenizer(TokenizerError {
                message: "decode failed".to_string(),
                source: Some(source),
            })
        })
    }
}

#[derive(Debug, Default)]
pub struct NullTokenizer;

impl Tokenizer for NullTokenizer {
    fn encode(&self, _text: &str) -> Result<Vec<TokenId>> {
        Err(OcelotlError::Unsupported(UnsupportedError {
            feature: "tokenizer_encode".to_string(),
            requested: None,
            supported: vec![],
        }))
    }

    fn decode(&self, _tokens: &[TokenId]) -> Result<String> {
        Err(OcelotlError::Unsupported(UnsupportedError {
            feature: "tokenizer_decode".to_string(),
            requested: None,
            supported: vec![],
        }))
    }
}

// ---------------------------------------------------------------------------
// Test-fixture helpers (M2.2)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod test_fixtures {
    use std::path::PathBuf;

    /// Resolve a fixture under `<repo>/fixtures/tokenizer/` by name.
    pub fn tokenizer_fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/tokenizer")
            .join(name)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::tokenizer_fixture_path;

    #[test]
    fn json_tokenizer_loads_tiny_wordlevel_fixture_and_encodes_known_input() {
        let path = tokenizer_fixture_path("tiny_wordlevel.json");
        let tok = JsonTokenizer::from_json_path(&path)
            .expect("tiny_wordlevel.json fixture should load via the Ocelotl tokenizer trait");

        let ids = tok
            .encode("hello world")
            .expect("encoding a known input against the tiny wordlevel fixture should succeed");

        // Pinned by fixtures/tokenizer/README.md: hello -> 1, world -> 2.
        assert_eq!(ids, vec![TokenId(1), TokenId(2)]);
    }
}
