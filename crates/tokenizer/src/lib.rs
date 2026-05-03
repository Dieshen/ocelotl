//! Tokenizer and chat-template boundary.

pub use ocelotl_core::TokenId;
use ocelotl_core::{OcelotlError, Result, UnsupportedError};

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>>;
    fn decode(&self, tokens: &[TokenId]) -> Result<String>;
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
