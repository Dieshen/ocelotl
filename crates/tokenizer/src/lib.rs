//! Tokenizer and chat-template boundary.

use ocelotl_core::{OcelotlError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenId(pub u32);

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>>;
    fn decode(&self, tokens: &[TokenId]) -> Result<String>;
}

#[derive(Debug, Default)]
pub struct NullTokenizer;

impl Tokenizer for NullTokenizer {
    fn encode(&self, _text: &str) -> Result<Vec<TokenId>> {
        Err(OcelotlError::Unsupported(
            "no tokenizer implementation configured".to_string(),
        ))
    }

    fn decode(&self, _tokens: &[TokenId]) -> Result<String> {
        Err(OcelotlError::Unsupported(
            "no tokenizer implementation configured".to_string(),
        ))
    }
}
