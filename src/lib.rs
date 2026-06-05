//! Root crate for the Ocelotl inference runtime workspace.

pub mod chat;
pub mod gemma4;

pub use chat::{ChatModel, ChatResponse};
pub use gemma4::{gemma4_gguf_tokenizer_spec_from_metadata, load_gemma4_gguf_tokenizer};

pub mod prelude {
    pub use ocelotl_core::{GenerationOptions, Result, TokenId};

    pub use crate::chat::{ChatModel, ChatResponse};
    pub use crate::gemma4::{gemma4_gguf_tokenizer_spec_from_metadata, load_gemma4_gguf_tokenizer};
}

/// Current public crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
