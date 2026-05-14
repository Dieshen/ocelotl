//! Root crate for the Ocelotl inference runtime workspace.

pub mod chat;

pub use chat::{ChatModel, ChatResponse};

pub mod prelude {
    pub use ocelotl_core::{GenerationOptions, Result, TokenId};

    pub use crate::chat::{ChatModel, ChatResponse};
}

/// Current public crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
