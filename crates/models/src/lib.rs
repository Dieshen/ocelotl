//! Model-family implementations.
//!
//! Each family is its own submodule (`qwen`, `gemma`, `whisper`). Family
//! types are NOT re-exported at the crate root; call sites use the family
//! namespace explicitly:
//!
//! - `ocelotl_models::qwen::Qwen2_5Model`
//! - `ocelotl_models::gemma::Gemma4Config`
//! - `ocelotl_models::whisper::WhisperModel`

pub mod gemma;
pub mod qwen;
pub mod whisper;
