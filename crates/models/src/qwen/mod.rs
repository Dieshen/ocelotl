//! Qwen-family model implementations.
//!
//! Each family is its own submodule (`qwen2_5`, `qwen3_5`). The publicly
//! visible family types are re-exported at this level so call sites read
//! `ocelotl_models::qwen::Qwen2_5Model` rather than exposing the internal
//! per-family file layout (`qwen2_5/config.rs`, `qwen2_5/model.rs`, etc.).

pub mod qwen2_5;
pub mod qwen3_5;

pub use qwen2_5::{
    Qwen2_5Config, Qwen2_5LayerWeights, Qwen2_5Model, Qwen2_5Weights, required_tensor_names,
    transpose_2d, validate_qwen2_5_tensors,
};
pub use qwen3_5::{Qwen3_5Config, parse_qwen3_5_config_json};
