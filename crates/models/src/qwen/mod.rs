//! Qwen-family model implementations.
//!
//! Keep Qwen-specific metadata, tensor validation, weight layout, and forward
//! semantics under this module. Root-level re-exports in `ocelotl_models`
//! preserve the current public API while making room for future families such
//! as Gemma without flattening every implementation file into `src/`.

pub mod qwen2_5;
pub mod qwen2_5_model;
pub mod qwen2_5_tensors;
pub mod qwen3_5;

pub use qwen2_5::Qwen2_5Config;
pub use qwen2_5_model::{Qwen2_5LayerWeights, Qwen2_5Model, Qwen2_5Weights, transpose_2d};
pub use qwen2_5_tensors::{required_tensor_names, validate_qwen2_5_tensors};
pub use qwen3_5::{Qwen3_5Config, parse_qwen3_5_config_json};
