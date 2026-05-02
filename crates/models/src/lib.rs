//! Model-family implementations.

use ocelotl_core::{GenerationOptions, ModelInfo, Result};

pub trait CausalLanguageModel: Send + Sync {
    fn info(&self) -> &ModelInfo;
    fn generate(&self, prompt: &[u32], options: &GenerationOptions) -> Result<Vec<u32>>;
}
