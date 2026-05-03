//! Model-family implementations.

use ocelotl_core::{GenerationOptions, ModelMetadata, Result};

pub trait CausalLanguageModel: Send + Sync {
    fn info(&self) -> &ModelMetadata;
    fn generate(&self, prompt: &[u32], options: &GenerationOptions) -> Result<Vec<u32>>;
}
