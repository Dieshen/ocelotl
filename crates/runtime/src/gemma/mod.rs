//! Gemma-family runtime entry points.
//!
//! MF.7 supports only `Gemma4TextModel`: a text-only synthetic dense subset.
//! Real Gemma4 GGUF artifacts remain rejected at the model boundary until the
//! multimodal, sliding-window pattern/shared-KV, quantized-origin, and full
//! parity policies land.

use ocelotl_core::{Result, TokenId};
use ocelotl_models::gemma::Gemma4TextModel;

use crate::greedy_sample;

/// Run Gemma4 text prefill on the given model and return final-position logits.
pub fn prefill(model: &Gemma4TextModel, tokens: &[TokenId]) -> Result<Vec<f32>> {
    model.prefill(tokens)
}

/// Run one Gemma4 text decode step by prefill + greedy argmax.
pub fn decode_one_token(model: &Gemma4TextModel, tokens: &[TokenId]) -> Result<TokenId> {
    let logits = prefill(model, tokens)?;
    greedy_sample(&logits)
}
