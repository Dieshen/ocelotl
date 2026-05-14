//! Qwen-family runtime entry points.

use ocelotl_core::{InvalidRequestError, OcelotlError, Result, RuntimeError, TokenId};
use ocelotl_models::Qwen2_5Model;

use crate::{
    greedy_sample,
    kv_cache::{ContiguousKvCache, PagedKvCache, PagedKvCacheAllocator},
};

/// Run prefill on the given model and return final-position logits.
///
/// This is the public M3.7 entry point. It is a thin shim over
/// `Qwen2_5Model::prefill` that exists so callers (server code, tests,
/// future decoder loops) reach prefill through `ocelotl_runtime::prefill`
/// rather than reaching directly into the model crate. The shape of this
/// API is intentionally model-typed for now: M3 targets only Qwen2.5.
/// When a second text model family lands, this can move behind a shared model
/// interface. Until then, generic abstraction would be premature (the M3 design
/// doc's "keep generic abstractions minimal until a second family is
/// implemented" line).
///
/// # Errors
///
/// Propagates `OcelotlError` from the model's prefill verbatim:
/// - `InvalidRequest` for empty prompts, out-of-range token ids, or
///   prompt length exceeding `context_length`.
/// - `Kernel` for unreachable shape violations (would indicate the
///   `Qwen2_5Model::new` length checks have a bug).
pub fn prefill(model: &Qwen2_5Model, tokens: &[TokenId]) -> Result<Vec<f32>> {
    model.prefill(tokens)
}

/// Run one decode step: prefill the prompt, sample the next token greedily,
/// return that token id.
///
/// This is the public M3.8 entry point. It is the minimum composition of
/// two existing public APIs:
///
/// 1. `runtime::prefill` (M3.7) -- final-position logits for the prompt.
/// 2. `runtime::greedy_sample` (M1.8) -- argmax over the logits with a
///    lowest-token-id tie-break.
///
/// # Why call `runtime::prefill` rather than `Qwen2_5Model::prefill` directly
///
/// The M3.8 brief's "Done when" line is "decode does not bypass runtime or
/// model APIs used by prefill". Going through the runtime's own `prefill`
/// shim keeps a single hop between the public decode API and the model:
/// any future addition to the runtime's `prefill` (cache plumbing,
/// tracing, request-state hooks) is automatically inherited by decode.
/// Reaching into `Qwen2_5Model::prefill` directly here would create a
/// second path that those hooks would silently miss.
///
/// # State handling (no KV cache)
///
/// M3 is the correctness-first reference path. This function does not
/// reuse any prefill state across calls -- each call to `decode_one_token`
/// pays the full O(prompt_len) prefill cost. KV-cache reuse is M5/M6
/// work and will land as a separate API (likely `decode_with_cache`)
/// rather than complicating this signature now. The current shape
/// (`(&Qwen2_5Model, &[TokenId]) -> Result<TokenId>`) is intentionally
/// thin so a future cache-aware path can be added alongside without
/// breaking callers.
///
/// # Errors
///
/// Propagates errors from the composed APIs verbatim:
/// - `InvalidRequest` from `runtime::prefill` for empty prompts,
///   out-of-range token ids, or prompt length exceeding `context_length`.
/// - `Runtime` from `greedy_sample` if the logits vector is empty
///   (unreachable with a validly constructed model -- `vocab_size > 0`
///   is enforced at `Qwen2_5Config::try_from`).
pub fn decode_one_token(model: &Qwen2_5Model, tokens: &[TokenId]) -> Result<TokenId> {
    let logits = prefill(model, tokens)?;
    greedy_sample(&logits)
}

/// Runtime-owned contiguous Qwen2.5 cache state.
///
/// `last_logits` is the sampling surface for the next token. The cache stores
/// K/V for every token already consumed by the model. Keeping both together in
/// runtime prevents callers from mixing logits from one request with cache
/// storage from another request.
#[derive(Debug, Clone)]
pub struct Qwen2_5ContiguousCacheState {
    cache: ContiguousKvCache,
    last_logits: Vec<f32>,
}

impl Qwen2_5ContiguousCacheState {
    pub fn cache(&self) -> &ContiguousKvCache {
        &self.cache
    }

    pub fn last_logits(&self) -> &[f32] {
        &self.last_logits
    }

    pub fn is_released(&self) -> bool {
        self.cache.is_released()
    }

    pub fn release(&mut self) {
        self.cache.release();
        self.last_logits.clear();
    }
}

/// Prefill a Qwen2.5 prompt and populate a request-scoped contiguous KV cache.
pub fn prepare_qwen2_5_contiguous_cache(
    model: &Qwen2_5Model,
    tokens: &[TokenId],
) -> Result<Qwen2_5ContiguousCacheState> {
    let mut cache = ContiguousKvCache::for_qwen2_5(model.config())?;
    let last_logits = model.prefill_with_cache(tokens, &mut cache)?;
    Ok(Qwen2_5ContiguousCacheState { cache, last_logits })
}

/// Greedily sample one token from cached state, append its K/V, and return it.
pub fn decode_one_token_with_contiguous_cache(
    model: &Qwen2_5Model,
    state: &mut Qwen2_5ContiguousCacheState,
) -> Result<TokenId> {
    let token = greedy_sample(&state.last_logits)?;
    state.last_logits = model.decode_token_with_cache(token, &mut state.cache)?;
    Ok(token)
}

/// Runtime-owned paged Qwen2.5 cache state.
///
/// The cache is optional only so `release_into` can return physical pages to
/// the allocator exactly once while leaving a visible released state behind.
#[derive(Debug, Clone)]
pub struct Qwen2_5PagedCacheState {
    cache: Option<PagedKvCache>,
    last_logits: Vec<f32>,
}

impl Qwen2_5PagedCacheState {
    pub fn cache(&self) -> Option<&PagedKvCache> {
        self.cache.as_ref()
    }

    pub fn last_logits(&self) -> &[f32] {
        &self.last_logits
    }

    pub fn is_released(&self) -> bool {
        self.cache.is_none()
    }

    pub fn release_into(&mut self, allocator: &mut PagedKvCacheAllocator) {
        if let Some(cache) = self.cache.take() {
            allocator.release(cache);
        }
        self.last_logits.clear();
    }
}

/// Prefill a Qwen2.5 prompt and populate a request-scoped paged KV cache.
///
/// Allocation happens in runtime, not in the model. If prefill fails after page
/// allocation, the pages are returned before the error escapes.
pub fn prepare_qwen2_5_paged_cache(
    model: &Qwen2_5Model,
    tokens: &[TokenId],
    allocator: &mut PagedKvCacheAllocator,
    capacity_tokens: usize,
) -> Result<Qwen2_5PagedCacheState> {
    if tokens.len() > capacity_tokens {
        return Err(invalid_request(
            "kv_cache.capacity",
            &format!(
                "prompt length {} exceeds requested paged cache capacity {capacity_tokens}",
                tokens.len()
            ),
        ));
    }

    let mut cache = allocator.allocate(capacity_tokens)?;
    match model.prefill_with_cache(tokens, &mut cache) {
        Ok(last_logits) => Ok(Qwen2_5PagedCacheState {
            cache: Some(cache),
            last_logits,
        }),
        Err(err) => {
            allocator.release(cache);
            Err(err)
        }
    }
}

/// Greedily sample one token from paged cached state, append its K/V, and return it.
pub fn decode_one_token_with_paged_cache(
    model: &Qwen2_5Model,
    state: &mut Qwen2_5PagedCacheState,
) -> Result<TokenId> {
    let token = greedy_sample(&state.last_logits)?;
    let cache = state
        .cache
        .as_mut()
        .ok_or_else(|| runtime_error("paged Qwen2.5 cache state has been released"))?;
    state.last_logits = model.decode_token_with_cache(token, cache)?;
    Ok(token)
}

fn invalid_request(field: &str, message: &str) -> OcelotlError {
    OcelotlError::InvalidRequest(InvalidRequestError {
        field: field.to_string(),
        message: message.to_string(),
    })
}

fn runtime_error(message: impl Into<String>) -> OcelotlError {
    OcelotlError::Runtime(RuntimeError {
        message: message.into(),
    })
}
