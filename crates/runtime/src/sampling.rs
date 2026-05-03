//! Token sampling strategies.
//!
//! M1 ships a single deterministic strategy: greedy argmax. Other modes
//! (top-k, top-p, temperature-scaled multinomial) are deferred to M2+ and
//! will arrive as a deliberate dispatch addition. Until then, requests that
//! ask for any non-greedy mode are rejected upstream by
//! `crate::validate_request` so the sampler itself stays mode-free.

use ocelotl_core::{OcelotlError, Result, RuntimeError, TokenId};

/// Pick the token id whose logit is largest.
///
/// # Errors
///
/// Returns `RuntimeError` when `logits` is empty — a vocab of size zero is
/// nonsensical and caller code should never reach the sampler with one.
///
/// # Preconditions
///
/// Logits are expected to be finite. NaN handling is undefined for M1: the
/// underlying `f32::partial_cmp` returns `None` on NaN, and the iterator
/// adapter falls back to a comparator that treats unordered pairs as
/// `Equal`, so a NaN may or may not "win" depending on its position. A
/// future hardening pass will reject NaN explicitly.
pub fn greedy_sample(logits: &[f32]) -> Result<TokenId> {
    if logits.is_empty() {
        return Err(OcelotlError::Runtime(RuntimeError {
            message: "greedy_sample requires at least one logit".to_string(),
        }));
    }

    let (idx, _) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .expect("non-empty slice guaranteed by the empty check above");

    Ok(TokenId(idx as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sample_picks_index_of_largest_logit() {
        // Peak is at index 3.
        let logits = [0.1, 0.5, 0.3, 0.9, 0.2];

        let token = greedy_sample(&logits).expect("non-empty logits must sample");

        assert_eq!(token, TokenId(3));
    }
}
