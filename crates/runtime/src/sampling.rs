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
/// Tie-break: when multiple logits share the maximum value, the **lowest
/// token id** wins. This matches the NumPy/PyTorch `argmax` convention and
/// is the contract downstream code (and reproducibility tests) rely on.
///
/// # Errors
///
/// Returns `RuntimeError` when `logits` is empty — a vocab of size zero is
/// nonsensical and caller code should never reach the sampler with one.
///
/// # Preconditions
///
/// Logits are expected to be finite. NaN handling is unspecified for M1:
/// because the scan uses `>` (which is `false` for any NaN comparison), a
/// NaN logit can never displace the running best, so a NaN never "wins"
/// unless every logit is NaN — in which case index 0 is returned. A future
/// hardening pass will reject NaN explicitly with a typed error.
pub fn greedy_sample(logits: &[f32]) -> Result<TokenId> {
    if logits.is_empty() {
        return Err(OcelotlError::Runtime(RuntimeError {
            message: "greedy_sample requires at least one logit".to_string(),
        }));
    }

    // Strict `>` is load-bearing for tie-break: `Iterator::max_by` returns
    // the *last* element among equals, which would give the highest index
    // on a tie — the opposite of the NumPy/PyTorch argmax contract. A
    // hand-rolled scan that only updates on a strict improvement keeps the
    // first occurrence and therefore the lowest token id.
    let mut best_idx = 0usize;
    let mut best_val = logits[0];
    for (idx, &val) in logits.iter().enumerate().skip(1) {
        if val > best_val {
            best_idx = idx;
            best_val = val;
        }
    }

    Ok(TokenId(best_idx as u32))
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

    #[test]
    fn greedy_sample_breaks_ties_by_lowest_token_id() {
        // Three-way tie at indices 1, 2, 3 — lowest index must win.
        // This is the NumPy/PyTorch argmax convention and the contract
        // downstream reproducibility tests rely on. The current
        // implementation gets this for free because `Iterator::max_by`
        // returns the *first* element comparing as the maximum; this test
        // pins the behavior so a future refactor cannot quietly change it.
        let logits = [0.1, 0.5, 0.5, 0.5, 0.2];

        let token = greedy_sample(&logits).expect("non-empty logits must sample");

        assert_eq!(token, TokenId(1));
    }

    #[test]
    fn greedy_sample_rejects_empty_logits_with_runtime_error() {
        let logits: [f32; 0] = [];

        let err = greedy_sample(&logits).expect_err("empty logits must error");

        match err {
            OcelotlError::Runtime(rt) => {
                assert!(
                    rt.message.contains("at least one logit"),
                    "expected message to mention the precondition, got {:?}",
                    rt.message
                );
            }
            other => panic!("expected Runtime error, got {other:?}"),
        }
    }
}
