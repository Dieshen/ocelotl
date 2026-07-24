//! Sequence pooling for embedding models: reduce per-token hidden states
//! `[seq_len, dim]` (row-major) to a single `[dim]` sentence embedding.
//!
//! EmbeddingGemma (and pplx-embed) use **mean pooling** (`pooling_type = 1` in
//! the GGUF) followed by projection heads; the final embedding is typically
//! **L2-normalized** so cosine similarity is a dot product. Both steps live
//! here as small, testable primitives the model forward composes.

use crate::{kernel_err, Result};

/// Mean-pool token hidden states: `out[d] = mean_i hidden[i, d]`.
///
/// `hidden` is row-major `[seq_len, dim]`. Every token row is averaged — the
/// caller owns padding exclusion (ocelotl runs one real sequence per forward,
/// no padding). This matches llama.cpp `pooling_type = 1` (mean) for a single
/// input sequence.
pub fn mean_pool(hidden: &[f32], seq_len: usize, dim: usize) -> Result<Vec<f32>> {
    if seq_len == 0 || dim == 0 {
        return Err(kernel_err(
            "mean_pool seq_len and dim must be non-zero".to_string(),
        ));
    }
    let expected = seq_len
        .checked_mul(dim)
        .ok_or_else(|| kernel_err("mean_pool seq_len*dim overflows usize".to_string()))?;
    if hidden.len() != expected {
        return Err(kernel_err(format!(
            "mean_pool hidden.len()={} does not match seq_len*dim={expected}",
            hidden.len()
        )));
    }
    let mut out = vec![0.0_f32; dim];
    for i in 0..seq_len {
        let base = i * dim;
        for (d, o) in out.iter_mut().enumerate() {
            *o += hidden[base + d];
        }
    }
    let inv = 1.0_f32 / seq_len as f32;
    for o in &mut out {
        *o *= inv;
    }
    Ok(out)
}

/// L2-normalize a vector in place: `v /= ||v||_2`. A zero vector is left
/// unchanged (no division by zero) — cosine against it is undefined anyway.
pub fn l2_normalize(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    if norm_sq > 0.0 {
        let inv = 1.0_f32 / norm_sq.sqrt();
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_averages_token_rows() {
        // 2 tokens, dim 3: rows [1,2,3] and [3,4,5] → mean [2,3,4].
        let hidden = [1.0_f32, 2.0, 3.0, 3.0, 4.0, 5.0];
        let out = mean_pool(&hidden, 2, 3).expect("well-formed mean_pool");
        assert_eq!(out, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn mean_pool_rejects_length_mismatch() {
        let hidden = [1.0_f32, 2.0, 3.0];
        assert!(mean_pool(&hidden, 2, 3).is_err());
    }

    #[test]
    fn l2_normalize_makes_unit_norm() {
        let mut v = [3.0_f32, 4.0]; // norm 5 → [0.6, 0.8]
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector_is_noop() {
        let mut v = [0.0_f32, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, [0.0, 0.0]);
    }
}
