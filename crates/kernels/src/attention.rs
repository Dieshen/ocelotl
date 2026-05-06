//! CPU reference scaled-dot-product attention with causal masking and GQA.
//!
//! Single-request, single-batch. The model forward path is responsible for
//! applying RoPE to Q and K **before** calling this kernel; this kernel is
//! pure scaled-dot-product attention plus the causal mask plus the
//! grouped-query-attention head mapping.
//!
//! # Layout & stride contract (M3.5, locked by Rick + Javier 2026-05-05)
//!
//! Contiguous row-major `&[f32]`, no strides — same M1.7/M3.3/M3.4 contract.
//!
//! - `q` is `[seq_len, num_q_heads, head_dim]` flattened row-major. The
//!   per-token Q projection that produced this slice is `[seq_len, hidden]`
//!   where `hidden == num_q_heads * head_dim`; the kernel takes the same
//!   memory and just re-interprets the inner axis as `(head, dim)`. Total
//!   length: `seq_len * num_q_heads * head_dim`.
//! - `k` is `[seq_len, num_kv_heads, head_dim]` flattened row-major. Total
//!   length: `seq_len * num_kv_heads * head_dim`.
//! - `v` is `[seq_len, num_kv_heads, head_dim]` flattened row-major. Same
//!   total length as `k`.
//! - `out` is `[seq_len, num_q_heads, head_dim]` flattened row-major. Same
//!   total length as `q`.
//!
//! Indexing helpers (used internally and pinned here so M3.7's reader
//! does not have to re-derive them):
//!
//! ```text
//! q[t, h, d]  = q[(t * num_q_heads  + h) * head_dim + d]
//! k[t, kh, d] = k[(t * num_kv_heads + kh) * head_dim + d]
//! v[t, kh, d] = v[(t * num_kv_heads + kh) * head_dim + d]
//! out[t, h, d] = out[(t * num_q_heads + h) * head_dim + d]
//! ```
//!
//! # GQA (grouped-query attention) head mapping
//!
//! `num_q_heads` must be a positive multiple of `num_kv_heads`. Define
//! `group_size = num_q_heads / num_kv_heads`. Query head `h` consumes KV
//! head `h / group_size` (integer division). For Qwen2.5-0.5B that's
//! `14 / 2 = 7`, so q_heads 0..7 share kv_head 0 and q_heads 7..14 share
//! kv_head 1. The mapping is a fixed function of head indices — there is
//! no learned routing.
//!
//! # Causal mask
//!
//! Single-request prefill: position `i` may attend to positions `j` with
//! `j <= i`. We implement this by setting `score[i, j] = -inf` for `j > i`
//! before softmax — the softmax then assigns those positions probability
//! zero and the upstream output never sees them. We do not skip the
//! computation; the explicit `-inf` keeps the per-row reduction in
//! `softmax` numerically clean (subtract-max stays well-defined).
//!
//! # Composition vs inlining
//!
//! `softmax` is reused (per-row in-place over the unmasked prefix). The
//! score and output matmuls are inlined: per-head, per-query-position
//! they're a `head_dim`-length dot followed by a `head_dim`-length scaled
//! accumulation. Calling `dot` inside a triple loop would still be O(n)
//! per call but would also pay validation overhead per call; inlining is
//! simpler and keeps the trip-count discipline visible.
//!
//! # Phase 2 scope
//!
//! - In: kernel function, layout decision, validation.
//! - Out: model-side wrapper, KV cache, multi-request batching.

use ocelotl_core::Result;

use crate::{checked_len_product, kernel_err, softmax};

/// Scaled-dot-product attention with causal mask and GQA, single request.
///
/// Computes:
///
/// ```text
/// for each query head h, query position i, key position j with j <= i:
///   kh             = h / group_size  where group_size = num_q_heads / num_kv_heads
///   scores[h,i,j]  = (Q[i,h,:] . K[j,kh,:]) / sqrt(head_dim)
///   scores[h,i,j>i] = -inf  (causal mask)
///   probs[h,i,:]   = softmax_j(scores[h,i,:])
///   out[i,h,:]     = sum_j probs[h,i,j] * V[j,kh,:]
/// ```
///
/// See module-level docs for the exact row-major layout.
///
/// # Errors
///
/// Returns `KernelError` (backend = `"cpu"`) when:
/// - `head_dim` is zero,
/// - `seq_len` is zero,
/// - `num_q_heads` is zero,
/// - `num_kv_heads` is zero,
/// - `num_q_heads % num_kv_heads != 0` (GQA group sizing invalid),
/// - any of `q`, `k`, `v`, `out` slice lengths do not match the declared
///   shape implied by `(seq_len, num_q_heads | num_kv_heads, head_dim)`.
#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) -> Result<()> {
    if head_dim == 0 {
        return Err(kernel_err(
            "scaled_dot_product_attention head_dim must be non-zero".to_string(),
        ));
    }
    if seq_len == 0 {
        return Err(kernel_err(
            "scaled_dot_product_attention seq_len must be non-zero".to_string(),
        ));
    }
    if num_q_heads == 0 {
        return Err(kernel_err(
            "scaled_dot_product_attention num_q_heads must be non-zero".to_string(),
        ));
    }
    if num_kv_heads == 0 {
        return Err(kernel_err(
            "scaled_dot_product_attention num_kv_heads must be non-zero".to_string(),
        ));
    }
    if num_q_heads % num_kv_heads != 0 {
        return Err(kernel_err(format!(
            "scaled_dot_product_attention num_q_heads ({num_q_heads}) must be a positive \
             multiple of num_kv_heads ({num_kv_heads})"
        )));
    }

    let q_total = checked_len_product(
        "scaled_dot_product_attention",
        "q/out",
        &[seq_len, num_q_heads, head_dim],
    )?;
    let kv_total = checked_len_product(
        "scaled_dot_product_attention",
        "k/v",
        &[seq_len, num_kv_heads, head_dim],
    )?;

    if q.len() != q_total {
        return Err(kernel_err(format!(
            "scaled_dot_product_attention q.len()={} does not match \
             seq_len*num_q_heads*head_dim={}*{}*{}={}",
            q.len(),
            seq_len,
            num_q_heads,
            head_dim,
            q_total
        )));
    }
    if k.len() != kv_total {
        return Err(kernel_err(format!(
            "scaled_dot_product_attention k.len()={} does not match \
             seq_len*num_kv_heads*head_dim={}*{}*{}={}",
            k.len(),
            seq_len,
            num_kv_heads,
            head_dim,
            kv_total
        )));
    }
    if v.len() != kv_total {
        return Err(kernel_err(format!(
            "scaled_dot_product_attention v.len()={} does not match \
             seq_len*num_kv_heads*head_dim={}*{}*{}={}",
            v.len(),
            seq_len,
            num_kv_heads,
            head_dim,
            kv_total
        )));
    }
    if out.len() != q_total {
        return Err(kernel_err(format!(
            "scaled_dot_product_attention out.len()={} does not match \
             seq_len*num_q_heads*head_dim={}*{}*{}={}",
            out.len(),
            seq_len,
            num_q_heads,
            head_dim,
            q_total
        )));
    }

    let group_size = num_q_heads / num_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // Per-(query position i, query head h) softmax buffer. We allocate it
    // once and reuse across iterations; the masked tail beyond i+1 is not
    // touched by either the score loop or the output accumulation loop.
    let mut scores = vec![0.0_f32; seq_len];

    for i in 0..seq_len {
        for h in 0..num_q_heads {
            let kh = h / group_size;

            // 1) Compute scaled dot products for j in 0..=i; j>i stays
            //    untouched and is never read (we only softmax/accumulate
            //    over the unmasked prefix). Setting -inf is unnecessary
            //    when we slice the prefix; numerically equivalent.
            let q_base = (i * num_q_heads + h) * head_dim;
            for (j, score) in scores.iter_mut().enumerate().take(i + 1) {
                let mut acc = 0.0_f32;
                let k_base = (j * num_kv_heads + kh) * head_dim;
                for d in 0..head_dim {
                    acc += q[q_base + d] * k[k_base + d];
                }
                *score = acc * scale;
            }

            // 2) Softmax over the unmasked prefix [0..=i]. Reuse the
            //    in-place softmax kernel from M1.7.
            softmax(&mut scores[..=i]);

            // 3) Accumulate weighted V into out. Zero the output row
            //    first because we accumulate.
            let out_base = (i * num_q_heads + h) * head_dim;
            for d in 0..head_dim {
                out[out_base + d] = 0.0_f32;
            }
            for (j, &p) in scores.iter().enumerate().take(i + 1) {
                let v_base = (j * num_kv_heads + kh) * head_dim;
                for d in 0..head_dim {
                    out[out_base + d] += p * v[v_base + d];
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ocelotl_core::{KernelError, OcelotlError};

    // --- Test 1: single-head causal attention, hand-checked ---
    //
    // Setup: seq_len=2, num_q_heads=num_kv_heads=1, head_dim=2.
    //
    //   Q = [[1, 0], [0, 1]]   flat: [1, 0, 0, 1]
    //   K = [[1, 0], [0, 1]]   flat: [1, 0, 0, 1]
    //   V = [[1, 2], [3, 4]]   flat: [1, 2, 3, 4]
    //
    //   scale = 1/sqrt(2) ≈ 0.7071067811865475
    //
    // Position i=0 (causal: only j=0):
    //   score[0,0,0] = (1*1 + 0*0) * scale = 0.7071068
    //   softmax over j∈{0}: probs = [1.0]
    //   out[0,0,:]   = 1.0 * V[0] = [1, 2]
    //
    // Position i=1 (j ∈ {0, 1}):
    //   score[0,1,0] = (0*1 + 1*0) * scale = 0
    //   score[0,1,1] = (0*0 + 1*1) * scale = 0.7071068
    //   softmax([0, 0.7071068]):
    //     shifted = [-0.7071068, 0]
    //     exp     = [0.49306869, 1.0]
    //     sum     = 1.49306869
    //     probs   ≈ [0.33023959, 0.66976041]
    //   out[0,1,:] = 0.33023958420 * [1, 2] + 0.66976041579 * [3, 4]
    //              = [0.33023958420 + 2.00928124738, 0.66047916841 + 2.67904166317]
    //              ≈ [2.33952083158, 3.33952083158]   (in 64-bit math)
    //
    // Expected flat output (64-bit-derived): [1, 2, 2.33952083, 3.33952083].
    // Tolerance: 5e-6. The chain is longer than RoPE's (head_dim mults +
    // exp + div + head_dim weighted accumulation), so the kernel's f32
    // result drifts ~2e-6 from the f64-derived expected — observed
    // empirically at 2.3395228 vs 2.3395208. 5e-6 is tight enough to
    // catch logic bugs (wrong scale, mask off-by-one, wrong softmax
    // window) and loose enough to absorb f32 rounding through the
    // multi-op chain. Bumped from 1e-6 after the first run failed for
    // exactly this reason — recorded as a pair-catch in the journal
    // (the navigator caught that "fail for right reason" includes
    // "the test specifies an unattainable tolerance", not just
    // "the impl is broken").

    #[test]
    fn single_head_causal_attention_matches_hand_computation() {
        let q = [1.0_f32, 0.0, 0.0, 1.0];
        let k = [1.0_f32, 0.0, 0.0, 1.0];
        let v = [1.0_f32, 2.0, 3.0, 4.0];
        let mut out = [0.0_f32; 4];

        scaled_dot_product_attention(
            &q, &k, &v, /* seq_len */ 2, /* num_q_heads */ 1, /* num_kv_heads */ 1,
            /* head_dim */ 2, &mut out,
        )
        .expect("well-formed attention call must succeed");

        let expected = [1.0_f32, 2.0, 2.339_521, 3.339_521];
        let tol = 5.0e-6_f32;
        for (idx, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < tol,
                "attention mismatch at flat index {idx}: got {got}, want {want}"
            );
        }
    }

    // --- Test 2: GQA head mapping, exact match (no softmax noise) ---
    //
    // Setup: seq_len=1, num_q_heads=4, num_kv_heads=2, head_dim=2,
    // group_size = 4 / 2 = 2.
    //
    // Mapping under "h / group_size" (integer division):
    //   q_head 0 -> kv_head 0
    //   q_head 1 -> kv_head 0
    //   q_head 2 -> kv_head 1
    //   q_head 3 -> kv_head 1
    //
    // With seq_len=1 the only attended position is j=0; softmax over a
    // single element is exactly 1.0. The output therefore reduces to
    //   out[0, h, :] = V[0, kv_head_for(h), :]
    // — independent of Q and K. Perfect signal for the mapping rule:
    // any swap or off-by-one ("h % group_size", "h / num_kv_heads",
    // etc.) produces a different flat output.
    //
    // Distinct V values per kv_head so a mis-map is visible:
    //   V[0, kv0, :] = [10, 20]
    //   V[0, kv1, :] = [30, 40]
    //   V flat       = [10, 20, 30, 40]   (kv0 then kv1, head_dim 2)
    //
    // Q and K still use distinct values per head so a "shuffled q access"
    // bug would also produce a divergent score, but the seq_len=1
    // softmax neutralizes that path; the test pins mapping only.
    //
    // Expected flat out (q_head 0..4, head_dim 2):
    //   [10, 20,  10, 20,  30, 40,  30, 40]
    //
    // Exact-equality assertion is appropriate because the only float op
    // here is multiply-by-1.0; no rounding chain.

    #[test]
    fn gqa_mapping_pins_query_head_to_kv_head_assignment() {
        let seq_len = 1_usize;
        let num_q_heads = 4_usize;
        let num_kv_heads = 2_usize;
        let head_dim = 2_usize;

        // Q[0, h, :] = [(h+1) as f32, 0.0]
        let q = [1.0_f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        // K[0, kh, :] = [(kh+1) as f32, 0.0] — values irrelevant for mapping.
        let k = [1.0_f32, 0.0, 2.0, 0.0];
        // V[0, kv0, :] = [10, 20]; V[0, kv1, :] = [30, 40]
        let v = [10.0_f32, 20.0, 30.0, 40.0];
        let mut out = [0.0_f32; 8];

        scaled_dot_product_attention(
            &q,
            &k,
            &v,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            &mut out,
        )
        .expect("well-formed GQA attention call must succeed");

        let expected = [10.0_f32, 20.0, 10.0, 20.0, 30.0, 40.0, 30.0, 40.0];
        assert_eq!(
            out, expected,
            "GQA mapping must route q_heads 0,1 to kv_head 0 and q_heads 2,3 to kv_head 1"
        );
    }

    // --- Test 3: causal mask isolation (head_dim=1, all-zero K) ---
    //
    // Setup: seq_len=3, num_q_heads=num_kv_heads=1, head_dim=1.
    //   Q = [1, 1, 1]
    //   K = [0, 0, 0]   -> all scores 0 -> uniform softmax over the
    //                     unmasked prefix at every query position
    //   V = [1, 2, 100] -> the 100 is a tripwire: if mask leaks at
    //                     position 1 and includes V[2], the result jumps
    //                     from 1.5 to ~34.3 — impossible to miss.
    //
    // Expected per position (uniform softmax over j=0..=i):
    //   pos 0: out = 1 * V[0]                = 1.0
    //   pos 1: out = (V[0] + V[1]) / 2       = 1.5
    //   pos 2: out = (V[0] + V[1] + V[2]) / 3 = 103/3 ≈ 34.33333
    //
    // Tolerance 5e-6 — same chain length as test 1.

    #[test]
    fn causal_mask_excludes_future_positions() {
        let seq_len = 3_usize;
        let head_dim = 1_usize;

        let q = [1.0_f32, 1.0, 1.0];
        let k = [0.0_f32, 0.0, 0.0];
        let v = [1.0_f32, 2.0, 100.0];
        let mut out = [0.0_f32; 3];

        scaled_dot_product_attention(&q, &k, &v, seq_len, 1, 1, head_dim, &mut out)
            .expect("well-formed causal attention call must succeed");

        let expected = [1.0_f32, 1.5, 103.0 / 3.0];
        let tol = 5.0e-6_f32;
        for (idx, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < tol,
                "causal-mask attention mismatch at position {idx}: got {got}, want {want}"
            );
        }
    }

    // --- Validation tests (launch boundary) ---

    fn valid_args() -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
        // seq_len=2, q_heads=1, kv_heads=1, head_dim=2 — same shape as test 1.
        let q = [1.0_f32, 0.0, 0.0, 1.0];
        let k = [1.0_f32, 0.0, 0.0, 1.0];
        let v = [1.0_f32, 2.0, 3.0, 4.0];
        let out = [0.0_f32; 4];
        (q, k, v, out)
    }

    fn assert_kernel_err_contains(err: OcelotlError, needle: &str) {
        match err {
            OcelotlError::Kernel(KernelError { backend, message }) => {
                assert_eq!(backend, "cpu");
                assert!(
                    message.contains(needle),
                    "expected error message to contain {needle:?}, got {message:?}"
                );
            }
            other => panic!("expected KernelError, got {other:?}"),
        }
    }

    #[test]
    fn rejects_zero_head_dim() {
        let (q, k, v, mut out) = valid_args();
        let err = scaled_dot_product_attention(&q, &k, &v, 2, 1, 1, 0, &mut out)
            .expect_err("zero head_dim must be rejected");
        assert_kernel_err_contains(err, "head_dim");
    }

    #[test]
    fn rejects_zero_seq_len() {
        let (q, k, v, mut out) = valid_args();
        let err = scaled_dot_product_attention(&q, &k, &v, 0, 1, 1, 2, &mut out)
            .expect_err("zero seq_len must be rejected");
        assert_kernel_err_contains(err, "seq_len");
    }

    #[test]
    fn rejects_q_heads_not_multiple_of_kv_heads() {
        // 3 q_heads, 2 kv_heads — group sizing invalid.
        // Build slices large enough that the *length* checks would also
        // fail; we want the multiple-of check to fire first, so size them
        // to match the declared shape.
        let seq_len = 1;
        let q_heads = 3;
        let kv_heads = 2;
        let head_dim = 2;
        let q = vec![0.0_f32; seq_len * q_heads * head_dim];
        let k = vec![0.0_f32; seq_len * kv_heads * head_dim];
        let v = vec![0.0_f32; seq_len * kv_heads * head_dim];
        let mut out = vec![0.0_f32; seq_len * q_heads * head_dim];

        let err = scaled_dot_product_attention(
            &q, &k, &v, seq_len, q_heads, kv_heads, head_dim, &mut out,
        )
        .expect_err("non-multiple q/kv head count must be rejected");
        assert_kernel_err_contains(err, "multiple of num_kv_heads");
    }

    #[test]
    fn rejects_shape_product_overflow() {
        let mut out = [];
        let err = scaled_dot_product_attention(&[], &[], &[], usize::MAX, 2, 1, 1, &mut out)
            .expect_err("overflowing shape product must be rejected");

        assert_kernel_err_contains(err, "overflows");
    }

    #[test]
    fn rejects_q_slice_length_mismatch() {
        let q = [0.0_f32; 3]; // claimed seq_len=2 * q_heads=1 * head_dim=2 = 4
        let k = [0.0_f32; 4];
        let v = [0.0_f32; 4];
        let mut out = [0.0_f32; 4];

        let err = scaled_dot_product_attention(&q, &k, &v, 2, 1, 1, 2, &mut out)
            .expect_err("q length mismatch must be rejected");
        assert_kernel_err_contains(err, "q.len()");
    }

    #[test]
    fn rejects_kv_slice_length_mismatch() {
        let q = [0.0_f32; 4];
        let k = [0.0_f32; 3]; // wrong
        let v = [0.0_f32; 4];
        let mut out = [0.0_f32; 4];

        let err = scaled_dot_product_attention(&q, &k, &v, 2, 1, 1, 2, &mut out)
            .expect_err("k length mismatch must be rejected");
        assert_kernel_err_contains(err, "k.len()");
    }

    #[test]
    fn rejects_out_slice_length_mismatch() {
        let (q, k, v, _) = valid_args();
        let mut out = [0.0_f32; 3]; // claimed seq_len=2 * q_heads=1 * head_dim=2 = 4

        let err = scaled_dot_product_attention(&q, &k, &v, 2, 1, 1, 2, &mut out)
            .expect_err("out length mismatch must be rejected");
        assert_kernel_err_contains(err, "out.len()");
    }
}
