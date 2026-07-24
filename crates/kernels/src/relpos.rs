//! Relative positional encoding primitives (Transformer-XL / Conformer style).
//!
//! Parakeet's encoder uses `self_attention_model: rel_pos` — **sinusoidal
//! relative** positions, not rotary. `crate::rope` is not applicable and its
//! presence must not suggest positional encoding is already handled.
//!
//! These are the two pieces that are pure index/table work and can be pinned by
//! hand before any weights exist. Composing them into the full attention score
//! (`((q + pos_bias_u)·kᵀ + rel_shift((q + pos_bias_v)·pᵀ)) / √d`) is deliberately
//! left to the encoder assembly, where it can be checked against the reference
//! encoder's tensors — writing it here would be writing code no test could fail.

use crate::{Result, kernel_err};

/// Convert a `[rows, 2*rows-1]` relative-position score matrix into the
/// `[rows, rows]` matrix indexed by absolute `(query, key)`.
///
/// `out[i][j] = input[i][(rows - 1) - i + j]`
///
/// That mapping is the pad-reshape-slice trick written out directly. The trick —
/// pad one zero column, view as `[2*rows, rows]`, drop the first row, view back
/// as `[rows, 2*rows-1]`, take the leading `rows` columns — is an in-place way to
/// achieve the same shift, and tracing it for `rows = 3` gives exactly the
/// formula above (see the hand-worked test). Doing the indexing explicitly costs
/// the same O(rows²) and is far harder to get silently wrong.
///
/// Row `i` reads a window starting at `(rows-1) - i`, i.e. the window slides left
/// as the query index advances; that is what makes column `j` mean "key `j`"
/// uniformly across rows.
pub fn rel_shift(input: &[f32], rows: usize, out: &mut [f32]) -> Result<()> {
    if rows == 0 {
        return Err(kernel_err("rel_shift rows must be non-zero"));
    }
    let width = 2 * rows - 1;
    if input.len() != rows * width {
        return Err(kernel_err(format!(
            "rel_shift input.len()={} does not match rows*(2*rows-1)={}",
            input.len(),
            rows * width
        )));
    }
    if out.len() != rows * rows {
        return Err(kernel_err(format!(
            "rel_shift out.len()={} does not match rows*rows={}",
            out.len(),
            rows * rows
        )));
    }
    for i in 0..rows {
        let base = (rows - 1) - i;
        let src = i * width;
        let dst = i * rows;
        for j in 0..rows {
            out[dst + j] = input[src + base + j];
        }
    }
    Ok(())
}

/// Sinusoidal relative-position table of shape `[2*len-1, dim]`.
///
/// Row `r` encodes relative offset `pos = (len - 1) - r`, so the table runs from
/// `+(len-1)` down through `0` to `-(len-1)` — the descending order NeMo's
/// `RelPositionalEncoding` produces, and the order [`rel_shift`] expects, since
/// row 0 of the score matrix must correspond to the largest positive offset.
///
/// Within a row the standard interleaving applies:
/// `pe[2i] = sin(pos / base^(2i/dim))`, `pe[2i+1] = cos(pos / base^(2i/dim))`.
///
/// The angle is computed in `f64`: for `len` in the thousands and small `2i/dim`,
/// `pos / base^(2i/dim)` reaches the thousands, and `sin`/`cos` on arguments that
/// large in `f32` lose mantissa to range reduction — the same failure that cost
/// 200x in the Parakeet mel frontend's DFT.
pub fn sinusoidal_rel_pos_table(len: usize, dim: usize, base: f32) -> Result<Vec<f32>> {
    if len == 0 || dim == 0 {
        return Err(kernel_err(
            "sinusoidal_rel_pos_table len and dim must be non-zero",
        ));
    }
    if dim % 2 != 0 {
        return Err(kernel_err(format!(
            "sinusoidal_rel_pos_table dim must be even, got {dim}"
        )));
    }
    if !base.is_finite() || base <= 0.0 {
        return Err(kernel_err(format!(
            "sinusoidal_rel_pos_table base must be finite and positive, got {base}"
        )));
    }
    let rows = 2 * len - 1;
    let mut table = vec![0.0_f32; rows * dim];
    for r in 0..rows {
        let pos = (len as f64 - 1.0) - r as f64;
        let row = r * dim;
        for i in 0..dim / 2 {
            let inv = (base as f64).powf(-2.0 * i as f64 / dim as f64);
            let angle = pos * inv;
            table[row + 2 * i] = angle.sin() as f32;
            table[row + 2 * i + 1] = angle.cos() as f32;
        }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rel_shift_matches_the_hand_traced_three_by_five_case() {
        // Derived by tracing the pad-reshape-slice trick for rows = 3
        // (width = 5). Label input[i][k] as the value 10*i + k.
        //   pad a zero column      -> rows of width 6
        //   view as [6, 3]         -> [0,a00,a01] [a02,a03,a04]
        //                             [0,a10,a11] [a12,a13,a14]
        //                             [0,a20,a21] [a22,a23,a24]
        //   drop the first row, view as [3, 5], take the leading 3 columns:
        //     row0 = a02,a03,a04
        //     row1 = a11,a12,a13
        //     row2 = a20,a21,a22
        let input: Vec<f32> = (0..3)
            .flat_map(|i| (0..5).map(move |k| (10 * i + k) as f32))
            .collect();
        let mut out = vec![0.0_f32; 9];
        rel_shift(&input, 3, &mut out).expect("rel_shift");
        assert_eq!(out, vec![2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0]);
    }

    #[test]
    fn rel_shift_matches_the_hand_worked_four_by_seven_case() {
        // rows = 4, width = 7; out[i][j] = input[i][3 - i + j].
        //   row0 = a03,a04,a05,a06
        //   row1 = a12,a13,a14,a15
        //   row2 = a21,a22,a23,a24
        //   row3 = a30,a31,a32,a33
        let input: Vec<f32> = (0..4)
            .flat_map(|i| (0..7).map(move |k| (10 * i + k) as f32))
            .collect();
        let mut out = vec![0.0_f32; 16];
        rel_shift(&input, 4, &mut out).expect("rel_shift");
        assert_eq!(
            out,
            vec![
                3.0, 4.0, 5.0, 6.0, // i = 0
                12.0, 13.0, 14.0, 15.0, // i = 1
                21.0, 22.0, 23.0, 24.0, // i = 2
                30.0, 31.0, 32.0, 33.0, // i = 3
            ]
        );
    }

    #[test]
    fn rel_shift_diagonal_is_the_zero_offset_column() {
        // The load-bearing property: the diagonal out[i][i] must all come from
        // the SAME source column, the centre (rows-1) — that is what "relative
        // offset 0" means. A transposed or off-by-one shift breaks this while
        // still producing a plausible-looking matrix.
        let rows = 5;
        let width = 2 * rows - 1;
        // Mark only the centre column.
        let mut input = vec![0.0_f32; rows * width];
        for i in 0..rows {
            input[i * width + (rows - 1)] = 1.0;
        }
        let mut out = vec![0.0_f32; rows * rows];
        rel_shift(&input, rows, &mut out).expect("rel_shift");
        for i in 0..rows {
            for j in 0..rows {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(
                    out[i * rows + j],
                    expected,
                    "centre column did not land on the diagonal at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn rel_shift_rejects_shape_mismatch() {
        let mut out = vec![0.0_f32; 9];
        assert!(rel_shift(&[0.0; 10], 3, &mut out).is_err());
    }

    #[test]
    fn sinusoidal_table_centre_row_is_offset_zero() {
        // Row (len-1) is relative offset 0 => sin(0)=0, cos(0)=1 everywhere.
        let len = 4;
        let dim = 8;
        let t = sinusoidal_rel_pos_table(len, dim, 10_000.0).expect("table");
        assert_eq!(t.len(), (2 * len - 1) * dim);
        let centre = (len - 1) * dim;
        for i in 0..dim / 2 {
            assert_eq!(t[centre + 2 * i], 0.0, "sin at offset 0");
            assert_eq!(t[centre + 2 * i + 1], 1.0, "cos at offset 0");
        }
    }

    #[test]
    fn sinusoidal_table_is_ordered_descending_from_positive_to_negative() {
        // Row 0 must be the LARGEST POSITIVE offset and the last row the most
        // negative. Check via the sin channel's sign at i = 0 (inv = 1), where
        // sin(pos) for pos = +1 and -1 must be equal and opposite.
        let t = sinusoidal_rel_pos_table(2, 4, 10_000.0).expect("table");
        let dim = 4;
        // rows = 3: offsets +1, 0, -1.
        let sin_first = t[dim * 0];
        let sin_last = t[dim * 2];
        assert!(
            (sin_first - 1.0_f32.sin()).abs() < 1e-6,
            "row 0 should be +1"
        );
        assert!(
            (sin_last + 1.0_f32.sin()).abs() < 1e-6,
            "last row should be -1"
        );
        assert!(
            sin_first > 0.0 && sin_last < 0.0,
            "ordering is not descending"
        );
    }

    #[test]
    fn sinusoidal_table_rejects_odd_dim_and_bad_base() {
        assert!(sinusoidal_rel_pos_table(4, 7, 10_000.0).is_err());
        assert!(sinusoidal_rel_pos_table(4, 8, 0.0).is_err());
    }
}
