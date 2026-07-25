//! Tensor layout helpers shared across model families.
//!
//! `transpose_2d` lived privately inside `qwen2_5` and is needed again by the
//! Parakeet encoder (which moves between `[channels][time]` and `[time][channels]`
//! at every convolution-module boundary). It is promoted here rather than copied:
//! a bug can only hide in the *difference* between two copies of one idea, so
//! `qwen` now re-exports this single implementation.

/// Transpose a row-major `[rows][cols]` matrix into `[cols][rows]`.
///
/// Note this is a pure re-indexing of a 2-D buffer — it is NOT the GGUF
/// `[out][in]` weight conversion, which is a *different* concern that happens to
/// use the same operation. `linear_out_by_in` consumes the raw `[out][in]` layout
/// directly and needs no transpose at all.
pub fn transpose_2d(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), rows * cols);
    let mut dst = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_2d_matches_hand_written_result() {
        // [[1,2,3],
        //  [4,5,6]]  ->  [[1,4],
        //                 [2,5],
        //                 [3,6]]
        let src = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(transpose_2d(&src, 2, 3), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_2d_is_an_involution_on_the_swapped_shape() {
        let src: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let once = transpose_2d(&src, 3, 4);
        assert_eq!(transpose_2d(&once, 4, 3), src);
    }

    #[test]
    fn transpose_2d_handles_degenerate_shapes() {
        let src = [7.0_f32];
        assert_eq!(transpose_2d(&src, 1, 1), vec![7.0]);
        let row = [1.0_f32, 2.0, 3.0];
        assert_eq!(transpose_2d(&row, 1, 3), vec![1.0, 2.0, 3.0]);
    }
}
