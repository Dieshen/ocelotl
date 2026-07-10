use ocelotl_core::{OcelotlError, Result, UnsupportedError};

use crate::{checked_len_product, kernel_err};

const QK_K: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q5_K_BLOCK_BYTES: usize = 176;
const Q6_K_BLOCK_BYTES: usize = 210;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlKQuantKind {
    Q4K,
    Q5K,
    Q6K,
}

#[derive(Debug, Clone, Copy)]
pub struct GgmlKQuantMatrixRef<'a> {
    /// GGUF logical input width. Must be block-aligned.
    pub input_features: usize,
    /// GGUF logical output width.
    pub output_features: usize,
    pub kind: GgmlKQuantKind,
    /// Raw GGUF K-quant tensor bytes in output-row order.
    pub data: &'a [u8],
}

pub fn linear_q8_k_k_quant(
    x: &[f32],
    rows: usize,
    matrix: GgmlKQuantMatrixRef<'_>,
    out: &mut [f32],
) -> Result<()> {
    if matrix.kind == GgmlKQuantKind::Q4K {
        return Err(unsupported_k_quant_projection(matrix.kind));
    }

    validate_k_quant_linear(x, rows, matrix, out)?;

    match matrix.kind {
        GgmlKQuantKind::Q5K => linear_q8_k_q5_k(x, rows, matrix, out),
        GgmlKQuantKind::Q6K => linear_q8_k_q6_k(x, rows, matrix, out),
        GgmlKQuantKind::Q4K => unreachable!("Q4_K returns Unsupported before validation"),
    }
}

fn unsupported_k_quant_projection(kind: GgmlKQuantKind) -> OcelotlError {
    OcelotlError::Unsupported(UnsupportedError {
        feature: "ggml_k_quant_projection".to_string(),
        requested: Some(format!("{kind:?}")),
        supported: vec!["Q5K".to_string(), "Q6K".to_string()],
    })
}

fn validate_k_quant_linear(
    x: &[f32],
    rows: usize,
    matrix: GgmlKQuantMatrixRef<'_>,
    out: &[f32],
) -> Result<()> {
    if matrix.input_features == 0 || matrix.output_features == 0 {
        return Err(kernel_err(
            "linear_q8_k_k_quant dimensions must be non-zero",
        ));
    }
    if matrix.input_features % QK_K != 0 {
        return Err(kernel_err(format!(
            "linear_q8_k_k_quant input_features {} is not divisible by Q8_K/K-quant block size {QK_K}",
            matrix.input_features
        )));
    }
    let x_expected =
        checked_len_product("linear_q8_k_k_quant", "x", &[rows, matrix.input_features])?;
    let out_expected = checked_len_product(
        "linear_q8_k_k_quant",
        "out",
        &[rows, matrix.output_features],
    )?;
    if x.len() != x_expected {
        return Err(kernel_err(format!(
            "linear_q8_k_k_quant x.len()={} does not match rows*input_features={}*{}={x_expected}",
            x.len(),
            rows,
            matrix.input_features
        )));
    }
    if out.len() != out_expected {
        return Err(kernel_err(format!(
            "linear_q8_k_k_quant out.len()={} does not match rows*output_features={}*{}={out_expected}",
            out.len(),
            rows,
            matrix.output_features
        )));
    }

    let block_bytes = match matrix.kind {
        GgmlKQuantKind::Q4K => Q4_K_BLOCK_BYTES,
        GgmlKQuantKind::Q5K => Q5_K_BLOCK_BYTES,
        GgmlKQuantKind::Q6K => Q6_K_BLOCK_BYTES,
    };
    let blocks_per_output = matrix.input_features / QK_K;
    let expected_blocks = matrix
        .output_features
        .checked_mul(blocks_per_output)
        .ok_or_else(|| kernel_err("linear_q8_k_k_quant block count overflows usize"))?;
    let expected_bytes = expected_blocks
        .checked_mul(block_bytes)
        .ok_or_else(|| kernel_err("linear_q8_k_k_quant raw weight byte length overflows usize"))?;
    if matrix.data.len() != expected_bytes {
        return Err(kernel_err(format!(
            "linear_q8_k_k_quant raw weight bytes {} do not match expected {expected_bytes}",
            matrix.data.len()
        )));
    }
    Ok(())
}

fn linear_q8_k_q5_k(
    x: &[f32],
    rows: usize,
    matrix: GgmlKQuantMatrixRef<'_>,
    out: &mut [f32],
) -> Result<()> {
    let blocks_per_output = matrix.input_features / QK_K;
    let row_bytes = blocks_per_output * Q5_K_BLOCK_BYTES;

    for row in 0..rows {
        let x_start = row * matrix.input_features;
        let q8_blocks = quantize_row_q8_k(&x[x_start..x_start + matrix.input_features])?;
        for output in 0..matrix.output_features {
            let weight_start = output * row_bytes;
            out[row * matrix.output_features + output] = vec_dot_q5_k_q8_k(
                &matrix.data[weight_start..weight_start + row_bytes],
                &q8_blocks,
            );
        }
    }
    Ok(())
}

fn linear_q8_k_q6_k(
    x: &[f32],
    rows: usize,
    matrix: GgmlKQuantMatrixRef<'_>,
    out: &mut [f32],
) -> Result<()> {
    let blocks_per_output = matrix.input_features / QK_K;
    let row_bytes = blocks_per_output * Q6_K_BLOCK_BYTES;

    for row in 0..rows {
        let x_start = row * matrix.input_features;
        let q8_blocks = quantize_row_q8_k(&x[x_start..x_start + matrix.input_features])?;
        for output in 0..matrix.output_features {
            let weight_start = output * row_bytes;
            out[row * matrix.output_features + output] = vec_dot_q6_k_q8_k(
                &matrix.data[weight_start..weight_start + row_bytes],
                &q8_blocks,
            );
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct Q8KBlock {
    d: f32,
    qs: [i8; QK_K],
    sums16: [i32; QK_K / 16],
}

fn quantize_row_q8_k(input: &[f32]) -> Result<Vec<Q8KBlock>> {
    debug_assert_eq!(input.len() % QK_K, 0);
    let mut blocks = Vec::with_capacity(input.len() / QK_K);
    for block_input in input.chunks_exact(QK_K) {
        let mut max = 0.0_f32;
        let mut amax = 0.0_f32;
        for value in block_input {
            if !value.is_finite() {
                return Err(kernel_err(
                    "linear_q8_k_k_quant input contains non-finite value",
                ));
            }
            let abs = value.abs();
            if abs > amax {
                amax = abs;
                max = *value;
            }
        }

        let mut block = Q8KBlock {
            d: 0.0,
            qs: [0; QK_K],
            sums16: [0; QK_K / 16],
        };
        if amax == 0.0 {
            blocks.push(block);
            continue;
        }

        let iscale = -127.0 / max;
        for (idx, value) in block_input.iter().enumerate() {
            let scaled = iscale * *value;
            if !scaled.is_finite() || scaled.abs() > 4_194_303.0 {
                return Err(kernel_err(
                    "linear_q8_k_k_quant Q8_K scaled value is out of range",
                ));
            }
            let quant = nearest_int(scaled).min(127);
            if !(-128..=127).contains(&quant) {
                return Err(kernel_err("linear_q8_k_k_quant Q8_K value does not fit i8"));
            }
            block.qs[idx] = quant as i8;
        }
        for group in 0..QK_K / 16 {
            let start = group * 16;
            block.sums16[group] = block.qs[start..start + 16]
                .iter()
                .map(|value| i32::from(*value))
                .sum();
        }
        block.d = 1.0 / iscale;
        blocks.push(block);
    }
    Ok(blocks)
}

fn vec_dot_q5_k_q8_k(raw_q5_k_row: &[u8], q8_blocks: &[Q8KBlock]) -> f32 {
    debug_assert_eq!(raw_q5_k_row.len(), q8_blocks.len() * Q5_K_BLOCK_BYTES);
    let mut sums = [0.0_f32; 8];
    let mut sumf = 0.0_f32;

    for (raw_block, q8) in raw_q5_k_row
        .chunks_exact(Q5_K_BLOCK_BYTES)
        .zip(q8_blocks.iter())
    {
        let d = f16_le_at(raw_block, 0) * q8.d;
        let dmin = f16_le_at(raw_block, 2) * q8.d;
        let scales = &raw_block[4..16];
        let qh = &raw_block[16..48];
        let qs = &raw_block[48..176];
        let mut quants = [0_i32; QK_K];
        let mut aux32 = [0_i32; 8];

        let mut high_bit = 1_u8;
        for pair in 0..QK_K / 64 {
            let q_base = pair * 32;
            let value_base = pair * 64;
            for lane in 0..32 {
                quants[value_base + lane] = i32::from(qs[q_base + lane] & 0x0f)
                    + if (qh[lane] & high_bit) != 0 { 16 } else { 0 };
            }
            high_bit <<= 1;
            for lane in 0..32 {
                quants[value_base + 32 + lane] = i32::from(qs[q_base + lane] >> 4)
                    + if (qh[lane] & high_bit) != 0 { 16 } else { 0 };
            }
            high_bit <<= 1;
        }

        let mut sumi = 0_i32;
        for group16 in 0..QK_K / 16 {
            let (_, min) = get_scale_min_k4(group16 / 2, scales);
            sumi += q8.sums16[group16] * i32::from(min);
        }
        sumf -= dmin * sumi as f32;

        for group32 in 0..QK_K / 32 {
            let (scale, _) = get_scale_min_k4(group32, scales);
            let value_start = group32 * 32;
            for lane in 0..32 {
                aux32[lane % 8] += i32::from(scale)
                    * i32::from(q8.qs[value_start + lane])
                    * quants[value_start + lane];
            }
        }

        for (lane, aux) in aux32.iter().enumerate() {
            sums[lane] += d * *aux as f32;
        }
    }

    for lane_sum in sums {
        sumf += lane_sum;
    }
    sumf
}

fn vec_dot_q6_k_q8_k(raw_q6_k_row: &[u8], q8_blocks: &[Q8KBlock]) -> f32 {
    debug_assert_eq!(raw_q6_k_row.len(), q8_blocks.len() * Q6_K_BLOCK_BYTES);
    let mut sums = [0.0_f32; 8];
    let mut sumf = 0.0_f32;

    for (raw_block, q8) in raw_q6_k_row
        .chunks_exact(Q6_K_BLOCK_BYTES)
        .zip(q8_blocks.iter())
    {
        let ql = &raw_block[0..128];
        let qh = &raw_block[128..192];
        let scales = &raw_block[192..208];
        let mut aux8 = [0_i8; QK_K];

        for super_group in 0..2 {
            let ql_base = super_group * 64;
            let qh_base = super_group * 32;
            let value_base = super_group * 128;
            for l in 0..32 {
                let high = qh[qh_base + l];
                aux8[value_base + l] =
                    (((ql[ql_base + l] & 0x0f) | ((high & 0x03) << 4)) as i8) - 32;
                aux8[value_base + l + 32] =
                    (((ql[ql_base + l + 32] & 0x0f) | (((high >> 2) & 0x03) << 4)) as i8) - 32;
                aux8[value_base + l + 64] =
                    (((ql[ql_base + l] >> 4) | (((high >> 4) & 0x03) << 4)) as i8) - 32;
                aux8[value_base + l + 96] =
                    (((ql[ql_base + l + 32] >> 4) | (((high >> 6) & 0x03) << 4)) as i8) - 32;
            }
        }

        let mut aux32 = [0_i32; 8];
        for (group16, scale_byte) in scales.iter().enumerate().take(QK_K / 16) {
            let scale = i8::from_ne_bytes([*scale_byte]);
            let value_start = group16 * 16;
            for l in 0..16 {
                let lane = l % 8;
                aux32[lane] += i32::from(scale)
                    * i32::from(q8.qs[value_start + l])
                    * i32::from(aux8[value_start + l]);
            }
        }

        let d = f16_le_at(raw_block, 208) * q8.d;
        for (lane, aux) in aux32.iter().enumerate() {
            sums[lane] += d * *aux as f32;
        }
    }

    for lane_sum in sums {
        sumf += lane_sum;
    }
    sumf
}

fn get_scale_min_k4(index: usize, scales: &[u8]) -> (u8, u8) {
    debug_assert_eq!(scales.len(), 12);
    debug_assert!(index < 8);
    if index < 4 {
        (scales[index] & 0x3f, scales[index + 4] & 0x3f)
    } else {
        (
            (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    }
}

fn nearest_int(value: f32) -> i32 {
    let bits = (value + 12_582_912.0).to_bits();
    ((bits & 0x007f_ffff) as i32) - 0x0040_0000
}

fn f16_le_at(data: &[u8], offset: usize) -> f32 {
    let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
    f16_bits_to_f32(bits)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1f;
    let frac = (bits & 0x03ff) as u32;

    let f32_bits = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut frac_norm = frac;
                let mut exp_unbiased = -14_i32;
                while (frac_norm & 0x0400) == 0 {
                    frac_norm <<= 1;
                    exp_unbiased -= 1;
                }
                frac_norm &= 0x03ff;
                let exp32 = (exp_unbiased + 127) as u32;
                sign | (exp32 << 23) | (frac_norm << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => {
            let exp32 = u32::from(exp) + (127 - 15);
            sign | (exp32 << 23) | (frac << 13)
        }
    };
    f32::from_bits(f32_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q6_single_weight_block() -> Vec<u8> {
        let mut block = vec![0_u8; Q6_K_BLOCK_BYTES];
        block[0] = 0x01;
        block[128] = 0x02;
        block[192] = 1;
        block[208..210].copy_from_slice(&0x3c00_u16.to_le_bytes());
        block
    }

    fn q5_single_weight_block() -> Vec<u8> {
        let mut block = vec![0_u8; Q5_K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        block[4] = 1;
        block[48] = 0x01;
        block
    }

    fn q5_single_min_block() -> Vec<u8> {
        let mut block = vec![0_u8; Q5_K_BLOCK_BYTES];
        block[2..4].copy_from_slice(&0x3c00_u16.to_le_bytes());
        block[8] = 1;
        block
    }

    fn single_output_matrix(data: &[u8], kind: GgmlKQuantKind) -> GgmlKQuantMatrixRef<'_> {
        GgmlKQuantMatrixRef {
            input_features: QK_K,
            output_features: 1,
            kind,
            data,
        }
    }

    #[test]
    fn q6_k_q8_k_projection_matches_hand_checked_single_weight() {
        let weight = q6_single_weight_block();
        let mut x = vec![0.0_f32; QK_K * 2];
        x[0] = 1.0;
        x[QK_K] = 2.0;
        let mut out = vec![0.0_f32; 2];

        linear_q8_k_k_quant(
            &x,
            2,
            single_output_matrix(&weight, GgmlKQuantKind::Q6K),
            &mut out,
        )
        .expect("Q6_K x Q8_K projection must succeed");

        assert_eq!(out, vec![1.0, 2.0]);
    }

    #[test]
    fn q5_k_q8_k_projection_matches_hand_checked_single_weight() {
        let weight = q5_single_weight_block();
        let mut x = vec![0.0_f32; QK_K * 2];
        x[0] = 1.0;
        x[QK_K] = 2.0;
        let mut out = vec![0.0_f32; 2];

        linear_q8_k_k_quant(
            &x,
            2,
            single_output_matrix(&weight, GgmlKQuantKind::Q5K),
            &mut out,
        )
        .expect("Q5_K x Q8_K projection must succeed");

        assert_eq!(out, vec![1.0, 2.0]);
    }

    #[test]
    fn q5_k_q8_k_projection_applies_min_term() {
        let weight = q5_single_min_block();
        let mut x = vec![0.0_f32; QK_K];
        x[0] = 1.0;
        let mut out = vec![0.0_f32; 1];

        linear_q8_k_k_quant(
            &x,
            1,
            single_output_matrix(&weight, GgmlKQuantKind::Q5K),
            &mut out,
        )
        .expect("Q5_K min term projection must succeed");

        assert_eq!(out, vec![-1.0]);
    }

    #[test]
    fn q6_k_q8_k_projection_rejects_bad_weight_byte_length() {
        let weight = vec![0_u8; Q6_K_BLOCK_BYTES - 1];
        let x = vec![0.0_f32; QK_K];
        let mut out = vec![0.0_f32; 1];

        let err = linear_q8_k_k_quant(
            &x,
            1,
            single_output_matrix(&weight, GgmlKQuantKind::Q6K),
            &mut out,
        )
        .expect_err("bad Q6_K byte length must fail");

        match err {
            OcelotlError::Kernel(kernel) => assert!(kernel.message.contains("raw weight bytes")),
            other => panic!("expected KernelError for bad Q6_K bytes, got {other:?}"),
        }
    }

    #[test]
    fn q6_k_q8_k_projection_rejects_unaligned_input_width() {
        let weight = q6_single_weight_block();
        let matrix = GgmlKQuantMatrixRef {
            input_features: QK_K - 1,
            output_features: 1,
            kind: GgmlKQuantKind::Q6K,
            data: &weight,
        };
        let x = vec![0.0_f32; QK_K - 1];
        let mut out = vec![0.0_f32; 1];

        let err = linear_q8_k_k_quant(&x, 1, matrix, &mut out)
            .expect_err("unaligned input width must fail");

        match err {
            OcelotlError::Kernel(kernel) => assert!(kernel.message.contains("block size")),
            other => panic!("expected KernelError for bad input width, got {other:?}"),
        }
    }

    #[test]
    fn q4_k_projection_remains_explicitly_unsupported() {
        let q4_weight = vec![0_u8; Q4_K_BLOCK_BYTES - 1];
        let x = vec![0.0_f32; QK_K];
        let mut out = vec![0.0_f32; 1];

        let err = linear_q8_k_k_quant(
            &x,
            1,
            single_output_matrix(&q4_weight, GgmlKQuantKind::Q4K),
            &mut out,
        )
        .expect_err("Q4_K native projection is not implemented in this slice");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(unsupported.feature, "ggml_k_quant_projection");
                assert_eq!(unsupported.requested.as_deref(), Some("Q4K"));
                assert_eq!(unsupported.supported, vec!["Q5K", "Q6K"]);
            }
            other => panic!("expected Unsupported for Q4K, got {other:?}"),
        }
    }
}
