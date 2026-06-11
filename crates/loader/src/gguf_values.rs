//! Value loading for GGUF tensors.
//!
//! GGUF value loading starts deliberately narrow: dense F32/F16/BF16 tensors can
//! be converted into Ocelotl-owned `LoadedTensor` values. K-quantized tensor
//! loading is available only through the explicit dequantizing API so callers
//! cannot accidentally treat quantized payloads as dense values.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use ocelotl_core::{InvalidModelError, IoError, OcelotlError, Result, UnsupportedError};

use crate::{
    GgmlTensorType, GgufManifest, GgufTensorEntry, inspect_gguf,
    safetensors_inspect::SupportedDtype, safetensors_values::LoadedTensor,
};

const Q4_K_BLOCK_BYTES: usize = 144;
const Q5_K_BLOCK_BYTES: usize = 176;
const Q6_K_BLOCK_BYTES: usize = 210;

pub fn load_gguf_tensor_f32(path: &Path, tensor_name: &str) -> Result<LoadedTensor> {
    let manifest = inspect_gguf(path)?;
    let mut file = File::open(path).map_err(|source| io_error(path, source))?;
    load_tensor_from_manifest(
        path,
        &manifest,
        &mut file,
        tensor_name,
        GgufValueMode::DenseOnly,
    )
}

pub fn load_gguf_tensors_f32<S: AsRef<str>>(
    path: &Path,
    tensor_names: &[S],
) -> Result<Vec<LoadedTensor>> {
    let manifest = inspect_gguf(path)?;
    let mut file = File::open(path).map_err(|source| io_error(path, source))?;
    tensor_names
        .iter()
        .map(|name| {
            load_tensor_from_manifest(
                path,
                &manifest,
                &mut file,
                name.as_ref(),
                GgufValueMode::DenseOnly,
            )
        })
        .collect()
}

pub fn load_gguf_tensor_dequantized_f32(path: &Path, tensor_name: &str) -> Result<LoadedTensor> {
    let manifest = inspect_gguf(path)?;
    let mut file = File::open(path).map_err(|source| io_error(path, source))?;
    load_tensor_from_manifest(
        path,
        &manifest,
        &mut file,
        tensor_name,
        GgufValueMode::DenseOrKQuantized,
    )
}

pub fn load_gguf_tensors_dequantized_f32<S: AsRef<str>>(
    path: &Path,
    tensor_names: &[S],
) -> Result<Vec<LoadedTensor>> {
    let manifest = inspect_gguf(path)?;
    let mut file = File::open(path).map_err(|source| io_error(path, source))?;
    tensor_names
        .iter()
        .map(|name| {
            load_tensor_from_manifest(
                path,
                &manifest,
                &mut file,
                name.as_ref(),
                GgufValueMode::DenseOrKQuantized,
            )
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgufValueMode {
    DenseOnly,
    DenseOrKQuantized,
}

fn load_tensor_from_manifest(
    path: &Path,
    manifest: &GgufManifest,
    file: &mut File,
    tensor_name: &str,
    mode: GgufValueMode,
) -> Result<LoadedTensor> {
    let tensor = manifest
        .tensors
        .iter()
        .find(|tensor| tensor.name == tensor_name)
        .ok_or_else(|| {
            OcelotlError::from(InvalidModelError {
                path: Some(path.to_path_buf()),
                field: Some(tensor_name.to_string()),
                message: format!("required tensor `{tensor_name}` not found in GGUF file"),
            })
        })?;
    load_tensor_bytes(path, file, tensor, mode)
}

fn load_tensor_bytes(
    path: &Path,
    file: &mut File,
    tensor: &GgufTensorEntry,
    mode: GgufValueMode,
) -> Result<LoadedTensor> {
    let tensor_name = tensor.name.as_str();
    let byte_len = tensor
        .byte_len
        .ok_or_else(|| unsupported_gguf_tensor_type(tensor.tensor_type, tensor_name, mode))?;
    let byte_len: usize = byte_len.try_into().map_err(|_| {
        invalid_gguf_values(
            path,
            Some(tensor_name),
            format!("tensor `{tensor_name}` byte length {byte_len} does not fit in usize"),
        )
    })?;

    file.seek(SeekFrom::Start(tensor.file_offset))
        .map_err(|source| io_error(path, source))?;
    let mut data = vec![0_u8; byte_len];
    file.read_exact(&mut data)
        .map_err(|source| io_error(path, source))?;

    let (dtype, values) = match tensor.tensor_type {
        GgmlTensorType::F32 => (SupportedDtype::F32, decode_f32(&data, path, tensor_name)?),
        GgmlTensorType::F16 => (SupportedDtype::F16, decode_f16(&data, path, tensor_name)?),
        GgmlTensorType::BF16 => (SupportedDtype::BF16, decode_bf16(&data, path, tensor_name)?),
        GgmlTensorType::Q4K | GgmlTensorType::Q5K | GgmlTensorType::Q6K
            if mode == GgufValueMode::DenseOrKQuantized =>
        {
            let element_count = element_count_from_shape(path, tensor)?;
            (
                SupportedDtype::F32,
                dequantize_k_quant_payload_f32(
                    path,
                    tensor_name,
                    tensor.tensor_type,
                    &data,
                    &tensor.shape,
                    element_count,
                )?,
            )
        }
        other => return Err(unsupported_gguf_tensor_type(other, tensor_name, mode)),
    };

    Ok(LoadedTensor {
        name: tensor.name.clone(),
        shape: tensor.shape.clone(),
        dtype,
        values,
    })
}

fn unsupported_gguf_tensor_type(
    tensor_type: GgmlTensorType,
    tensor_name: &str,
    mode: GgufValueMode,
) -> OcelotlError {
    let mut supported = vec!["F32".into(), "F16".into(), "BF16".into()];
    if mode == GgufValueMode::DenseOrKQuantized {
        supported.extend(["Q4K".into(), "Q5K".into(), "Q6K".into()]);
    }
    OcelotlError::from(UnsupportedError {
        feature: "gguf_tensor_type".to_string(),
        requested: Some(format!("{tensor_type:?} (tensor `{tensor_name}`)")),
        supported,
    })
}

fn element_count_from_shape(path: &Path, tensor: &GgufTensorEntry) -> Result<usize> {
    tensor.shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            invalid_gguf_values(
                path,
                Some(&tensor.name),
                format!(
                    "GGUF tensor `{}` element count overflows usize",
                    tensor.name
                ),
            )
        })
    })
}

fn decode_f32(data: &[u8], path: &Path, tensor_name: &str) -> Result<Vec<f32>> {
    let chunks = exact_chunks(data, 4, path, tensor_name)?;
    Ok(chunks
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("chunk size is 4")))
        .collect())
}

fn decode_bf16(data: &[u8], path: &Path, tensor_name: &str) -> Result<Vec<f32>> {
    let chunks = exact_chunks(data, 2, path, tensor_name)?;
    Ok(chunks
        .map(|bytes| {
            let bits = u16::from_le_bytes(bytes.try_into().expect("chunk size is 2"));
            f32::from_bits((bits as u32) << 16)
        })
        .collect())
}

fn decode_f16(data: &[u8], path: &Path, tensor_name: &str) -> Result<Vec<f32>> {
    let chunks = exact_chunks(data, 2, path, tensor_name)?;
    Ok(chunks
        .map(|bytes| {
            let bits = u16::from_le_bytes(bytes.try_into().expect("chunk size is 2"));
            f16_bits_to_f32(bits)
        })
        .collect())
}

fn exact_chunks<'a>(
    data: &'a [u8],
    elem_size: usize,
    path: &Path,
    tensor_name: &str,
) -> Result<std::slice::ChunksExact<'a, u8>> {
    let chunks = data.chunks_exact(elem_size);
    if !chunks.remainder().is_empty() {
        return Err(invalid_gguf_values(
            path,
            Some(tensor_name),
            format!(
                "GGUF tensor `{tensor_name}` has malformed payload: {} data bytes is not divisible by element size {elem_size}",
                data.len()
            ),
        ));
    }
    Ok(chunks)
}

fn dequantize_k_quant_payload_f32(
    path: &Path,
    tensor_name: &str,
    tensor_type: GgmlTensorType,
    data: &[u8],
    shape: &[usize],
    element_count: usize,
) -> Result<Vec<f32>> {
    let layout = tensor_type.quant_layout().ok_or_else(|| {
        unsupported_gguf_tensor_type(tensor_type, tensor_name, GgufValueMode::DenseOrKQuantized)
    })?;
    let block_element_count: usize = layout.block_element_count.try_into().map_err(|_| {
        invalid_gguf_values(
            path,
            Some(tensor_name),
            format!("GGUF tensor `{tensor_name}` quant block element count does not fit usize"),
        )
    })?;
    let block_byte_len: usize = layout.block_byte_len.try_into().map_err(|_| {
        invalid_gguf_values(
            path,
            Some(tensor_name),
            format!("GGUF tensor `{tensor_name}` quant block byte length does not fit usize"),
        )
    })?;
    let row_element_count = shape.first().copied().ok_or_else(|| {
        invalid_gguf_values(
            path,
            Some(tensor_name),
            format!("GGUF tensor `{tensor_name}` has no row dimension"),
        )
    })?;
    if row_element_count % block_element_count != 0 {
        return Err(invalid_gguf_values(
            path,
            Some(tensor_name),
            format!(
                "GGUF tensor `{tensor_name}` row element count {row_element_count} is not divisible by {:?} quant block size {block_element_count}",
                tensor_type
            ),
        ));
    }
    if element_count % block_element_count != 0 {
        return Err(invalid_gguf_values(
            path,
            Some(tensor_name),
            format!(
                "GGUF tensor `{tensor_name}` element count {element_count} is not divisible by {:?} quant block size {block_element_count}",
                tensor_type
            ),
        ));
    }
    let block_count = element_count / block_element_count;
    let expected_len = block_count.checked_mul(block_byte_len).ok_or_else(|| {
        invalid_gguf_values(
            path,
            Some(tensor_name),
            format!("GGUF tensor `{tensor_name}` quantized payload length overflows usize"),
        )
    })?;
    if data.len() != expected_len {
        return Err(invalid_gguf_values(
            path,
            Some(tensor_name),
            format!(
                "GGUF tensor `{tensor_name}` has malformed {:?} payload: {} data bytes, expected {expected_len}",
                tensor_type,
                data.len()
            ),
        ));
    }

    let mut values = Vec::with_capacity(element_count);
    match tensor_type {
        GgmlTensorType::Q4K => dequantize_q4_k_blocks(data, &mut values),
        GgmlTensorType::Q5K => dequantize_q5_k_blocks(data, &mut values),
        GgmlTensorType::Q6K => dequantize_q6_k_blocks(data, &mut values),
        other => {
            return Err(unsupported_gguf_tensor_type(
                other,
                tensor_name,
                GgufValueMode::DenseOrKQuantized,
            ));
        }
    }
    debug_assert_eq!(values.len(), element_count);
    Ok(values)
}

fn dequantize_q4_k_blocks(data: &[u8], values: &mut Vec<f32>) {
    for block in data.chunks_exact(Q4_K_BLOCK_BYTES) {
        let d = f16_le_at(block, 0);
        let dmin = f16_le_at(block, 2);
        let scales = &block[4..16];
        let qs = &block[16..144];

        let mut scale_index = 0usize;
        let mut q_offset = 0usize;
        for _ in 0..4 {
            let q = &qs[q_offset..q_offset + 32];
            let (sc1, m1) = get_scale_min_k4(scale_index, scales);
            let (sc2, m2) = get_scale_min_k4(scale_index + 1, scales);
            let d1 = d * f32::from(sc1);
            let min1 = dmin * f32::from(m1);
            let d2 = d * f32::from(sc2);
            let min2 = dmin * f32::from(m2);

            values.extend(q.iter().map(|byte| d1 * f32::from(byte & 0x0f) - min1));
            values.extend(q.iter().map(|byte| d2 * f32::from(byte >> 4) - min2));

            scale_index += 2;
            q_offset += 32;
        }
    }
}

fn dequantize_q5_k_blocks(data: &[u8], values: &mut Vec<f32>) {
    for block in data.chunks_exact(Q5_K_BLOCK_BYTES) {
        let d = f16_le_at(block, 0);
        let dmin = f16_le_at(block, 2);
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];

        let mut scale_index = 0usize;
        let mut q_offset = 0usize;
        let mut high_bit_low_nibble = 1u8;
        let mut high_bit_high_nibble = 2u8;
        for _ in 0..4 {
            let q = &qs[q_offset..q_offset + 32];
            let (sc1, m1) = get_scale_min_k4(scale_index, scales);
            let (sc2, m2) = get_scale_min_k4(scale_index + 1, scales);
            let d1 = d * f32::from(sc1);
            let min1 = dmin * f32::from(m1);
            let d2 = d * f32::from(sc2);
            let min2 = dmin * f32::from(m2);

            values.extend(q.iter().zip(qh.iter()).map(|(byte, high)| {
                let quant = (byte & 0x0f)
                    + if (high & high_bit_low_nibble) != 0 {
                        16
                    } else {
                        0
                    };
                d1 * f32::from(quant) - min1
            }));
            values.extend(q.iter().zip(qh.iter()).map(|(byte, high)| {
                let quant = (byte >> 4)
                    + if (high & high_bit_high_nibble) != 0 {
                        16
                    } else {
                        0
                    };
                d2 * f32::from(quant) - min2
            }));

            scale_index += 2;
            q_offset += 32;
            high_bit_low_nibble <<= 2;
            high_bit_high_nibble <<= 2;
        }
    }
}

fn dequantize_q6_k_blocks(data: &[u8], values: &mut Vec<f32>) {
    for block in data.chunks_exact(Q6_K_BLOCK_BYTES) {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16_le_at(block, 208);

        for super_group in 0..2 {
            let ql_base = super_group * 64;
            let qh_base = super_group * 32;
            let scale_base = super_group * 8;
            let value_base = values.len();
            values.resize(value_base + 128, 0.0);

            for l in 0..32 {
                let scale_index = l / 16;
                let high = qh[qh_base + l];
                let q1 = i32::from((ql[ql_base + l] & 0x0f) | ((high & 0x03) << 4)) - 32;
                let q2 =
                    i32::from((ql[ql_base + l + 32] & 0x0f) | (((high >> 2) & 0x03) << 4)) - 32;
                let q3 = i32::from((ql[ql_base + l] >> 4) | (((high >> 4) & 0x03) << 4)) - 32;
                let q4 = i32::from((ql[ql_base + l + 32] >> 4) | (((high >> 6) & 0x03) << 4)) - 32;

                let sc1 = i8::from_ne_bytes([scales[scale_base + scale_index]]);
                let sc2 = i8::from_ne_bytes([scales[scale_base + scale_index + 2]]);
                let sc3 = i8::from_ne_bytes([scales[scale_base + scale_index + 4]]);
                let sc4 = i8::from_ne_bytes([scales[scale_base + scale_index + 6]]);

                values[value_base + l] = d * f32::from(sc1) * q1 as f32;
                values[value_base + l + 32] = d * f32::from(sc2) * q2 as f32;
                values[value_base + l + 64] = d * f32::from(sc3) * q3 as f32;
                values[value_base + l + 96] = d * f32::from(sc4) * q4 as f32;
            }
        }
    }
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
                let mut exp_unbiased = -14i32;
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
            let exp32 = (exp as u32) + (127 - 15);
            sign | (exp32 << 23) | (frac << 13)
        }
    };
    f32::from_bits(f32_bits)
}

fn io_error(path: &Path, source: std::io::Error) -> OcelotlError {
    OcelotlError::from(IoError {
        path: Some(path.to_path_buf()),
        source,
    })
}

fn invalid_gguf_values(path: &Path, field: Option<&str>, message: String) -> OcelotlError {
    OcelotlError::from(InvalidModelError {
        path: Some(path.to_path_buf()),
        field: field.map(str::to_string),
        message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SupportedDtype;
    use std::path::{Path, PathBuf};

    const GGUF_MAGIC: &[u8; 4] = b"GGUF";
    const GGUF_VERSION: u32 = 3;
    const ALIGNMENT: usize = 32;
    const QK_K: usize = 256;

    struct FixtureTensor<'a> {
        name: &'a str,
        shape: &'a [u64],
        raw_type: u32,
        payload: Vec<u8>,
    }

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ocelotl_gguf_values_{}_{}.gguf",
            std::process::id(),
            name
        ));
        p
    }

    fn write_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn write_u64(out: &mut Vec<u8>, value: u64) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn write_string(out: &mut Vec<u8>, value: &str) {
        write_u64(out, value.len() as u64);
        out.extend_from_slice(value.as_bytes());
    }

    fn write_string_metadata(out: &mut Vec<u8>, key: &str, value: &str) {
        write_string(out, key);
        write_u32(out, 8);
        write_string(out, value);
    }

    fn write_u32_metadata(out: &mut Vec<u8>, key: &str, value: u32) {
        write_string(out, key);
        write_u32(out, 4);
        write_u32(out, value);
    }

    fn align_len(len: usize) -> usize {
        len.next_multiple_of(ALIGNMENT)
    }

    fn write_fixture(path: &Path, tensors: &[FixtureTensor<'_>]) {
        let mut payloads = Vec::new();
        let mut next_offset = 0usize;
        for tensor in tensors {
            next_offset = align_len(next_offset);
            payloads.push((next_offset, tensor.payload.as_slice()));
            next_offset += tensor.payload.len();
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        write_u32(&mut bytes, GGUF_VERSION);
        write_u64(&mut bytes, tensors.len() as u64);
        write_u64(&mut bytes, 3);
        write_string_metadata(&mut bytes, "general.architecture", "gemma4");
        write_string_metadata(&mut bytes, "general.name", "tiny gguf values");
        write_u32_metadata(&mut bytes, "general.alignment", ALIGNMENT as u32);

        for (tensor, (offset, _payload)) in tensors.iter().zip(payloads.iter()) {
            write_string(&mut bytes, tensor.name);
            write_u32(&mut bytes, tensor.shape.len() as u32);
            for dim in tensor.shape {
                write_u64(&mut bytes, *dim);
            }
            write_u32(&mut bytes, tensor.raw_type);
            write_u64(&mut bytes, *offset as u64);
        }

        while bytes.len() % ALIGNMENT != 0 {
            bytes.push(0);
        }
        let data_start = bytes.len();
        bytes.resize(data_start + next_offset, 0);
        for (offset, payload) in payloads {
            let start = data_start + offset;
            bytes[start..start + payload.len()].copy_from_slice(payload);
        }

        std::fs::write(path, bytes).expect("write GGUF value fixture");
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn u16_bytes(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn one_f16() -> [u8; 2] {
        0x3c00_u16.to_le_bytes()
    }

    fn half_f16() -> [u8; 2] {
        0x3800_u16.to_le_bytes()
    }

    fn q4_k_block() -> Vec<u8> {
        let mut block = vec![0; Q4_K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&one_f16());
        block[2..4].copy_from_slice(&half_f16());
        block[4] = 2;
        block[5] = 3;
        block[8] = 2;
        block[9] = 4;
        block[16] = 0x21;
        block
    }

    fn expected_q4_k_values() -> Vec<f32> {
        let mut expected = vec![0.0; QK_K];
        expected[0..32].fill(-1.0);
        expected[32..64].fill(-2.0);
        expected[0] = 1.0;
        expected[32] = 4.0;
        expected
    }

    fn q4_k_all_groups_block() -> Vec<u8> {
        let mut block = vec![0; Q4_K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&one_f16());
        block[2..4].copy_from_slice(&half_f16());
        block[4..16].copy_from_slice(&[
            0x41, 0x42, 0x83, 0xc4, 0x45, 0x46, 0x87, 0xc8, 0x31, 0x42, 0x21, 0xef,
        ]);
        block[16] = 0x21;
        block[48] = 0x43;
        block[80] = 0x65;
        block[112] = 0xfe;
        block
    }

    fn q5_k_block() -> Vec<u8> {
        let mut block = vec![0; Q5_K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&one_f16());
        block[2..4].copy_from_slice(&half_f16());
        block[4] = 1;
        block[5] = 2;
        block[8] = 2;
        block[9] = 4;
        block[16] = 0x03;
        block[48] = 0x21;
        block
    }

    fn expected_q5_k_values() -> Vec<f32> {
        let mut expected = vec![0.0; QK_K];
        expected[0..32].fill(-1.0);
        expected[32..64].fill(-2.0);
        expected[0] = 16.0;
        expected[32] = 34.0;
        expected
    }

    fn q6_k_block() -> Vec<u8> {
        let mut block = vec![0; Q6_K_BLOCK_BYTES];
        block[0] = 0x21;
        block[32] = 0x43;
        block[192] = 1;
        block[194] = 2;
        block[196] = 3;
        block[198] = 4;
        block[208..210].copy_from_slice(&one_f16());
        block
    }

    fn expected_q6_k_values() -> Vec<f32> {
        let mut expected = vec![0.0; QK_K];
        expected[0..16].fill(-32.0);
        expected[32..48].fill(-64.0);
        expected[64..80].fill(-96.0);
        expected[96..112].fill(-128.0);
        expected[0] = -31.0;
        expected[32] = -58.0;
        expected[64] = -90.0;
        expected[96] = -112.0;
        expected
    }

    #[test]
    fn load_gguf_tensor_f32_loads_f32_values_and_metadata() {
        let path = tmp_path("f32_values");
        write_fixture(
            &path,
            &[FixtureTensor {
                name: "dense.weight",
                shape: &[2, 2],
                raw_type: 0,
                payload: f32_bytes(&[1.0, -2.5, 0.0, 3.25]),
            }],
        );

        let loaded = load_gguf_tensor_f32(&path, "dense.weight").expect("load F32 tensor");

        assert_eq!(loaded.name, "dense.weight");
        assert_eq!(loaded.shape, vec![2, 2]);
        assert_eq!(loaded.dtype, SupportedDtype::F32);
        assert_eq!(loaded.values, vec![1.0, -2.5, 0.0, 3.25]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_gguf_tensor_f32_converts_bf16_values() {
        let path = tmp_path("bf16_values");
        write_fixture(
            &path,
            &[FixtureTensor {
                name: "dense.weight",
                shape: &[4],
                raw_type: 30,
                payload: u16_bytes(&[0x3f80, 0xc020, 0x0000, 0x7f80]),
            }],
        );

        let loaded = load_gguf_tensor_f32(&path, "dense.weight").expect("load BF16 tensor");

        assert_eq!(loaded.dtype, SupportedDtype::BF16);
        assert_eq!(loaded.values, vec![1.0, -2.5, 0.0, f32::INFINITY]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_gguf_tensor_f32_converts_f16_values() {
        let path = tmp_path("f16_values");
        write_fixture(
            &path,
            &[FixtureTensor {
                name: "dense.weight",
                shape: &[6],
                raw_type: 1,
                payload: u16_bytes(&[0x3c00, 0xc100, 0x0000, 0x0400, 0x0001, 0x7c00]),
            }],
        );

        let loaded = load_gguf_tensor_f32(&path, "dense.weight").expect("load F16 tensor");

        assert_eq!(loaded.dtype, SupportedDtype::F16);
        assert_eq!(loaded.values[0], 1.0);
        assert_eq!(loaded.values[1], -2.5);
        assert_eq!(loaded.values[2], 0.0);
        assert_eq!(loaded.values[3], f32::from_bits(0x3880_0000));
        assert_eq!(loaded.values[4], f32::from_bits(0x3380_0000));
        assert_eq!(loaded.values[5], f32::INFINITY);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_gguf_tensors_f32_loads_many_tensors_in_requested_order() {
        let path = tmp_path("many_values");
        write_fixture(
            &path,
            &[
                FixtureTensor {
                    name: "first",
                    shape: &[2],
                    raw_type: 0,
                    payload: f32_bytes(&[1.0, 2.0]),
                },
                FixtureTensor {
                    name: "second",
                    shape: &[1],
                    raw_type: 0,
                    payload: f32_bytes(&[-3.5]),
                },
            ],
        );

        let names = vec!["second".to_string(), "first".to_string()];
        let loaded = load_gguf_tensors_f32(&path, &names).expect("load many GGUF tensors");

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "second");
        assert_eq!(loaded[0].values, vec![-3.5]);
        assert_eq!(loaded[1].name, "first");
        assert_eq!(loaded[1].values, vec![1.0, 2.0]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_gguf_tensor_f32_returns_invalid_model_for_missing_tensor() {
        let path = tmp_path("missing_tensor");
        write_fixture(
            &path,
            &[FixtureTensor {
                name: "present",
                shape: &[1],
                raw_type: 0,
                payload: f32_bytes(&[1.0]),
            }],
        );

        let err = load_gguf_tensor_f32(&path, "absent").expect_err("missing tensor must fail");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.path.as_deref(), Some(path.as_path()));
                assert_eq!(invalid.field.as_deref(), Some("absent"));
                assert!(invalid.message.contains("absent"));
            }
            other => panic!("expected InvalidModel for missing tensor, got {other:?}"),
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_gguf_tensor_f32_returns_unsupported_for_quantized_tensor() {
        let path = tmp_path("q4k_unsupported");
        write_fixture(
            &path,
            &[FixtureTensor {
                name: "quant.weight",
                shape: &[256],
                raw_type: 12,
                payload: vec![0; 144],
            }],
        );

        let err = load_gguf_tensor_f32(&path, "quant.weight").expect_err("Q4K must be unsupported");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(unsupported.feature, "gguf_tensor_type");
                assert!(unsupported.requested.as_deref().is_some_and(|requested| {
                    requested.contains("Q4K") && requested.contains("quant.weight")
                }));
            }
            other => panic!("expected Unsupported for Q4K, got {other:?}"),
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_gguf_tensor_dequantized_f32_rejects_k_quant_row_not_block_multiple() {
        let path = tmp_path("q4k_bad_row");
        write_fixture(
            &path,
            &[FixtureTensor {
                name: "quant.weight",
                shape: &[128, 2],
                raw_type: 12,
                payload: q4_k_block(),
            }],
        );

        let err = load_gguf_tensor_dequantized_f32(&path, "quant.weight")
            .expect_err("K-quant row width must be block-aligned");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("quant.weight"));
                assert!(invalid.message.contains("row element count 128"));
                assert!(invalid.message.contains("256"));
            }
            other => panic!("expected InvalidModel for bad K-quant row width, got {other:?}"),
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_gguf_tensor_dequantized_f32_loads_q4_k_all_scale_min_groups() {
        let path = tmp_path("q4k_all_groups");
        write_fixture(
            &path,
            &[FixtureTensor {
                name: "quant.weight",
                shape: &[256],
                raw_type: 12,
                payload: q4_k_all_groups_block(),
            }],
        );

        let loaded = load_gguf_tensor_dequantized_f32(&path, "quant.weight")
            .expect("load dequantized Q4_K tensor");

        assert_eq!(loaded.values[0], -1.5);
        assert_eq!(loaded.values[32], 1.0);
        assert_eq!(loaded.values[64], 5.5);
        assert_eq!(loaded.values[96], 12.0);
        assert_eq!(loaded.values[128], 75.5);
        assert_eq!(loaded.values[160], 98.0);
        assert_eq!(loaded.values[192], 445.0);
        assert_eq!(loaded.values[224], 914.0);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_gguf_tensor_dequantized_f32_loads_q4_k_values() {
        let path = tmp_path("q4k_dequant");
        write_fixture(
            &path,
            &[FixtureTensor {
                name: "quant.weight",
                shape: &[256],
                raw_type: 12,
                payload: q4_k_block(),
            }],
        );

        let loaded = load_gguf_tensor_dequantized_f32(&path, "quant.weight")
            .expect("load dequantized Q4_K tensor");

        assert_eq!(loaded.name, "quant.weight");
        assert_eq!(loaded.shape, vec![256]);
        assert_eq!(loaded.dtype, SupportedDtype::F32);
        assert_eq!(loaded.values, expected_q4_k_values());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_gguf_tensor_dequantized_f32_loads_q5_k_values() {
        let path = tmp_path("q5k_dequant");
        write_fixture(
            &path,
            &[FixtureTensor {
                name: "quant.weight",
                shape: &[256],
                raw_type: 13,
                payload: q5_k_block(),
            }],
        );

        let loaded = load_gguf_tensor_dequantized_f32(&path, "quant.weight")
            .expect("load dequantized Q5_K tensor");

        assert_eq!(loaded.shape, vec![256]);
        assert_eq!(loaded.dtype, SupportedDtype::F32);
        assert_eq!(loaded.values, expected_q5_k_values());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_gguf_tensor_dequantized_f32_loads_q6_k_values() {
        let path = tmp_path("q6k_dequant");
        write_fixture(
            &path,
            &[FixtureTensor {
                name: "quant.weight",
                shape: &[256],
                raw_type: 14,
                payload: q6_k_block(),
            }],
        );

        let loaded = load_gguf_tensor_dequantized_f32(&path, "quant.weight")
            .expect("load dequantized Q6_K tensor");

        assert_eq!(loaded.shape, vec![256]);
        assert_eq!(loaded.dtype, SupportedDtype::F32);
        assert_eq!(loaded.values, expected_q6_k_values());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_gguf_tensors_dequantized_f32_loads_dense_and_quantized_values() {
        let path = tmp_path("many_dequant");
        write_fixture(
            &path,
            &[
                FixtureTensor {
                    name: "dense",
                    shape: &[1],
                    raw_type: 0,
                    payload: f32_bytes(&[3.0]),
                },
                FixtureTensor {
                    name: "quant",
                    shape: &[256],
                    raw_type: 12,
                    payload: q4_k_block(),
                },
            ],
        );

        let names = vec!["quant", "dense"];
        let loaded =
            load_gguf_tensors_dequantized_f32(&path, &names).expect("load mixed GGUF tensors");

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "quant");
        assert_eq!(loaded[0].values, expected_q4_k_values());
        assert_eq!(loaded[1].name, "dense");
        assert_eq!(loaded[1].dtype, SupportedDtype::F32);
        assert_eq!(loaded[1].values, vec![3.0]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_gguf_tensor_f32_returns_io_for_missing_file() {
        let path = tmp_path("missing_file");

        let err = load_gguf_tensor_f32(&path, "weight").expect_err("missing file fails");

        match err {
            OcelotlError::Io(io) => {
                assert_eq!(io.path.as_deref(), Some(path.as_path()));
                assert_eq!(io.source.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected Io for missing file, got {other:?}"),
        }
    }
}
