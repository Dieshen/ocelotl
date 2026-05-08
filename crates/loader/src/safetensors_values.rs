//! Value loading for individual safetensors tensors.
//!
//! This module is the value-loading counterpart to `safetensors_inspect`: it
//! reads one named tensor and converts the supported artifact dtypes into an
//! Ocelotl-owned `Vec<f32>` without exposing safetensors crate types.

use std::{fs, path::Path};

use ocelotl_core::{InvalidModelError, IoError, OcelotlError, Result, UnsupportedError};

use crate::SupportedDtype;

#[derive(Debug, Clone, PartialEq)]
pub struct LoadedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: SupportedDtype,
    pub values: Vec<f32>,
}

pub fn load_safetensors_tensor_f32(path: &Path, tensor_name: &str) -> Result<LoadedTensor> {
    let bytes = fs::read(path).map_err(|source| {
        OcelotlError::from(IoError {
            path: Some(path.to_path_buf()),
            source,
        })
    })?;

    let tensors = safetensors::SafeTensors::deserialize(&bytes).map_err(|source| {
        invalid_safetensors(
            path,
            None,
            format!("failed to parse safetensors file: {source}"),
        )
    })?;
    let tensor = tensors.tensor(tensor_name).map_err(|source| {
        OcelotlError::from(InvalidModelError {
            path: Some(path.to_path_buf()),
            field: Some(tensor_name.to_string()),
            message: format!(
                "required tensor `{tensor_name}` not found in safetensors file: {source}"
            ),
        })
    })?;

    let dtype = map_dtype(tensor.dtype(), tensor_name)?;
    let data = tensor.data();
    let values = match dtype {
        SupportedDtype::F32 => decode_f32(data, path, tensor_name)?,
        SupportedDtype::BF16 => decode_bf16(data, path, tensor_name)?,
        SupportedDtype::F16 => decode_f16(data, path, tensor_name)?,
    };

    Ok(LoadedTensor {
        name: tensor_name.to_string(),
        shape: tensor.shape().to_vec(),
        dtype,
        values,
    })
}

fn map_dtype(raw: safetensors::Dtype, tensor_name: &str) -> Result<SupportedDtype> {
    Ok(match raw {
        safetensors::Dtype::F32 => SupportedDtype::F32,
        safetensors::Dtype::F16 => SupportedDtype::F16,
        safetensors::Dtype::BF16 => SupportedDtype::BF16,
        other => {
            return Err(OcelotlError::from(UnsupportedError {
                feature: "safetensors_dtype".to_string(),
                requested: Some(format!("{other} (tensor `{tensor_name}`)")),
                supported: vec!["F32".into(), "F16".into(), "BF16".into()],
            }));
        }
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
        return Err(invalid_safetensors(
            path,
            Some(tensor_name),
            format!(
                "safetensors tensor `{tensor_name}` has malformed payload: {} data bytes is not divisible by element size {elem_size}",
                data.len()
            ),
        ));
    }
    Ok(chunks)
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

fn invalid_safetensors(path: &Path, field: Option<&str>, message: String) -> OcelotlError {
    OcelotlError::from(InvalidModelError {
        path: Some(path.to_path_buf()),
        field: field.map(str::to_string),
        message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::BTreeMap, io::Write, path::PathBuf};

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ocelotl_w_asr_7_{}_{}.safetensors",
            std::process::id(),
            name
        ));
        p
    }

    fn build_value_fixture(path: &Path, tensors: &[(&str, &str, &[usize], Vec<u8>)]) {
        let mut header_map: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        let mut data = Vec::new();

        for (name, dtype, shape, payload) in tensors {
            let begin = data.len();
            data.extend(payload);
            let end = data.len();
            header_map.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [begin, end],
                }),
            );
        }

        let header_json = serde_json::to_string(&header_map).expect("serialize header");
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut file = fs::File::create(path).expect("create fixture file");
        file.write_all(&header_len.to_le_bytes())
            .expect("write header length");
        file.write_all(header_bytes).expect("write header");
        file.write_all(&data).expect("write data section");
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

    #[test]
    fn load_safetensors_tensor_f32_loads_f32_values_and_metadata() {
        let path = tmp_path("f32_values");
        build_value_fixture(
            &path,
            &[("weight", "F32", &[2, 2], f32_bytes(&[1.0, -2.5, 0.0, 3.25]))],
        );

        let loaded = load_safetensors_tensor_f32(&path, "weight").expect("load F32 tensor");

        assert_eq!(loaded.name, "weight");
        assert_eq!(loaded.shape, vec![2, 2]);
        assert_eq!(loaded.dtype, SupportedDtype::F32);
        assert_eq!(loaded.values, vec![1.0, -2.5, 0.0, 3.25]);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_safetensors_tensor_f32_converts_bf16_values() {
        let path = tmp_path("bf16_values");
        build_value_fixture(
            &path,
            &[(
                "weight",
                "BF16",
                &[4],
                u16_bytes(&[0x3f80, 0xc020, 0x0000, 0x7f80]),
            )],
        );

        let loaded = load_safetensors_tensor_f32(&path, "weight").expect("load BF16 tensor");

        assert_eq!(loaded.dtype, SupportedDtype::BF16);
        assert_eq!(loaded.values, vec![1.0, -2.5, 0.0, f32::INFINITY]);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_safetensors_tensor_f32_converts_f16_values() {
        let path = tmp_path("f16_values");
        build_value_fixture(
            &path,
            &[(
                "weight",
                "F16",
                &[6],
                u16_bytes(&[0x3c00, 0xc100, 0x0000, 0x0400, 0x0001, 0x7c00]),
            )],
        );

        let loaded = load_safetensors_tensor_f32(&path, "weight").expect("load F16 tensor");

        assert_eq!(loaded.dtype, SupportedDtype::F16);
        assert_eq!(loaded.values[0], 1.0);
        assert_eq!(loaded.values[1], -2.5);
        assert_eq!(loaded.values[2], 0.0);
        assert_eq!(loaded.values[3], f32::from_bits(0x3880_0000));
        assert_eq!(loaded.values[4], f32::from_bits(0x3380_0000));
        assert_eq!(loaded.values[5], f32::INFINITY);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_safetensors_tensor_f32_preserves_f16_nan() {
        let path = tmp_path("f16_nan");
        build_value_fixture(&path, &[("weight", "F16", &[1], u16_bytes(&[0x7e00]))]);

        let loaded = load_safetensors_tensor_f32(&path, "weight").expect("load F16 tensor");

        assert!(loaded.values[0].is_nan());

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_safetensors_tensor_f32_returns_io_for_missing_file() {
        let path = tmp_path("missing_file");

        let err = load_safetensors_tensor_f32(&path, "weight").expect_err("missing file fails");

        match err {
            OcelotlError::Io(io) => {
                assert_eq!(io.path.as_deref(), Some(path.as_path()));
                assert_eq!(io.source.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected Io for missing file, got {other:?}"),
        }
    }

    #[test]
    fn load_safetensors_tensor_f32_returns_invalid_model_for_missing_tensor() {
        let path = tmp_path("missing_tensor");
        build_value_fixture(&path, &[("present", "F32", &[1], f32_bytes(&[1.0]))]);

        let err =
            load_safetensors_tensor_f32(&path, "absent").expect_err("missing tensor must fail");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.path.as_deref(), Some(path.as_path()));
                assert_eq!(invalid.field.as_deref(), Some("absent"));
                assert!(invalid.message.contains("absent"));
            }
            other => panic!("expected InvalidModel for missing tensor, got {other:?}"),
        }

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_safetensors_tensor_f32_returns_unsupported_for_unsupported_dtype() {
        let path = tmp_path("unsupported_dtype");
        build_value_fixture(
            &path,
            &[("weight", "F64", &[1], 1.0f64.to_le_bytes().to_vec())],
        );

        let err =
            load_safetensors_tensor_f32(&path, "weight").expect_err("F64 must be unsupported");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(unsupported.feature, "safetensors_dtype");
                assert!(unsupported.requested.as_deref().is_some_and(|requested| {
                    requested.contains("F64") && requested.contains("weight")
                }));
            }
            other => panic!("expected Unsupported for F64, got {other:?}"),
        }

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_safetensors_tensor_f32_returns_invalid_model_for_malformed_payload() {
        let path = tmp_path("malformed_payload");
        build_value_fixture(&path, &[("weight", "F32", &[1], vec![0, 0, 0])]);

        let err =
            load_safetensors_tensor_f32(&path, "weight").expect_err("malformed payload fails");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.path.as_deref(), Some(path.as_path()));
                assert!(invalid.message.contains("safetensors"));
            }
            other => panic!("expected InvalidModel for malformed payload, got {other:?}"),
        }

        let _ = fs::remove_file(&path);
    }
}
