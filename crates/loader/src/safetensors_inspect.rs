//! Header-only inspection of safetensors artifacts.
//!
//! The loader reads tensor names, shapes, dtypes, and byte ranges from a
//! safetensors file *without* loading or executing weights. M2.6 (paired,
//! cross-crate) maps the resulting `SafetensorsManifest` into
//! `ocelotl-core::ModelMetadata`; this module deliberately stops at the
//! artifact-shape boundary and does not import model-family knowledge.

use std::path::Path;

use ocelotl_core::{InvalidModelError, OcelotlError, Result, UnsupportedError};

/// Dtypes the loader currently accepts for inspected tensors. Anything else in
/// a header is rejected with `OcelotlError::Unsupported` so the caller can
/// surface "this artifact uses dtype X, we support only Y" without leaking
/// `safetensors::Dtype` through the public API.
///
/// Kept intentionally narrow for M2.5 — quantized dtypes are explicitly
/// out-of-scope per `docs/milestones/m2-loader-tokenizer.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportedDtype {
    F32,
    F16,
    BF16,
}

/// One tensor's header-level identity. The byte range is relative to the
/// start of the safetensors **data section** (i.e. the byte buffer that
/// follows the JSON header), matching the `data_offsets` semantics in the
/// safetensors spec. M2.6 will use these to map artifact tensors to model
/// weights without ever needing to re-parse the header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: SupportedDtype,
    pub byte_range: (usize, usize),
}

/// Header-derived view of a safetensors artifact: the tensors it declares and
/// the total byte length of its data section. No tensor bytes are read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsManifest {
    pub tensors: Vec<TensorEntry>,
    pub data_len: usize,
}

/// Inspect a safetensors file and return its tensor manifest. Does **not**
/// load or interpret tensor weights — only the header is parsed.
pub fn inspect_safetensors(path: &Path) -> Result<SafetensorsManifest> {
    let bytes = std::fs::read(path).map_err(|source| {
        OcelotlError::from(InvalidModelError {
            path: Some(path.to_path_buf()),
            field: None,
            message: format!("failed to read safetensors file: {source}"),
        })
    })?;

    let (_header_size, metadata) =
        safetensors::SafeTensors::read_metadata(&bytes).map_err(|e| {
            OcelotlError::from(InvalidModelError {
                path: Some(path.to_path_buf()),
                field: None,
                message: format!("failed to parse safetensors header: {e}"),
            })
        })?;

    // `metadata.tensors()` returns a HashMap; iterate `offset_keys()` so the
    // order is deterministic by data layout, which is what most callers want
    // when they are about to allocate weight buffers in declared order.
    let mut tensors = Vec::with_capacity(metadata.offset_keys().len());
    for name in metadata.offset_keys() {
        let info = metadata.info(&name).ok_or_else(|| {
            // Should be unreachable: offset_keys() and info() are populated
            // from the same map. Surface as InvalidModel rather than panic so
            // an upstream safetensors bug doesn't crash the loader.
            OcelotlError::from(InvalidModelError {
                path: Some(path.to_path_buf()),
                field: Some(name.clone()),
                message: format!(
                    "safetensors header listed tensor `{name}` in offset_keys but had no info"
                ),
            })
        })?;
        let dtype = map_dtype(info.dtype, &name, path)?;
        tensors.push(TensorEntry {
            name,
            shape: info.shape.clone(),
            dtype,
            byte_range: info.data_offsets,
        });
    }

    Ok(SafetensorsManifest {
        tensors,
        data_len: metadata.data_len(),
    })
}

/// Map a safetensors dtype to the supported subset, returning a typed
/// `Unsupported` error otherwise. The error reports the *requested* dtype as
/// the safetensors crate's `Display` form (e.g. `"F32"`, `"Q4_0"`-style names
/// in future versions) so logs are human-readable without needing to import
/// `safetensors::Dtype` to interpret them.
fn map_dtype(raw: safetensors::Dtype, tensor_name: &str, _path: &Path) -> Result<SupportedDtype> {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Build a minimal safetensors file at `path` with the given tensor specs.
    /// The data section is filled with zero bytes of the right length — we
    /// only test header-level inspection, so tensor *values* are irrelevant
    /// and writing zeros keeps the fixture deterministic.
    ///
    /// `tensors` is `(name, dtype_str, shape)`. Tensors are laid out in the
    /// data section in the order given.
    fn build_fixture(path: &Path, tensors: &[(&str, &str, &[usize])]) {
        use std::collections::BTreeMap;
        use std::io::Write;

        // Compute byte offsets per tensor.
        let mut header_map: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        let mut cursor: usize = 0;
        for (name, dtype, shape) in tensors {
            let elem_size: usize = match *dtype {
                "F32" => 4,
                "F16" | "BF16" => 2,
                "I8" | "U8" | "BOOL" => 1,
                "F64" | "I64" | "U64" => 8,
                other => panic!("unsupported dtype in test fixture: {other}"),
            };
            let n_elems: usize = if shape.is_empty() {
                1
            } else {
                shape.iter().product()
            };
            let n_bytes = n_elems * elem_size;
            let begin = cursor;
            let end = cursor + n_bytes;
            cursor = end;
            header_map.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [begin, end],
                }),
            );
        }
        let total_data_bytes = cursor;

        let header_json = serde_json::to_string(&header_map).expect("serialize header");
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut file = std::fs::File::create(path).expect("create fixture file");
        file.write_all(&header_len.to_le_bytes())
            .expect("write header length");
        file.write_all(header_bytes).expect("write header");
        let zeros = vec![0u8; total_data_bytes];
        file.write_all(&zeros).expect("write data section");
    }

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ocelotl_m2_5_{}_{}.safetensors",
            std::process::id(),
            name
        ));
        p
    }

    #[test]
    fn inspect_safetensors_returns_tensor_metadata_for_known_good_fixture() {
        let path = tmp_path("happy");
        // build_fixture lays tensors out in the data section in the order
        // given here: token_embedding at offset 0, then lm_head. The
        // safetensors parser sorts by data_offsets on read, so the
        // returned order matches our input order.
        build_fixture(
            &path,
            &[
                ("token_embedding", "F32", &[16, 4]),
                ("lm_head", "F16", &[4, 16]),
            ],
        );

        let manifest = inspect_safetensors(&path).expect("known-good fixture must inspect");

        assert_eq!(manifest.tensors.len(), 2);

        let tok = &manifest.tensors[0];
        assert_eq!(tok.name, "token_embedding");
        assert_eq!(tok.shape, vec![16, 4]);
        assert_eq!(tok.dtype, SupportedDtype::F32);
        // F32, 16 * 4 = 64 elems * 4 bytes = 256 bytes, starting at 0.
        assert_eq!(tok.byte_range, (0, 256));

        let lm_head = &manifest.tensors[1];
        assert_eq!(lm_head.name, "lm_head");
        assert_eq!(lm_head.shape, vec![4, 16]);
        assert_eq!(lm_head.dtype, SupportedDtype::F16);
        // F16, 4 * 16 = 64 elems * 2 bytes = 128 bytes, after token_embedding's 256.
        assert_eq!(lm_head.byte_range, (256, 384));

        assert_eq!(manifest.data_len, 384);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn inspect_safetensors_rejects_truncated_header_with_invalid_model_error() {
        // Write a file that claims a 1024-byte header but contains only 4
        // bytes after the length prefix. The safetensors crate must reject
        // this; we expect that rejection to surface as a typed
        // OcelotlError::InvalidModel carrying the fixture path so callers
        // can report which artifact failed.
        use std::io::Write;
        let path = tmp_path("truncated_header");
        let mut file = std::fs::File::create(&path).expect("create truncated fixture");
        let claimed_header_len: u64 = 1024;
        file.write_all(&claimed_header_len.to_le_bytes())
            .expect("write claimed header length");
        file.write_all(b"oops").expect("write truncated body");
        drop(file);

        let err =
            inspect_safetensors(&path).expect_err("truncated safetensors header must be rejected");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(
                    invalid.path.as_deref(),
                    Some(path.as_path()),
                    "expected fixture path on InvalidModel, got {:?}",
                    invalid.path,
                );
                assert!(
                    invalid.message.contains("safetensors header"),
                    "expected message to mention safetensors header parsing, got {:?}",
                    invalid.message,
                );
            }
            other => {
                panic!("expected OcelotlError::InvalidModel for truncated header, got {other:?}")
            }
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn inspect_safetensors_rejects_unsupported_dtype_with_typed_unsupported_error() {
        // F64 is a valid safetensors dtype (so the header parses cleanly) but
        // outside Ocelotl's supported subset for M2.5. The contract: this is
        // an Unsupported error (not InvalidModel) carrying the requested
        // dtype name and the supported list, so callers can render
        // "we support [F32, F16, BF16]; got F64".
        let path = tmp_path("unsupported_dtype");
        build_fixture(&path, &[("attention_bias", "F64", &[2, 4])]);

        let err =
            inspect_safetensors(&path).expect_err("F64 dtype must be rejected with Unsupported");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(
                    unsupported.feature, "safetensors_dtype",
                    "expected feature == \"safetensors_dtype\", got {:?}",
                    unsupported.feature,
                );
                let requested = unsupported
                    .requested
                    .as_deref()
                    .expect("Unsupported error must carry the requested dtype");
                assert!(
                    requested.contains("F64"),
                    "expected requested to mention F64, got {requested:?}",
                );
                assert!(
                    requested.contains("attention_bias"),
                    "expected requested to mention the offending tensor name, got {requested:?}",
                );
                assert!(
                    unsupported.supported.iter().any(|s| s == "F32"),
                    "expected F32 in supported list, got {:?}",
                    unsupported.supported,
                );
            }
            other => {
                panic!("expected OcelotlError::Unsupported for F64 dtype, got {other:?}")
            }
        }

        let _ = std::fs::remove_file(&path);
    }
}
