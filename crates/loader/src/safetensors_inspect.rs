//! Header-only inspection of safetensors artifacts.
//!
//! The loader reads tensor names, shapes, dtypes, and byte ranges from a
//! safetensors file *without* loading or executing weights. M2.6 (paired,
//! cross-crate) maps the resulting `SafetensorsManifest` into
//! `ocelotl-core::ModelMetadata`; this module deliberately stops at the
//! artifact-shape boundary and does not import model-family knowledge.

use std::{
    fs::File,
    io::{ErrorKind, Read},
    path::Path,
};

use ocelotl_core::{InvalidModelError, IoError, OcelotlError, Result, UnsupportedError};

const SAFETENSORS_HEADER_LEN_BYTES: usize = std::mem::size_of::<u64>();
// Mirrors safetensors 0.7.0's header cap. Keep this local so header-only
// inspection does not need to allocate a whole-file buffer just to reuse the
// upstream parser's length gate.
const SAFETENSORS_MAX_HEADER_SIZE: usize = 100_000_000;

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
    // Per docs/design/errors.md and .local/workflow/crossing-crate-boundaries.md,
    // file-read failures map to `Io`, not `InvalidModel`. The artifact only
    // becomes "invalid" once we can read it and find malformed contents — a
    // missing or unreadable file is a storage problem, not a malformed
    // artifact. Decided in M2.6 (paired); previous M2.5 mapping to
    // InvalidModel was wrong.
    let mut file = File::open(path).map_err(|source| io_error(path, source))?;
    let file_len = file
        .metadata()
        .map_err(|source| io_error(path, source))?
        .len();

    let mut header_len_bytes = [0u8; SAFETENSORS_HEADER_LEN_BYTES];
    read_exact_header_part(&mut file, path, &mut header_len_bytes)?;
    let header_len_u64 = u64::from_le_bytes(header_len_bytes);
    let header_len: usize = header_len_u64.try_into().map_err(|_| {
        invalid_safetensors_header(
            path,
            format!("safetensors header length {header_len_u64} does not fit in usize"),
        )
    })?;
    if header_len > SAFETENSORS_MAX_HEADER_SIZE {
        return Err(invalid_safetensors_header(
            path,
            format!(
                "safetensors header length {header_len} exceeds max {SAFETENSORS_MAX_HEADER_SIZE}"
            ),
        ));
    }

    let header_end = SAFETENSORS_HEADER_LEN_BYTES
        .checked_add(header_len)
        .ok_or_else(|| {
            invalid_safetensors_header(
                path,
                format!("safetensors header length {header_len} overflows file offset"),
            )
        })?;
    if file_len < header_end as u64 {
        return Err(invalid_safetensors_header(
            path,
            format!(
                "safetensors header is truncated: file is {file_len} bytes, header ends at {header_end}"
            ),
        ));
    }

    let mut header_bytes = vec![0u8; header_len];
    read_exact_header_part(&mut file, path, &mut header_bytes)?;
    let header_str = std::str::from_utf8(&header_bytes).map_err(|e| {
        invalid_safetensors_header(path, format!("failed to parse safetensors header: {e}"))
    })?;
    let metadata: safetensors::tensor::Metadata =
        serde_json::from_str(header_str).map_err(|e| {
            invalid_safetensors_header(path, format!("failed to parse safetensors header: {e}"))
        })?;

    let expected_file_len = (header_end as u64)
        .checked_add(metadata.data_len() as u64)
        .ok_or_else(|| {
            invalid_safetensors_header(
                path,
                format!(
                    "safetensors data length {} overflows file length",
                    metadata.data_len()
                ),
            )
        })?;
    if expected_file_len != file_len {
        return Err(invalid_safetensors_header(
            path,
            format!(
                "safetensors file length mismatch: header declares {} data bytes, expected total {expected_file_len}, got {file_len}",
                metadata.data_len()
            ),
        ));
    }

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

fn read_exact_header_part(file: &mut File, path: &Path, buf: &mut [u8]) -> Result<()> {
    file.read_exact(buf).map_err(|source| {
        if source.kind() == ErrorKind::UnexpectedEof {
            invalid_safetensors_header(path, "safetensors header is truncated".to_string())
        } else {
            io_error(path, source)
        }
    })
}

fn io_error(path: &Path, source: std::io::Error) -> OcelotlError {
    OcelotlError::from(IoError {
        path: Some(path.to_path_buf()),
        source,
    })
}

fn invalid_safetensors_header(path: &Path, message: String) -> OcelotlError {
    OcelotlError::from(InvalidModelError {
        path: Some(path.to_path_buf()),
        field: None,
        message,
    })
}

/// Verify that every tensor name in `required` appears in the manifest.
/// Returns an `OcelotlError::InvalidModel` for the *first* missing tensor,
/// with `field` set to the missing name so callers can surface
/// "this artifact is missing tensor X" without parsing the message.
///
/// `path` is optional so callers that built a manifest from memory can still
/// use this helper; tests against on-disk fixtures should pass `Some(path)`
/// so the InvalidModel error carries the artifact location.
///
/// Kept deliberately small for M2.5: M2.6 will own the model-family-specific
/// list of required tensor names (e.g. the full Qwen2.5 weight set). This
/// helper just answers "are all these names present?" against any manifest.
pub fn require_tensors(
    manifest: &SafetensorsManifest,
    required: &[&str],
    path: Option<&Path>,
) -> Result<()> {
    for name in required {
        if !manifest.tensors.iter().any(|t| t.name == *name) {
            return Err(OcelotlError::from(InvalidModelError {
                path: path.map(|p| p.to_path_buf()),
                field: Some((*name).to_string()),
                message: format!("required tensor `{name}` not found in safetensors header"),
            }));
        }
    }
    Ok(())
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

    /// Build a deliberately malformed safetensors file where one tensor's
    /// `data_offsets` byte length does not match `shape_product * dtype_size`.
    ///
    /// `tensors` is `(name, dtype_str, shape, declared_byte_len)`. The
    /// `declared_byte_len` is what gets written into `data_offsets[1] -
    /// data_offsets[0]` regardless of what the shape × dtype implies. The
    /// data section is sized to `declared_byte_len` so the file itself is
    /// internally consistent at the byte level — the *header* is the bug.
    ///
    /// M2.5's `build_fixture` deliberately keeps shape and offsets
    /// consistent; this peer helper exists specifically to construct the
    /// shape-vs-offsets disagreement that M2.7 covers.
    fn build_shape_mismatch_fixture(path: &Path, tensors: &[(&str, &str, &[usize], usize)]) {
        use std::collections::BTreeMap;
        use std::io::Write;

        let mut header_map: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        let mut cursor: usize = 0;
        for (name, dtype, shape, declared_byte_len) in tensors {
            let begin = cursor;
            let end = cursor + *declared_byte_len;
            cursor = end;
            header_map.insert(
                (*name).to_string(),
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

    #[test]
    fn inspect_safetensors_returns_io_error_when_file_does_not_exist() {
        // Per docs/design/errors.md and crossing-crate-boundaries.md, a
        // missing artifact file is a storage/IO problem, not a malformed
        // model. The error must be `Io`, carry the requested path, and
        // preserve the underlying io::Error as a source so callers can
        // distinguish NotFound from PermissionDenied.
        let path = tmp_path("definitely_does_not_exist");
        // No build_fixture call — the file is intentionally absent.
        let err = inspect_safetensors(&path).expect_err("missing file must fail");

        match err {
            OcelotlError::Io(io) => {
                assert_eq!(
                    io.path.as_deref(),
                    Some(path.as_path()),
                    "expected the missing path on the Io error, got {:?}",
                    io.path,
                );
                assert_eq!(
                    io.source.kind(),
                    std::io::ErrorKind::NotFound,
                    "expected the underlying io::Error kind to be NotFound, got {:?}",
                    io.source.kind(),
                );
            }
            other => panic!("expected OcelotlError::Io for missing file, got {other:?}"),
        }
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
    fn inspect_safetensors_accepts_large_data_section_without_reading_tensor_bytes() {
        use std::io::Write;

        let path = tmp_path("large_data_header_only");
        let data_len: u64 = 4 * 1024 * 1024;
        let header = serde_json::json!({
            "large_weight": {
                "dtype": "F32",
                "shape": [1024, 1024],
                "data_offsets": [0, data_len],
            }
        });
        let header_json = serde_json::to_string(&header).expect("serialize header");
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut file = std::fs::File::create(&path).expect("create large-data fixture");
        file.write_all(&header_len.to_le_bytes())
            .expect("write header length");
        file.write_all(header_bytes).expect("write header");
        let total_len = SAFETENSORS_HEADER_LEN_BYTES as u64 + header_len + data_len;
        file.set_len(total_len)
            .expect("extend data section without writing tensor bytes");
        drop(file);

        let manifest =
            inspect_safetensors(&path).expect("header-only inspection must not need data bytes");

        assert_eq!(manifest.data_len, data_len as usize);
        assert_eq!(manifest.tensors.len(), 1);
        assert_eq!(manifest.tensors[0].name, "large_weight");
        assert_eq!(manifest.tensors[0].byte_range, (0, data_len as usize));

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

    #[test]
    fn require_tensors_returns_invalid_model_error_when_a_required_tensor_is_missing() {
        // The manifest declares only `embed`. We require both `embed` and
        // `lm_head`. The contract: missing tensors are an InvalidModel
        // error (the artifact is internally incomplete relative to a
        // model-family expectation), with `field` carrying the missing
        // tensor name so callers can surface "this artifact is missing
        // tensor X".
        let path = tmp_path("missing_tensor");
        build_fixture(&path, &[("embed", "F32", &[2, 4])]);

        let manifest = inspect_safetensors(&path).expect("fixture inspects");
        let err = require_tensors(&manifest, &["embed", "lm_head"], Some(&path))
            .expect_err("require_tensors must reject when a required tensor is absent");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(
                    invalid.path.as_deref(),
                    Some(path.as_path()),
                    "expected fixture path on InvalidModel, got {:?}",
                    invalid.path,
                );
                assert_eq!(
                    invalid.field.as_deref(),
                    Some("lm_head"),
                    "expected the missing tensor name in field, got {:?}",
                    invalid.field,
                );
                assert!(
                    invalid.message.contains("lm_head"),
                    "expected message to mention the missing tensor, got {:?}",
                    invalid.message,
                );
            }
            other => {
                panic!("expected OcelotlError::InvalidModel for missing tensor, got {other:?}")
            }
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn require_tensors_returns_ok_when_all_required_tensors_are_present() {
        let path = tmp_path("all_present");
        build_fixture(
            &path,
            &[("embed", "F32", &[2, 4]), ("lm_head", "F32", &[4, 2])],
        );

        let manifest = inspect_safetensors(&path).expect("fixture inspects");
        require_tensors(&manifest, &["embed", "lm_head"], Some(&path))
            .expect("all required tensors are present");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn inspect_safetensors_rejects_shape_offsets_mismatch_with_invalid_model_error() {
        // Header declares shape [4, 8] and dtype F32 (32 elems × 4 bytes =
        // 128 bytes), but `data_offsets: [0, 64]` claims only 64 bytes for
        // the tensor's payload. The header is internally inconsistent. The
        // M2.7 contract: this surfaces as OcelotlError::InvalidModel
        // carrying the fixture path so callers can report which artifact
        // failed and which tensor was inconsistent.
        //
        // Note: this is a header-vs-header inconsistency (declared shape
        // disagrees with declared byte_range), not a header-vs-file-size
        // truncation — the file's data section is sized to match
        // data_offsets, so the bug is purely in the metadata.
        let path = tmp_path("shape_offsets_mismatch");
        build_shape_mismatch_fixture(
            &path,
            // shape [4, 8] × F32 implies 128 bytes; we declare 64.
            &[("attention_weight", "F32", &[4, 8], 64)],
        );

        let err = inspect_safetensors(&path)
            .expect_err("safetensors header with shape vs data_offsets mismatch must be rejected");

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
                panic!(
                    "expected OcelotlError::InvalidModel for shape/offsets mismatch, got {other:?}"
                )
            }
        }

        let _ = std::fs::remove_file(&path);
    }
}
