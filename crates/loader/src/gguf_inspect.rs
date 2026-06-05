//! Header-only inspection of GGUF artifacts.
//!
//! GGUF carries model metadata, tokenizer metadata, tensor descriptors, and
//! tensor payloads in one file. The loader boundary here intentionally stops
//! before payload reads: it parses the bounded metadata and tensor-info region,
//! normalizes it into Ocelotl-owned structs, and validates declared tensor
//! offsets against the file length.

use std::{
    fs::File,
    io::{ErrorKind, Read, Seek, SeekFrom},
    path::Path,
};

use ocelotl_core::{InvalidModelError, IoError, OcelotlError, Result, UnsupportedError};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const SUPPORTED_GGUF_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_METADATA_ENTRIES: u64 = 200_000;
const MAX_TENSORS: u64 = 2_000_000;
const MAX_METADATA_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_KEY_BYTES: u64 = 65_535;
const MAX_TENSOR_NAME_BYTES: u64 = 65_535;
const MAX_ARRAY_ELEMENTS: u64 = 2_000_000;
const MAX_ARRAY_DEPTH: usize = 4;
const MAX_TENSOR_DIMS: u32 = 8;

#[derive(Debug, Clone, PartialEq)]
pub struct GgufManifest {
    pub version: u32,
    pub metadata: Vec<GgufMetadataEntry>,
    pub tensors: Vec<GgufTensorEntry>,
    pub alignment: u64,
    pub data_start: u64,
    pub file_len: u64,
}

impl GgufManifest {
    pub fn metadata_value(&self, key: &str) -> Option<&GgufMetadataValue> {
        self.metadata
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| &entry.value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufMetadataEntry {
    pub key: String,
    pub value: GgufMetadataValue,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GgufMetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array {
        element_type: GgufMetadataType,
        len: u64,
    },
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufMetadataType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl GgufMetadataType {
    fn from_raw(raw: u32, path: &Path, field: &str) -> Result<Self> {
        Ok(match raw {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            other => {
                return Err(invalid_gguf(
                    path,
                    Some(field),
                    format!("unknown GGUF metadata value type {other}"),
                ));
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub tensor_type: GgmlTensorType,
    /// Offset relative to the GGUF tensor-data section.
    pub offset: u64,
    /// Absolute byte offset in the file.
    pub file_offset: u64,
    /// Header-derived byte length when the tensor type has a fixed byte size
    /// Ocelotl can validate without implementing that quantization.
    pub byte_len: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufTokenizerMetadata {
    pub model: Option<String>,
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub token_types: Vec<i32>,
    pub merges: Vec<String>,
    pub chat_template: Option<String>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub unknown_token_id: Option<u32>,
    pub padding_token_id: Option<u32>,
    pub mask_token_id: Option<u32>,
    pub add_bos_token: Option<bool>,
    pub add_space_prefix: Option<bool>,
}

impl GgufTokenizerMetadata {
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlQuantLayout {
    pub block_element_count: u64,
    pub block_byte_len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlTensorType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    I8,
    I16,
    I32,
    I64,
    F64,
    Iq1M,
    BF16,
    Tq1_0,
    Tq2_0,
    MXFP4,
    Unknown(u32),
}

impl GgmlTensorType {
    fn from_raw(raw: u32) -> Self {
        match raw {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            19 => Self::Iq1S,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::Iq1M,
            30 => Self::BF16,
            34 => Self::Tq1_0,
            35 => Self::Tq2_0,
            39 => Self::MXFP4,
            other => Self::Unknown(other),
        }
    }

    pub fn quant_layout(self) -> Option<GgmlQuantLayout> {
        // Current ggml K-quant constants, verified against
        // ggml/src/ggml-common.h on 2026-06-05:
        // QK_K = 256, K_SCALE_SIZE = 12.
        //
        // block_q4_K = 2 * half + K_SCALE_SIZE + QK_K / 2 = 144 bytes.
        // block_q5_K = 2 * half + K_SCALE_SIZE + QK_K / 2 + QK_K / 8 = 176 bytes.
        // block_q6_K = half + QK_K / 16 + 3 * QK_K / 4 = 210 bytes.
        match self {
            Self::Q4K => Some(GgmlQuantLayout {
                block_element_count: 256,
                block_byte_len: 144,
            }),
            Self::Q5K => Some(GgmlQuantLayout {
                block_element_count: 256,
                block_byte_len: 176,
            }),
            Self::Q6K => Some(GgmlQuantLayout {
                block_element_count: 256,
                block_byte_len: 210,
            }),
            _ => None,
        }
    }

    fn fixed_element_size(self) -> Option<u64> {
        match self {
            Self::F32 | Self::I32 => Some(4),
            Self::F16 | Self::BF16 | Self::I16 => Some(2),
            Self::I8 => Some(1),
            Self::I64 | Self::F64 => Some(8),
            _ => None,
        }
    }

    fn byte_len_for_element_count(
        self,
        element_count: u64,
        path: &Path,
        tensor_name: &str,
    ) -> Result<Option<u64>> {
        if let Some(elem_size) = self.fixed_element_size() {
            return element_count
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    invalid_gguf(
                        path,
                        Some(tensor_name),
                        format!("GGUF tensor `{tensor_name}` byte length overflows u64"),
                    )
                })
                .map(Some);
        }

        if let Some(layout) = self.quant_layout() {
            if element_count % layout.block_element_count != 0 {
                return Err(invalid_gguf(
                    path,
                    Some(tensor_name),
                    format!(
                        "GGUF tensor `{tensor_name}` element count {element_count} is not divisible by {:?} quant block size {}",
                        self, layout.block_element_count
                    ),
                ));
            }
            return (element_count / layout.block_element_count)
                .checked_mul(layout.block_byte_len)
                .ok_or_else(|| {
                    invalid_gguf(
                        path,
                        Some(tensor_name),
                        format!("GGUF tensor `{tensor_name}` byte length overflows u64"),
                    )
                })
                .map(Some);
        }

        Ok(None)
    }
}

/// Inspect a GGUF file and return metadata plus tensor descriptors. Does not
/// read tensor payload bytes.
pub fn inspect_gguf(path: &Path) -> Result<GgufManifest> {
    let file = File::open(path).map_err(|source| io_error(path, source))?;
    let file_len = file
        .metadata()
        .map_err(|source| io_error(path, source))?
        .len();
    let mut reader = GgufReader {
        file,
        path,
        file_len,
    };

    let mut magic = [0u8; 4];
    reader.read_exact_header_part(&mut magic)?;
    if &magic != GGUF_MAGIC {
        return Err(invalid_gguf(
            path,
            Some("magic"),
            format!("invalid GGUF magic bytes: {magic:?}"),
        ));
    }

    let version = reader.read_u32()?;
    if version != SUPPORTED_GGUF_VERSION {
        return Err(OcelotlError::from(UnsupportedError {
            feature: "gguf_version".to_string(),
            requested: Some(version.to_string()),
            supported: vec![SUPPORTED_GGUF_VERSION.to_string()],
        }));
    }

    let tensor_count = reader.read_u64()?;
    let metadata_count = reader.read_u64()?;
    if tensor_count > MAX_TENSORS {
        return Err(invalid_gguf(
            path,
            Some("tensor_count"),
            format!("GGUF tensor_count {tensor_count} exceeds max {MAX_TENSORS}"),
        ));
    }
    if metadata_count > MAX_METADATA_ENTRIES {
        return Err(invalid_gguf(
            path,
            Some("metadata_kv_count"),
            format!("GGUF metadata_kv_count {metadata_count} exceeds max {MAX_METADATA_ENTRIES}"),
        ));
    }

    let mut metadata = Vec::with_capacity(metadata_count as usize);
    for _ in 0..metadata_count {
        let key = reader.read_string("metadata key", MAX_KEY_BYTES)?;
        let raw_type = reader.read_u32()?;
        let value_type = GgufMetadataType::from_raw(raw_type, path, &key)?;
        let value = reader.read_metadata_value(value_type, 0)?;
        metadata.push(GgufMetadataEntry { key, value });
    }

    let alignment = alignment_from_metadata(path, &metadata)?;
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = reader.read_string("tensor name", MAX_TENSOR_NAME_BYTES)?;
        let n_dims = reader.read_u32()?;
        if n_dims == 0 || n_dims > MAX_TENSOR_DIMS {
            return Err(invalid_gguf(
                path,
                Some(&name),
                format!("GGUF tensor `{name}` has unsupported dimension count {n_dims}"),
            ));
        }

        let mut shape = Vec::with_capacity(n_dims as usize);
        let mut element_count = 1_u64;
        for _ in 0..n_dims {
            let dim = reader.read_u64()?;
            if dim == 0 {
                return Err(invalid_gguf(
                    path,
                    Some(&name),
                    format!("GGUF tensor `{name}` has a zero dimension"),
                ));
            }
            let dim_usize: usize = dim.try_into().map_err(|_| {
                invalid_gguf(
                    path,
                    Some(&name),
                    format!("GGUF tensor `{name}` dimension {dim} does not fit in usize"),
                )
            })?;
            element_count = element_count.checked_mul(dim).ok_or_else(|| {
                invalid_gguf(
                    path,
                    Some(&name),
                    format!("GGUF tensor `{name}` element count overflows u64"),
                )
            })?;
            shape.push(dim_usize);
        }

        let tensor_type = GgmlTensorType::from_raw(reader.read_u32()?);
        let offset = reader.read_u64()?;
        if offset % alignment != 0 {
            return Err(invalid_gguf(
                path,
                Some(&name),
                format!("GGUF tensor `{name}` offset {offset} is not aligned to {alignment} bytes"),
            ));
        }
        let byte_len = tensor_type.byte_len_for_element_count(element_count, path, &name)?;

        tensors.push(GgufTensorEntry {
            name,
            shape,
            tensor_type,
            offset,
            file_offset: 0,
            byte_len,
        });
    }

    let tensor_info_end = reader.position()?;
    let data_start = align_offset(path, tensor_info_end, alignment)?;
    if data_start > file_len {
        return Err(invalid_gguf(
            path,
            Some("tensor_data"),
            format!(
                "GGUF tensor data section starts at {data_start}, beyond file length {file_len}"
            ),
        ));
    }

    for tensor in &mut tensors {
        let file_offset = data_start.checked_add(tensor.offset).ok_or_else(|| {
            invalid_gguf(
                path,
                Some(&tensor.name),
                format!("GGUF tensor `{}` file offset overflows u64", tensor.name),
            )
        })?;
        if file_offset > file_len {
            return Err(invalid_gguf(
                path,
                Some(&tensor.name),
                format!(
                    "GGUF tensor `{}` starts at file offset {file_offset}, beyond file length {file_len}",
                    tensor.name
                ),
            ));
        }
        if let Some(byte_len) = tensor.byte_len {
            let tensor_end = file_offset.checked_add(byte_len).ok_or_else(|| {
                invalid_gguf(
                    path,
                    Some(&tensor.name),
                    format!("GGUF tensor `{}` end offset overflows u64", tensor.name),
                )
            })?;
            if tensor_end > file_len {
                return Err(invalid_gguf(
                    path,
                    Some(&tensor.name),
                    format!(
                        "GGUF tensor `{}` byte range {file_offset}..{tensor_end} exceeds file length {file_len}",
                        tensor.name
                    ),
                ));
            }
        }
        tensor.file_offset = file_offset;
    }

    Ok(GgufManifest {
        version,
        metadata,
        tensors,
        alignment,
        data_start,
        file_len,
    })
}

/// Extract embedded GGUF tokenizer metadata without retaining tokenizer arrays
/// in the default `inspect_gguf` manifest path.
///
/// This reads only metadata entries and intentionally does not validate tensor
/// descriptors or tensor payload ranges. Use `inspect_gguf` when the model
/// tensor contract must also be proven.
pub fn inspect_gguf_tokenizer(path: &Path) -> Result<GgufTokenizerMetadata> {
    let file = File::open(path).map_err(|source| io_error(path, source))?;
    let file_len = file
        .metadata()
        .map_err(|source| io_error(path, source))?
        .len();
    let mut reader = GgufReader {
        file,
        path,
        file_len,
    };

    let mut magic = [0u8; 4];
    reader.read_exact_header_part(&mut magic)?;
    if &magic != GGUF_MAGIC {
        return Err(invalid_gguf(
            path,
            Some("magic"),
            format!("invalid GGUF magic bytes: {magic:?}"),
        ));
    }

    let version = reader.read_u32()?;
    if version != SUPPORTED_GGUF_VERSION {
        return Err(OcelotlError::from(UnsupportedError {
            feature: "gguf_version".to_string(),
            requested: Some(version.to_string()),
            supported: vec![SUPPORTED_GGUF_VERSION.to_string()],
        }));
    }

    let tensor_count = reader.read_u64()?;
    let metadata_count = reader.read_u64()?;
    if tensor_count > MAX_TENSORS {
        return Err(invalid_gguf(
            path,
            Some("tensor_count"),
            format!("GGUF tensor_count {tensor_count} exceeds max {MAX_TENSORS}"),
        ));
    }
    if metadata_count > MAX_METADATA_ENTRIES {
        return Err(invalid_gguf(
            path,
            Some("metadata_kv_count"),
            format!("GGUF metadata_kv_count {metadata_count} exceeds max {MAX_METADATA_ENTRIES}"),
        ));
    }

    let mut metadata = GgufTokenizerMetadata {
        model: None,
        tokens: Vec::new(),
        scores: Vec::new(),
        token_types: Vec::new(),
        merges: Vec::new(),
        chat_template: None,
        bos_token_id: None,
        eos_token_id: None,
        unknown_token_id: None,
        padding_token_id: None,
        mask_token_id: None,
        add_bos_token: None,
        add_space_prefix: None,
    };

    for _ in 0..metadata_count {
        let key = reader.read_string("metadata key", MAX_KEY_BYTES)?;
        let raw_type = reader.read_u32()?;
        let value_type = GgufMetadataType::from_raw(raw_type, path, &key)?;

        match key.as_str() {
            "tokenizer.ggml.tokens" => {
                metadata.tokens = reader.read_metadata_string_array(value_type, &key)?;
            }
            "tokenizer.ggml.scores" => {
                metadata.scores = reader.read_metadata_f32_array(value_type, &key)?;
            }
            "tokenizer.ggml.token_type" => {
                metadata.token_types = reader.read_metadata_i32_array(value_type, &key)?;
            }
            "tokenizer.ggml.merges" => {
                metadata.merges = reader.read_metadata_string_array(value_type, &key)?;
            }
            "tokenizer.ggml.model" => {
                metadata.model = Some(expect_metadata_string(
                    path,
                    &key,
                    reader.read_metadata_value(value_type, 0)?,
                )?);
            }
            "tokenizer.chat_template" => {
                metadata.chat_template = Some(expect_metadata_string(
                    path,
                    &key,
                    reader.read_metadata_value(value_type, 0)?,
                )?);
            }
            "tokenizer.ggml.bos_token_id" => {
                metadata.bos_token_id = Some(expect_metadata_u32(
                    path,
                    &key,
                    reader.read_metadata_value(value_type, 0)?,
                )?);
            }
            "tokenizer.ggml.eos_token_id" => {
                metadata.eos_token_id = Some(expect_metadata_u32(
                    path,
                    &key,
                    reader.read_metadata_value(value_type, 0)?,
                )?);
            }
            "tokenizer.ggml.unknown_token_id" => {
                metadata.unknown_token_id = Some(expect_metadata_u32(
                    path,
                    &key,
                    reader.read_metadata_value(value_type, 0)?,
                )?);
            }
            "tokenizer.ggml.padding_token_id" => {
                metadata.padding_token_id = Some(expect_metadata_u32(
                    path,
                    &key,
                    reader.read_metadata_value(value_type, 0)?,
                )?);
            }
            "tokenizer.ggml.mask_token_id" => {
                metadata.mask_token_id = Some(expect_metadata_u32(
                    path,
                    &key,
                    reader.read_metadata_value(value_type, 0)?,
                )?);
            }
            "tokenizer.ggml.add_bos_token" => {
                metadata.add_bos_token = Some(expect_metadata_bool(
                    path,
                    &key,
                    reader.read_metadata_value(value_type, 0)?,
                )?);
            }
            "tokenizer.ggml.add_space_prefix" => {
                metadata.add_space_prefix = Some(expect_metadata_bool(
                    path,
                    &key,
                    reader.read_metadata_value(value_type, 0)?,
                )?);
            }
            _ => {
                reader.skip_metadata_value(value_type, 0)?;
            }
        }
    }

    Ok(metadata)
}

struct GgufReader<'a> {
    file: File,
    path: &'a Path,
    file_len: u64,
}

impl GgufReader<'_> {
    fn read_exact_header_part(&mut self, buf: &mut [u8]) -> Result<()> {
        self.file.read_exact(buf).map_err(|source| {
            if source.kind() == ErrorKind::UnexpectedEof {
                invalid_gguf(
                    self.path,
                    Some("header"),
                    "GGUF header is truncated".to_string(),
                )
            } else {
                io_error(self.path, source)
            }
        })
    }

    fn read_u8(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.read_exact_header_part(&mut buf)?;
        Ok(buf[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(i8::from_le_bytes([self.read_u8()?]))
    }

    fn read_u16(&mut self) -> Result<u16> {
        let mut buf = [0u8; 2];
        self.read_exact_header_part(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(&mut self) -> Result<i16> {
        let mut buf = [0u8; 2];
        self.read_exact_header_part(&mut buf)?;
        Ok(i16::from_le_bytes(buf))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.read_exact_header_part(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32(&mut self) -> Result<i32> {
        let mut buf = [0u8; 4];
        self.read_exact_header_part(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_bool(&mut self) -> Result<bool> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(invalid_gguf(
                self.path,
                Some("bool"),
                format!("GGUF bool value must be 0 or 1, got {other}"),
            )),
        }
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.read_exact_header_part(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64(&mut self) -> Result<i64> {
        let mut buf = [0u8; 8];
        self.read_exact_header_part(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_string(&mut self, field: &str, max_len: u64) -> Result<String> {
        let len = self.read_u64()?;
        if len > max_len || len > MAX_METADATA_STRING_BYTES {
            return Err(invalid_gguf(
                self.path,
                Some(field),
                format!("GGUF {field} length {len} exceeds max {max_len}"),
            ));
        }
        let len_usize: usize = len.try_into().map_err(|_| {
            invalid_gguf(
                self.path,
                Some(field),
                format!("GGUF {field} length {len} does not fit in usize"),
            )
        })?;
        let mut bytes = vec![0u8; len_usize];
        self.read_exact_header_part(&mut bytes)?;
        String::from_utf8(bytes).map_err(|source| {
            invalid_gguf(
                self.path,
                Some(field),
                format!("GGUF {field} is not UTF-8: {source}"),
            )
        })
    }

    fn skip_string(&mut self, field: &str, max_len: u64) -> Result<()> {
        let len = self.read_u64()?;
        if len > max_len || len > MAX_METADATA_STRING_BYTES {
            return Err(invalid_gguf(
                self.path,
                Some(field),
                format!("GGUF {field} length {len} exceeds max {max_len}"),
            ));
        }
        self.skip_bytes(len)
    }

    fn read_metadata_value(
        &mut self,
        value_type: GgufMetadataType,
        depth: usize,
    ) -> Result<GgufMetadataValue> {
        Ok(match value_type {
            GgufMetadataType::U8 => GgufMetadataValue::U8(self.read_u8()?),
            GgufMetadataType::I8 => GgufMetadataValue::I8(self.read_i8()?),
            GgufMetadataType::U16 => GgufMetadataValue::U16(self.read_u16()?),
            GgufMetadataType::I16 => GgufMetadataValue::I16(self.read_i16()?),
            GgufMetadataType::U32 => GgufMetadataValue::U32(self.read_u32()?),
            GgufMetadataType::I32 => GgufMetadataValue::I32(self.read_i32()?),
            GgufMetadataType::F32 => GgufMetadataValue::F32(self.read_f32()?),
            GgufMetadataType::Bool => GgufMetadataValue::Bool(self.read_bool()?),
            GgufMetadataType::String => GgufMetadataValue::String(
                self.read_string("metadata string", MAX_METADATA_STRING_BYTES)?,
            ),
            GgufMetadataType::Array => {
                let (element_type, len) = self.read_array_header(depth)?;
                for _ in 0..len {
                    self.skip_metadata_value(element_type, depth + 1)?;
                }
                GgufMetadataValue::Array { element_type, len }
            }
            GgufMetadataType::U64 => GgufMetadataValue::U64(self.read_u64()?),
            GgufMetadataType::I64 => GgufMetadataValue::I64(self.read_i64()?),
            GgufMetadataType::F64 => GgufMetadataValue::F64(self.read_f64()?),
        })
    }

    fn skip_metadata_value(&mut self, value_type: GgufMetadataType, depth: usize) -> Result<()> {
        match value_type {
            GgufMetadataType::U8 | GgufMetadataType::I8 | GgufMetadataType::Bool => {
                self.skip_bytes(1)
            }
            GgufMetadataType::U16 | GgufMetadataType::I16 => self.skip_bytes(2),
            GgufMetadataType::U32 | GgufMetadataType::I32 | GgufMetadataType::F32 => {
                self.skip_bytes(4)
            }
            GgufMetadataType::U64 | GgufMetadataType::I64 | GgufMetadataType::F64 => {
                self.skip_bytes(8)
            }
            GgufMetadataType::String => {
                self.skip_string("metadata string", MAX_METADATA_STRING_BYTES)
            }
            GgufMetadataType::Array => {
                let (element_type, len) = self.read_array_header(depth)?;
                for _ in 0..len {
                    self.skip_metadata_value(element_type, depth + 1)?;
                }
                Ok(())
            }
        }
    }

    fn read_metadata_string_array(
        &mut self,
        value_type: GgufMetadataType,
        key: &str,
    ) -> Result<Vec<String>> {
        let len = self.expect_array_element_type(value_type, key, GgufMetadataType::String)?;
        let len_usize = array_len_to_usize(self.path, key, len)?;
        let mut values = Vec::with_capacity(len_usize);
        for _ in 0..len {
            values.push(self.read_string(key, MAX_METADATA_STRING_BYTES)?);
        }
        Ok(values)
    }

    fn read_metadata_f32_array(
        &mut self,
        value_type: GgufMetadataType,
        key: &str,
    ) -> Result<Vec<f32>> {
        let len = self.expect_array_element_type(value_type, key, GgufMetadataType::F32)?;
        let len_usize = array_len_to_usize(self.path, key, len)?;
        let mut values = Vec::with_capacity(len_usize);
        for _ in 0..len {
            values.push(self.read_f32()?);
        }
        Ok(values)
    }

    fn read_metadata_i32_array(
        &mut self,
        value_type: GgufMetadataType,
        key: &str,
    ) -> Result<Vec<i32>> {
        let len = self.expect_array_element_type(value_type, key, GgufMetadataType::I32)?;
        let len_usize = array_len_to_usize(self.path, key, len)?;
        let mut values = Vec::with_capacity(len_usize);
        for _ in 0..len {
            values.push(self.read_i32()?);
        }
        Ok(values)
    }

    fn expect_array_element_type(
        &mut self,
        value_type: GgufMetadataType,
        key: &str,
        expected: GgufMetadataType,
    ) -> Result<u64> {
        if value_type != GgufMetadataType::Array {
            return Err(invalid_gguf(
                self.path,
                Some(key),
                format!("GGUF metadata `{key}` must be an array, got {value_type:?}"),
            ));
        }
        let (element_type, len) = self.read_array_header(0)?;
        if element_type != expected {
            return Err(invalid_gguf(
                self.path,
                Some(key),
                format!(
                    "GGUF metadata `{key}` array must contain {expected:?}, got {element_type:?}"
                ),
            ));
        }
        Ok(len)
    }

    fn read_array_header(&mut self, depth: usize) -> Result<(GgufMetadataType, u64)> {
        if depth >= MAX_ARRAY_DEPTH {
            return Err(invalid_gguf(
                self.path,
                Some("metadata array"),
                format!("GGUF metadata array nesting exceeds max depth {MAX_ARRAY_DEPTH}"),
            ));
        }
        let element_type =
            GgufMetadataType::from_raw(self.read_u32()?, self.path, "metadata array")?;
        let len = self.read_u64()?;
        if len > MAX_ARRAY_ELEMENTS {
            return Err(invalid_gguf(
                self.path,
                Some("metadata array"),
                format!("GGUF metadata array length {len} exceeds max {MAX_ARRAY_ELEMENTS}"),
            ));
        }
        Ok((element_type, len))
    }

    fn skip_bytes(&mut self, len: u64) -> Result<()> {
        let pos = self.position()?;
        let end = pos.checked_add(len).ok_or_else(|| {
            invalid_gguf(
                self.path,
                Some("header"),
                format!("GGUF skip length {len} overflows file offset"),
            )
        })?;
        if end > self.file_len {
            return Err(invalid_gguf(
                self.path,
                Some("header"),
                format!(
                    "GGUF metadata/tensor-info region is truncated: attempted to seek to {end}, file length is {}",
                    self.file_len
                ),
            ));
        }
        self.file
            .seek(SeekFrom::Start(end))
            .map_err(|source| io_error(self.path, source))?;
        Ok(())
    }

    fn position(&mut self) -> Result<u64> {
        self.file
            .stream_position()
            .map_err(|source| io_error(self.path, source))
    }
}

fn expect_metadata_string(path: &Path, key: &str, value: GgufMetadataValue) -> Result<String> {
    match value {
        GgufMetadataValue::String(value) => Ok(value),
        other => Err(invalid_gguf(
            path,
            Some(key),
            format!("GGUF metadata `{key}` must be a string, got {other:?}"),
        )),
    }
}

fn expect_metadata_u32(path: &Path, key: &str, value: GgufMetadataValue) -> Result<u32> {
    match value {
        GgufMetadataValue::U32(value) => Ok(value),
        GgufMetadataValue::U64(value) => value.try_into().map_err(|_| {
            invalid_gguf(
                path,
                Some(key),
                format!("GGUF metadata `{key}` value {value} does not fit in u32"),
            )
        }),
        other => Err(invalid_gguf(
            path,
            Some(key),
            format!("GGUF metadata `{key}` must be a uint32, got {other:?}"),
        )),
    }
}

fn expect_metadata_bool(path: &Path, key: &str, value: GgufMetadataValue) -> Result<bool> {
    match value {
        GgufMetadataValue::Bool(value) => Ok(value),
        other => Err(invalid_gguf(
            path,
            Some(key),
            format!("GGUF metadata `{key}` must be a bool, got {other:?}"),
        )),
    }
}

fn array_len_to_usize(path: &Path, key: &str, len: u64) -> Result<usize> {
    len.try_into().map_err(|_| {
        invalid_gguf(
            path,
            Some(key),
            format!("GGUF metadata `{key}` array length {len} does not fit in usize"),
        )
    })
}

fn alignment_from_metadata(path: &Path, metadata: &[GgufMetadataEntry]) -> Result<u64> {
    let alignment = match metadata
        .iter()
        .find(|entry| entry.key == "general.alignment")
        .map(|entry| &entry.value)
    {
        Some(GgufMetadataValue::U32(value)) => *value as u64,
        Some(GgufMetadataValue::U64(value)) => *value,
        Some(other) => {
            return Err(invalid_gguf(
                path,
                Some("general.alignment"),
                format!("GGUF general.alignment must be uint32 or uint64, got {other:?}"),
            ));
        }
        None => DEFAULT_ALIGNMENT,
    };

    if alignment == 0 || alignment % 8 != 0 {
        return Err(invalid_gguf(
            path,
            Some("general.alignment"),
            format!("GGUF alignment must be a positive multiple of 8, got {alignment}"),
        ));
    }
    Ok(alignment)
}

fn align_offset(path: &Path, offset: u64, alignment: u64) -> Result<u64> {
    let remainder = offset % alignment;
    if remainder == 0 {
        return Ok(offset);
    }
    offset.checked_add(alignment - remainder).ok_or_else(|| {
        invalid_gguf(
            path,
            Some("alignment"),
            "GGUF aligned offset overflows u64".into(),
        )
    })
}

fn io_error(path: &Path, source: std::io::Error) -> OcelotlError {
    OcelotlError::from(IoError {
        path: Some(path.to_path_buf()),
        source,
    })
}

fn invalid_gguf(path: &Path, field: Option<&str>, message: String) -> OcelotlError {
    OcelotlError::from(InvalidModelError {
        path: Some(path.to_path_buf()),
        field: field.map(str::to_string),
        message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("ocelotl_mf2_{}_{}.gguf", std::process::id(), name));
        p
    }

    fn repo_root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
    }

    fn local_gemma4_gguf_path() -> std::path::PathBuf {
        if let Ok(path) = std::env::var("OCELOTL_GEMMA4_GGUF_PATH") {
            return std::path::PathBuf::from(path);
        }
        repo_root()
            .join("local-artifacts")
            .join("gemma4_e4b_it_q4_k_m")
            .join("google_gemma-4-E4B-it-Q4_K_M.gguf")
    }

    fn write_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn write_i32(out: &mut Vec<u8>, value: i32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn write_f32(out: &mut Vec<u8>, value: f32) {
        out.extend_from_slice(&value.to_bits().to_le_bytes());
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

    fn write_bool_metadata(out: &mut Vec<u8>, key: &str, value: bool) {
        write_string(out, key);
        write_u32(out, 7);
        out.push(u8::from(value));
    }

    fn write_string_array_metadata(out: &mut Vec<u8>, key: &str, values: &[&str]) {
        write_string(out, key);
        write_u32(out, 9);
        write_u32(out, 8);
        write_u64(out, values.len() as u64);
        for value in values {
            write_string(out, value);
        }
    }

    fn write_f32_array_metadata(out: &mut Vec<u8>, key: &str, values: &[f32]) {
        write_string(out, key);
        write_u32(out, 9);
        write_u32(out, 6);
        write_u64(out, values.len() as u64);
        for value in values {
            write_f32(out, *value);
        }
    }

    fn write_i32_array_metadata(out: &mut Vec<u8>, key: &str, values: &[i32]) {
        write_string(out, key);
        write_u32(out, 9);
        write_u32(out, 5);
        write_u64(out, values.len() as u64);
        for value in values {
            write_i32(out, *value);
        }
    }

    fn write_u32_array_metadata(out: &mut Vec<u8>, key: &str, values: &[u32]) {
        write_string(out, key);
        write_u32(out, 9);
        write_u32(out, 4);
        write_u64(out, values.len() as u64);
        for value in values {
            write_u32(out, *value);
        }
    }

    fn write_tokenizer_fixture(path: &Path) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        write_u32(&mut bytes, SUPPORTED_GGUF_VERSION);
        write_u64(&mut bytes, 0);
        write_u64(&mut bytes, 14);

        write_string_metadata(&mut bytes, "general.architecture", "gemma4");
        write_string_metadata(&mut bytes, "tokenizer.ggml.model", "gemma4");
        write_string_array_metadata(
            &mut bytes,
            "tokenizer.ggml.tokens",
            &["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "Hello"],
        );
        write_f32_array_metadata(
            &mut bytes,
            "tokenizer.ggml.scores",
            &[0.0, 0.1, 0.2, 0.3, 0.4, -1.25],
        );
        write_i32_array_metadata(&mut bytes, "tokenizer.ggml.token_type", &[3, 3, 3, 2, 3, 1]);
        write_string_array_metadata(&mut bytes, "tokenizer.ggml.merges", &["H e", "He llo"]);
        write_string_metadata(
            &mut bytes,
            "tokenizer.chat_template",
            "{{ bos_token }}{{ messages[0].content }}",
        );
        write_u32_metadata(&mut bytes, "tokenizer.ggml.bos_token_id", 2);
        write_u32_metadata(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_u32_metadata(&mut bytes, "tokenizer.ggml.unknown_token_id", 3);
        write_u32_metadata(&mut bytes, "tokenizer.ggml.padding_token_id", 0);
        write_u32_metadata(&mut bytes, "tokenizer.ggml.mask_token_id", 4);
        write_bool_metadata(&mut bytes, "tokenizer.ggml.add_bos_token", true);
        write_bool_metadata(&mut bytes, "tokenizer.ggml.add_space_prefix", false);

        std::fs::write(path, bytes).expect("write GGUF tokenizer fixture");
    }

    fn write_bad_tokenizer_array_fixture(path: &Path) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        write_u32(&mut bytes, SUPPORTED_GGUF_VERSION);
        write_u64(&mut bytes, 0);
        write_u64(&mut bytes, 1);
        write_u32_array_metadata(&mut bytes, "tokenizer.ggml.tokens", &[1]);
        std::fs::write(path, bytes).expect("write bad GGUF tokenizer fixture");
    }

    fn write_minimal_fixture(path: &Path, tensor_offset: u64, tensor_data_len: usize) {
        write_single_tensor_fixture(
            path,
            "blk.0.attn_q.weight",
            &[2, 2],
            0,
            tensor_offset,
            tensor_data_len,
        );
    }

    fn write_single_tensor_fixture(
        path: &Path,
        tensor_name: &str,
        shape: &[u64],
        raw_tensor_type: u32,
        tensor_offset: u64,
        tensor_data_len: usize,
    ) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        write_u32(&mut bytes, SUPPORTED_GGUF_VERSION);
        write_u64(&mut bytes, 1);
        write_u64(&mut bytes, 4);

        write_string_metadata(&mut bytes, "general.architecture", "gemma4");
        write_string_metadata(&mut bytes, "general.name", "tiny synthetic gemma4");
        write_u32_metadata(&mut bytes, "general.alignment", 32);
        write_string_array_metadata(&mut bytes, "tokenizer.ggml.tokens", &["<bos>", "<eos>"]);

        write_string(&mut bytes, tensor_name);
        write_u32(&mut bytes, shape.len() as u32);
        for dim in shape {
            write_u64(&mut bytes, *dim);
        }
        write_u32(&mut bytes, raw_tensor_type);
        write_u64(&mut bytes, tensor_offset);

        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend(std::iter::repeat_n(0u8, tensor_data_len));

        std::fs::write(path, bytes).expect("write GGUF fixture");
    }

    #[test]
    fn inspect_gguf_parses_minimal_v3_manifest_without_payload_read() {
        let path = tmp_path("minimal");
        write_minimal_fixture(&path, 0, 16);

        let manifest = inspect_gguf(&path).expect("minimal GGUF fixture must inspect");

        assert_eq!(manifest.version, SUPPORTED_GGUF_VERSION);
        assert_eq!(manifest.alignment, 32);
        assert_eq!(
            manifest.metadata_value("general.architecture"),
            Some(&GgufMetadataValue::String("gemma4".to_string()))
        );
        assert_eq!(
            manifest.metadata_value("tokenizer.ggml.tokens"),
            Some(&GgufMetadataValue::Array {
                element_type: GgufMetadataType::String,
                len: 2,
            })
        );

        assert_eq!(manifest.tensors.len(), 1);
        let tensor = &manifest.tensors[0];
        assert_eq!(tensor.name, "blk.0.attn_q.weight");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.tensor_type, GgmlTensorType::F32);
        assert_eq!(tensor.offset, 0);
        assert_eq!(tensor.byte_len, Some(16));
        assert_eq!(tensor.file_offset, manifest.data_start);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn ggml_k_quant_layouts_match_current_ggml_common_h_contract() {
        assert_eq!(
            GgmlTensorType::Q4K.quant_layout(),
            Some(GgmlQuantLayout {
                block_element_count: 256,
                block_byte_len: 144
            })
        );
        assert_eq!(
            GgmlTensorType::Q5K.quant_layout(),
            Some(GgmlQuantLayout {
                block_element_count: 256,
                block_byte_len: 176
            })
        );
        assert_eq!(
            GgmlTensorType::Q6K.quant_layout(),
            Some(GgmlQuantLayout {
                block_element_count: 256,
                block_byte_len: 210
            })
        );
        assert_eq!(GgmlTensorType::F32.quant_layout(), None);
    }

    #[test]
    fn inspect_gguf_computes_q4_q5_q6_k_quantized_byte_lengths() {
        for (suffix, raw_type, tensor_type, expected_byte_len) in [
            ("q4k", 12, GgmlTensorType::Q4K, 144),
            ("q5k", 13, GgmlTensorType::Q5K, 176),
            ("q6k", 14, GgmlTensorType::Q6K, 210),
        ] {
            let path = tmp_path(suffix);
            write_single_tensor_fixture(
                &path,
                "blk.0.attn_q.weight",
                &[256],
                raw_type,
                0,
                expected_byte_len as usize,
            );

            let manifest = inspect_gguf(&path).expect("K-quant fixture must inspect");
            let tensor = &manifest.tensors[0];
            assert_eq!(tensor.tensor_type, tensor_type);
            assert_eq!(tensor.byte_len, Some(expected_byte_len));

            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn inspect_gguf_rejects_k_quant_tensor_with_non_block_multiple_elements() {
        let path = tmp_path("q4k_bad_block_count");
        write_single_tensor_fixture(&path, "blk.0.attn_q.weight", &[255], 12, 0, 144);

        let err = inspect_gguf(&path).expect_err("bad K-quant block count must fail");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("blk.0.attn_q.weight"));
                assert!(invalid.message.contains("not divisible"));
                assert!(invalid.message.contains("256"));
            }
            other => panic!("expected InvalidModel for bad K-quant block count, got {other:?}"),
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn inspect_gguf_rejects_short_k_quant_payload_with_byte_range_error() {
        let path = tmp_path("q6k_short_payload");
        write_single_tensor_fixture(&path, "blk.0.attn_q.weight", &[256], 14, 0, 209);

        let err = inspect_gguf(&path).expect_err("short Q6_K payload must fail");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("blk.0.attn_q.weight"));
                assert!(invalid.message.contains("exceeds file length"));
            }
            other => panic!("expected InvalidModel for short K-quant payload, got {other:?}"),
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn inspect_gguf_returns_io_error_when_file_does_not_exist() {
        let path = tmp_path("missing");

        let err = inspect_gguf(&path).expect_err("missing GGUF file must fail");

        match err {
            OcelotlError::Io(io) => {
                assert_eq!(io.path.as_deref(), Some(path.as_path()));
                assert_eq!(io.source.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected Io for missing GGUF file, got {other:?}"),
        }
    }

    #[test]
    fn inspect_gguf_rejects_truncated_header_with_invalid_model() {
        let path = tmp_path("truncated");
        std::fs::write(&path, b"GGUF\x03").expect("write truncated fixture");

        let err = inspect_gguf(&path).expect_err("truncated GGUF header must fail");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.path.as_deref(), Some(path.as_path()));
                assert_eq!(invalid.field.as_deref(), Some("header"));
                assert!(invalid.message.contains("truncated"));
            }
            other => panic!("expected InvalidModel for truncated GGUF header, got {other:?}"),
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn inspect_gguf_rejects_unsupported_version_with_typed_unsupported() {
        let path = tmp_path("unsupported_version");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        write_u32(&mut bytes, 99);
        write_u64(&mut bytes, 0);
        write_u64(&mut bytes, 0);
        std::fs::write(&path, bytes).expect("write unsupported-version fixture");

        let err = inspect_gguf(&path).expect_err("unsupported GGUF version must fail");

        match err {
            OcelotlError::Unsupported(unsupported) => {
                assert_eq!(unsupported.feature, "gguf_version");
                assert_eq!(unsupported.requested.as_deref(), Some("99"));
                assert!(unsupported.supported.iter().any(|s| s == "3"));
            }
            other => panic!("expected Unsupported for GGUF version, got {other:?}"),
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn inspect_gguf_rejects_bad_tensor_offset_before_payload_read() {
        let path = tmp_path("bad_offset");
        write_minimal_fixture(&path, 64, 16);

        let err = inspect_gguf(&path).expect_err("bad tensor offset must fail");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.path.as_deref(), Some(path.as_path()));
                assert_eq!(invalid.field.as_deref(), Some("blk.0.attn_q.weight"));
                assert!(
                    invalid.message.contains("exceeds file length")
                        || invalid.message.contains("beyond file length"),
                    "expected tensor offset/file length message, got {:?}",
                    invalid.message,
                );
            }
            other => panic!("expected InvalidModel for bad tensor offset, got {other:?}"),
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn inspect_gguf_rejects_oversized_metadata_string() {
        let path = tmp_path("oversized_string");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        write_u32(&mut bytes, SUPPORTED_GGUF_VERSION);
        write_u64(&mut bytes, 0);
        write_u64(&mut bytes, 1);
        write_string(&mut bytes, "general.name");
        write_u32(&mut bytes, 8);
        write_u64(&mut bytes, MAX_METADATA_STRING_BYTES + 1);
        std::fs::write(&path, bytes).expect("write oversized-string fixture");

        let err = inspect_gguf(&path).expect_err("oversized metadata string must fail");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("metadata string"));
                assert!(invalid.message.contains("exceeds max"));
            }
            other => panic!("expected InvalidModel for oversized metadata, got {other:?}"),
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn inspect_gguf_rejects_unaligned_tensor_offsets() {
        let path = tmp_path("unaligned_offset");
        write_minimal_fixture(&path, 1, 16);

        let err = inspect_gguf(&path).expect_err("unaligned tensor offset must fail");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("blk.0.attn_q.weight"));
                assert!(invalid.message.contains("not aligned"));
            }
            other => panic!("expected InvalidModel for unaligned tensor offset, got {other:?}"),
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn inspect_gguf_tokenizer_extracts_tiny_synthetic_metadata_arrays() {
        let path = tmp_path("tokenizer_metadata");
        write_tokenizer_fixture(&path);

        let tokenizer =
            inspect_gguf_tokenizer(&path).expect("tiny tokenizer metadata must inspect");

        assert_eq!(tokenizer.model.as_deref(), Some("gemma4"));
        assert_eq!(
            tokenizer.tokens,
            vec!["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "Hello"]
        );
        assert_eq!(tokenizer.token_count(), 6);
        assert_eq!(tokenizer.scores, vec![0.0, 0.1, 0.2, 0.3, 0.4, -1.25]);
        assert_eq!(tokenizer.token_types, vec![3, 3, 3, 2, 3, 1]);
        assert_eq!(tokenizer.merges, vec!["H e", "He llo"]);
        assert_eq!(
            tokenizer.chat_template.as_deref(),
            Some("{{ bos_token }}{{ messages[0].content }}")
        );
        assert_eq!(tokenizer.bos_token_id, Some(2));
        assert_eq!(tokenizer.eos_token_id, Some(1));
        assert_eq!(tokenizer.unknown_token_id, Some(3));
        assert_eq!(tokenizer.padding_token_id, Some(0));
        assert_eq!(tokenizer.mask_token_id, Some(4));
        assert_eq!(tokenizer.add_bos_token, Some(true));
        assert_eq!(tokenizer.add_space_prefix, Some(false));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn inspect_gguf_tokenizer_rejects_wrong_token_array_element_type() {
        let path = tmp_path("bad_tokenizer_array");
        write_bad_tokenizer_array_fixture(&path);

        let err =
            inspect_gguf_tokenizer(&path).expect_err("wrong token array type must be rejected");

        match err {
            OcelotlError::InvalidModel(invalid) => {
                assert_eq!(invalid.field.as_deref(), Some("tokenizer.ggml.tokens"));
                assert!(invalid.message.contains("String"));
                assert!(invalid.message.contains("U32"));
            }
            other => panic!("expected InvalidModel for wrong token array type, got {other:?}"),
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    #[ignore = "requires local-artifacts/gemma4_e4b_it_q4_k_m/google_gemma-4-E4B-it-Q4_K_M.gguf or OCELOTL_GEMMA4_GGUF_PATH"]
    fn local_gemma4_q4_k_m_gguf_header_contract_is_well_formed() {
        let path = local_gemma4_gguf_path();
        assert!(
            path.exists(),
            "missing Gemma4 GGUF artifact at {}; set OCELOTL_GEMMA4_GGUF_PATH or see docs/artifact-preparation.md",
            path.display()
        );

        let manifest = inspect_gguf(&path).expect("local Gemma4 GGUF header must inspect");
        assert_eq!(
            manifest.metadata_value("general.architecture"),
            Some(&GgufMetadataValue::String("gemma4".to_string()))
        );
        assert!(
            manifest
                .metadata_value("tokenizer.ggml.tokens")
                .is_some_and(|value| matches!(
                    value,
                    GgufMetadataValue::Array {
                        element_type: GgufMetadataType::String,
                        len
                    } if *len > 0
                )),
            "Gemma4 GGUF should carry embedded tokenizer tokens metadata"
        );
        assert!(
            !manifest.tensors.is_empty(),
            "Gemma4 GGUF should declare tensor descriptors"
        );
        assert!(
            manifest.tensors.iter().any(|tensor| matches!(
                tensor.tensor_type,
                GgmlTensorType::Q4K | GgmlTensorType::Q5K | GgmlTensorType::Q6K
            ) && tensor.byte_len.is_some()),
            "Q4_K_M Gemma4 GGUF should include K-quantized tensors with validated byte lengths"
        );
        assert!(
            manifest
                .tensors
                .iter()
                .all(|tensor| tensor.byte_len.is_some()),
            "selected Gemma4 Q4_K_M artifact should contain only dense or Q4_K/Q5_K/Q6_K tensor types whose byte ranges Ocelotl validates"
        );
    }

    #[test]
    #[ignore = "requires local-artifacts/gemma4_e4b_it_q4_k_m/google_gemma-4-E4B-it-Q4_K_M.gguf or OCELOTL_GEMMA4_GGUF_PATH"]
    fn local_gemma4_q4_k_m_gguf_tokenizer_metadata_is_extractable() {
        let path = local_gemma4_gguf_path();
        assert!(
            path.exists(),
            "missing Gemma4 GGUF artifact at {}; set OCELOTL_GEMMA4_GGUF_PATH or see docs/artifact-preparation.md",
            path.display()
        );

        let tokenizer =
            inspect_gguf_tokenizer(&path).expect("local Gemma4 tokenizer metadata must inspect");

        assert_eq!(tokenizer.model.as_deref(), Some("gemma4"));
        assert_eq!(tokenizer.token_count(), 262_144);
        assert_eq!(
            tokenizer.scores.len(),
            tokenizer.tokens.len(),
            "Gemma4 GGUF tokenizer scores should align one-for-one with tokens"
        );
        assert_eq!(
            tokenizer.token_types.len(),
            tokenizer.tokens.len(),
            "Gemma4 GGUF tokenizer token types should align one-for-one with tokens"
        );
        assert!(
            !tokenizer.merges.is_empty(),
            "Gemma4 GGUF tokenizer should include BPE merge metadata"
        );
        assert!(
            tokenizer
                .chat_template
                .as_deref()
                .is_some_and(|template| template.contains("messages")),
            "Gemma4 GGUF tokenizer should expose a chat template mentioning messages"
        );
        assert_eq!(tokenizer.bos_token_id, Some(2));
        assert_eq!(tokenizer.eos_token_id, Some(1));
        assert_eq!(tokenizer.unknown_token_id, Some(3));
        assert_eq!(tokenizer.padding_token_id, Some(0));
        assert_eq!(tokenizer.mask_token_id, Some(4));
        assert_eq!(tokenizer.add_bos_token, Some(true));
        assert_eq!(tokenizer.add_space_prefix, Some(false));
    }
}
