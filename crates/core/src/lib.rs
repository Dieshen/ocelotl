//! Core types and contracts shared across Ocelotl crates.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, OcelotlError>;

// ---------------------------------------------------------------------------
// Top-level error enum
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum OcelotlError {
    #[error(transparent)]
    InvalidModel(#[from] InvalidModelError),
    #[error(transparent)]
    InvalidRequest(#[from] InvalidRequestError),
    #[error(transparent)]
    Unsupported(#[from] UnsupportedError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error(transparent)]
    Kernel(#[from] KernelError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Io(#[from] IoError),
}

// ---------------------------------------------------------------------------
// Error structs
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
#[error(
    "invalid model artifact{}{}: {message}",
    self.path.as_ref().map(|p| format!(" at `{}`", p.display())).unwrap_or_default(),
    self.field.as_deref().map(|f| format!(" (field `{f}`)")).unwrap_or_default(),
)]
pub struct InvalidModelError {
    pub path: Option<PathBuf>,
    pub field: Option<String>,
    pub message: String,
}

#[derive(Debug, Error)]
#[error("invalid request field `{field}`: {message}")]
pub struct InvalidRequestError {
    pub field: String,
    pub message: String,
}

#[derive(Debug, Error)]
#[error(
    "unsupported feature `{feature}`{}: supported = [{}]",
    self.requested.as_deref().map(|r| format!(" (requested `{r}`)")).unwrap_or_default(),
    self.supported.join(", ")
)]
pub struct UnsupportedError {
    pub feature: String,
    pub requested: Option<String>,
    pub supported: Vec<String>,
}

#[derive(Debug, Error)]
#[error("tokenizer error: {message}")]
pub struct TokenizerError {
    pub message: String,
    #[source]
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

#[derive(Debug, Error)]
#[error("kernel error on backend `{backend}`: {message}")]
pub struct KernelError {
    pub backend: String,
    pub message: String,
}

#[derive(Debug, Error)]
#[error("runtime error: {message}")]
pub struct RuntimeError {
    pub message: String,
}

#[derive(Debug, Error)]
#[error("IO error{}: {source}", self.path.as_ref().map(|p| format!(" on `{}`", p.display())).unwrap_or_default())]
pub struct IoError {
    pub path: Option<PathBuf>,
    #[source]
    pub source: std::io::Error,
}

impl From<std::io::Error> for IoError {
    fn from(source: std::io::Error) -> Self {
        Self { path: None, source }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    // --- UnsupportedError ---

    #[test]
    fn unsupported_error_display_mentions_feature_requested_and_supported() {
        let err = OcelotlError::Unsupported(UnsupportedError {
            feature: "rope_scaling".to_string(),
            requested: Some("yarn".to_string()),
            supported: vec!["linear".to_string(), "dynamic".to_string()],
        });

        let rendered = format!("{err}");

        assert!(
            rendered.contains("rope_scaling"),
            "expected feature name in display, got {rendered:?}"
        );
        assert!(
            rendered.contains("yarn"),
            "expected requested value in display, got {rendered:?}"
        );
        assert!(
            rendered.contains("linear") && rendered.contains("dynamic"),
            "expected supported list in display, got {rendered:?}"
        );
    }

    #[test]
    fn unsupported_error_can_be_constructed_with_no_requested_value() {
        let err = OcelotlError::Unsupported(UnsupportedError {
            feature: "paged_kv_cache".to_string(),
            requested: None,
            supported: vec!["contiguous".to_string()],
        });

        let rendered = format!("{err}");

        assert!(rendered.contains("paged_kv_cache"));
        assert!(rendered.contains("contiguous"));
        assert!(
            !rendered.contains("requested"),
            "should not mention requested when None, got {rendered:?}"
        );
    }

    // --- InvalidModelError ---

    #[test]
    fn invalid_model_error_display_includes_path_field_and_message() {
        let err = OcelotlError::InvalidModel(InvalidModelError {
            path: Some(PathBuf::from("/models/qwen.safetensors")),
            field: Some("num_hidden_layers".to_string()),
            message: "expected positive integer".to_string(),
        });

        let rendered = format!("{err}");

        assert!(
            rendered.contains("qwen.safetensors"),
            "expected path in display, got {rendered:?}"
        );
        assert!(
            rendered.contains("num_hidden_layers"),
            "expected field in display, got {rendered:?}"
        );
        assert!(
            rendered.contains("expected positive integer"),
            "expected message in display, got {rendered:?}"
        );
    }

    #[test]
    fn invalid_model_error_works_with_no_path_or_field() {
        let err = OcelotlError::InvalidModel(InvalidModelError {
            path: None,
            field: None,
            message: "malformed header".to_string(),
        });

        let rendered = format!("{err}");

        assert!(
            rendered.contains("malformed header"),
            "expected message in display, got {rendered:?}"
        );
    }

    // --- InvalidRequestError ---

    #[test]
    fn invalid_request_error_display_includes_field_and_message() {
        let err = OcelotlError::InvalidRequest(InvalidRequestError {
            field: "max_new_tokens".to_string(),
            message: "must be greater than zero".to_string(),
        });

        let rendered = format!("{err}");

        assert!(
            rendered.contains("max_new_tokens"),
            "expected field in display, got {rendered:?}"
        );
        assert!(
            rendered.contains("must be greater than zero"),
            "expected message in display, got {rendered:?}"
        );
    }

    // --- TokenizerError ---

    #[test]
    fn tokenizer_error_display_includes_message() {
        let err = OcelotlError::Tokenizer(TokenizerError {
            message: "unknown special token <|im_start|>".to_string(),
            source: None,
        });

        let rendered = format!("{err}");

        assert!(
            rendered.contains("unknown special token"),
            "expected message in display, got {rendered:?}"
        );
    }

    #[test]
    fn tokenizer_error_preserves_source_when_present() {
        let inner = std::io::Error::new(std::io::ErrorKind::InvalidData, "bad utf8");
        let err = TokenizerError {
            message: "failed to load tokenizer".to_string(),
            source: Some(Box::new(inner)),
        };

        assert!(err.source().is_some(), "expected source to be preserved");
    }

    // --- KernelError ---

    #[test]
    fn kernel_error_display_includes_backend_and_message() {
        let err = OcelotlError::Kernel(KernelError {
            backend: "cpu".to_string(),
            message: "unsupported stride layout".to_string(),
        });

        let rendered = format!("{err}");

        assert!(
            rendered.contains("cpu"),
            "expected backend in display, got {rendered:?}"
        );
        assert!(
            rendered.contains("unsupported stride layout"),
            "expected message in display, got {rendered:?}"
        );
    }

    // --- RuntimeError ---

    #[test]
    fn runtime_error_display_includes_message() {
        let err = OcelotlError::Runtime(RuntimeError {
            message: "KV cache allocation failed".to_string(),
        });

        let rendered = format!("{err}");

        assert!(
            rendered.contains("KV cache allocation failed"),
            "expected message in display, got {rendered:?}"
        );
    }

    // --- IoError ---

    #[test]
    fn io_error_display_includes_path() {
        let err = OcelotlError::Io(IoError {
            path: Some(PathBuf::from("/weights/model.safetensors")),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "file missing"),
        });

        let rendered = format!("{err}");

        assert!(
            rendered.contains("model.safetensors"),
            "expected path in display, got {rendered:?}"
        );
    }

    #[test]
    fn io_error_preserves_source() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let err = OcelotlError::Io(IoError {
            path: None,
            source: io_err,
        });

        assert!(
            err.source().is_some(),
            "expected io::Error to be reachable via source()"
        );
    }

    // --- M1.2 newtype tests ---

    #[test]
    fn token_id_wraps_and_unwraps_correctly() {
        let id = TokenId(1234);
        assert_eq!(id.0, 1234);
    }

    #[test]
    fn seq_len_wraps_and_unwraps_correctly() {
        let s = SeqLen(512);
        assert_eq!(s.0, 512);
    }

    #[test]
    fn batch_size_wraps_and_unwraps_correctly() {
        let b = BatchSize(8);
        assert_eq!(b.0, 8);
    }

    #[test]
    fn context_len_wraps_and_unwraps_correctly() {
        let c = ContextLen(4096);
        assert_eq!(c.0, 4096);
    }

    // --- GenerateResponse (M1.9) ---

    #[test]
    fn generate_response_round_trips_through_serde_json() {
        // The server crate will eventually serialize this to JSON, so we pin
        // the wire shape here: a `tokens` array of u32-shaped TokenIds. If
        // someone changes the field name or token representation, this test
        // fails before the server contract silently drifts.
        let resp = GenerateResponse {
            tokens: vec![TokenId(7), TokenId(42), TokenId(0)],
        };

        let json = serde_json::to_string(&resp).expect("serialize");
        assert!(
            json.contains("\"tokens\""),
            "expected tokens field in JSON, got {json:?}"
        );
        assert!(json.contains("7") && json.contains("42"));

        let round_tripped: GenerateResponse = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(round_tripped, resp);
    }

    // Type safety is enforced at compile time — TokenId and SeqLen are distinct
    // types; passing one where the other is expected is a type error.

    // --- M1.3 metadata fixture test ---

    #[test]
    fn qwen2_5_tiny_synthetic_fixture_deserializes_correctly() {
        #[derive(Debug, serde::Deserialize)]
        struct MetadataFixture {
            model: ModelMetadata,
        }

        let fixture_path =
            crate::test_fixtures::metadata_fixture_path("qwen2_5_tiny_synthetic.json");

        let json = std::fs::read_to_string(&fixture_path)
            .expect("fixture file must be readable from workspace root");

        let fixture: MetadataFixture =
            serde_json::from_str(&json).expect("fixture must deserialize without error");

        let m = fixture.model;
        assert_eq!(m.architecture, "qwen2");
        assert_eq!(m.vocab_size, 32);
        assert_eq!(m.num_hidden_layers, 2);
        assert_eq!(m.hidden_size, 16);
        assert_eq!(m.intermediate_size, 32);
        assert_eq!(m.num_attention_heads, 4);
        assert_eq!(m.num_key_value_heads, 2);
        assert_eq!(m.head_dim, 4);
        assert_eq!(m.context_length, 128);
        assert_eq!(m.dtype, DType::F32);
        assert!((m.rope_theta - 1_000_000.0_f64).abs() < 1e-6);
        assert!((m.rms_norm_eps - 1e-6_f64).abs() < 1e-12);
    }

    #[test]
    fn kv_cache_layout_calculates_shape_and_bytes() {
        let layout = KvCacheLayout::new(2, 3, 5, 4, DType::F32, Device::Cpu)
            .expect("valid layout must construct");

        assert_eq!(layout.values_per_position().unwrap(), 12);
        assert_eq!(layout.values_per_layer_tensor().unwrap(), 60);
        assert_eq!(layout.values_per_all_key_tensors().unwrap(), 120);
        assert_eq!(layout.total_values().unwrap(), 240);
        assert_eq!(layout.total_bytes().unwrap(), 960);
        assert_eq!(layout.layer_position_offset(1, 2).unwrap(), 84);
    }

    #[test]
    fn paged_kv_layout_maps_positions_and_rejects_bad_tables() {
        let base = KvCacheLayout::new(1, 2, 6, 4, DType::F32, Device::Cpu).unwrap();
        let layout = PagedKvCacheLayout::new(base, 4, 3).unwrap();

        assert_eq!(layout.required_pages_for_tokens(0).unwrap(), 0);
        assert_eq!(layout.required_pages_for_tokens(1).unwrap(), 1);
        assert_eq!(layout.required_pages_for_tokens(6).unwrap(), 2);
        assert_eq!(layout.logical_page_and_offset(5).unwrap(), (1, 1));
        assert_eq!(layout.values_per_page_tensor().unwrap(), 32);
        layout.validate_page_table(&[0, 2]).unwrap();

        let duplicate = layout
            .validate_page_table(&[1, 1])
            .expect_err("duplicate physical pages must be rejected");
        assert!(format!("{duplicate}").contains("appears more than once"));

        let out_of_range = layout
            .validate_page_table(&[0, 3])
            .expect_err("out-of-range physical pages must be rejected");
        assert!(format!("{out_of_range}").contains("out of range"));
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Gpu { ordinal: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    F16,
    BF16,
    Q4,
    Q8,
}

// ---------------------------------------------------------------------------
// KV cache contracts (M5/M6)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheLayout {
    pub num_layers: usize,
    pub num_key_value_heads: usize,
    pub capacity_tokens: usize,
    pub head_dim: usize,
    pub dtype: DType,
    pub device: Device,
}

impl KvCacheLayout {
    pub fn new(
        num_layers: usize,
        num_key_value_heads: usize,
        capacity_tokens: usize,
        head_dim: usize,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        if num_layers == 0 {
            return Err(runtime_err("kv cache num_layers must be > 0"));
        }
        if num_key_value_heads == 0 {
            return Err(runtime_err("kv cache num_key_value_heads must be > 0"));
        }
        if capacity_tokens == 0 {
            return Err(runtime_err("kv cache capacity_tokens must be > 0"));
        }
        if head_dim == 0 {
            return Err(runtime_err("kv cache head_dim must be > 0"));
        }

        Ok(Self {
            num_layers,
            num_key_value_heads,
            capacity_tokens,
            head_dim,
            dtype,
            device,
        })
    }

    pub fn values_per_position(&self) -> Result<usize> {
        checked_product(
            "kv cache values_per_position",
            &[self.num_key_value_heads, self.head_dim],
        )
    }

    pub fn values_per_layer_tensor(&self) -> Result<usize> {
        checked_product(
            "kv cache values_per_layer_tensor",
            &[
                self.capacity_tokens,
                self.num_key_value_heads,
                self.head_dim,
            ],
        )
    }

    pub fn values_per_all_key_tensors(&self) -> Result<usize> {
        checked_product(
            "kv cache values_per_all_key_tensors",
            &[
                self.num_layers,
                self.capacity_tokens,
                self.num_key_value_heads,
                self.head_dim,
            ],
        )
    }

    pub fn total_values(&self) -> Result<usize> {
        checked_product(
            "kv cache total_values",
            &[2, self.values_per_all_key_tensors()?],
        )
    }

    pub fn bytes_per_value(&self) -> usize {
        match self.dtype {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::Q4 | DType::Q8 => 1,
        }
    }

    pub fn total_bytes(&self) -> Result<usize> {
        checked_product(
            "kv cache total_bytes",
            &[self.total_values()?, self.bytes_per_value()],
        )
    }

    pub fn layer_position_offset(&self, layer: usize, position: usize) -> Result<usize> {
        if layer >= self.num_layers {
            return Err(runtime_err(format!(
                "kv cache layer {layer} out of range for {} layers",
                self.num_layers
            )));
        }
        if position >= self.capacity_tokens {
            return Err(runtime_err(format!(
                "kv cache position {position} out of range for capacity {}",
                self.capacity_tokens
            )));
        }
        let layer_base = checked_product(
            "kv cache layer offset",
            &[layer, self.values_per_layer_tensor()?],
        )?;
        let position_base = checked_product(
            "kv cache position offset",
            &[position, self.values_per_position()?],
        )?;
        layer_base.checked_add(position_base).ok_or_else(|| {
            runtime_err("kv cache layer position offset overflowed usize".to_string())
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PagedKvCacheLayout {
    pub base: KvCacheLayout,
    pub page_size_tokens: usize,
    pub physical_pages: usize,
}

impl PagedKvCacheLayout {
    pub fn new(
        base: KvCacheLayout,
        page_size_tokens: usize,
        physical_pages: usize,
    ) -> Result<Self> {
        if page_size_tokens == 0 {
            return Err(runtime_err("paged kv page_size_tokens must be > 0"));
        }
        if physical_pages == 0 {
            return Err(runtime_err("paged kv physical_pages must be > 0"));
        }
        let layout = Self {
            base,
            page_size_tokens,
            physical_pages,
        };
        let required = layout.required_pages_for_tokens(layout.base.capacity_tokens)?;
        if required > physical_pages {
            return Err(runtime_err(format!(
                "paged kv capacity requires {required} pages but allocator has {physical_pages}"
            )));
        }
        Ok(layout)
    }

    pub fn required_pages_for_tokens(&self, tokens: usize) -> Result<usize> {
        if tokens == 0 {
            return Ok(0);
        }
        tokens
            .checked_add(self.page_size_tokens - 1)
            .and_then(|v| v.checked_div(self.page_size_tokens))
            .ok_or_else(|| runtime_err("paged kv page count overflowed usize".to_string()))
    }

    pub fn logical_page_and_offset(&self, position: usize) -> Result<(usize, usize)> {
        if position >= self.base.capacity_tokens {
            return Err(runtime_err(format!(
                "paged kv position {position} out of range for capacity {}",
                self.base.capacity_tokens
            )));
        }
        Ok((
            position / self.page_size_tokens,
            position % self.page_size_tokens,
        ))
    }

    pub fn values_per_page_tensor(&self) -> Result<usize> {
        checked_product(
            "paged kv values_per_page_tensor",
            &[self.page_size_tokens, self.base.values_per_position()?],
        )
    }

    pub fn validate_page_table(&self, table: &[usize]) -> Result<()> {
        let required = self.required_pages_for_tokens(self.base.capacity_tokens)?;
        if table.len() < required {
            return Err(runtime_err(format!(
                "paged kv page table has {} pages but capacity requires {required}",
                table.len()
            )));
        }
        let mut seen = std::collections::HashSet::with_capacity(table.len());
        for &page in table {
            if page >= self.physical_pages {
                return Err(runtime_err(format!(
                    "paged kv physical page {page} out of range for {} pages",
                    self.physical_pages
                )));
            }
            if !seen.insert(page) {
                return Err(runtime_err(format!(
                    "paged kv physical page {page} appears more than once"
                )));
            }
        }
        Ok(())
    }
}

pub trait KvCacheStore {
    fn layout(&self) -> &KvCacheLayout;
    fn len_tokens(&self) -> usize;
    fn set_len_tokens(&mut self, len_tokens: usize) -> Result<()>;
    fn write_layer_kv(
        &mut self,
        layer: usize,
        position: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<()>;
    fn read_layer_keys(&self, layer: usize, len_tokens: usize, out: &mut [f32]) -> Result<()>;
    fn read_layer_values(&self, layer: usize, len_tokens: usize, out: &mut [f32]) -> Result<()>;
}

fn checked_product(label: &str, dims: &[usize]) -> Result<usize> {
    dims.iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| runtime_err(format!("{label} overflowed usize for dims {dims:?}")))
}

fn runtime_err(message: impl Into<String>) -> OcelotlError {
    OcelotlError::Runtime(RuntimeError {
        message: message.into(),
    })
}

// ---------------------------------------------------------------------------
// Domain newtypes (M1.2)
// ---------------------------------------------------------------------------

/// A vocabulary token identifier. Wraps `u32` to prevent mixing with other integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TokenId(pub u32);

/// Length of a sequence (number of tokens).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SeqLen(pub usize);

/// Number of concurrent requests in a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BatchSize(pub usize);

/// Maximum context length supported by a model or requested by a caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContextLen(pub usize);

// ---------------------------------------------------------------------------
// Model metadata (M1.3)
// ---------------------------------------------------------------------------

/// Full model configuration as loaded from a metadata artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub architecture: String,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub context_length: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub dtype: DType,
    #[serde(default)]
    pub tokenizer_model_hint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationOptions {
    pub max_new_tokens: usize,
    pub temperature: Option<f32>,
}

/// The result of a generation call: the tokens the runtime produced, in
/// emission order. Token-level (not text-level) because detokenization is the
/// caller's job — the runtime owns no tokenizer. The server crate maps this
/// to JSON; the loader and kernel crates don't import it. Lives in
/// `ocelotl-core` so every consumer agrees on the shape without depending on
/// the runtime crate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub tokens: Vec<TokenId>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Test-fixture helpers (M1.5)
// ---------------------------------------------------------------------------

/// Test-only helpers shared across crates. Gated behind the `test-fixtures`
/// feature so the helpers stay out of the production API surface.
///
/// Enable from another crate's `[dev-dependencies]`:
/// ```toml
/// ocelotl-core = { workspace = true, features = ["test-fixtures"] }
/// ```
#[cfg(any(test, feature = "test-fixtures"))]
pub mod test_fixtures {
    use std::path::PathBuf;

    /// Resolve a fixture under `<repo>/fixtures/metadata/` by name.
    ///
    /// Both `ocelotl-core` and `ocelotl-loader` live at `crates/<name>/` and
    /// therefore share the same `../../fixtures/metadata/` relative path from
    /// their `CARGO_MANIFEST_DIR`. `CARGO_MANIFEST_DIR` here resolves to the
    /// `ocelotl-core` crate, but the absolute target path is identical for
    /// any sibling crate at the same depth.
    pub fn metadata_fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/metadata")
            .join(name)
    }
}
