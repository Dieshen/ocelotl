# Interface Sketches

This document sketches the initial Rust API surface Ocelotl should grow toward.
It is intentionally not a final API contract. The purpose is to reduce M1/M2
thrash by making crate boundaries concrete before implementation starts.

## Design Rules

- Public interfaces use Ocelotl-owned types from `ocelotl-core` where practical.
- Implementation crates may wrap external libraries, but should not leak them
  across crate boundaries.
- Runtime APIs should be stable enough for server and tests without exposing
  model or kernel internals.
- Kernel APIs should pass explicit shape, dtype, and layout data.
- Unsupported behavior returns typed errors, not `String`-only failures.

## Core Types

`ocelotl-core` should own stable shared types:

```rust
pub type Result<T> = std::result::Result<T, OcelotlError>;

pub enum OcelotlError {
    InvalidModel(InvalidModelError),
    Unsupported(UnsupportedError),
    InvalidRequest(InvalidRequestError),
    Tokenizer(TokenizerError),
    Kernel(KernelError),
    Runtime(RuntimeError),
    Io(IoError),
}

pub enum Device {
    Cpu,
    Gpu { ordinal: usize },
}

pub enum DType {
    F32,
    F16,
    BF16,
    Q4,
    Q8,
}

pub struct ModelInfo {
    pub architecture: Architecture,
    pub context_length: usize,
    pub dtype: DType,
    pub layers: LayerInfo,
    pub attention: AttentionInfo,
}
```

M1 can keep simpler structs, but the direction should be typed categories rather
than unstructured strings.

## Loader Boundary

`ocelotl-loader` should normalize files into metadata and verified tensor access.
It should not construct runtime sessions.

```rust
pub trait ModelLoader: Send + Sync {
    fn inspect(&self, source: &ModelSource) -> ocelotl_core::Result<ModelManifest>;
    fn open(&self, source: &ModelSource) -> ocelotl_core::Result<ModelArtifact>;
}

pub struct ModelSource {
    pub path: std::path::PathBuf,
}

pub struct ModelManifest {
    pub info: ModelInfo,
    pub tokenizer: Option<TokenizerSource>,
    pub tensors: Vec<TensorManifest>,
}

pub struct ModelArtifact {
    pub manifest: ModelManifest,
    pub tensors: Box<dyn TensorStore>,
}

pub trait TensorStore: Send + Sync {
    fn tensor(&self, name: &str) -> ocelotl_core::Result<TensorView<'_>>;
}
```

`TensorView` should keep dtype, shape, and data together so model code cannot
accidentally skip validation.

## Tokenizer Boundary

`ocelotl-tokenizer` should expose Ocelotl token IDs and hide Hugging Face
`tokenizers` internals.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenId(pub u32);

pub trait Tokenizer: Send + Sync {
    fn encode(&self, input: EncodeInput<'_>) -> ocelotl_core::Result<Vec<TokenId>>;
    fn decode(&self, tokens: &[TokenId], options: DecodeOptions) -> ocelotl_core::Result<String>;
}

pub enum EncodeInput<'a> {
    Text(&'a str),
    Messages(&'a [ChatMessage]),
}

pub struct DecodeOptions {
    pub skip_special_tokens: bool,
}
```

Chat-template rendering should be explicit and fixture-tested.

## Kernel Boundary

`ocelotl-kernels` should own backend dispatch. Model and runtime code should
call Ocelotl operations, not CubeCL, CubeK, Burn, or vendor APIs directly.

```rust
pub trait KernelBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn device(&self) -> Device;
    fn capabilities(&self) -> KernelCapabilities;
}

pub trait MatmulKernel: KernelBackend {
    fn matmul(&self, problem: MatmulProblem<'_>) -> ocelotl_core::Result<()>;
}

pub trait AttentionKernel: KernelBackend {
    fn prefill_attention(&self, problem: AttentionProblem<'_>) -> ocelotl_core::Result<()>;
    fn decode_attention(&self, problem: DecodeAttentionProblem<'_>) -> ocelotl_core::Result<()>;
}

pub trait KvCacheKernel: KernelBackend {
    fn write_kv(&self, problem: KvWriteProblem<'_>) -> ocelotl_core::Result<()>;
    fn read_kv(&self, problem: KvReadProblem<'_>) -> ocelotl_core::Result<()>;
}
```

Problem structs should include explicit shape, stride, dtype, and layout. Avoid
implicit global layout assumptions.

## Model Boundary

`ocelotl-models` owns architecture semantics. It consumes verified metadata and
kernel traits. Keep the concrete family types first (`Qwen2_5Model`,
`WhisperModel`, future Gemma types); do not introduce a shared text-generation
trait until at least two real text families need the same runtime call shape.

The model should not allocate scheduler state or hide downloads. Local
family-level load helpers may compose `ocelotl-loader` with family-specific
tensor mapping, but file format parsing and network fetches stay outside the
model crate.

## Runtime Boundary

`ocelotl-runtime` owns request lifecycle and resource ownership.

```rust
pub struct Runtime {
    // model, tokenizer, kernels, cache allocator, scheduler policy
}

impl Runtime {
    pub fn load(config: RuntimeConfig) -> ocelotl_core::Result<Self>;
    pub fn generate(&self, request: GenerateRequest) -> ocelotl_core::Result<GenerateResponse>;
    pub fn generate_stream(&self, request: GenerateRequest) -> TokenStream;
}

pub struct GenerateRequest {
    pub input: PromptInput,
    pub options: GenerationOptions,
}

pub struct GenerateResponse {
    pub text: String,
    pub token_ids: Vec<TokenId>,
    pub finish_reason: FinishReason,
}
```

M1 can use synchronous generation. Async streaming belongs later, once resource
cleanup and cancellation are explicit.

## Server Boundary

`ocelotl-server` adapts transport to runtime.

```rust
pub trait RuntimeHandle: Send + Sync {
    fn generate(&self, request: GenerateRequest) -> ocelotl_core::Result<GenerateResponse>;
}

pub struct ServerConfig {
    pub bind_addr: String,
}
```

Server tests should use mock runtime handles before real models.

## M1 Minimum

The first implementation does not need every trait above. It should at least
avoid painting the project into a corner:

- Keep runtime request/response types Ocelotl-owned.
- Keep tokenizer output as `TokenId`, not raw external types.
- Keep model metadata typed enough to validate shape/dtype/context.
- Keep kernel inputs explicit even for CPU reference code.
