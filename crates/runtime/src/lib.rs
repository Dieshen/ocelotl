//! Request lifecycle and generation runtime.
//!
//! Family-specific entry points live in submodules and are intentionally NOT
//! re-exported at the crate root. Call sites use the family namespace
//! explicitly:
//!
//! - `ocelotl_runtime::qwen::prefill`, `decode_one_token`, cache helpers,
//!   `generate_qwen_batch`, `QwenGreedyModel`, `qwen2_5_kv_layout`.
//! - `ocelotl_runtime::gemma::prefill`, `decode_one_token` for the MF.7
//!   Gemma4 text-only synthetic subset.
//! - `ocelotl_runtime::whisper::transcribe`, `prepare_whisper_transcription`,
//!   `plan_transcription_chunks`, etc.
//!
//! Generic primitives (`greedy_sample`, KV cache structs, scheduler) live at
//! the crate root.

pub mod gemma;
mod kv_cache;
pub mod qwen;
mod sampling;
mod scheduler;
pub mod whisper;

use std::sync::Arc;

pub use kv_cache::{ContiguousKvCache, PagedKvCache, PagedKvCacheAllocator};
pub use sampling::greedy_sample;
pub use scheduler::{
    ContinuousBatchScheduler, GreedyDecodeModel, ScheduledGenerationRequest,
    ScheduledGenerationResponse, SchedulerConfig, SchedulerEvent, SchedulerRequestState,
};

use ocelotl_core::Result;
#[cfg(feature = "cubecl-wgpu")]
use ocelotl_kernels::CubeClKernelBackend;
use ocelotl_kernels::{CpuKernelBackend, KernelBackend};
use ocelotl_models::{
    gemma::{Gemma4Config, Gemma4TextModel, Gemma4TextWeights},
    qwen::{Qwen2_5Config, Qwen2_5Model, Qwen2_5Weights},
};

/// Convenience builder that pairs a kernel backend with the factories that
/// construct real-path models against it. Backed by `CpuKernelBackend` by
/// default; `cubecl_wgpu` opens a CubeCL/WGPU-backed builder when the
/// `cubecl-wgpu` feature is on.
pub struct Runtime<B: KernelBackend = CpuKernelBackend> {
    backend: B,
}

impl Runtime<CpuKernelBackend> {
    pub fn cpu() -> Self {
        Self {
            backend: CpuKernelBackend::default(),
        }
    }

    pub fn optimized_cpu() -> Self {
        Self {
            backend: CpuKernelBackend::optimized(),
        }
    }

    pub fn qwen2_5_model(
        &self,
        config: Qwen2_5Config,
        weights: Qwen2_5Weights,
    ) -> Result<Qwen2_5Model> {
        Qwen2_5Model::with_kernel_backend(config, weights, Arc::new(self.backend.clone()))
    }

    pub fn gemma4_text_model(
        &self,
        config: Gemma4Config,
        weights: Gemma4TextWeights,
    ) -> Result<Gemma4TextModel> {
        Gemma4TextModel::with_kernel_backend(config, weights, Arc::new(self.backend.clone()))
    }
}

#[cfg(feature = "cubecl-wgpu")]
impl Runtime<CubeClKernelBackend> {
    pub fn cubecl_wgpu(ordinal: usize) -> Self {
        Self {
            backend: CubeClKernelBackend::new_gpu(ordinal),
        }
    }

    pub fn qwen2_5_model(
        &self,
        config: Qwen2_5Config,
        weights: Qwen2_5Weights,
    ) -> Result<Qwen2_5Model> {
        Qwen2_5Model::with_kernel_backend(config, weights, Arc::new(self.backend.clone()))
    }

    pub fn gemma4_text_model(
        &self,
        config: Gemma4Config,
        weights: Gemma4TextWeights,
    ) -> Result<Gemma4TextModel> {
        Gemma4TextModel::with_kernel_backend(config, weights, Arc::new(self.backend.clone()))
    }
}

impl<B: KernelBackend> Runtime<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use ocelotl_core::{DType, KvCacheStore, OcelotlError, TokenId};

    #[derive(Debug)]
    struct CountingAttentionBackend {
        inner: CpuKernelBackend,
        full_attention_calls: Arc<AtomicUsize>,
        incremental_attention_calls: Arc<AtomicUsize>,
    }

    impl KernelBackend for CountingAttentionBackend {
        fn name(&self) -> &'static str {
            self.inner.name()
        }

        fn context(&self) -> &ocelotl_kernels::KernelContext {
            self.inner.context()
        }

        fn matmul(
            &self,
            a: &[f32],
            a_shape: (usize, usize),
            b: &[f32],
            b_shape: (usize, usize),
            out: &mut [f32],
        ) -> ocelotl_core::Result<()> {
            self.inner.matmul(a, a_shape, b, b_shape, out)
        }

        #[allow(clippy::too_many_arguments)]
        fn linear_out_by_in(
            &self,
            x: &[f32],
            rows: usize,
            in_features: usize,
            weight_out_by_in: &[f32],
            out_features: usize,
            bias: Option<&[f32]>,
            out: &mut [f32],
        ) -> ocelotl_core::Result<()> {
            self.inner.linear_out_by_in(
                x,
                rows,
                in_features,
                weight_out_by_in,
                out_features,
                bias,
                out,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn scaled_dot_product_attention(
            &self,
            q: &[f32],
            k: &[f32],
            v: &[f32],
            seq_len: usize,
            num_q_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            out: &mut [f32],
        ) -> ocelotl_core::Result<()> {
            self.full_attention_calls.fetch_add(1, Ordering::Relaxed);
            self.inner.scaled_dot_product_attention(
                q,
                k,
                v,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                out,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn scaled_dot_product_attention_incremental(
            &self,
            q: &[f32],
            k: &[f32],
            v: &[f32],
            seq_len: usize,
            num_q_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            out: &mut [f32],
        ) -> ocelotl_core::Result<()> {
            self.incremental_attention_calls
                .fetch_add(1, Ordering::Relaxed);
            self.inner.scaled_dot_product_attention_incremental(
                q,
                k,
                v,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                out,
            )
        }

        fn rope_apply_inplace(
            &self,
            x: &mut [f32],
            head_dim: usize,
            position: usize,
            theta: f32,
        ) -> ocelotl_core::Result<()> {
            self.inner.rope_apply_inplace(x, head_dim, position, theta)
        }

        fn rmsnorm(
            &self,
            x: &[f32],
            rows: usize,
            hidden: usize,
            weight: &[f32],
            epsilon: f32,
            out: &mut [f32],
        ) -> ocelotl_core::Result<()> {
            self.inner.rmsnorm(x, rows, hidden, weight, epsilon, out)
        }

        #[allow(clippy::too_many_arguments)]
        fn mlp_gated_silu(
            &self,
            x: &[f32],
            rows: usize,
            hidden: usize,
            intermediate: usize,
            gate_w: &[f32],
            up_w: &[f32],
            down_w: &[f32],
            gate_buf: &mut [f32],
            up_buf: &mut [f32],
            out: &mut [f32],
        ) -> ocelotl_core::Result<()> {
            self.inner.mlp_gated_silu(
                x,
                rows,
                hidden,
                intermediate,
                gate_w,
                up_w,
                down_w,
                gate_buf,
                up_buf,
                out,
            )
        }

        fn vec_add(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> ocelotl_core::Result<()> {
            self.inner.vec_add(a, b, out)
        }
    }

    // Pull in family-specific entry points the tests exercise. They live in
    // submodules now; importing them once here keeps every test below readable.
    use super::qwen::{
        decode_one_token, decode_one_token_with_contiguous_cache,
        decode_one_token_with_paged_cache, prefill, prepare_qwen2_5_contiguous_cache,
        prepare_qwen2_5_paged_cache, qwen2_5_kv_layout,
    };

    fn tiny_qwen_config_and_weights() -> (Qwen2_5Config, Qwen2_5Weights) {
        use ocelotl_models::qwen::{Qwen2_5LayerWeights, transpose_2d};

        let cfg = Qwen2_5Config {
            vocab_size: 8,
            num_hidden_layers: 1,
            hidden_size: 4,
            intermediate_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            context_length: 16,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            dtype: DType::F32,
        };
        let h = cfg.hidden_size;
        let v = cfg.vocab_size;
        let q_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        let i_size = cfg.intermediate_size;
        let embed: Vec<f32> = (0..v * h).map(|i| (i as f32) * 0.01).collect();
        let lm_head_w = transpose_2d(&embed, v, h);
        let weights = Qwen2_5Weights {
            embed_tokens: embed,
            layers: vec![Qwen2_5LayerWeights {
                q_proj_w: vec![0.01; h * q_out],
                q_proj_b: vec![0.0; q_out],
                k_proj_w: vec![0.01; h * kv_out],
                k_proj_b: vec![0.0; kv_out],
                v_proj_w: vec![0.01; h * kv_out],
                v_proj_b: vec![0.0; kv_out],
                o_proj_w: vec![0.01; q_out * h],
                input_layernorm_w: vec![1.0; h],
                post_attention_layernorm_w: vec![1.0; h],
                gate_proj_w: vec![0.01; h * i_size],
                up_proj_w: vec![0.01; h * i_size],
                down_proj_w: vec![0.01; i_size * h],
            }],
            final_norm_w: vec![1.0; h],
            lm_head_w,
            tie_word_embeddings: true,
        };

        (cfg, weights)
    }

    fn tiny_gemma4_text_config_and_weights() -> (Gemma4Config, Gemma4TextWeights) {
        use ocelotl_models::gemma::{Gemma4Quantization, Gemma4TextLayerWeights};

        let cfg = Gemma4Config {
            context_length: 16,
            block_count: 1,
            embedding_length: 4,
            embedding_length_per_layer_input: 4,
            feed_forward_length: 8,
            attention_head_count: 2,
            attention_head_count_kv: 1,
            attention_key_length: 2,
            attention_value_length: 2,
            attention_key_length_swa: 2,
            attention_value_length_swa: 2,
            rope_dimension_count: 2,
            rope_dimension_count_swa: 2,
            rope_freq_base: 10_000.0,
            rope_freq_base_swa: 10_000.0,
            rms_norm_eps: 1e-6,
            attention_sliding_window: None,
            attention_shared_kv_layers: None,
            attention_sliding_window_pattern: None,
            final_logit_softcap: None,
            tokenizer_model: Some("gemma4".to_string()),
            tokenizer_token_count: 8,
            quantization: Gemma4Quantization::Unquantized,
            has_quantized_tensors: false,
            tensor_count: 23,
            multimodal: false,
        };
        let h = cfg.embedding_length;
        let v = cfg.tokenizer_token_count;
        let q_out = cfg.attention_head_count * cfg.attention_key_length;
        let kv_out = cfg.attention_head_count_kv * cfg.attention_key_length;
        let f = cfg.feed_forward_length;
        let epl = cfg.embedding_length_per_layer_input;
        let per_layer_width = cfg.block_count * epl;
        let token_embd: Vec<f32> = (0..v * h).map(|i| (i as f32) * 0.01).collect();
        let mut lm_head_w = vec![0.0_f32; h * v];
        for token in 0..v {
            for feature in 0..h {
                lm_head_w[feature * v + token] = token_embd[token * h + feature];
            }
        }
        let weights = Gemma4TextWeights {
            token_embd,
            layers: vec![Gemma4TextLayerWeights {
                attn_norm_w: vec![1.0; h],
                attn_q_w: vec![0.01; h * q_out],
                attn_k_w: vec![0.01; h * kv_out],
                attn_v_w: vec![0.01; h * kv_out],
                attn_o_w: vec![0.01; q_out * h],
                attn_q_norm_w: vec![1.0; cfg.attention_key_length],
                attn_k_norm_w: vec![1.0; cfg.attention_key_length],
                ffn_norm_w: vec![1.0; h],
                ffn_gate_w: vec![0.01; h * f],
                ffn_up_w: vec![0.01; h * f],
                ffn_down_w: vec![0.01; f * h],
                attn_post_norm_w: vec![1.0; h],
                ffn_post_norm_w: vec![1.0; h],
                per_layer_inp_gate_w: vec![0.0; h * epl],
                per_layer_proj_w: vec![0.0; epl * h],
                per_layer_post_norm_w: vec![1.0; h],
                layer_output_scale: 1.0,
            }],
            per_layer_token_embd: vec![0.0; v * per_layer_width],
            per_layer_model_proj_w: vec![0.0; h * per_layer_width],
            per_layer_proj_norm_w: vec![1.0; epl],
            rope_freqs: vec![1.0; cfg.rope_dimension_count / 2],
            output_norm_w: vec![1.0; h],
            lm_head_w,
            tie_word_embeddings: true,
        };

        (cfg, weights)
    }

    #[test]
    fn optimized_cpu_runtime_selects_optimized_kernel_backend() {
        let runtime = Runtime::optimized_cpu();

        assert_eq!(
            runtime.backend().mode(),
            ocelotl_kernels::CpuKernelMode::Optimized
        );
    }

    #[test]
    fn cpu_runtime_builds_qwen_model_with_selected_cpu_backend() {
        let runtime = Runtime::optimized_cpu();
        let (cfg, weights) = tiny_qwen_config_and_weights();

        let model = runtime
            .qwen2_5_model(cfg, weights)
            .expect("runtime should construct Qwen model");

        assert_eq!(model.kernel_backend().name(), "cpu");
        assert_eq!(
            model.execution_backend().context().device,
            ocelotl_core::Device::Cpu
        );
    }

    #[test]
    fn cpu_runtime_builds_gemma4_text_model_with_selected_cpu_backend() {
        let runtime = Runtime::optimized_cpu();
        let (cfg, weights) = tiny_gemma4_text_config_and_weights();

        let model = runtime
            .gemma4_text_model(cfg, weights)
            .expect("runtime should construct Gemma4 text model");

        assert_eq!(model.kernel_backend().name(), "cpu");
        assert_eq!(
            model.execution_backend().context().device,
            ocelotl_core::Device::Cpu
        );
    }

    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn cubecl_wgpu_runtime_builds_qwen_model_with_gpu_execution_backend_without_launch() {
        let runtime = Runtime::cubecl_wgpu(0);
        let (cfg, weights) = tiny_qwen_config_and_weights();

        let model = runtime
            .qwen2_5_model(cfg, weights)
            .expect("runtime should construct CubeCL-backed Qwen model");

        assert_eq!(model.execution_backend().name(), "cubecl");
        assert_eq!(
            model.execution_backend().context().device,
            ocelotl_core::Device::Gpu { ordinal: 0 }
        );
    }

    #[cfg(feature = "cubecl-wgpu")]
    #[test]
    fn cubecl_wgpu_runtime_builds_gemma4_text_model_with_gpu_backend_without_launch() {
        let runtime = Runtime::cubecl_wgpu(0);
        let (cfg, weights) = tiny_gemma4_text_config_and_weights();

        let model = runtime
            .gemma4_text_model(cfg, weights)
            .expect("runtime should construct CubeCL-backed Gemma4 text model");

        assert_eq!(model.execution_backend().name(), "cubecl");
        assert_eq!(
            model.execution_backend().context().device,
            ocelotl_core::Device::Gpu { ordinal: 0 }
        );
    }

    // --- prefill (M3.7 wiring) ---

    #[test]
    fn prefill_returns_logits_through_runtime_api_surface() {
        // M3.7 contract: the runtime exposes prefill on the public API
        // so callers do not bypass the runtime layer to reach the model.
        // We keep the model construction inline here because the runtime
        // crate doesn't own model fixtures; the goal of this test is to
        // pin the API shape, not to re-validate prefill numerics (that's
        // pinned in the models crate's tiny-synthetic integration test).
        use ocelotl_models::qwen::{
            Qwen2_5Config, Qwen2_5LayerWeights, Qwen2_5Model, Qwen2_5Weights, transpose_2d,
        };
        let cfg = Qwen2_5Config {
            vocab_size: 8,
            num_hidden_layers: 1,
            hidden_size: 4,
            intermediate_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            context_length: 16,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            dtype: DType::F32,
        };
        let h = cfg.hidden_size;
        let v = cfg.vocab_size;
        let q_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        let i_size = cfg.intermediate_size;
        let embed: Vec<f32> = (0..v * h).map(|i| (i as f32) * 0.01).collect();
        let lm_head_w = transpose_2d(&embed, v, h);
        let weights = Qwen2_5Weights {
            embed_tokens: embed,
            layers: vec![Qwen2_5LayerWeights {
                q_proj_w: vec![0.01; h * q_out],
                q_proj_b: vec![0.0; q_out],
                k_proj_w: vec![0.01; h * kv_out],
                k_proj_b: vec![0.0; kv_out],
                v_proj_w: vec![0.01; h * kv_out],
                v_proj_b: vec![0.0; kv_out],
                o_proj_w: vec![0.01; q_out * h],
                input_layernorm_w: vec![1.0; h],
                post_attention_layernorm_w: vec![1.0; h],
                gate_proj_w: vec![0.01; h * i_size],
                up_proj_w: vec![0.01; h * i_size],
                down_proj_w: vec![0.01; i_size * h],
            }],
            final_norm_w: vec![1.0; h],
            lm_head_w,
            tie_word_embeddings: true,
        };
        let model = Qwen2_5Model::new(cfg, weights).expect("tiny model must construct");

        let logits = prefill(&model, &[TokenId(1), TokenId(2)])
            .expect("prefill via runtime API must succeed");

        assert_eq!(logits.len(), 8);
        for v in &logits {
            assert!(v.is_finite(), "prefill logits must be finite");
        }
    }

    #[test]
    fn prefill_propagates_invalid_request_for_empty_prompt() {
        // A runtime-level public API must surface model-level validation
        // failures verbatim. An empty prompt is InvalidRequest at the
        // model boundary (Qwen2_5Model::prefill); the runtime wrapper
        // must not swallow or remap it.
        use ocelotl_models::qwen::{
            Qwen2_5Config, Qwen2_5LayerWeights, Qwen2_5Model, Qwen2_5Weights, transpose_2d,
        };
        let cfg = Qwen2_5Config {
            vocab_size: 4,
            num_hidden_layers: 1,
            hidden_size: 2,
            intermediate_size: 4,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            context_length: 8,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            dtype: DType::F32,
        };
        let h = cfg.hidden_size;
        let v = cfg.vocab_size;
        let q_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        let i_size = cfg.intermediate_size;
        let embed: Vec<f32> = vec![0.1; v * h];
        let lm_head_w = transpose_2d(&embed, v, h);
        let weights = Qwen2_5Weights {
            embed_tokens: embed,
            layers: vec![Qwen2_5LayerWeights {
                q_proj_w: vec![0.1; h * q_out],
                q_proj_b: vec![0.0; q_out],
                k_proj_w: vec![0.1; h * kv_out],
                k_proj_b: vec![0.0; kv_out],
                v_proj_w: vec![0.1; h * kv_out],
                v_proj_b: vec![0.0; kv_out],
                o_proj_w: vec![0.1; q_out * h],
                input_layernorm_w: vec![1.0; h],
                post_attention_layernorm_w: vec![1.0; h],
                gate_proj_w: vec![0.1; h * i_size],
                up_proj_w: vec![0.1; h * i_size],
                down_proj_w: vec![0.1; i_size * h],
            }],
            final_norm_w: vec![1.0; h],
            lm_head_w,
            tie_word_embeddings: true,
        };
        let model = Qwen2_5Model::new(cfg, weights).unwrap();

        let err = prefill(&model, &[]).expect_err("empty prompt must be rejected");

        match err {
            OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "tokens");
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    // --- decode_one_token (M3.8 wiring) ---

    /// Build the same tiny Qwen2.5 model used by `prefill_returns_logits_through_runtime_api_surface`.
    /// Kept inline rather than extracted to a helper to keep these tests
    /// self-contained: the runtime crate doesn't own model fixtures, and a
    /// helper would only have the two callers in this module.
    fn tiny_model_for_decode() -> ocelotl_models::qwen::Qwen2_5Model {
        use ocelotl_models::qwen::{
            Qwen2_5Config, Qwen2_5LayerWeights, Qwen2_5Model, Qwen2_5Weights, transpose_2d,
        };
        let cfg = Qwen2_5Config {
            vocab_size: 8,
            num_hidden_layers: 1,
            hidden_size: 4,
            intermediate_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            context_length: 16,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            dtype: DType::F32,
        };
        let h = cfg.hidden_size;
        let v = cfg.vocab_size;
        let q_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        let i_size = cfg.intermediate_size;
        let embed: Vec<f32> = (0..v * h).map(|i| (i as f32) * 0.01).collect();
        let lm_head_w = transpose_2d(&embed, v, h);
        let weights = Qwen2_5Weights {
            embed_tokens: embed,
            layers: vec![Qwen2_5LayerWeights {
                q_proj_w: vec![0.01; h * q_out],
                q_proj_b: vec![0.0; q_out],
                k_proj_w: vec![0.01; h * kv_out],
                k_proj_b: vec![0.0; kv_out],
                v_proj_w: vec![0.01; h * kv_out],
                v_proj_b: vec![0.0; kv_out],
                o_proj_w: vec![0.01; q_out * h],
                input_layernorm_w: vec![1.0; h],
                post_attention_layernorm_w: vec![1.0; h],
                gate_proj_w: vec![0.01; h * i_size],
                up_proj_w: vec![0.01; h * i_size],
                down_proj_w: vec![0.01; i_size * h],
            }],
            final_norm_w: vec![1.0; h],
            lm_head_w,
            tie_word_embeddings: true,
        };
        Qwen2_5Model::new(cfg, weights).expect("tiny model must construct")
    }

    #[test]
    fn decode_one_token_returns_a_token_id_for_valid_prompt() {
        // Smallest meaningful contract for M3.8: decode_one_token returns
        // a TokenId that is within the model's vocabulary. Pins the API
        // shape (Vec<f32> -> TokenId) without yet asserting numerics; the
        // pinning test below covers the specific value.
        let model = tiny_model_for_decode();

        let token = decode_one_token(&model, &[TokenId(1), TokenId(2)])
            .expect("valid prompt must produce a token");

        assert!(
            (token.0 as usize) < model.config().vocab_size,
            "decoded token must be within vocabulary, got {token:?}",
        );
    }

    #[test]
    fn contiguous_cache_decode_matches_no_cache_decode_and_appends_token() {
        let model = tiny_model_for_decode();
        let prompt = [TokenId(1), TokenId(2)];
        let expected = decode_one_token(&model, &prompt).unwrap();

        let mut state =
            prepare_qwen2_5_contiguous_cache(&model, &prompt).expect("cached prefill must succeed");

        assert_eq!(state.cache().len_tokens(), prompt.len());
        assert_eq!(
            state.cache().key_at(0, 0).unwrap().len(),
            model.config().num_key_value_heads * model.config().head_dim
        );
        assert!(state.last_logits().iter().all(|v| v.is_finite()));

        let actual = decode_one_token_with_contiguous_cache(&model, &mut state)
            .expect("cached decode must succeed");

        assert_eq!(actual, expected);
        assert_eq!(state.cache().len_tokens(), prompt.len() + 1);
    }

    #[test]
    fn contiguous_cache_decode_uses_one_query_incremental_attention() {
        let (config, weights) = tiny_qwen_config_and_weights();
        let full_attention_calls = Arc::new(AtomicUsize::new(0));
        let incremental_attention_calls = Arc::new(AtomicUsize::new(0));
        let model = Qwen2_5Model::with_kernel_backend(
            config,
            weights,
            Arc::new(CountingAttentionBackend {
                inner: CpuKernelBackend::scalar(),
                full_attention_calls: Arc::clone(&full_attention_calls),
                incremental_attention_calls: Arc::clone(&incremental_attention_calls),
            }),
        )
        .expect("counting backend model must construct");

        let mut state =
            prepare_qwen2_5_contiguous_cache(&model, &[TokenId(1), TokenId(2), TokenId(3)])
                .expect("cached prefill must succeed");
        let full_after_prefill = full_attention_calls.load(Ordering::Relaxed);

        decode_one_token_with_contiguous_cache(&model, &mut state)
            .expect("cached decode must succeed");

        assert_eq!(
            full_attention_calls.load(Ordering::Relaxed),
            full_after_prefill,
            "cached decode must not construct or execute full-sequence causal attention"
        );
        assert_eq!(
            incremental_attention_calls.load(Ordering::Relaxed),
            model.config().num_hidden_layers,
            "cached decode must execute one incremental attention row per layer"
        );
    }

    #[test]
    fn contiguous_cache_rejects_capacity_overflow_before_advancing_len() {
        let model = tiny_model_for_decode();
        let layout = qwen2_5_kv_layout(model.config(), 2).unwrap();
        let mut cache = ContiguousKvCache::new(layout).unwrap();

        let err = model
            .prefill_with_cache(&[TokenId(1), TokenId(2), TokenId(3)], &mut cache)
            .expect_err("prompt beyond cache capacity must be rejected");

        match err {
            OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "kv_cache.capacity");
            }
            other => panic!("expected InvalidRequest(kv_cache.capacity), got {other:?}"),
        }
        assert_eq!(cache.len_tokens(), 0);
    }

    #[test]
    fn contiguous_cache_release_clears_runtime_state_on_cancellation() {
        let model = tiny_model_for_decode();
        let mut state = prepare_qwen2_5_contiguous_cache(&model, &[TokenId(1), TokenId(2)])
            .expect("cached prefill must succeed");

        state.release();

        assert!(state.is_released());
        assert!(state.last_logits().is_empty());
    }

    #[test]
    fn paged_cache_decode_matches_no_cache_decode_across_page_boundary() {
        let model = tiny_model_for_decode();
        let prompt = [TokenId(1), TokenId(2), TokenId(3)];
        let expected = decode_one_token(&model, &prompt).unwrap();
        let mut allocator = PagedKvCacheAllocator::for_qwen2_5(model.config(), 4, 2, 3)
            .expect("allocator must construct");

        let mut state = prepare_qwen2_5_paged_cache(&model, &prompt, &mut allocator, 4)
            .expect("paged cached prefill must succeed");

        let cache = state.cache().expect("state must hold a cache");
        assert_eq!(cache.page_table(), &[0, 1]);
        assert_eq!(cache.physical_page_for_position(2).unwrap(), 1);
        assert_eq!(cache.len_tokens(), prompt.len());
        assert_eq!(allocator.free_page_count(), 1);

        let actual = decode_one_token_with_paged_cache(&model, &mut state)
            .expect("paged cached decode must succeed");

        assert_eq!(actual, expected);
        assert_eq!(
            state.cache().expect("cache remains live").len_tokens(),
            prompt.len() + 1
        );

        state.release_into(&mut allocator);
        assert!(state.is_released());
        assert_eq!(allocator.free_page_count(), 3);
    }

    #[test]
    fn paged_cache_releases_pages_when_prefill_fails_after_allocation() {
        let model = tiny_model_for_decode();
        let mut allocator = PagedKvCacheAllocator::for_qwen2_5(model.config(), 4, 2, 3)
            .expect("allocator must construct");

        let err = prepare_qwen2_5_paged_cache(&model, &[TokenId(99)], &mut allocator, 4)
            .expect_err("invalid token should fail prefill after allocation");

        match err {
            OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "tokens");
            }
            other => panic!("expected InvalidRequest(tokens), got {other:?}"),
        }
        assert_eq!(allocator.free_page_count(), 3);
    }
}
