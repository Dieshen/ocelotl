//! Qwen2.5 prefill: full-prompt forward returning final-token logits.
//!
//! Composes (per layer):
//! `RMSNorm -> q/k/v projections + biases -> RoPE on Q and K ->
//!  scaled-dot-product attention -> o_proj -> residual add ->
//!  RMSNorm -> gated SiLU MLP -> residual add`.
//! Then a final RMSNorm and the lm_head projection on the last position only.
//!
//! When a `KvCacheStore` is supplied, post-RoPE K/V are written into the
//! cache per layer. Cache length is updated only after the full prefill
//! succeeds, so capacity errors cannot leave a partially advanced request.

use ocelotl_core::{InvalidRequestError, KvCacheStore, OcelotlError, Result, TokenId};

use super::model::{Qwen2_5Model, add_bias_per_row, validate_cache_layout_for_model};

impl Qwen2_5Model {
    /// Run prefill over the prompt and return the logits at the final
    /// position over the vocabulary.
    ///
    /// Returns a `Vec<f32>` of length `vocab_size`.
    ///
    /// # Errors
    ///
    /// - `InvalidRequest` if `tokens` is empty.
    /// - `InvalidRequest` if any token id is `>= vocab_size`.
    /// - `InvalidRequest` if `tokens.len() > config.context_length`.
    /// - Propagates `KernelError` from the underlying kernels for
    ///   shape/length violations (these should be unreachable when the
    ///   `Qwen2_5Model::new` length checks have run).
    pub fn prefill(&self, tokens: &[TokenId]) -> Result<Vec<f32>> {
        self.prefill_impl(tokens, None)
    }

    /// Run prefill and write each layer's post-RoPE K/V tensors into a
    /// caller-owned cache.
    ///
    /// The cache is runtime-owned but model-written. Its layout must match the
    /// Qwen2.5 config exactly. Cache length is updated only after the full
    /// prefill succeeds, so capacity errors cannot leave a partially advanced
    /// request state.
    pub fn prefill_with_cache(
        &self,
        tokens: &[TokenId],
        cache: &mut dyn KvCacheStore,
    ) -> Result<Vec<f32>> {
        validate_cache_layout_for_model(&self.config, cache.layout(), tokens.len())?;
        cache.set_len_tokens(0)?;
        self.prefill_impl(tokens, Some(cache))
    }

    fn prefill_impl(
        &self,
        tokens: &[TokenId],
        mut cache: Option<&mut dyn KvCacheStore>,
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(OcelotlError::InvalidRequest(InvalidRequestError {
                field: "tokens".to_string(),
                message: "Qwen2_5Model::prefill requires at least one token".to_string(),
            }));
        }
        let cfg = &self.config;
        if tokens.len() > cfg.context_length {
            return Err(OcelotlError::InvalidRequest(InvalidRequestError {
                field: "tokens".to_string(),
                message: format!(
                    "prompt length {} exceeds context_length {}",
                    tokens.len(),
                    cfg.context_length,
                ),
            }));
        }
        for (idx, t) in tokens.iter().enumerate() {
            if (t.0 as usize) >= cfg.vocab_size {
                return Err(OcelotlError::InvalidRequest(InvalidRequestError {
                    field: "tokens".to_string(),
                    message: format!(
                        "token id {} at position {} is out of range for vocab_size {}",
                        t.0, idx, cfg.vocab_size,
                    ),
                }));
            }
        }

        let seq = tokens.len();
        let h = cfg.hidden_size;
        let q_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        let i_size = cfg.intermediate_size;
        let v = cfg.vocab_size;
        let eps = cfg.rms_norm_eps as f32;
        let theta = cfg.rope_theta as f32;

        // Step 1: token embedding lookup.
        // hidden has shape [seq, hidden_size]. embed_tokens is laid out as
        // [vocab, hidden] row-major; row `t` is the embedding for token id t.
        // Inlined rather than introducing a kernel: it's a copy of one row
        // per token, no math, no shape decisions.
        let mut hidden = vec![0.0_f32; seq * h];
        for (t_idx, tok) in tokens.iter().enumerate() {
            let src = (tok.0 as usize) * h;
            let dst = t_idx * h;
            hidden[dst..dst + h].copy_from_slice(&self.weights.embed_tokens[src..src + h]);
        }

        // Scratch buffers reused across layers/projections.
        let mut norm_buf = vec![0.0_f32; seq * h];
        let mut q_buf = vec![0.0_f32; seq * q_out];
        let mut k_buf = vec![0.0_f32; seq * kv_out];
        let mut v_buf = vec![0.0_f32; seq * kv_out];
        let mut attn_out = vec![0.0_f32; seq * q_out];
        let mut o_buf = vec![0.0_f32; seq * h];
        let mut residual_buf = vec![0.0_f32; seq * h];
        let mut gate_buf = vec![0.0_f32; seq * i_size];
        let mut up_buf = vec![0.0_f32; seq * i_size];
        let mut mlp_out = vec![0.0_f32; seq * h];

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            // Save residual before attention.
            residual_buf.copy_from_slice(&hidden);

            // Pre-attention RMSNorm.
            self.kernels.rmsnorm(
                &hidden,
                seq,
                h,
                &layer.input_layernorm_w,
                eps,
                &mut norm_buf,
            )?;

            // Q = norm @ q_proj_w  ([seq,h] @ [h,q_out])
            self.kernels
                .matmul(&norm_buf, (seq, h), &layer.q_proj_w, (h, q_out), &mut q_buf)?;
            add_bias_per_row(&mut q_buf, &layer.q_proj_b, seq, q_out);

            // K = norm @ k_proj_w  ([seq,h] @ [h,kv_out])
            self.kernels.matmul(
                &norm_buf,
                (seq, h),
                &layer.k_proj_w,
                (h, kv_out),
                &mut k_buf,
            )?;
            add_bias_per_row(&mut k_buf, &layer.k_proj_b, seq, kv_out);

            // V = norm @ v_proj_w  ([seq,h] @ [h,kv_out])
            self.kernels.matmul(
                &norm_buf,
                (seq, h),
                &layer.v_proj_w,
                (h, kv_out),
                &mut v_buf,
            )?;
            add_bias_per_row(&mut v_buf, &layer.v_proj_b, seq, kv_out);

            // Apply RoPE to Q and K, per-position. Both have layout
            // [seq, num_heads, head_dim] flattened row-major; rope expects
            // a slice of `num_heads * head_dim` per position.
            for pos in 0..seq {
                let q_start = pos * q_out;
                self.kernels.rope_apply_inplace(
                    &mut q_buf[q_start..q_start + q_out],
                    cfg.head_dim,
                    pos,
                    theta,
                )?;
                let k_start = pos * kv_out;
                self.kernels.rope_apply_inplace(
                    &mut k_buf[k_start..k_start + kv_out],
                    cfg.head_dim,
                    pos,
                    theta,
                )?;
            }
            if let Some(cache) = &mut cache {
                for pos in 0..seq {
                    let start = pos * kv_out;
                    cache.write_layer_kv(
                        layer_idx,
                        pos,
                        &k_buf[start..start + kv_out],
                        &v_buf[start..start + kv_out],
                    )?;
                }
            }

            // Scaled-dot-product attention.
            self.kernels.scaled_dot_product_attention(
                &q_buf,
                &k_buf,
                &v_buf,
                seq,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                &mut attn_out,
            )?;

            // O = attn_out @ o_proj_w  ([seq,q_out] @ [q_out,h])
            self.kernels.matmul(
                &attn_out,
                (seq, q_out),
                &layer.o_proj_w,
                (q_out, h),
                &mut o_buf,
            )?;

            // Residual: hidden = residual + o_buf.
            self.kernels.vec_add(&residual_buf, &o_buf, &mut hidden)?;

            // Save residual before MLP.
            residual_buf.copy_from_slice(&hidden);

            // Post-attention RMSNorm.
            self.kernels.rmsnorm(
                &hidden,
                seq,
                h,
                &layer.post_attention_layernorm_w,
                eps,
                &mut norm_buf,
            )?;

            // MLP: out = down(silu(gate(x)) * up(x)).
            self.kernels.mlp_gated_silu(
                &norm_buf,
                seq,
                h,
                i_size,
                &layer.gate_proj_w,
                &layer.up_proj_w,
                &layer.down_proj_w,
                &mut gate_buf,
                &mut up_buf,
                &mut mlp_out,
            )?;

            // Residual: hidden = residual + mlp_out.
            self.kernels.vec_add(&residual_buf, &mlp_out, &mut hidden)?;
        }

        // Final RMSNorm over the full sequence.
        self.kernels.rmsnorm(
            &hidden,
            seq,
            h,
            &self.weights.final_norm_w,
            eps,
            &mut norm_buf,
        )?;

        // lm_head over the last position only. norm_buf last row is
        // [h] -> a `[1, h] @ [h, v]` matmul into a `[1, v]` output.
        let last_start = (seq - 1) * h;
        let last_row = &norm_buf[last_start..last_start + h];
        let mut logits = vec![0.0_f32; v];
        self.kernels.matmul(
            last_row,
            (1, h),
            &self.weights.lm_head_w,
            (h, v),
            &mut logits,
        )?;

        if let Some(cache) = cache {
            cache.set_len_tokens(seq)?;
        }

        Ok(logits)
    }
}
