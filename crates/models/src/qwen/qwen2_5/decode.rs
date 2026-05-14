//! Qwen2.5 single-token decode with KV cache reuse.
//!
//! `decode_token_with_cache` is the cached counterpart to `prefill`: it runs a
//! one-position forward pass that reuses previously written K/V from the
//! caller-owned cache, appends the new K/V for the supplied token, and
//! returns logits for the *following* position.

use ocelotl_core::{InvalidRequestError, KvCacheStore, OcelotlError, Result, TokenId};

use super::model::{Qwen2_5Model, add_bias_per_row, validate_cache_layout_for_model};

impl Qwen2_5Model {
    /// Decode a single already-selected token against an existing KV cache and
    /// append that token's K/V at the next cache position.
    ///
    /// This returns logits for the *following* token. The public runtime
    /// cached-decode helper samples from the prior prefill/decode logits, then
    /// calls this method to advance cache state for future decode steps.
    pub fn decode_token_with_cache(
        &self,
        token: TokenId,
        cache: &mut dyn KvCacheStore,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let position = cache.len_tokens();
        let next_len = position.checked_add(1).ok_or_else(|| {
            OcelotlError::InvalidRequest(InvalidRequestError {
                field: "kv_cache.capacity".to_string(),
                message: "cache position overflows usize".to_string(),
            })
        })?;
        validate_cache_layout_for_model(cfg, cache.layout(), next_len)?;
        if (token.0 as usize) >= cfg.vocab_size {
            return Err(OcelotlError::InvalidRequest(InvalidRequestError {
                field: "token".to_string(),
                message: format!(
                    "token id {} is out of range for vocab_size {}",
                    token.0, cfg.vocab_size
                ),
            }));
        }

        let h = cfg.hidden_size;
        let q_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        let i_size = cfg.intermediate_size;
        let v = cfg.vocab_size;
        let eps = cfg.rms_norm_eps as f32;
        let theta = cfg.rope_theta as f32;

        let src = (token.0 as usize) * h;
        let mut hidden = self.weights.embed_tokens[src..src + h].to_vec();

        let mut norm_buf = vec![0.0_f32; h];
        let mut q_buf = vec![0.0_f32; q_out];
        let mut k_buf = vec![0.0_f32; kv_out];
        let mut v_buf = vec![0.0_f32; kv_out];
        let mut k_cache = vec![0.0_f32; next_len * kv_out];
        let mut v_cache = vec![0.0_f32; next_len * kv_out];
        let mut q_context = vec![0.0_f32; next_len * q_out];
        let mut attn_context = vec![0.0_f32; next_len * q_out];
        let mut o_buf = vec![0.0_f32; h];
        let mut residual_buf = vec![0.0_f32; h];
        let mut gate_buf = vec![0.0_f32; i_size];
        let mut up_buf = vec![0.0_f32; i_size];
        let mut mlp_out = vec![0.0_f32; h];

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            residual_buf.copy_from_slice(&hidden);
            self.kernels
                .rmsnorm(&hidden, 1, h, &layer.input_layernorm_w, eps, &mut norm_buf)?;

            self.kernels
                .matmul(&norm_buf, (1, h), &layer.q_proj_w, (h, q_out), &mut q_buf)?;
            add_bias_per_row(&mut q_buf, &layer.q_proj_b, 1, q_out);
            self.kernels
                .matmul(&norm_buf, (1, h), &layer.k_proj_w, (h, kv_out), &mut k_buf)?;
            add_bias_per_row(&mut k_buf, &layer.k_proj_b, 1, kv_out);
            self.kernels
                .matmul(&norm_buf, (1, h), &layer.v_proj_w, (h, kv_out), &mut v_buf)?;
            add_bias_per_row(&mut v_buf, &layer.v_proj_b, 1, kv_out);

            self.kernels
                .rope_apply_inplace(&mut q_buf, cfg.head_dim, position, theta)?;
            self.kernels
                .rope_apply_inplace(&mut k_buf, cfg.head_dim, position, theta)?;

            cache.write_layer_kv(layer_idx, position, &k_buf, &v_buf)?;
            cache.read_layer_keys(layer_idx, next_len, &mut k_cache)?;
            cache.read_layer_values(layer_idx, next_len, &mut v_cache)?;

            q_context.fill(0.0);
            let q_start = position * q_out;
            q_context[q_start..q_start + q_out].copy_from_slice(&q_buf);
            attn_context.fill(0.0);
            self.kernels.scaled_dot_product_attention(
                &q_context,
                &k_cache,
                &v_cache,
                next_len,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                &mut attn_context,
            )?;

            let attn_start = position * q_out;
            self.kernels.matmul(
                &attn_context[attn_start..attn_start + q_out],
                (1, q_out),
                &layer.o_proj_w,
                (q_out, h),
                &mut o_buf,
            )?;

            self.kernels.vec_add(&residual_buf, &o_buf, &mut hidden)?;
            residual_buf.copy_from_slice(&hidden);

            self.kernels.rmsnorm(
                &hidden,
                1,
                h,
                &layer.post_attention_layernorm_w,
                eps,
                &mut norm_buf,
            )?;
            self.kernels.mlp_gated_silu(
                &norm_buf,
                1,
                h,
                i_size,
                &layer.gate_proj_w,
                &layer.up_proj_w,
                &layer.down_proj_w,
                &mut gate_buf,
                &mut up_buf,
                &mut mlp_out,
            )?;
            self.kernels.vec_add(&residual_buf, &mlp_out, &mut hidden)?;
        }

        self.kernels.rmsnorm(
            &hidden,
            1,
            h,
            &self.weights.final_norm_w,
            eps,
            &mut norm_buf,
        )?;

        let mut logits = vec![0.0_f32; v];
        self.kernels.matmul(
            &norm_buf,
            (1, h),
            &self.weights.lm_head_w,
            (h, v),
            &mut logits,
        )?;
        cache.set_len_tokens(next_len)?;

        Ok(logits)
    }
}
