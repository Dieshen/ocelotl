use super::*;

#[derive(Debug, Clone)]
pub struct CpuKernelBackend {
    context: KernelContext,
    mode: CpuKernelMode,
    /// Optional thread pool. `None` means single-threaded execution, which is
    /// the parity oracle the default tests rely on. A pool is built when the
    /// caller asks for `threads >= 2` via `with_mode_and_threads`.
    pool: Option<Arc<rayon::ThreadPool>>,
}

impl Default for CpuKernelBackend {
    fn default() -> Self {
        Self::scalar()
    }
}

impl CpuKernelBackend {
    pub fn scalar() -> Self {
        Self::new_supported_mode(CpuKernelMode::Scalar)
    }

    pub fn optimized() -> Self {
        Self::new_supported_mode(CpuKernelMode::Optimized)
    }

    /// Construct a serial CPU backend after validating runtime CPU features.
    ///
    /// `CpuKernelMode::Avx2` is accepted only when the current x86_64 host
    /// advertises both AVX2 and FMA. Keeping this public constructor fallible
    /// prevents safe callers from creating a backend that could later execute
    /// unsupported `#[target_feature]` code.
    pub fn with_mode(mode: CpuKernelMode) -> Result<Self> {
        validate_mode_supported(mode)?;
        Ok(Self::new_supported_mode(mode))
    }

    /// Build a backend after its mode invariant has already been established.
    /// Scalar and Optimized have no runtime feature requirements; every AVX2
    /// caller must pass through `validate_mode_supported` first.
    fn new_supported_mode(mode: CpuKernelMode) -> Self {
        Self {
            context: KernelContext {
                device: Device::Cpu,
            },
            mode,
            pool: None,
        }
    }

    /// Compatibility alias for the now-checked `with_mode` constructor.
    pub fn with_mode_checked(mode: CpuKernelMode) -> Result<Self> {
        Self::with_mode(mode)
    }

    /// Construct a backend that runs hot kernels (currently `linear_out_by_in`
    /// and any caller that opts in via `cpu_thread_pool()`) across `threads`
    /// worker threads. `threads <= 1` is equivalent to `with_mode_checked`.
    /// The pool is built once and reused for the lifetime of this backend.
    pub fn with_mode_and_threads(mode: CpuKernelMode, threads: usize) -> Result<Self> {
        validate_mode_supported(mode)?;
        if threads <= 1 {
            return Ok(Self::new_supported_mode(mode));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("ocelotl-cpu-{i}"))
            .build()
            .map_err(|err| {
                OcelotlError::Kernel(KernelError {
                    backend: "cpu".to_string(),
                    message: format!("failed to build rayon thread pool: {err}"),
                })
            })?;
        Ok(Self {
            context: KernelContext {
                device: Device::Cpu,
            },
            mode,
            pool: Some(Arc::new(pool)),
        })
    }

    pub fn mode(&self) -> CpuKernelMode {
        self.mode
    }

    /// Number of worker threads, or 1 if running serially.
    pub fn num_threads(&self) -> usize {
        self.pool.as_ref().map_or(1, |p| p.current_num_threads())
    }

    pub fn matmul(
        &self,
        a: &[f32],
        a_shape: (usize, usize),
        b: &[f32],
        b_shape: (usize, usize),
        out: &mut [f32],
    ) -> Result<()> {
        let m = a_shape.0;
        if let Some(pool) = &self.pool {
            if m >= PARALLEL_MATMUL_MIN_ROWS {
                return matmul_parallel(pool, self.mode, a, a_shape, b, b_shape, out);
            }
        }
        match self.mode {
            CpuKernelMode::Scalar => matmul(a, a_shape, b, b_shape, out),
            CpuKernelMode::Optimized => matmul_optimized(a, a_shape, b, b_shape, out),
            // matmul is used by the Qwen-shaped GEMM; AVX2 today only
            // accelerates `linear_out_by_in` (the [out, in] Whisper weight
            // layout). Other matmul callers fall back to the optimized
            // scalar path until AVX2 covers them.
            CpuKernelMode::Avx2 => matmul_optimized(a, a_shape, b, b_shape, out),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn linear_out_by_in(
        &self,
        x: &[f32],
        rows: usize,
        in_features: usize,
        weight_out_by_in: &[f32],
        out_features: usize,
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        if let Some(pool) = &self.pool {
            if rows == 1 && out_features >= PARALLEL_LINEAR_MIN_OUTPUTS {
                return linear_out_by_in_output_parallel(
                    pool,
                    self.mode,
                    x,
                    in_features,
                    weight_out_by_in,
                    out_features,
                    bias,
                    out,
                );
            }
            if rows >= PARALLEL_LINEAR_MIN_ROWS {
                return linear_out_by_in_parallel(
                    pool,
                    self.mode,
                    x,
                    rows,
                    in_features,
                    weight_out_by_in,
                    out_features,
                    bias,
                    out,
                );
            }
        }
        match self.mode {
            CpuKernelMode::Scalar => linear_out_by_in(
                x,
                rows,
                in_features,
                weight_out_by_in,
                out_features,
                bias,
                out,
            ),
            CpuKernelMode::Optimized => linear_out_by_in_optimized(
                x,
                rows,
                in_features,
                weight_out_by_in,
                out_features,
                bias,
                out,
            ),
            CpuKernelMode::Avx2 => linear_out_by_in_avx2(
                x,
                rows,
                in_features,
                weight_out_by_in,
                out_features,
                bias,
                out,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn scaled_dot_product_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        out: &mut [f32],
    ) -> Result<()> {
        if let Some(pool) = &self.pool {
            if seq_len >= PARALLEL_SDPA_MIN_SEQ {
                return attention::scaled_dot_product_attention_parallel(
                    pool,
                    self.mode,
                    q,
                    k,
                    v,
                    seq_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    out,
                );
            }
        }
        match self.mode {
            CpuKernelMode::Scalar => attention::scaled_dot_product_attention(
                q,
                k,
                v,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                out,
            ),
            CpuKernelMode::Optimized | CpuKernelMode::Avx2 => {
                // Same fallback rationale as matmul: this kernel is only
                // exercised by the Qwen path right now. AVX2 currently
                // accelerates the Whisper-shaped `linear_out_by_in`; the
                // Qwen-shaped attention falls back to optimized scalar.
                attention::scaled_dot_product_attention_optimized(
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
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn scaled_dot_product_attention_with_scale(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        out: &mut [f32],
    ) -> Result<()> {
        match self.mode {
            CpuKernelMode::Scalar => attention::scaled_dot_product_attention_with_scale(
                q,
                k,
                v,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                scale,
                out,
            ),
            CpuKernelMode::Optimized | CpuKernelMode::Avx2 => {
                attention::scaled_dot_product_attention_optimized_with_scale(
                    q,
                    k,
                    v,
                    seq_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    out,
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn scaled_dot_product_attention_windowed(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        out: &mut [f32],
    ) -> Result<()> {
        match self.mode {
            CpuKernelMode::Scalar => attention::scaled_dot_product_attention_windowed(
                q,
                k,
                v,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                sliding_window,
                out,
            ),
            CpuKernelMode::Optimized | CpuKernelMode::Avx2 => {
                attention::scaled_dot_product_attention_windowed_optimized(
                    q,
                    k,
                    v,
                    seq_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    sliding_window,
                    out,
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn scaled_dot_product_attention_windowed_with_scale(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        scale: f32,
        out: &mut [f32],
    ) -> Result<()> {
        match self.mode {
            CpuKernelMode::Scalar => attention::scaled_dot_product_attention_windowed_with_scale(
                q,
                k,
                v,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                sliding_window,
                scale,
                out,
            ),
            CpuKernelMode::Optimized | CpuKernelMode::Avx2 => {
                attention::scaled_dot_product_attention_windowed_optimized_with_scale(
                    q,
                    k,
                    v,
                    seq_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    sliding_window,
                    scale,
                    out,
                )
            }
        }
    }
}

impl KernelBackend for CpuKernelBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn context(&self) -> &KernelContext {
        &self.context
    }

    fn cpu_thread_pool(&self) -> Option<&rayon::ThreadPool> {
        self.pool.as_deref()
    }

    fn matmul(
        &self,
        a: &[f32],
        a_shape: (usize, usize),
        b: &[f32],
        b_shape: (usize, usize),
        out: &mut [f32],
    ) -> Result<()> {
        CpuKernelBackend::matmul(self, a, a_shape, b, b_shape, out)
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
    ) -> Result<()> {
        CpuKernelBackend::linear_out_by_in(
            self,
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
    ) -> Result<()> {
        CpuKernelBackend::scaled_dot_product_attention(
            self,
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
    fn scaled_dot_product_attention_with_scale(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        out: &mut [f32],
    ) -> Result<()> {
        CpuKernelBackend::scaled_dot_product_attention_with_scale(
            self,
            q,
            k,
            v,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            out,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn scaled_dot_product_attention_windowed(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        out: &mut [f32],
    ) -> Result<()> {
        CpuKernelBackend::scaled_dot_product_attention_windowed(
            self,
            q,
            k,
            v,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            out,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn scaled_dot_product_attention_windowed_with_scale(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        scale: f32,
        out: &mut [f32],
    ) -> Result<()> {
        CpuKernelBackend::scaled_dot_product_attention_windowed_with_scale(
            self,
            q,
            k,
            v,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            scale,
            out,
        )
    }

    fn rope_apply_inplace(
        &self,
        x: &mut [f32],
        head_dim: usize,
        position: usize,
        theta: f32,
    ) -> Result<()> {
        rope_apply_inplace(x, head_dim, position, theta)
    }

    fn rmsnorm(
        &self,
        x: &[f32],
        rows: usize,
        hidden: usize,
        weight: &[f32],
        epsilon: f32,
        out: &mut [f32],
    ) -> Result<()> {
        rmsnorm::rmsnorm(x, rows, hidden, weight, epsilon, out)
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
    ) -> Result<()> {
        mlp::mlp_gated_silu(
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

    #[allow(clippy::too_many_arguments)]
    fn mlp_gated_gelu(
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
    ) -> Result<()> {
        mlp::mlp_gated_gelu(
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

    fn vec_add(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        vec_add(a, b, out)
    }

    /// CPU override: when every handle is host-resident (the common case
    /// for the CPU backend), borrow the underlying `Vec<f32>` slices directly
    /// and call the existing `linear_out_by_in` — no readback, no extra
    /// allocations. Falls back to the trait default for the rare device-on-
    /// CPU case (which produces a readback through the default impl).
    #[allow(clippy::too_many_arguments)]
    fn linear_d(
        &self,
        x: &DeviceTensor,
        rows: usize,
        in_features: usize,
        weight: &DeviceTensor,
        out_features: usize,
        bias: Option<&DeviceTensor>,
        out: &DeviceTensor,
    ) -> Result<()> {
        // All inputs are host-resident in practice on this backend; borrow
        // their slices directly. If any side is somehow device-resident we
        // delegate to the default impl which forces readback.
        let (x_borrow, w_borrow, out_borrow) = match (
            x.borrow_host_slice(),
            weight.borrow_host_slice(),
            out.borrow_host_slice_mut(),
        ) {
            (Ok(x_b), Ok(w_b), Ok(out_b)) => (x_b, w_b, out_b),
            _ => {
                let x_host = x.to_host_owned()?;
                let weight_host = weight.to_host_owned()?;
                let bias_host = bias.map(DeviceTensor::to_host_owned).transpose()?;
                let mut out_buf = vec![0.0_f32; rows * out_features];
                CpuKernelBackend::linear_out_by_in(
                    self,
                    &x_host,
                    rows,
                    in_features,
                    &weight_host,
                    out_features,
                    bias_host.as_deref(),
                    &mut out_buf,
                )?;
                return out.write_from_host_slice(&out_buf);
            }
        };
        let mut out_borrow = out_borrow;
        match bias {
            Some(b) => {
                let b_borrow = b.borrow_host_slice()?;
                CpuKernelBackend::linear_out_by_in(
                    self,
                    &x_borrow,
                    rows,
                    in_features,
                    &w_borrow,
                    out_features,
                    Some(&b_borrow),
                    &mut out_borrow,
                )
            }
            None => CpuKernelBackend::linear_out_by_in(
                self,
                &x_borrow,
                rows,
                in_features,
                &w_borrow,
                out_features,
                None,
                &mut out_borrow,
            ),
        }
    }

    /// CPU override: borrow both host slices and do an elementwise add.
    /// Falls through to the trait default if either side is device-resident
    /// (shouldn't happen on the CPU backend, but the default path stays
    /// correct).
    fn add_inplace_d(&self, lhs: &DeviceTensor, rhs: &DeviceTensor) -> Result<()> {
        let lhs_borrow = lhs.borrow_host_slice_mut();
        let rhs_borrow = rhs.borrow_host_slice();
        match (lhs_borrow, rhs_borrow) {
            (Ok(mut l), Ok(r)) => {
                if l.len() != r.len() {
                    return Err(kernel_err(format!(
                        "add_inplace_d length mismatch: lhs={} rhs={}",
                        l.len(),
                        r.len()
                    )));
                }
                for (lv, rv) in l.iter_mut().zip(r.iter()) {
                    *lv += *rv;
                }
                Ok(())
            }
            _ => {
                let mut lhs_host = lhs.to_host_owned()?;
                let rhs_host = rhs.to_host_owned()?;
                if lhs_host.len() != rhs_host.len() {
                    return Err(kernel_err(format!(
                        "add_inplace_d length mismatch: lhs={} rhs={}",
                        lhs_host.len(),
                        rhs_host.len()
                    )));
                }
                for (l, r) in lhs_host.iter_mut().zip(rhs_host.iter()) {
                    *l += *r;
                }
                lhs.write_from_host_slice(&lhs_host)
            }
        }
    }

    /// CPU override: borrow the host slice and apply the Whisper-exact
    /// GELU in place. Bit-identical to `gelu_inplace` in the models crate
    /// because both call the same `gelu_whisper_scalar` math.
    fn gelu_inplace_d(&self, x: &DeviceTensor) -> Result<()> {
        match x.borrow_host_slice_mut() {
            Ok(mut borrow) => {
                for v in borrow.iter_mut() {
                    *v = gelu_whisper_scalar(*v);
                }
                Ok(())
            }
            Err(_) => {
                let mut host = x.to_host_owned()?;
                for v in host.iter_mut() {
                    *v = gelu_whisper_scalar(*v);
                }
                x.write_from_host_slice(&host)
            }
        }
    }

    /// CPU override: borrow host slices and run the scalar Whisper-shape
    /// LayerNorm directly. Falls through to the trait default for the
    /// device-on-CPU case.
    #[allow(clippy::too_many_arguments)]
    fn layer_norm_d(
        &self,
        x: &DeviceTensor,
        rows: usize,
        hidden: usize,
        weight: &DeviceTensor,
        bias: &DeviceTensor,
        eps: f32,
        out: &DeviceTensor,
    ) -> Result<()> {
        validate_layer_norm_shapes(x, rows, hidden, weight, bias, out)?;
        let (x_b, w_b, b_b, out_b) = match (
            x.borrow_host_slice(),
            weight.borrow_host_slice(),
            bias.borrow_host_slice(),
            out.borrow_host_slice_mut(),
        ) {
            (Ok(x_b), Ok(w_b), Ok(b_b), Ok(out_b)) => (x_b, w_b, b_b, out_b),
            _ => {
                let x_host = x.to_host_owned()?;
                let w_host = weight.to_host_owned()?;
                let b_host = bias.to_host_owned()?;
                let mut out_buf = vec![0.0_f32; rows * hidden];
                layer_norm_whisper_scalar(
                    &x_host,
                    rows,
                    hidden,
                    &w_host,
                    &b_host,
                    eps,
                    &mut out_buf,
                );
                return out.write_from_host_slice(&out_buf);
            }
        };
        let mut out_b = out_b;
        layer_norm_whisper_scalar(&x_b, rows, hidden, &w_b, &b_b, eps, &mut out_b);
        Ok(())
    }

    /// CPU override: borrow host slices and run encoder attention directly.
    /// A configured CPU thread pool partitions query rows; each row keeps the
    /// same per-head scalar softmax order as the serial oracle. Falls through
    /// to the trait default if any operand is device-resident (shouldn't happen
    /// on the CPU backend, but the default path stays correct).
    #[allow(clippy::too_many_arguments)]
    fn attention_encoder_d(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        seq: usize,
        n_head: usize,
        head_dim: usize,
        scale: f32,
        output: &DeviceTensor,
    ) -> Result<()> {
        validate_attention_encoder_shapes(q, k, v, seq, n_head, head_dim, output)?;
        let (q_b, k_b, v_b, out_b) = match (
            q.borrow_host_slice(),
            k.borrow_host_slice(),
            v.borrow_host_slice(),
            output.borrow_host_slice_mut(),
        ) {
            (Ok(q_b), Ok(k_b), Ok(v_b), Ok(out_b)) => (q_b, k_b, v_b, out_b),
            _ => {
                let q_host = q.to_host_owned()?;
                let k_host = k.to_host_owned()?;
                let v_host = v.to_host_owned()?;
                let mut out_buf = vec![0.0_f32; seq * n_head * head_dim];
                attention_encoder_scalar(
                    &q_host,
                    &k_host,
                    &v_host,
                    seq,
                    n_head,
                    head_dim,
                    scale,
                    &mut out_buf,
                );
                return output.write_from_host_slice(&out_buf);
            }
        };
        let mut out_b = out_b;
        if let Some(pool) = &self.pool {
            if seq >= PARALLEL_SDPA_MIN_SEQ {
                attention_encoder_parallel(
                    pool, self.mode, &q_b, &k_b, &v_b, seq, n_head, head_dim, scale, &mut out_b,
                );
                return Ok(());
            }
        }
        attention_encoder_scalar(&q_b, &k_b, &v_b, seq, n_head, head_dim, scale, &mut out_b);
        Ok(())
    }

    /// CPU override: borrow host slices and accumulate the positional
    /// embedding into `x` in place. Falls through to the trait default
    /// for the device-on-CPU case.
    fn add_positional_embedding_d(
        &self,
        x: &DeviceTensor,
        rows: usize,
        cols: usize,
        pe: &DeviceTensor,
        pe_rows: usize,
        start_pos: usize,
    ) -> Result<()> {
        validate_add_positional_embedding_shapes(x, rows, cols, pe, pe_rows, start_pos)?;
        let (x_b, pe_b) = match (x.borrow_host_slice_mut(), pe.borrow_host_slice()) {
            (Ok(x_b), Ok(pe_b)) => (x_b, pe_b),
            _ => {
                let mut x_host = x.to_host_owned()?;
                let pe_host = pe.to_host_owned()?;
                for row in 0..rows {
                    let dst_start = row * cols;
                    let src_start = (start_pos + row) * cols;
                    for col in 0..cols {
                        x_host[dst_start + col] += pe_host[src_start + col];
                    }
                }
                return x.write_from_host_slice(&x_host);
            }
        };
        let mut x_b = x_b;
        for row in 0..rows {
            let dst_start = row * cols;
            let src_start = (start_pos + row) * cols;
            for col in 0..cols {
                x_b[dst_start + col] += pe_b[src_start + col];
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_decoder_incremental_cache_d(
        &self,
        q: &DeviceTensor,
        key_cache: &DeviceTensor,
        value_cache: &DeviceTensor,
        visible_seq: usize,
        cache_capacity: usize,
        n_head: usize,
        head_dim: usize,
        scale: f32,
        output: &DeviceTensor,
    ) -> Result<()> {
        validate_attention_decoder_incremental_cache_shapes(
            q,
            key_cache,
            value_cache,
            visible_seq,
            cache_capacity,
            n_head,
            head_dim,
            output,
        )?;
        let state = n_head * head_dim;
        let visible_len = visible_seq * state;
        let (q_b, key_b, value_b, out_b) = match (
            q.borrow_host_slice(),
            key_cache.borrow_host_slice(),
            value_cache.borrow_host_slice(),
            output.borrow_host_slice_mut(),
        ) {
            (Ok(q_b), Ok(key_b), Ok(value_b), Ok(out_b)) => (q_b, key_b, value_b, out_b),
            _ => {
                let q_host = q.to_host_owned()?;
                let key_host = key_cache.to_host_owned()?;
                let value_host = value_cache.to_host_owned()?;
                let mut out_buf = vec![0.0_f32; state];
                attention_decoder_incremental_cache_host(
                    self.mode,
                    &q_host,
                    &key_host[..visible_len],
                    &value_host[..visible_len],
                    visible_seq,
                    n_head,
                    head_dim,
                    scale,
                    &mut out_buf,
                );
                return output.write_from_host_slice(&out_buf);
            }
        };
        let mut out_b = out_b;
        let key_prefix = &key_b[..visible_len];
        let value_prefix = &value_b[..visible_len];
        if let Some(pool) = &self.pool {
            if visible_seq >= PARALLEL_SDPA_MIN_SEQ && n_head > 1 {
                attention_decoder_incremental_cache_parallel(
                    pool,
                    self.mode,
                    &q_b,
                    key_prefix,
                    value_prefix,
                    visible_seq,
                    n_head,
                    head_dim,
                    scale,
                    &mut out_b,
                );
                return Ok(());
            }
        }
        attention_decoder_incremental_cache_host(
            self.mode,
            &q_b,
            key_prefix,
            value_prefix,
            visible_seq,
            n_head,
            head_dim,
            scale,
            &mut out_b,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_decoder_incremental_cache_append_d(
        &self,
        q: &DeviceTensor,
        key_cache: &DeviceTensor,
        value_cache: &DeviceTensor,
        new_k: &DeviceTensor,
        new_v: &DeviceTensor,
        past_seq: usize,
        cache_capacity: usize,
        n_head: usize,
        head_dim: usize,
        scale: f32,
        output: &DeviceTensor,
    ) -> Result<()> {
        validate_attention_decoder_incremental_cache_append_shapes(
            q,
            key_cache,
            value_cache,
            new_k,
            new_v,
            past_seq,
            cache_capacity,
            n_head,
            head_dim,
            output,
        )?;
        let state = n_head * head_dim;
        let dst_offset = past_seq.checked_mul(state).ok_or_else(|| {
            kernel_err(
                "attention_decoder_incremental_cache_append_d past_seq*state overflowed usize",
            )
        })?;
        {
            let maybe_borrows = (
                new_k.borrow_host_slice(),
                new_v.borrow_host_slice(),
                key_cache.borrow_host_slice_mut(),
                value_cache.borrow_host_slice_mut(),
            );
            match maybe_borrows {
                (Ok(new_k_b), Ok(new_v_b), Ok(mut key_b), Ok(mut value_b)) => {
                    key_b[dst_offset..dst_offset + state].copy_from_slice(&new_k_b);
                    value_b[dst_offset..dst_offset + state].copy_from_slice(&new_v_b);
                }
                _ => {
                    self.copy_into_d(new_k, key_cache, dst_offset)?;
                    self.copy_into_d(new_v, value_cache, dst_offset)?;
                }
            }
        }
        self.attention_decoder_incremental_cache_d(
            q,
            key_cache,
            value_cache,
            past_seq + 1,
            cache_capacity,
            n_head,
            head_dim,
            scale,
            output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_decoder_cross_d(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        q_seq: usize,
        kv_seq: usize,
        n_head: usize,
        head_dim: usize,
        scale: f32,
        output: &DeviceTensor,
    ) -> Result<()> {
        validate_attention_decoder_cross_shapes(q, k, v, q_seq, kv_seq, n_head, head_dim, output)?;
        let (q_b, k_b, v_b, out_b) = match (
            q.borrow_host_slice(),
            k.borrow_host_slice(),
            v.borrow_host_slice(),
            output.borrow_host_slice_mut(),
        ) {
            (Ok(q_b), Ok(k_b), Ok(v_b), Ok(out_b)) => (q_b, k_b, v_b, out_b),
            _ => {
                let q_host = q.to_host_owned()?;
                let k_host = k.to_host_owned()?;
                let v_host = v.to_host_owned()?;
                let mut out_buf = vec![0.0_f32; q_seq * n_head * head_dim];
                attention_decoder_cross_host(
                    self.mode,
                    &q_host,
                    &k_host,
                    &v_host,
                    q_seq,
                    kv_seq,
                    n_head,
                    head_dim,
                    scale,
                    &mut out_buf,
                );
                return output.write_from_host_slice(&out_buf);
            }
        };
        let mut out_b = out_b;
        if let Some(pool) = &self.pool {
            if kv_seq >= PARALLEL_SDPA_MIN_SEQ && q_seq * n_head > 1 {
                attention_decoder_cross_parallel(
                    pool, self.mode, &q_b, &k_b, &v_b, q_seq, kv_seq, n_head, head_dim, scale,
                    &mut out_b,
                );
                return Ok(());
            }
        }
        attention_decoder_cross_host(
            self.mode, &q_b, &k_b, &v_b, q_seq, kv_seq, n_head, head_dim, scale, &mut out_b,
        );
        Ok(())
    }
}
