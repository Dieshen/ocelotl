//! Model-family implementations.
//!
//! Each family is its own submodule (`qwen`, `gemma`, `whisper`). Family
//! types are NOT re-exported at the crate root; call sites use the family
//! namespace explicitly:
//!
//! - `ocelotl_models::qwen::Qwen2_5Model`
//! - `ocelotl_models::gemma::Gemma4Config`
//! - `ocelotl_models::whisper::WhisperModel`

pub mod gemma;
pub mod parakeet;
pub mod qwen;
pub mod whisper;

use std::sync::Arc;

use ocelotl_kernels::{CpuKernelBackend, CpuKernelMode, KernelBackend};

/// Build the CPU kernel backend the bidirectional encoders run on.
///
/// Picks the fastest mode the host supports — `Avx2` (with FMA) on capable
/// x86_64, else the autovectorizable `Optimized` scalar path — and gives it a
/// rayon pool sized to the machine. For the encoders the AVX2 microkernel of
/// `linear_out_by_in` is the dominant win (contiguous dot products over the raw
/// GGUF `[out][in]` weight layout); the pool only engages for long sequences
/// (`rows >= 32`), so short single-sentence embeds run serial-SIMD and bulk
/// throughput comes from parallelizing across sentences at the caller.
pub(crate) fn cpu_embedding_backend() -> Arc<dyn KernelBackend> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    #[cfg(target_arch = "x86_64")]
    let mode = if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        CpuKernelMode::Avx2
    } else {
        CpuKernelMode::Optimized
    };
    #[cfg(not(target_arch = "x86_64"))]
    let mode = CpuKernelMode::Optimized;
    match CpuKernelBackend::with_mode_and_threads(mode, threads) {
        Ok(backend) => Arc::new(backend),
        Err(_) => Arc::new(CpuKernelBackend::optimized()),
    }
}
