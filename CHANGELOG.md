# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); pre-1.0 minor
versions may make breaking changes.

## [0.1.0] — 2026-07-25

First tagged release. A local, single-process Rust library for inference from
trusted local artifacts, with **no FFI** — every kernel, loader, and tokenizer
path is Rust.

### Supported

Scope is set by which surfaces have independent-reference parity evidence
captured at this commit, not by which were planned first.

**Text embeddings** — `gemma::embedding::EmbeddingGemmaModel`,
`qwen::pplx_embed::PplxEmbedModel`. Mean-pooled, L2-normalized.

- Cosine ≥ 0.9999997 vs `llama-embedding`, with full retrieval-ranking agreement.
- CPU (AVX2 + rayon): 13.1 ms/embed on bulk — **2.4× faster than llama.cpp**.
- GPU (CubeCL, WGPU or HIP/ROCm): fully device-resident forward; 5.9 ms/embed at
  batch 256, matching llama.cpp on the same card.

**Parakeet TDT 0.6B ASR** — `parakeet::model::ParakeetModel`. 16 kHz mono,
English + 24 other European languages.

- **Token-exact and frame-exact against two independent implementations**: the
  ONNX export and `parakeet.cpp`. Tokens, per-token frame indices, and text all
  identical.
- Real-time factor **0.095** on 12 threads (0.288 single-threaded), Ryzen 7 7840HS.
- Long-form chunking available with bounded degradation; see the caveat below.

### Not supported — do not present these as working

- **No network server.** M8 has not started: there is no supported endpoint,
  streaming transport, authentication, or external error contract. This is a
  library, not a service.
- **Gemma4 text generation** — deferred to 0.2.0. It executes and selects the
  same top-1 token as llama.cpp with 18–19/20 top-20 overlap, but full-logit
  parity does not meet its numeric contract. The investigation ruled out every
  specific suspect and concluded the residual is the independent-f32
  reproducibility floor, so this is a contract decision rather than a bug.
- **Gemma4 multimodal** (image/audio/video input) — unsupported.
- **Whisper ASR** — deferred to 0.2.0. It executes, but a bundle rebuilt at this
  commit diverges from a pure-greedy whisper.cpp reference by one token (26 of 27
  match). Cause unresolved.
- **Whisper timestamps and streaming**, multilingual quality claims, and any WER
  threshold — out of scope.
- **Unproved hardware backends.** GPU support is exercised on AMD RX 6600 via
  WGPU/Vulkan and HIP/ROCm. Nothing else is validated.

### Known limitations

- **Parakeet chunking is a pessimization below ~40 minutes of audio.** It is
  slower *and* less accurate than a single window at ordinary lengths (13.1 s vs
  11.5 s on a 121 s clip, for 4.2% token error). It exists to bound the attention
  score transient at extreme lengths. Use `encode_audio` unless past
  `MAX_AUDIO_SECONDS`.
- **Single-threaded frontend and decode.** Only the encoder GEMMs use the thread
  pool; the mel frontend and greedy decode do not, and are ~19% of a 12-thread run.
- **Parakeet is ~1.8× slower than `parakeet.cpp`** single-threaded. The gap is
  understood: per-head gather overhead and 576 small-GEMM launches per encode.
- **GPU is not useful for the Parakeet encoder** — measured 9–10× slower than
  threaded CPU, because an 11 s utterance is only 138 rows and cannot fill a
  device. Batched inference would change this.
- Eager dequantized fallback weights are retained beside native quantized
  sidecars in the Gemma loader; correctness-first, but the peak-memory cost must
  be bounded before that path is claimed.

### Testing

589 offline tests run by default. 43 opt-in tests cover local weights, real GPU
launch, and external reference binaries; all are env-gated and documented, so a
default `cargo test` needs no network, no weights, and no GPU.

### Licensing

`NOTICE` records third-party model weight licences, including the **CC-BY-4.0**
obligation on the Parakeet weights (attribution plus a statement of changes).
Ocelotl's own code is MIT OR Apache-2.0.
