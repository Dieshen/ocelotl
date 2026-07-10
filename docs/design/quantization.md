# Quantization Design

Quantization support should be added after the unquantized path is correct.
Quantized execution changes both loader behavior and kernel behavior, so it needs
its own validation gates.

## Goals

- Keep quantization metadata explicit.
- Reject unsupported quantization formats early.
- Compare quantized output against unquantized or trusted reference output.
- Avoid mixing incompatible quantization layouts in one generic path.

## Deferred Until After M4

M1-M4 should use f16, bf16, or f32 weights. Quantization belongs after the basic
CPU and GPU execution paths have parity.

## Metadata

Quantized formats should describe:

- Quantization family.
- Block size.
- Scale dtype.
- Zero-point behavior.
- Packing layout.
- Per-tensor, per-channel, or per-block parameters.

## Loader Contract

The loader should not simply expose opaque bytes. It should validate that the
runtime and kernels understand the quantization format before model execution.

### Current GGUF K-Quant Value Policy

As of 2026-06-05, GGUF inspection knows the selected Gemma4 Q4_K_M artifact's
ggml K-quant layouts and the loader has an explicit dequantizing value API:

| GGML type | Elements per block | Bytes per block | Value status | Execution status |
| --- | ---: | ---: | --- | --- |
| `Q4K` | 256 | 144 | exact dequant and bounded raw-byte load tested | native Q8_K text projection |
| `Q5K` | 256 | 176 | exact dequant and bounded raw-byte load tested | native Q8_K text projection |
| `Q6K` | 256 | 210 | exact dequant and bounded raw-byte load tested | native Q8_K text projection |

`inspect_gguf` rejects Q4K/Q5K/Q6K tensors whose row width or total element
count is not divisible by 256, and validates the computed byte range against
the file length. This follows ggml's row contract: a tensor such as `[128, 2]`
has 256 total elements but is not a valid K-quant row layout.
`load_gguf_tensor_f32` stays dense-only and still rejects quantized payloads with
a typed `Unsupported` error. Callers that intend to materialize quantized values
must use `load_gguf_tensor_dequantized_f32` or
`load_gguf_tensors_dequantized_f32`. The narrower
`load_gguf_k_quant_tensor_bytes` and `load_gguf_k_quant_tensors_bytes` APIs expose
only inspected Q4K/Q5K/Q6K payloads for model-owned native sidecars; dense and
unknown tensor types remain rejected at that boundary.

The Q4K/Q5K/Q6K dequantizers use ggml's current K-quant layouts and bit packing
from `ggml/src/ggml-common.h` and `ggml/src/ggml-quants.c`. Default tests pin
small exact dequantized vectors for each format, including Q4/Q5 scale/min
packing across all Q4_K scale/min groups, Q5 high-bit lanes, and Q6 signed
scales. MF.7 adds a dense F32/dequantized-F32 text subset; follow-up
text-forward slices handle mixed SWA/global layer widths, GGUF sliding-window
pattern metadata, sliding-window masking, shared-KV reuse, final logit softcap,
post-branch RMSNorms, per-layer embeddings, layer output scales, and Gemma4
RoPE frequency factors for that subset. Follow-up coverage also executes
explicitly dequantized Q4_K_M-origin text tensors in that path. Raw quantized
tensors cannot enter `Gemma4TextWeights`; native bytes are held in separate,
shape-validated text-projection sidecars. The native path now covers Q4_K,
Q5_K, and Q6_K matrices for attention, FFN gate/up/down, per-layer input/output,
and the tied output projection. Each kernel quantizes the F32 activation row to
Q8_K and follows the pinned llama.cpp AVX2 reduction order.

As of the 2026-07-10 local proof, the selected layer-0 Q/K/V/O inventory remains
Q6_K/Q5_K/Q6_K/Q5_K, but the complete text model now uses native sidecars for
all supported quantized projection matrices. With the pinned llama.cpp
reference forced to non-flash attention, F32 K/V cache, and `--no-repack`, the
real layer-0 trace matches through `l_out-0` with a largest sampled difference
of approximately `6.2e-5` at tolerance `0.05`. Default tests independently pin
hand-checked Q4_K/Q5_K/Q6_K values, scale/min behavior, malformed lengths, and
the pinned llama.cpp AVX2 reduction result.

The full 42-layer path reaches final logits, but parity is not closed. The
latest non-repacked reference comparison had max absolute error `2.1698594`,
mean absolute error `0.38767775`, RMS error `0.48177935`, identical top-1, and
18/20 top-20 overlap. Layer samples begin crossing `0.05` at layer 7 and drift
smoothly rather than failing at one projection. This remains a numeric-parity
blocker; the `0.05` tolerance must not be widened to hide accumulated drift.

The correctness-first loader currently materializes eager F32 fallbacks in
addition to native sidecars. That simplifies discriminators but duplicates
large weights. A production-alpha implementation should avoid dense
materialization where a validated native sidecar exists and measure peak memory,
load latency, and tokens per second on the selected artifact.

## Kernel Contract

Quantized kernels should document whether they dequantize eagerly, dequantize on
the fly, or use native quantized matmul. Each choice has different memory and
performance tradeoffs.

## Validation

Quantized tests should include:

- Shape validation.
- Known small tensors with exact expected dequantized values.
- End-to-end generation smoke tests.
- Parity against a trusted reference for representative prompts.
