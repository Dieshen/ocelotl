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
| `Q4K` | 256 | 144 | exact dequant tested | rejected |
| `Q5K` | 256 | 176 | exact dequant tested | rejected |
| `Q6K` | 256 | 210 | exact dequant tested | rejected |

`inspect_gguf` rejects Q4K/Q5K/Q6K tensors whose row width or total element
count is not divisible by 256, and validates the computed byte range against
the file length. This follows ggml's row contract: a tensor such as `[128, 2]`
has 256 total elements but is not a valid K-quant row layout.
`load_gguf_tensor_f32` stays dense-only and still rejects quantized payloads with
a typed `Unsupported` error. Callers that intend to materialize quantized values
must use `load_gguf_tensor_dequantized_f32` or
`load_gguf_tensors_dequantized_f32`.

The Q4K/Q5K/Q6K dequantizers use ggml's current K-quant layouts and bit packing
from `ggml/src/ggml-common.h` and `ggml/src/ggml-quants.c`. Default tests pin
small exact dequantized vectors for each format, including Q4/Q5 scale/min
packing across all Q4_K scale/min groups, Q5 high-bit lanes, and Q6 signed
scales. This is still a loader value contract, not a native quantized-kernel
execution claim. MF.7 adds a dense F32/dequantized-F32 text subset; follow-up
text-forward slices handle mixed SWA/global layer widths, GGUF sliding-window
pattern metadata, sliding-window masking, shared-KV reuse, final logit softcap,
post-branch RMSNorms, per-layer embeddings, layer output scales, and Gemma4
RoPE frequency factors for that subset. Follow-up coverage also executes
explicitly dequantized Q4_K_M-origin text tensors in that path. Raw quantized
GGUF tensors remain rejected before compute. The selected real Gemma4 GGUF
artifact remains blocked until multimodal handling and native quantized-kernel
or equivalent llama.cpp reference parity work exists.

As of 2026-06-11, the Gemma4 tensor-summary harness proves the GGUF matrix
layout bridge far enough to match llama.cpp at `inp_scaled` and `attn_norm-0`
for the selected Q4_K_M artifact. The first remaining mismatch is the layer-0
query projection `Qcur-0`, after the first K-quant matrix multiply. A native
diagnostic then proved the selected artifact stores `blk.0.attn_q.weight` as
Q6_K and that a ported llama.cpp-style Q6_K x Q8_K dot matches the `Qcur-0`
summary. Treat this as a production matrix-execution gap: scalar
Q4_K/Q5_K/Q6_K dequant tests remain green, and the native K-quant proof is
green, but eager F32 dequant plus F32 matmul is not a llama.cpp-equivalent
projection path.

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
