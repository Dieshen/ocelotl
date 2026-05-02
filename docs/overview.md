# Ocelotl Overview

Ocelotl is a Rust-first LLM inference runtime. The long-term goal is a local
runtime with explicit model, loader, tokenizer, kernel, KV-cache, scheduler,
and server boundaries. The near-term goal is narrower: prove one model path is
correct before expanding architecture coverage or serving throughput.

## Goals

- Provide a Rust-native runtime for decoder-only language model inference.
- Keep model semantics separate from runtime scheduling and kernel dispatch.
- Support a CPU reference path before relying on GPU kernels.
- Add portable GPU execution through a kernel boundary that can use CubeCL,
  CubeK, Burn, or hand-written Rust where each is appropriate.
- Make correctness measurable with fixture-based parity tests.
- Fail explicitly for unsupported model features instead of silently producing
  plausible but wrong output.

## Non-Goals

- Matching vLLM throughput in the first milestones.
- Supporting every model family at the start.
- Supporting every quantization format before the unquantized path is correct.
- Hiding model-specific behavior behind vague generic abstractions.
- Treating GPU execution as correct without CPU/reference parity.

## Initial Target

The first useful runtime should support one narrow decoder-only model family,
one process, one model loaded from disk, deterministic prefill and decode, and
a small set of offline validation fixtures.

Paged KV, continuous batching, OpenAI-compatible serving, quantized weights,
GGUF compatibility, and multi-GPU execution are follow-on milestones.
