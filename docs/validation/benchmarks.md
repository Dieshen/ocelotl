# Benchmarks

Benchmarks measure performance after correctness is established. They should not
replace parity tests.

## Benchmark Layers

- Loader latency.
- Tokenizer throughput.
- Single-kernel latency.
- Prefill throughput.
- Decode tokens per second.
- End-to-end request latency.
- Scheduler throughput under multiple requests.
- Memory use and cache pressure.

## Reporting

Benchmark reports should include:

- Hardware.
- OS and driver versions.
- Backend and feature flags.
- Model and dtype.
- Prompt length.
- Generated token count.
- Batch size or request count.
- Mean and percentile latency where relevant.

## Baselines

Keep separate baselines for:

- CPU reference.
- First GPU path.
- Optimized GPU path.
- Quantized path.
- External performance baselines such as whisper.cpp.

Do not compare unrelated models or quantization formats as if they were the same
benchmark.

## First Benchmark Target

After M3, measure CPU prefill and decode separately. After M4, compare GPU and
CPU for the same fixture and report both speed and parity results.

## Whisper ASR Baseline

For W-ASR.13, whisper.cpp is a performance baseline only, not the canonical
correctness oracle. The harness contract lives in
`docs/benchmarks/whisper-cpp.md`, with default-on schema tests in
`crates/models/tests/whisper_cpp_benchmark.rs` and example JSON fixtures under
`fixtures/benchmarks/`.
