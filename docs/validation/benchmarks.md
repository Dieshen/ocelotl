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

## Deterministic Criterion Targets

The first default-off microbenchmark pyramid uses Criterion 0.5 with synthetic,
network-free inputs:

```powershell
cargo bench -p ocelotl-kernels --bench kernel_primitives
cargo bench -p ocelotl-loader --bench loader_artifacts
cargo bench -p ocelotl-models --bench whisper_audio
cargo bench -p ocelotl-runtime --bench qwen_cached_decode
```

The targets currently cover:

- scalar `linear_out_by_in` and Q6_K x Q8_K projection;
- safetensors header inspection and F32 value loading;
- steady-state Whisper log-mel preprocessing after one-time Fourier setup;
- one Qwen2.5 contiguous-cache decode step with cloned setup state excluded
  from the measured routine.

Use `cargo bench --workspace --no-run` as the compile gate. Use `-- --test` on
an individual target for a fast execution smoke test without collecting a
baseline. Benchmark names include the operation and synthetic shape so future
shape additions remain comparable rather than silently replacing a workload.

Criterion baselines are developer-machine evidence until a controlled runner
records hardware, toolchain, lockfile, revision, and clean/dirty state. Do not
make cross-machine Criterion deltas a CI gate.

## Whisper ASR Baseline

For W-ASR.13, whisper.cpp is a performance baseline only, not the canonical
correctness oracle. The harness contract lives in
`docs/benchmarks/whisper-cpp.md`, with default-on schema tests in
`crates/models/tests/whisper_cpp_benchmark.rs` and example JSON fixtures under
`fixtures/benchmarks/`.
