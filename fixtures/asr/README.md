# Parakeet ASR parity fixtures

Audio fixtures and reference tensors are **not committed** — same convention as the
Whisper local-artifact proofs (`docs/status.md`: "the proof remains opt-in because
weights and reference binaries are not committed"). Everything here is reproducible
from the two commands below.

Three fixtures, not one, per the fixture-per-failure-mode rule — `jfk` alone cannot
catch a per-feature-normalization bug, because that failure only shows across clips
of differing length and level:

| fixture | length | level | stresses |
|---|---|---|---|
| `parity_jfk.wav` | ~11 s | ~-17 dBFS | ordinary speech, the decode loop |
| `parity_short.wav` | 0.75 s | ~-17 dBFS | frame-count off-by-one; the unbiased `T-1` std denominator at small `T` |
| `parity_long_quiet.wav` | 121 s | ~-30 dBFS | per-feature statistics and `T` arithmetic at scale |

## 1. Fixtures

```sh
curl -sL -o fixtures/asr/parity_jfk.wav \
  https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav
python3 fixtures/asr/make_fixtures.py     # derives parity_short + parity_long_quiet
```

## 2. Reference tensors (needs `onnxruntime`, once)

The oracle is `nemo128.onnx` from `istupakov/parakeet-tdt-0.6b-v3-onnx` — a stock
`ASRModel.export()` of the official NVIDIA checkpoint, 140 KB. It is the frontend
in isolation, which is why the frontend can be gated before any encoder exists.

```sh
curl -sL -o ~/models/parakeet/nemo128.onnx \
  https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/nemo128.onnx
python3 fixtures/asr/dump_reference.py    # writes ~/models/parakeet/ref/
OCELOTL_PARAKEET_REF_DIR=~/models/parakeet/ref \
  cargo test -p ocelotl-models --release --test parakeet_frontend_parity -- --ignored --nocapture
```

Model weights are CC-BY-4.0 (NVIDIA). Attribution is required for any distribution
that includes or downloads them — unlike Whisper's MIT, which required nothing.
