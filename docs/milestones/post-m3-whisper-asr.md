# Post-M3 Whisper ASR Track

Working shorthand: "M3.3 Whisper". This is not the closed M3.3 task in
`docs/tasks/m3-single-model-forward.md`; that task is CPU RMSNorm. This document
captures the proposed post-M3 speech-to-text track without reopening M3
acceptance history.

## Goal

Add a bounded Whisper speech-to-text path to Ocelotl using `whisper-burn` as
reference material and Burn as an internal implementation option.

The first useful result is not a general multimodal runtime. It is:

- 16 kHz mono audio input.
- Deterministic log-mel preprocessing.
- A Whisper-shaped encoder/decoder path.
- Token/text output through Ocelotl-owned APIs.
- Offline fixture parity for a tiny or synthetic model.

## Reference Material

- `https://github.com/Gadersd/whisper-burn`
- `https://huggingface.co/Gadersd/whisper-burn`

The upstream repo is useful because it already demonstrates a Rust Whisper
implementation using Burn, converted model files, audio preprocessing, tokenizer
special-token handling, and a WGPU option.

Treat it as reference or vendored source material, not a stable dependency:

- No published releases.
- Burn is pulled from git in upstream.
- Upstream uses older tokenizer dependencies.
- It assumes 16 kHz single-channel audio.
- Its artifact format and conversion flow are not Ocelotl's loader contract.

## Boundary

Whisper should not be folded into the existing Qwen2.5 causal-LLM path.

Preferred first shape:

- `ocelotl-models` owns Whisper model semantics.
- `ocelotl-loader` owns Whisper artifact manifests and metadata validation.
- `ocelotl-tokenizer` owns Whisper tokenizer/special-token loading behind
  Ocelotl-owned traits.
- `ocelotl-runtime` exposes Ocelotl-owned transcription request/result types.
- Burn tensor types stay internal to `ocelotl-models` or `ocelotl-kernels`.

If the audio surface grows, split a dedicated `ocelotl-audio` or `ocelotl-asr`
crate before letting audio preprocessing leak into runtime or model-family code.

## First Artifact Contract

The first Ocelotl-owned Whisper artifact contract should be a local bundle, not
the upstream `whisper-burn` crate as a transitive dependency:

```text
local-artifacts/whisper_tiny_en/
├── config.json
├── tokenizer.json
├── model.safetensors
└── reference/
    ├── sample_16khz_mono.wav
    └── expected_tokens.json
```

`whisper-burn` and its Hugging Face converted files can seed the converter or
fixture capture process, but Ocelotl's loader should validate Ocelotl-owned
metadata and tensor manifests. Burnpack can be evaluated later as a deployment
format; it should not be the only format that proves correctness.

Default tests should use committed tiny waveform/log-mel fixtures and synthetic
weights. Real `whisper_tiny_en` tests are `#[ignore]` and local-artifact gated.

## Non-Goals

- Replacing Qwen2.5 M3 runtime APIs.
- Using Gemma4 audio as the first ASR implementation.
- Timestamped segmentation in the first slice.
- Streaming transcription.
- Speaker diarization.
- Full Whisper model coverage.
- Public Burn types in Ocelotl APIs.
- Network-dependent default tests.

## TDD Plan

Write tests before implementation for:

- 16 kHz mono input acceptance and unsupported audio rejection.
- Deterministic log-mel output for a tiny committed waveform fixture.
- Whisper tokenizer special-token IDs and no-timestamp masking.
- Encoder/decoder shape checks with a tiny synthetic model.
- One short opt-in local-artifact parity test against converted `tiny.en` or
  another pinned reference model.

## Design Notes

### Audio preprocessing

Start with the constants and flow that Whisper expects:

- sample rate: 16 kHz
- channels: mono
- FFT size: 400
- hop length: 160
- mel bins: 80

The first default-on tests should use tiny committed audio fixtures and expected
mel values. Real audio/model tests stay `#[ignore]` until artifact policy is
documented.

### Decode policy

Whisper decode is not Qwen decode. It needs start-of-transcript, language, task,
timestamp/no-timestamp, end-of-text, and special-token masking rules. Keep those
rules explicit and fixture-tested instead of burying them in a generic tokenizer
wrapper.

### Burn usage

Burn is acceptable here because Whisper Burn already proves a plausible Rust
implementation shape. Keep Burn behind an internal backend alias or adapter.
Do not expose Burn tensors from runtime APIs and do not let Burn records become
the only artifact format Ocelotl understands.

## Acceptance Criteria

- A Whisper track has explicit docs, task backlog, and validation commands.
- Audio preprocessing rejects unsupported sample rates/channels before compute.
- Log-mel preprocessing has a deterministic fixture test.
- Whisper token startup/masking rules are fixture-tested.
- A tiny synthetic Whisper-shaped path proves encoder/decoder shape and decode
  flow without network access.
- Any real-model parity test is opt-in, local-artifact gated, and documented.
- Burn remains an internal implementation detail.

## Validation Commands

Initial docs-only validation:

```powershell
cargo fmt --all -- --check
cargo test --workspace
pwsh -NoProfile -File ci/check-offline.ps1
```

When code lands, add focused crate tests for whichever crate owns the first
slice.

## Known Risks

- Whisper can pull Ocelotl toward a broad multimodal runtime before the LLM
  path has GPU/KV/server maturity.
- Audio preprocessing parity is easy to get almost-right while still producing
  plausible transcripts.
- Burn API or record-format churn could infect Ocelotl if not kept behind a
  strict boundary.
- Timestamp behavior is subtle and should not be mixed into the first decode
  slice unless a fixture requires it.
