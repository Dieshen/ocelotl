# Post-M3 Whisper ASR Tasks

Working shorthand: "M3.3 Whisper". This is a post-M3 expansion track and does
not modify the closed M3.3 RMSNorm task.

## Kickoff Status

Kicked off 2026-05-08 as a post-M3 expansion track. W-ASR.1 is the senior-owned
boundary/docs task. W-ASR.2 and W-ASR.3 are the first implementation slices once
junior agents are explicitly dispatched.

Phase split:

- Phase 0: W-ASR.1 boundary and artifact contract.
- Phase 1: W-ASR.2 audio/log-mel fixture and W-ASR.3 tokenizer startup rules.
- Phase 2: W-ASR.4 tiny synthetic Whisper path.
- Phase 3: W-ASR.5 local-artifact parity and W-ASR.6 runtime API shape.

## W-ASR.1 Define The Whisper Boundary

- `Crates`: docs first; likely `ocelotl-models`, `ocelotl-loader`,
  `ocelotl-tokenizer`, and `ocelotl-runtime` later.
- `Test first`: no code test; add this milestone/task pair and link the
  boundary from `docs/roadmap.md`.
- `Done when`: docs name the first artifact format, API boundary, non-goals,
  and the rule that Burn types do not escape public Ocelotl APIs.

## W-ASR.2 Add Audio Fixture And Log-Mel Reference

- `Crates`: likely future `ocelotl-audio` or `ocelotl-models::whisper`.
- `Test first`: a tiny 16 kHz mono waveform fixture maps to pinned log-mel
  values.
- `Done when`: unsupported sample rates and non-mono input fail before compute,
  and the deterministic log-mel fixture passes offline.
- `Status note`: first slice landed in `ocelotl-models::whisper::audio` with a
  scalar CPU reference and no Burn dependency. If the audio surface grows beyond
  Whisper model-family semantics, split a dedicated audio crate before exposing
  it through runtime APIs.

## W-ASR.3 Pin Whisper Tokenizer Startup Rules

- `Crates`: `ocelotl-tokenizer`.
- `Test first`: a fixture asserts start-of-transcript, language, task,
  no-timestamp, timestamp, and end-of-text token handling.
- `Done when`: no-timestamp masking and special-token treatment are explicit and
  tested for the whole decode loop, not just the prompt prefix.

## W-ASR.4 Build A Tiny Synthetic Whisper Path

- `Crates`: `ocelotl-models`, `ocelotl-kernels` if needed.
- `Test first`: construct a tiny synthetic Whisper-shaped encoder/decoder and
  assert output shape plus one pinned token/logit fixture.
- `Done when`: the model path proves encoder, cross-attention, decoder, and
  token projection wiring without loading a real model.

## W-ASR.5 Add Opt-In Local-Artifact Parity

- `Crates`: `ocelotl-loader`, `ocelotl-models`, `ocelotl-tokenizer`.
- `Test first`: an ignored test checks for a local converted Whisper tiny model
  and skips with a clear message when missing.
- `Done when`: the test documents exact artifact paths and compares one short
  audio clip against a pinned reference transcript or token sequence.

## W-ASR.6 Add Runtime Transcription API Shape

- `Crates`: `ocelotl-runtime`.
- `Test first`: runtime rejects empty audio and unsupported audio metadata
  before model compute.
- `Done when`: runtime exposes Ocelotl-owned transcription request/result types
  and reaches the Whisper model through the same public lifecycle discipline as
  Qwen prefill/decode.

## Track Closure

The track closes when a default-on offline fixture proves audio preprocessing and
a tiny synthetic Whisper-shaped decode path, and an ignored local-artifact test
documents real-model parity.
