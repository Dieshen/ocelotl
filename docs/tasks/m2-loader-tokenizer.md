# M2 Tasks

M2 connects local artifact inspection and tokenizer behavior for the first target
model family. It should still avoid large checked-in model weights and networked
default tests.

## Entry Criteria

- M1 CPU reference contracts are in place.
- `docs/model-target.md` names the first real artifact candidate and license.
- Fixture policy in `docs/validation/fixtures.md` is current.

## Task List

- [ ] M2.1 Pin the first external model artifact revision.
  - Crates: docs and fixtures
  - Test first: add a metadata fixture or manifest field that records the expected model repository and revision.
  - Done when: tests and docs refer to an exact artifact identity, not a moving branch name.

- [ ] M2.2 Add tokenizer JSON loading behind the tokenizer boundary.
  - Crates: `ocelotl-tokenizer`
  - Test first: load a small local tokenizer fixture and assert exact token IDs for a basic prompt.
  - Done when: callers use an Ocelotl tokenizer trait or wrapper instead of depending directly on `tokenizers` APIs.

- [ ] M2.3 Fill the Qwen2.5 tokenizer fixture expectations.
  - Crates: `ocelotl-tokenizer`, fixtures
  - Test first: update `fixtures/tokenizer/qwen2_5_basic_prompt.json` with exact prompt text, token IDs, and decoded text.
  - Done when: encode and decode round trips are covered with explicit IDs.

- [ ] M2.4 Define chat-template behavior for the target model.
  - Crates: `ocelotl-tokenizer`
  - Test first: add a fixture for system/user/assistant messages and the exact rendered prompt or token sequence.
  - Done when: chat template output is deterministic and unsupported template features fail explicitly.

- [ ] M2.5 Inspect safetensors metadata without executing weights.
  - Crates: `ocelotl-loader`
  - Test first: add a tiny safetensors fixture and assert tensor names, shapes, dtypes, and byte ranges.
  - Done when: loader can enumerate tensor metadata and reject malformed or missing tensors without model execution.

- [ ] M2.6 Map artifact metadata into core model metadata.
  - Crates: `ocelotl-loader`, `ocelotl-core`
  - Test first: parse a local config fixture and assert the resulting `ModelMetadata` matches the M1 typed shape.
  - Done when: loader owns artifact parsing and core owns the validated metadata types.

- [ ] M2.7 Add malformed artifact tests.
  - Crates: `ocelotl-loader`, `ocelotl-tokenizer`
  - Test first: add fixtures for missing tokenizer file, invalid JSON, missing required tensor, unsupported dtype, and shape mismatch.
  - Done when: each malformed fixture produces a specific typed error.

- [ ] M2.8 Keep default tests offline.
  - Crates: workspace and CI
  - Test first: CI fails if a default test attempts to fetch model artifacts from the network.
  - Done when: optional artifact download helpers are ignored, feature-gated, or documented as manual commands.

- [ ] M2.9 Document artifact preparation.
  - Crates: docs only
  - Test first: docs mention exact local paths expected by tests before implementation relies on them.
  - Done when: contributors know how to place local tokenizer/config/safetensors files without committing large model weights.

## Exit Criteria

- Tokenizer encode/decode fixtures pass offline.
- Chat-template behavior for the target model is explicit and tested.
- Loader can inspect local safetensors metadata and reject malformed artifacts.
- Core metadata remains the shared contract, not loader-specific structs.
- No default validation command requires network access.

## Deferred

- Full weight loading into model execution.
- Quantized artifact support.
- Broad tokenizer-family support.
- Remote artifact management.
