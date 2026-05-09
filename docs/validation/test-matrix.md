# Test Matrix

This matrix maps milestones to required tests. Each milestone spec can add more,
but it should not provide less than this baseline.

| Milestone | Required Tests |
| --- | --- |
| M0 Skeleton | Workspace check, formatting, publish dry runs. |
| M1 CPU Reference | Unit tests for request validation, fixture test for deterministic CPU output, unsupported-config tests. |
| M2 Loader And Tokenizer | Loader good/bad fixtures, exact tokenizer ID fixtures, chat-template fixtures. |
| M3 Single Model Forward | Prefill logits parity, one-token decode parity, shape/dtype failure tests. |
| M4 GPU Kernel Path | CPU/GPU parity for each GPU kernel, unsupported device/dtype tests. |
| M5 Contiguous KV Cache | KV read/write position tests, prefill/decode cache parity, request isolation tests. |
| M6 Paged KV Cache | Page allocation tests, multi-page decode test, contiguous/paged parity. |
| M7 Continuous Batching | Scheduler ordering tests, cancellation tests, batched/unbatched parity. |
| M8 Server API | API request validation, runtime error mapping, streaming lifecycle tests. |
| Post-M3 Whisper ASR | Audio metadata/log-mel tests, tokenizer startup/mask tests, tiny synthetic Whisper path tests, ignored local-artifact parity harness. |

## Validation Tiers

Focused commands should be used while developing. Workspace commands are required
before merging.

Focused examples:

```powershell
cargo test -p ocelotl-loader
cargo test -p ocelotl-tokenizer
cargo test -p ocelotl-runtime
```

Workspace gate:

```powershell
cargo fmt --all
cargo test --workspace
cargo check --workspace
```

## Offline Rule

Default tests must not require network access. Any network-dependent test should
be ignored by default and documented with the exact command to run it.

## M1 Acceptance Traceability

The table below maps each acceptance criterion in
`docs/milestones/m1-cpu-reference.md` to the test (or test set) that proves it.
Reviewers should be able to read this table and confirm every M1 acceptance
bullet has a green test in `cargo test --workspace`. As tests land, fill in the
`status` column; placeholders cite the task that will land the test.

| # | Acceptance criterion | Test(s) proving it | Status |
| - | -------------------- | ------------------ | ------ |
| 1 | One supported model shape declared explicitly: tiny synthetic Qwen2.5-shaped decoder-only metadata loads into typed structs. | `ocelotl_core::tests::qwen2_5_tiny_synthetic_fixture_deserializes_correctly` (in `crates/core/src/lib.rs`). | green |
| 2 | Unsupported features fail before execution begins. | Loader: `ocelotl_loader::tests::{load_metadata_rejects_unknown_architecture_with_typed_unsupported_error, load_metadata_rejects_unknown_dtype_with_typed_unsupported_error, load_metadata_rejects_missing_required_field_with_invalid_model_error}`. Runtime: `ocelotl_runtime::tests::{validate_request_rejects_empty_prompt, validate_request_rejects_zero_max_new_tokens, validate_request_rejects_temperature_with_unsupported_sampling_mode, validate_request_rejects_context_overflow, validate_request_temperature_check_fires_before_other_violations}`. | green |
| 3 | Prefill and one-token decode run through `ocelotl-runtime`. | `ocelotl_runtime::m1_cpu_reference_smoke_produces_expected_token` (integration test at `crates/runtime/tests/m1_smoke.rs`) — wires `validate_request` → `tiny_synthetic_forward` → `greedy_sample` and asserts a pinned next-token. | green |
| 4 | A fixture test validates deterministic output without network access. | Same M1.9 smoke test as criterion 3 — runs offline by construction (no `--ignored`, no network calls, no model downloads). Determinism also pinned by `ocelotl_models::tests::tiny_synthetic_forward_is_deterministic_for_identical_inputs`. | green |
| 5 | Output is compared against a documented reference or committed fixture. | M1.9 smoke test asserts `TokenId(5)` for prompt `[TokenId(7)]` against `fixtures/logits/m1_smoke_expected.json`. The fixture's `expected_next_token` field is the pinned reference; updates require regeneration when `tiny_synthetic_forward` intentionally changes. | green |
| 6 | Shape, dtype, and context-length errors are explicit. | Dtype: `ocelotl_loader::tests::load_metadata_rejects_unknown_dtype_with_typed_unsupported_error`. Context-length: `ocelotl_runtime::tests::{validate_request_rejects_context_overflow, validate_request_accepts_request_exactly_filling_context}` (boundary pinned). Shape: kernels rejection tests `ocelotl_kernels::tests::{vec_add_rejects_mismatched_input_lengths, vec_add_rejects_mismatched_output_length, dot_rejects_mismatched_lengths, matmul_rejects_inner_dimension_mismatch, matmul_rejects_wrong_a_slice_length, matmul_rejects_wrong_output_length}`. Display: `ocelotl_core::tests::{invalid_model_error_display_includes_path_field_and_message, unsupported_error_display_mentions_feature_requested_and_supported, kernel_error_display_includes_backend_and_message}`. | green |

### Closure note (2026-05-03)

All six rows are green. M1 acceptance is provable from `cargo test --workspace` against the commit set on main through `b7cf755 test(runtime): add M1 smoke integration test through public API`. Total at M1 close: 17 core + 15 kernels + 5 loader + 4 models + 12 runtime unit + 1 runtime smoke + 4 kernel doctests = 58 tests passing.

### Note — offline by construction

M1 default tests are offline by construction; the offline-by-default principle
and its forward-looking implications for milestones that introduce network
access live in `docs/ci.md`.

## M2 Acceptance Traceability

The table below maps each acceptance criterion in
`docs/milestones/m2-loader-tokenizer.md` to the test (or test set) that proves
it. Same shape as the M1 table; reviewers should be able to confirm every M2
acceptance bullet has a green test in `cargo test --workspace`.

| # | Acceptance criterion | Test(s) proving it | Status |
| - | -------------------- | ------------------ | ------ |
| 1 | `ocelotl-loader` exposes normalized metadata for at least one supported Qwen2.5-shaped model fixture. | `ocelotl_loader::tests::parse_hf_config_maps_real_qwen2_5_config_into_model_metadata` (in `crates/loader/src/lib.rs`) — parses `fixtures/metadata/qwen2_5_0_5b_instruct_config.json` (real Qwen config at pinned SHA `7ae5576...`) and asserts the resulting `core::ModelMetadata` matches the M1 typed shape. | green |
| 2 | `ocelotl-loader` rejects malformed metadata fixtures. | Architecture: `load_metadata_rejects_unknown_architecture_with_typed_unsupported_error`. Dtype: `load_metadata_rejects_unknown_dtype_with_typed_unsupported_error`. Missing required field: `load_metadata_rejects_missing_required_field_with_invalid_model_error`. HF config gates: `parse_hf_config_rejects_unknown_model_type_with_unsupported`, `parse_hf_config_rejects_unknown_torch_dtype_with_unsupported`, `parse_hf_config_rejects_non_divisible_head_dim_with_invalid_model`. Missing file → `Io`: `load_metadata_returns_io_error_when_file_does_not_exist`. Safetensors header malformed/truncated/shape-mismatch/unsupported-dtype/missing-tensor: `inspect_safetensors_rejects_truncated_header_with_invalid_model_error`, `inspect_safetensors_rejects_unsupported_dtype_with_typed_unsupported_error`, `inspect_safetensors_rejects_shape_offsets_mismatch_with_invalid_model_error`, `require_tensors_returns_invalid_model_error_when_a_required_tensor_is_missing`, `inspect_safetensors_returns_io_error_when_file_does_not_exist`. The full inventory lives at `docs/validation/correctness.md` § Malformed Artifact Coverage (M2.7). | green |
| 3 | `ocelotl-tokenizer` encodes and decodes known fixtures exactly. | Tiny WordLevel boundary tests: `ocelotl_tokenizer::tests::{json_tokenizer_loads_tiny_wordlevel_fixture_and_encodes_known_input, json_tokenizer_decode_returns_known_text_for_known_ids, json_tokenizer_missing_file_returns_typed_tokenizer_error_with_path, json_tokenizer_malformed_json_returns_typed_tokenizer_error_with_path}`. Qwen2.5 fixture shape (default-on): `crates/tokenizer/tests/qwen2_5_basic_prompt.rs::fixture_is_well_formed_and_populated` asserts `expected_token_ids = [9707]` for `"Hello"`. Real-tokenizer round trip (`#[ignore]`'d, opt-in via `local-artifacts/`): `json_tokenizer_round_trips_qwen2_5_basic_prompt`. | green |
| 4 | Chat-template behavior is covered by a deterministic test. | Inline-fixture exact-bytes pin: `crates/tokenizer/tests/qwen2_5_chat_template.rs::chat_template_renders_inline_fixture_to_expected_bytes` (176-byte rendered output for system+user+assistant+user with `add_generation_prompt=true`). Determinism across calls: `chat_template_render_is_byte_for_byte_deterministic_across_repeated_calls`. Lenient-undefined contract pinned: `apply_treats_undefined_as_lenient_to_match_upstream_jinja2_semantics`. Unsupported-template-feature failures: `from_jinja_rejects_unsupported_statement_with_typed_unsupported_error`, `apply_surfaces_unknown_filter_via_typed_unsupported_error`, `from_jinja_distinguishes_genuine_syntax_error_from_unsupported_feature`. Upstream cross-check (`#[ignore]`'d): `inline_chat_template_matches_upstream_pinned_tokenizer_config`. | green |
| 5 | Runtime-facing metadata types contain enough fields for M3 model construction. | Provable now via `parse_hf_config_maps_real_qwen2_5_config_into_model_metadata`: the test asserts the produced `core::ModelMetadata` carries every field M3 will need (architecture, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, intermediate_size, max_position_embeddings, rope_theta, head_dim, dtype). Final proof lands when M3 constructs a real model from this metadata; M2 closes this criterion at "shape is sufficient", M3 will close it at "shape is correct". | green (shape) |
| 6 | Default tests do not require network access. | Static enforcement: `ci/check-offline.ps1` (added M2.8) scans all non-`#[ignore]` test code for forbidden patterns (`reqwest::`, `ureq::`, `hf_hub::`, `huggingface_hub`, literal `huggingface.co` URLs) and fails CI if any appear. Wired into `.github/workflows/ci.yml` between `cargo check` and `cargo test`. Behavioral: zero default tests fetch artifacts (verified by gate exit-0 on `0a63350`); the only network-dependent tests are the `#[ignore]`'d opt-in pair `json_tokenizer_round_trips_qwen2_5_basic_prompt` and `inline_chat_template_matches_upstream_pinned_tokenizer_config`, both gated on `local-artifacts/qwen2_5_0_5b_instruct/...` per `docs/artifact-preparation.md`. | green |

### Closure note (2026-05-03)

All six rows are green. M2 acceptance is provable from `cargo test --workspace`
against main through `0a63350 merge: M2.4 — chat-template behavior for
Qwen2.5 (dev-03)`. Total at M2 close: **85 default + 2 `#[ignore]`'d** tests
passing. Per-crate breakdown: 17 core + 15 kernels + 18 loader (5 baseline +
6 from M2.5 + 7 from M2.6 sweep+mapping) + 4 models + 12 runtime + 1 runtime
smoke + 10 tokenizer (4 from M2.2 + 6 from M2.4 unit) + 1 doctest in
tokenizer + 3 chat-template integration + 4 kernel doctests = 85 default;
plus 1 `#[ignore]`'d in `qwen2_5_basic_prompt` and 1 in `qwen2_5_chat_template`.

The M2.8 offline gate (`ci/check-offline.ps1`) extends acceptance enforcement
beyond `cargo test`: criterion 6 is now CI-blocking, not advisory.

## M3 Acceptance Traceability

The table below maps each acceptance criterion in
`docs/milestones/m3-single-model-forward.md` to the test (or test set) that
proves it. Same shape as M1/M2: reviewers should be able to confirm every M3
acceptance bullet has a green test in `cargo test --workspace`.

| # | Acceptance criterion | Test(s) proving it | Status |
| - | -------------------- | ------------------ | ------ |
| 1 | Qwen2.5-style dense decoder-only inference has explicit support. | `ocelotl_models::qwen::qwen2_5::tests::try_from_accepts_real_qwen2_5_0_5b_metadata` proves the real Qwen2.5 metadata contract (`model_type = "qwen2"`, GQA, BF16). `ocelotl_models::qwen::qwen2_5_model::tests::{new_rejects_layer_count_mismatch_with_invalid_model, new_rejects_wrong_embed_tokens_length_with_invalid_model, prefill_returns_one_logit_per_vocab_entry_for_single_token_prompt}` prove the concrete `Qwen2_5Model` construction and output surface. Tensor coverage is pinned by `ocelotl_models::qwen::qwen2_5_tensors::tests::{required_tensor_names_enumerates_full_set_for_tied_and_untied, validate_accepts_complete_manifest_with_tied_embeddings, validate_accepts_complete_manifest_with_untied_embeddings}`. | green |
| 2 | Prefill and decode are separate operations in the runtime path. | Prefill: `ocelotl_runtime::tests::prefill_returns_logits_through_runtime_api_surface` and `crates/models/tests/qwen2_5_tiny_synthetic_prefill.rs::prefill_matches_pinned_fixture_within_tolerance`. Decode: `ocelotl_runtime::tests::decode_one_token_returns_a_token_id_for_valid_prompt` and `crates/runtime/tests/qwen2_5_tiny_synthetic_decode.rs::{decode_one_token_is_deterministic_for_identical_inputs, decode_one_token_matches_pinned_argmax_of_m3_7_fixture, decode_one_token_propagates_invalid_request_for_empty_prompt}`. | green |
| 3 | Reference fixtures cover at least one short prompt. | `fixtures/logits/qwen2_5_tiny_synthetic_prefill.json` pins final-token logits for prompt `[3, 7, 11]` on the tiny synthetic Qwen2.5 shape. `crates/models/tests/qwen2_5_tiny_synthetic_prefill.rs::prefill_matches_pinned_fixture_within_tolerance` loads that fixture and compares every logit. `crates/runtime/tests/qwen2_5_tiny_synthetic_decode.rs::decode_one_token_matches_pinned_argmax_of_m3_7_fixture` derives `TokenId(16)` from the same fixture and the greedy-sampling contract. | green |
| 4 | Greedy output is deterministic. | `ocelotl_runtime::sampling::tests::{greedy_sample_picks_index_of_largest_logit, greedy_sample_breaks_ties_by_lowest_token_id}` pin the greedy contract. `crates/runtime/tests/qwen2_5_tiny_synthetic_decode.rs::decode_one_token_is_deterministic_for_identical_inputs` proves repeated decode calls return the same token through the public path. | green |
| 5 | Unsupported model metadata fails before execution. | `ocelotl_models::qwen::qwen2_5::tests::{try_from_rejects_non_qwen2_architecture_with_unsupported_error, try_from_rejects_non_divisible_gqa_grouping_with_invalid_model_error, try_from_rejects_odd_head_dim_with_invalid_model_error, try_from_rejects_zero_rope_theta_with_invalid_model_error, try_from_rejects_negative_rope_theta_with_invalid_model_error, try_from_rejects_zero_context_length_with_invalid_model_error, try_from_rejects_oversized_context_length_with_invalid_model_error, try_from_rejects_zero_vocab_size_with_invalid_model_error, try_from_rejects_one_vocab_size_with_invalid_model_error, try_from_rejects_unsupported_dtype_with_typed_unsupported_error, try_from_rejects_quantized_dtype_with_typed_unsupported_error}`. Tensor shape/name failures are pinned by `ocelotl_models::qwen::qwen2_5_tensors::tests::{validate_rejects_missing_tensor_with_invalid_model_error_naming_the_tensor, validate_rejects_wrong_shape_with_invalid_model_error_naming_the_tensor, validate_rejects_missing_lm_head_only_when_embeddings_are_untied}`. Runtime prompt gates are pinned by `ocelotl_models::qwen::qwen2_5_model::tests::{prefill_rejects_empty_prompt_with_invalid_request, prefill_rejects_prompt_longer_than_context_with_invalid_request, prefill_rejects_token_id_outside_vocab_with_invalid_request}`. | green |
| 6 | Tests document the reference source and tolerance. | `docs/validation/parity.md` names the M3 fixture source, `1e-4` prefill tolerance, exact-token decode rule, and deferred real-Qwen target tolerance. `crates/models/tests/qwen2_5_m3_parity_policy.rs::parity_doc_names_m3_sources_and_tolerances` makes that documentation executable by failing if the M3 parity doc stops naming the source, tolerance, or decode token. The fixture-side tolerance self-check remains in `prefill_matches_pinned_fixture_within_tolerance`. | green |

### Closure note (2026-05-05)

All six rows are green. M3 acceptance is provable from `cargo test --workspace`
against main through `f463f70 test(validation): formalize M3 parity policy
tripwires`. Total at M3 close: **157 default tests + 8 doctests passing**; the
2 M2 local-artifact tests remain `#[ignore]`'d opt-in checks. The M2.8 offline
gate (`ci/check-offline.ps1`) also passes, so the M3 default validation surface
remains offline by construction.

## Post-M3 Whisper ASR Acceptance Traceability

The table below maps `docs/milestones/post-m3-whisper-asr.md` acceptance
criteria to the tests and docs that prove the current post-M3 Whisper track.

| # | Acceptance criterion | Test(s) proving it | Status |
| - | -------------------- | ------------------ | ------ |
| 1 | A Whisper track has explicit docs, task backlog, and validation commands. | `docs/milestones/post-m3-whisper-asr.md`, `docs/tasks/post-m3-whisper-asr.md`, and the validation commands in the milestone doc. | green |
| 2 | Audio preprocessing rejects unsupported sample rates/channels before compute. | `ocelotl_models::whisper::audio::tests::{audio_metadata_rejects_unsupported_sample_rate_before_compute,audio_metadata_rejects_non_mono_before_compute}`. | green |
| 3 | Log-mel preprocessing has a deterministic fixture test. | `ocelotl_models::whisper::audio::tests::tiny_waveform_fixture_maps_to_pinned_log_mel_values`. | green |
| 4 | Whisper token startup/masking rules are fixture-tested. | `crates/tokenizer/tests/whisper_startup.rs::{whisper_transcribe_no_timestamps_startup_sequence_is_explicit,whisper_no_timestamps_decode_mask_suppresses_timestamps_and_prompt_specials}`. | green |
| 5 | A tiny synthetic Whisper-shaped path proves encoder/decoder shape and decode flow without network access. | `crates/models/tests/whisper_tiny_synthetic.rs::tiny_synthetic_whisper_path_matches_pinned_logits` plus the request/model validation tests in the same file. | green |
| 6 | Any real-model parity test is opt-in, local-artifact gated, and documented. | `crates/models/tests/whisper_local_artifact_parity.rs::{whisper_local_artifact_contract_lists_exact_required_paths,expected_tokens_schema_accepts_multilingual_documented_shape,expected_tokens_schema_accepts_english_only_documented_shape,decode_policy_uses_english_only_prompt_below_openai_multilingual_threshold,decode_policy_uses_multilingual_prompt_at_openai_multilingual_threshold,expected_tokens_schema_rejects_empty_reference_sequence,expected_tokens_schema_rejects_sequence_without_artifact_startup_prompt,wav_sample_reader_decodes_pcm16_mono_values,wav_sample_reader_decodes_ieee_float32_mono_values}` run by default. `local_whisper_tiny_en_artifact_contract_is_well_formed` is `#[ignore]`'d and names every required file under `local-artifacts/whisper_tiny_en`; when the local bundle exists, it derives the tokenizer/model family from `vocab_size`, loads real tensors, computes log-mel features from the reference WAV, runs the real Whisper adapter autoregressively with the matching no-timestamps decode mask, and compares exact token IDs. `docs/artifact-preparation.md` and `docs/validation/parity.md` document the bundle and artifact-blocked proof. | green (harness; exact proof artifact-blocked) |
| 7 | Burn remains an internal implementation detail. | Current W-ASR.2-W-ASR.8 code uses Ocelotl-owned structs/tests and does not expose Burn types. | green |
| 8 | Runtime exposes Ocelotl-owned transcription request/result types and reaches the Whisper model through the public lifecycle. | `crates/runtime/tests/whisper_transcription.rs::{transcribe_rejects_empty_audio_before_preprocessing_or_model_compute,transcribe_rejects_unsupported_audio_metadata_before_model_compute,transcribe_runs_tiny_synthetic_whisper_path_and_returns_token_plus_logits,transcribe_propagates_model_errors_after_runtime_audio_validation}`. | green (synthetic) |
| 9 | Loader can read safetensors tensor values without exposing foreign safetensors types. | `ocelotl_loader::safetensors_values::tests::{load_safetensors_tensor_f32_loads_f32_values_and_metadata,load_safetensors_tensor_f32_converts_bf16_values,load_safetensors_tensor_f32_converts_f16_values,load_safetensors_tensor_f32_preserves_f16_nan,load_safetensors_tensor_f32_returns_io_for_missing_file,load_safetensors_tensor_f32_returns_invalid_model_for_missing_tensor,load_safetensors_tensor_f32_returns_unsupported_for_unsupported_dtype,load_safetensors_tensor_f32_returns_invalid_model_for_malformed_payload}`. | green |
| 10 | Real Whisper config and tensor manifest validation fail before compute. | `ocelotl_models::whisper::config::tests::{parses_hf_tiny_en_config_shape,parses_openai_style_dims_shape,rejects_non_whisper_architecture,rejects_zero_dimension_before_compute,rejects_inconsistent_head_divisibility}` and `ocelotl_models::whisper::tensors::tests::{required_names_cover_real_whisper_tiny_en_families,untied_projection_requires_extra_projection_tensor,validate_accepts_complete_manifest,validate_rejects_invalid_config_before_manifest_walk,validate_rejects_missing_cross_attention_tensor,validate_rejects_wrong_shape,validate_rejects_wrong_dtype,validate_accepts_f16_manifest_when_config_is_f16}`. The ignored local-artifact harness now runs `parse_whisper_config_json` and `validate_whisper_tensors` against local `config.json` and `model.safetensors`. | green (contract) |

### Note - W-ASR.10 real parity limit

The W-ASR.10 ignored test now performs real output-token comparison when the
local Whisper tiny.en bundle is present. The repository does not commit
`local-artifacts/whisper_tiny_en`, so each contributor must supply that bundle
locally and run the ignored test to refresh the proof on their machine. The
senior run on 2026-05-09 verified the opt-in proof with:

```powershell
cargo test -p ocelotl-models --release --test whisper_local_artifact_parity -- --ignored --nocapture
```

### Note - W-ASR.6 runtime limit

The W-ASR.6 runtime API returns a greedy token plus logits from the tiny
synthetic Whisper model. It does not yet perform multi-token transcription,
timestamp handling, decode masking, or token-to-text decoding.

### Note - W-ASR.7/W-ASR.8 adapter groundwork limit

W-ASR.7 and W-ASR.8 unblock real Whisper adapter work by adding generic
safetensors value loading and a canonical OpenAI-style Whisper config/tensor
contract. W-ASR.9 adds the real forward adapter, and W-ASR.10 wires it into the
ignored local-artifact parity harness. W-ASR.8
intentionally defers HF/Burn-converted tensor-name aliases until a local
`model.safetensors` manifest proves which alternate names are needed.
