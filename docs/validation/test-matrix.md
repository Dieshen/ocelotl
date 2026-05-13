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
| 11 | English timestamp decode policy has explicit startup, masking, timestamp-time conversion, and token-level segment-boundary tests. | `crates/tokenizer/tests/whisper_timestamps.rs::{english_timestamp_enabled_startup_prompt_omits_no_timestamps_token,no_timestamps_mode_suppresses_timestamp_tokens,timestamp_enabled_mode_allows_timestamp_tokens_and_suppresses_prompt_specials,timestamp_token_to_time_uses_twenty_milliseconds_per_token_offset,timestamped_segments_parse_text_between_timestamp_boundaries}`. | green (policy; local timestamped parity deferred) |
| 12 | Transcript-only WER infrastructure normalizes transcripts, counts substitutions/insertions/deletions, and aggregates a tiny offline corpus without a quality threshold. | `ocelotl_models::whisper::wer::tests::{normalize_transcript_lowercases_strips_punctuation_and_folds_whitespace,edit_counts_cover_exact_match_substitution_insertion_and_deletion,corpus_report_aggregates_case_counts_without_thresholds,wer_rejects_reference_that_normalizes_to_empty,corpus_report_rejects_empty_case_list}`. | green (transcript-only) |
| 13 | A whisper.cpp benchmark harness can name both commands, local model/audio inputs, thread count, timing/output fields, and clear missing-binary skip behavior without requiring local artifacts in default tests. | `crates/models/tests/whisper_cpp_benchmark.rs::{whisper_cpp_benchmark_manifest_names_commands_inputs_and_threads,whisper_cpp_benchmark_record_names_timing_and_output_fields,missing_whisper_cpp_binary_record_has_clear_remediation}` validates the committed schema fixtures under `fixtures/benchmarks/`. `tools/whisper-cpp-bench.ps1` is opt-in and `docs/benchmarks/whisper-cpp.md` documents local prerequisites, exact invocation, and the rule that whisper.cpp is a performance baseline rather than a correctness oracle. | green (harness; local run opt-in) |
| 14 | Whisper config and tensor-contract logic is not tiny-only for the classic OpenAI Whisper sizes. | `ocelotl_models::whisper::config::tests::{parses_known_openai_whisper_size_dimensions,parses_non_tiny_hf_whisper_size_dimensions,rejects_oversized_audio_context_before_compute}` and `ocelotl_models::whisper::tensors::tests::{required_names_scale_with_known_openai_size_layers,validate_accepts_synthetic_manifests_for_non_tiny_openai_sizes,validate_rejects_tiny_state_tensor_shape_for_base_config_before_compute}` cover tiny/base/small/medium/large dimensions without loading large weights. | green (contract audit; output parity per size deferred) |
| 15 | Timestamped local-artifact parity has an explicit reference schema and opt-in proof path without weakening the no-timestamps proof. | `crates/models/tests/whisper_local_artifact_parity.rs::{timestamped_expected_tokens_schema_accepts_segments_for_english_only,timestamped_expected_tokens_schema_rejects_mismatched_segments}` validate `timestamps: true`, timestamp boundary tokens, and `expected_segments` by default. `local_whisper_tiny_en_timestamped_artifact_contract_is_well_formed` is `#[ignore]`'d and gated on `local-artifacts/whisper_tiny_en/reference/expected_tokens_timestamped.json`; it reuses the same local tiny.en bundle and preserves `local_whisper_tiny_en_artifact_contract_is_well_formed` for the W-ASR.10 no-timestamps proof. | green (schema; timestamped local proof artifact-blocked) |
| 16 | A local WER corpus runner can read a manifest, skip absent local corpus artifacts, and report per-sample plus aggregate WER without thresholds. | `crates/models/tests/whisper_wer_corpus_runner.rs::{wer_corpus_manifest_schema_names_local_artifacts_and_cases,wer_corpus_manifest_rejects_empty_cases_and_missing_skip_policy,missing_local_artifacts_are_reported_as_skip_reasons_without_loading,runner_report_formats_per_sample_and_aggregate_wer_without_thresholds}` plus the committed fixture `fixtures/wer/whisper_wer_corpus.example.json`. `local_whisper_wer_corpus_runner_reports_scores_when_artifacts_exist` is `#[ignore]`'d and runs the local corpus when artifacts exist. | green (runner; corpus artifacts opt-in) |
| 17 | Streaming/chunked transcription has an Ocelotl-owned deterministic chunk-planning contract before cache/state reuse. | `crates/runtime/tests/whisper_streaming.rs::{chunk_planner_pins_window_overlap_and_last_partial_chunk,chunk_planner_keeps_chunk_ranges_monotonic_in_samples_and_seconds,chunk_planner_rejects_zero_window,chunk_planner_rejects_overlap_equal_to_window,chunk_planner_rejects_unsupported_audio_metadata}` cover the public runtime types exported from `ocelotl_runtime::{ChunkedTranscriptionRequest,TranscriptionChunkingConfig,TranscriptionChunk,plan_transcription_chunks}`. | green (planning contract only) |
| 18 | The whisper.cpp benchmark harness times a dedicated Ocelotl transcription hook instead of the Rust test harness. | `crates/models/tests/whisper_cpp_benchmark.rs::whisper_cpp_benchmark_manifest_names_commands_inputs_and_threads` now asserts the Ocelotl command is `target/release/ocelotl.exe bench-whisper-transcribe ...`, not `cargo test`. `tools/whisper-cpp-bench.ps1 -DryRun` validates planned command records without local artifacts, and `src/main.rs` implements the narrow `bench-whisper-transcribe` hook for local runs. | green (timing hook; local numbers opt-in) |
| 19 | Classic Whisper size local-artifact parity harnesses exist for `tiny.en`, `base.en`, `small.en`, `medium.en`, and `large`, with independent skips and no large-weight loading in default tests. | `crates/models/tests/whisper_local_artifact_parity.rs::{classic_whisper_local_artifact_contract_lists_all_size_paths,classic_whisper_local_artifact_references_reuse_expected_token_schema}` prove all-size paths/schema by default. Ignored tests `local_whisper_{tiny_en,base_en,small_en,medium_en,large}_artifact_contract_is_well_formed` run each present bundle independently and skip absent bundles with remediation. The senior run on 2026-05-09 verified `tiny.en` through `OCELOTL_LOCAL_ARTIFACTS_DIR`; other size bundles were absent. | green (harness; per-size output parity opt-in) |
| 20 | The Ocelotl Whisper benchmark hook emits stage-level CPU timings. | `src/main.rs::tests::bench_whisper_output_reports_stage_timings` pins the `timings_ms` JSON schema for config parse, manifest validation, expected-token read, tensor load/model construction, WAV read, log-mel, audio encode, decode total, and per-token decode timing. `docs/benchmarks/whisper-cpp.md` records the 2026-05-12 tiny.en local comparison after encoded-audio reuse. | green (timing schema; local numbers opt-in) |
| 21 | Runtime can hold an Ocelotl-owned encoded-audio state for real Whisper transcription. | `crates/runtime/tests/whisper_real_transcription.rs::real_whisper_transcription_reuses_encoded_audio_and_matches_legacy_loop` prepares `WhisperTranscriptionState` once with `prepare_whisper_transcription`, verifies the encoded-audio state shape, and decodes from that state through `decode_whisper_transcription` using a separate `WhisperDecodeRequest`. `ocelotl_models::whisper::real::tests::{cached_audio_logits_match_legacy_forward_path,cached_audio_forward_rejects_wrong_state_size_before_compute}` pins the lower-level model seam. | green |
| 22 | Real Whisper decode loops can reuse encoder output without changing token parity. | `crates/runtime/tests/whisper_real_transcription.rs::real_whisper_transcription_reuses_encoded_audio_and_matches_legacy_loop` compares the runtime cached-audio decode loop against the legacy loop that recomputes `forward_next_token_logits(log_mel, tokens)`. The ignored local parity harness calls `encode_audio_features` once per audio input and now advances generation through the decoder state; the 2026-05-12 senior run verified `local_whisper_tiny_en_artifact_contract_is_well_formed` passed locally (`122.92s` debug harness). | green (tiny.en local proof opt-in) |
| 23 | CPU kernel selection is explicit, preserves the scalar fallback, and can route hot matmul/attention work through an optimized CPU backend without changing model outputs. | `ocelotl_kernels::tests::{cpu_backend_defaults_to_scalar_mode,cpu_backend_can_select_optimized_mode,optimized_matmul_matches_scalar_for_non_square_shape,optimized_attention_matches_scalar_backend,optimized_linear_out_by_in_matches_scalar_with_bias}` pin the backend-selection and kernel parity contracts. `ocelotl_models::qwen::qwen2_5_model::tests::optimized_cpu_backend_preserves_prefill_logits` and `ocelotl_models::whisper::real::tests::optimized_cpu_backend_preserves_forward_logits` prove model-level outputs remain within tolerance when optimized CPU mode is selected. `ocelotl_runtime::tests::optimized_cpu_runtime_selects_optimized_kernel_backend` pins the runtime constructor. | green |
| 24 | A fresh tiny.en whisper.cpp comparison is captured after W-ASR.20-W-ASR.23, and CPU follow-up gates are measurable. | `src/main.rs::tests::bench_args_default_to_scalar_and_accept_optimized_kernel_mode` pins the benchmark hook's scalar default plus optimized-mode opt-in. `crates/models/tests/whisper_cpp_benchmark.rs::whisper_cpp_benchmark_manifest_names_commands_inputs_and_threads` now asserts the example manifest passes `--cpu-kernel-mode optimized`. `docs/benchmarks/whisper-cpp.md` records the 2026-05-12 W-ASR.24 local run: Ocelotl optimized `16,648 ms` vs whisper.cpp `564 ms`, plus same-binary scalar `14,179 ms`, and defines correctness, regression, optimized-default, and CPU-competitiveness gates. | green (local benchmark captured; performance gates currently failing) |
| 25 | Whisper CPU benchmarks distinguish resident loaded-model latency from artifact/model-load setup. | `src/main.rs::tests::bench_whisper_output_reports_stage_timings` now pins `resident_model_ms.audio_to_tokens` and `resident_model_ms.mel_to_tokens` in the benchmark hook JSON. `docs/benchmarks/whisper-cpp.md` defines those formulas and identifies resident audio-to-tokens timing as the next CPU optimization denominator. Senior local release proof on 2026-05-12 passed exact token parity with scalar total `13,869 ms`, resident audio-to-tokens `9,925 ms`, and resident mel-to-tokens `9,172 ms`. | green |
| 26 | Whisper decoder cross-attention reuses K/V projected from encoded audio without changing generated tokens. | `ocelotl_models::whisper::real::tests::precomputed_cross_attention_matches_projected_cross_attention` proves cached K/V attention matches the projected-on-demand path. `cached_audio_logits_match_legacy_forward_path` verifies encoded audio still produces the same logits as the legacy wrapper. Senior local release proof on 2026-05-12 passed exact token parity and improved scalar total time to `8,582 ms`, resident audio-to-tokens to `4,635 ms`, and decode total to `1,519 ms`. | green |
| 27 | Whisper decoder self-attention reuses per-layer K/V while appending generated tokens one step at a time. | `ocelotl_models::whisper::real::tests::{incremental_self_attention_matches_full_causal_last_row,decoder_state_append_matches_full_context_logits,decoder_state_append_rejects_context_overflow_before_compute}` pin the incremental attention math, full-context logits equivalence, and context overflow gate. `crates/runtime/tests/whisper_real_transcription.rs::real_whisper_transcription_reuses_encoded_audio_and_matches_legacy_loop` exercises the runtime transcription path through the decoder state. Senior local release proof on 2026-05-12 passed exact token parity and improved scalar total time to `7,409 ms`, resident audio-to-tokens to `3,506 ms`, and decode total to `319 ms`. | green |
| 28 | Whisper local proof and benchmark paths load required safetensors values from one parsed archive instead of rereading the model once per tensor. | `ocelotl_loader::safetensors_values::tests::load_safetensors_tensors_f32_loads_many_tensors_from_one_archive` pins bulk value loading order and metadata. `src/main.rs`, `crates/models/tests/whisper_local_artifact_parity.rs`, and `crates/models/tests/whisper_wer_corpus_runner.rs` now call `load_safetensors_tensors_f32` for Whisper bundles. Senior local release proof on 2026-05-12 passed exact token parity and improved scalar total time to `3,620 ms` with tensor load + model construction at `61 ms`, clearing the first `<=10x whisper.cpp` wall-time gate. | green |
| 29 | Whisper log-mel preprocessing reuses a fixed STFT Fourier basis instead of recomputing trig terms inside every frame/bin/sample loop. | `ocelotl_models::whisper::audio::tests::{tiny_waveform_fixture_maps_to_pinned_log_mel_values,fourier_basis_matches_direct_trig_formula}` pin log-mel output compatibility and the cached-basis formula. Senior local release proof on 2026-05-12 passed exact token parity and improved scalar total time to `2,859 ms`, resident audio-to-tokens to `2,795 ms`, and log-mel to `45 ms`, moving Ocelotl to `~5.1x` whisper.cpp wall time. | green |
| 30 | Whisper benchmark timings split audio encode into encoder forward and cross-attention K/V precompute. | `ocelotl_models::whisper::real::tests::timed_audio_encode_matches_plain_audio_encode` proves the timed encode path returns the same encoded audio as the plain path. `src/main.rs::tests::bench_whisper_output_reports_stage_timings` pins `timings_ms.audio_encode_detail.{encoder,cross_attention_precompute}`. Senior local release proof on 2026-05-12 passed exact token parity and measured scalar `audio_encode = 2,387 ms`, `encoder = 2,153 ms`, and `cross_attention_precompute = 234 ms`. | green |
| 31 | Whisper attention context accumulation walks V rows contiguously without changing token parity. | `ocelotl_models::whisper::real::tests::{encoder_self_attention_does_not_apply_causal_mask,decoder_cross_attention_does_not_apply_causal_mask,precomputed_cross_attention_matches_projected_cross_attention,cached_audio_logits_match_legacy_forward_path}` pin the affected attention semantics. Senior local release proof on 2026-05-12 passed exact token parity and improved scalar total time to `2,559 ms`, resident audio-to-tokens to `2,496 ms`, and encoder timing to `1,896 ms`, moving Ocelotl to `~4.5x` whisper.cpp wall time. | green |
| 32 | Scalar CPU kernels clear the documented tiny.en `<=3x` whisper.cpp wall-time gate without changing token parity. | `ocelotl_kernels::tests::optimized_linear_out_by_in_matches_scalar_with_bias` keeps the linear kernel parity fence after four-output scalar unroll. `ocelotl_models::whisper::real::tests::{encoder_self_attention_does_not_apply_causal_mask,optimized_cpu_backend_preserves_forward_logits,cached_audio_logits_match_legacy_forward_path}` cover the affected Whisper attention/model path. Senior local release proof on 2026-05-12 passed exact token parity twice after the change (`1,646 ms`, then `1,623 ms`), improving the latest scalar total to `~2.88x` the `564 ms` whisper.cpp baseline. | green |
| 33 | All classic local Whisper sizes have a captured scalar CPU comparison against whisper.cpp after W-ASR.32. | `docs/benchmarks/whisper-cpp.md` records 2026-05-12 local runs for tiny.en, base.en, small.en, medium.en, and large-v2. Every Ocelotl row had `matches_expected = true`; the non-tiny references remain short expected-token contract checks, so the captured encoder timings are the main performance signal. Only tiny.en clears the existing `<=3x` wall-time gate; larger sizes remain `~4.1x` to `~5.5x` slower than whisper.cpp. | green (local benchmark captured; larger-size competitiveness failing) |
| 35 | Scalar `[out, in]` linear uses row/output tiling without breaking exact Whisper token parity. | `ocelotl_kernels::tests::scalar_linear_out_by_in_handles_row_and_output_tile_tails` pins 4-row/4-output tile edge handling, and existing scalar/optimized parity tests still cover the public backend path. Senior local release proofs on 2026-05-12 passed exact token parity for tiny.en (`1,073 ms`), base.en (`1,807 ms`), small.en (`6,345 ms`), medium.en (`20,941 ms`), and large-v2 (`40,416 ms`), clearing the `<=3x` whisper.cpp wall-time gate for all five recorded sizes. | green |

### Note - W-ASR.10 real parity limit

The W-ASR.10 ignored test now performs real output-token comparison when the
local Whisper tiny.en bundle is present. The repository does not commit
`local-artifacts/whisper_tiny_en`, so each contributor must supply that bundle
locally and run the ignored test to refresh the proof on their machine. The
senior run on 2026-05-09 verified the opt-in proof with:

```powershell
cargo test -p ocelotl-models --release --test whisper_local_artifact_parity -- --ignored --nocapture
```

### Note - W-ASR.6/W-ASR.22 runtime limit

The W-ASR.6 runtime API returns a greedy token plus logits from the tiny
synthetic Whisper model. W-ASR.22 adds a separate real-Whisper runtime path that
performs multi-token masked decode from a reusable encoded-audio state. The
runtime still does not own token-to-text decoding, WER thresholds, cross-chunk
state reuse, or decoder self-attention KV cache.

### Note - W-ASR.7/W-ASR.8 adapter groundwork limit

W-ASR.7 and W-ASR.8 unblock real Whisper adapter work by adding generic
safetensors value loading and a canonical OpenAI-style Whisper config/tensor
contract. W-ASR.9 adds the real forward adapter, and W-ASR.10 wires it into the
ignored local-artifact parity harness. W-ASR.8
intentionally defers HF/Burn-converted tensor-name aliases until a local
`model.safetensors` manifest proves which alternate names are needed.
