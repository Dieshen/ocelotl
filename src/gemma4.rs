use std::path::Path;

use ocelotl_core::{OcelotlError, Result, TokenId, TokenizerError};
use ocelotl_loader::{GgufTokenizerMetadata, inspect_gguf_tokenizer};
use ocelotl_tokenizer::{GgufBpeTokenizer, GgufBpeTokenizerSpec, GgufSpecialTokens, GgufTokenType};

pub fn gemma4_gguf_tokenizer_spec_from_metadata(
    metadata: &GgufTokenizerMetadata,
) -> Result<GgufBpeTokenizerSpec> {
    if metadata.model.as_deref() != Some("gemma4") {
        return Err(tokenizer_error(format!(
            "Gemma4 GGUF tokenizer requires tokenizer.ggml.model = \"gemma4\", got {:?}",
            metadata.model
        )));
    }
    if metadata.token_types.len() != metadata.tokens.len() {
        return Err(tokenizer_error(format!(
            "Gemma4 GGUF tokenizer requires token_type length {} to match token length {}",
            metadata.token_types.len(),
            metadata.tokens.len()
        )));
    }

    let mut spec = GgufBpeTokenizerSpec::new(metadata.tokens.clone(), metadata.merges.clone());
    spec.model = metadata.model.clone();
    spec.token_types = metadata
        .token_types
        .iter()
        .map(|raw| GgufTokenType::from_gguf_i32(*raw))
        .collect::<Result<Vec<_>>>()?;
    spec.special_tokens = GgufSpecialTokens {
        bos_token_id: metadata.bos_token_id.map(TokenId),
        eos_token_id: metadata.eos_token_id.map(TokenId),
        unknown_token_id: metadata.unknown_token_id.map(TokenId),
        padding_token_id: metadata.padding_token_id.map(TokenId),
        mask_token_id: metadata.mask_token_id.map(TokenId),
    };
    spec.add_bos_token = metadata.add_bos_token.unwrap_or(false);
    spec.add_space_prefix = metadata.add_space_prefix.unwrap_or(false);
    Ok(spec)
}

pub fn load_gemma4_gguf_tokenizer(path: impl AsRef<Path>) -> Result<GgufBpeTokenizer> {
    let metadata = inspect_gguf_tokenizer(path.as_ref())?;
    let spec = gemma4_gguf_tokenizer_spec_from_metadata(&metadata)?;
    GgufBpeTokenizer::from_spec(spec)
}

fn tokenizer_error(message: impl Into<String>) -> OcelotlError {
    OcelotlError::Tokenizer(TokenizerError {
        message: message.into(),
        source: None,
    })
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs,
        io::{Read, Seek, SeekFrom},
        path::{Path, PathBuf},
        process::Command,
        time::{SystemTime, UNIX_EPOCH},
    };

    use ocelotl_loader::{GgmlTensorType, inspect_gguf};
    use ocelotl_models::gemma::{
        Gemma4TextModel, Gemma4TextWeights, load_gemma4_dequantized_tensors_from_gguf,
    };
    use ocelotl_runtime::gemma::prefill;
    use ocelotl_tokenizer::Tokenizer;
    use serde::Deserialize;

    use super::*;

    const GEMMA4_GGUF_REVISION: &str = "c04cb322fd63e347db759a08b6249b867488ccf8";
    const GEMMA4_LOGITS_FIXTURE_NAME: &str = "gemma4_q4_k_m_basic_prompt_logits_reference";
    const QK_K: usize = 256;
    const Q4_K_BLOCK_BYTES: usize = 144;
    const Q6_K_BLOCK_BYTES: usize = 210;

    #[derive(Debug, Deserialize)]
    struct Gemma4TokenizerFixture {
        fixture_version: u32,
        name: String,
        source: String,
        input: String,
        expected_token_ids: Vec<u32>,
        expected_with_bos_token_ids: Vec<u32>,
        decoded: String,
        reference_status: String,
        llama_cpp_reference: LlamaCppTokenizerReference,
    }

    #[derive(Debug, Deserialize)]
    struct LlamaCppTokenizerReference {
        tool: String,
        revision: String,
        source: String,
        bos_free_command: Vec<String>,
        configured_bos_command: Vec<String>,
        expected_stdout_token_ids: Vec<u32>,
        expected_stdout_with_bos_token_ids: Vec<u32>,
    }

    #[derive(Debug, Deserialize)]
    struct Gemma4LogitsReferenceFixture {
        fixture_version: u32,
        name: String,
        source: String,
        input: String,
        token_ids: Vec<u32>,
        selected_logit_token_ids: Vec<u32>,
        tolerance: f32,
        reference_status: String,
        llama_cpp_reference: LlamaCppLogitsReference,
        ocelotl_reference: OcelotlLogitsReference,
        regeneration: String,
    }

    #[derive(Debug, Deserialize)]
    struct LlamaCppLogitsReference {
        tool: String,
        revision: String,
        source: String,
        command: Vec<String>,
    }

    #[derive(Debug, Deserialize)]
    struct OcelotlLogitsReference {
        command: String,
        notes: String,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct LlamaDebugTensorSummary {
        shape: Vec<usize>,
        sum: f32,
    }

    fn tiny_metadata() -> GgufTokenizerMetadata {
        GgufTokenizerMetadata {
            model: Some("gemma4".to_string()),
            tokens: [
                "<pad>",
                "</s>",
                "<s>",
                "<unk>",
                "<mask>",
                "H",
                "e",
                "l",
                "o",
                "He",
                "Hel",
                "Hell",
                "Hello",
                "<control>",
                "<tool_call>",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            scores: vec![],
            token_types: vec![3, 3, 3, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 4],
            merges: ["H e", "He l", "Hel l", "Hell o"]
                .into_iter()
                .map(String::from)
                .collect(),
            chat_template: Some("{{ bos_token }}{{ messages[0].content }}".to_string()),
            bos_token_id: Some(2),
            eos_token_id: Some(1),
            unknown_token_id: Some(3),
            padding_token_id: Some(0),
            mask_token_id: Some(4),
            add_bos_token: Some(true),
            add_space_prefix: Some(false),
        }
    }

    fn gemma4_tokenizer_fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("tokenizer")
            .join("gemma4_gguf_basic_prompt.json")
    }

    fn gemma4_logits_fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("logits")
            .join(format!("{GEMMA4_LOGITS_FIXTURE_NAME}.json"))
    }

    fn load_gemma4_tokenizer_fixture() -> Gemma4TokenizerFixture {
        let path = gemma4_tokenizer_fixture_path();
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read fixture at {}: {err}", path.display()));
        serde_json::from_str(&raw)
            .unwrap_or_else(|err| panic!("failed to parse fixture at {}: {err}", path.display()))
    }

    fn load_gemma4_logits_fixture() -> Gemma4LogitsReferenceFixture {
        let path = gemma4_logits_fixture_path();
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read fixture at {}: {err}", path.display()));
        serde_json::from_str(&raw)
            .unwrap_or_else(|err| panic!("failed to parse fixture at {}: {err}", path.display()))
    }

    #[test]
    fn gemma4_gguf_tokenizer_fixture_is_well_formed_and_populated() {
        let fixture = load_gemma4_tokenizer_fixture();

        assert_eq!(fixture.fixture_version, 1);
        assert_eq!(fixture.name, "gemma4_gguf_basic_prompt");
        assert!(
            fixture.source.contains(GEMMA4_GGUF_REVISION),
            "fixture source must reference pinned Gemma4 GGUF revision {GEMMA4_GGUF_REVISION}"
        );
        assert!(!fixture.input.is_empty(), "fixture input must be non-empty");
        assert!(
            !fixture.expected_token_ids.is_empty(),
            "Gemma4 GGUF expected_token_ids must be populated"
        );
        assert_eq!(
            fixture.expected_with_bos_token_ids.first().copied(),
            Some(2),
            "Gemma4 GGUF configured-BOS fixture must start with BOS id 2"
        );
        assert!(
            !fixture.decoded.is_empty(),
            "Gemma4 GGUF fixture must pin decoded text"
        );
        assert!(
            fixture.reference_status.contains("llama.cpp"),
            "fixture must describe the llama.cpp reference state"
        );
        assert_eq!(fixture.llama_cpp_reference.tool, "llama-tokenize");
        assert!(
            fixture.llama_cpp_reference.source.contains("llama.cpp"),
            "fixture must name the llama.cpp tokenizer source"
        );
        assert!(
            !fixture.llama_cpp_reference.revision.is_empty(),
            "fixture must declare the intended llama.cpp reference revision"
        );
        assert!(
            fixture
                .llama_cpp_reference
                .bos_free_command
                .iter()
                .any(|arg| arg == "--no-bos"),
            "BOS-free llama.cpp reference command must pass --no-bos"
        );
        assert!(
            !fixture
                .llama_cpp_reference
                .configured_bos_command
                .iter()
                .any(|arg| arg == "--no-bos"),
            "configured-BOS llama.cpp reference command must allow model BOS"
        );
        assert_eq!(
            fixture.llama_cpp_reference.expected_stdout_token_ids, fixture.expected_token_ids,
            "fixture llama.cpp BOS-free stdout must match pinned expected IDs"
        );
        assert_eq!(
            fixture
                .llama_cpp_reference
                .expected_stdout_with_bos_token_ids,
            fixture.expected_with_bos_token_ids,
            "fixture llama.cpp configured-BOS stdout must match pinned expected IDs"
        );
    }

    #[test]
    fn gemma4_q4_k_m_logits_reference_fixture_is_well_formed_and_populated() {
        let fixture = load_gemma4_logits_fixture();

        assert_eq!(fixture.fixture_version, 1);
        assert_eq!(fixture.name, GEMMA4_LOGITS_FIXTURE_NAME);
        assert!(
            fixture.source.contains(GEMMA4_GGUF_REVISION),
            "Gemma4 logits fixture must reference pinned GGUF revision {GEMMA4_GGUF_REVISION}"
        );
        assert_eq!(
            fixture.input, "Hello",
            "logits fixture must stay aligned with the tokenizer fixture prompt"
        );
        assert_eq!(
            fixture.token_ids,
            vec![2, 9259],
            "logits fixture must pin configured-BOS token IDs for the prompt"
        );
        assert!(
            !fixture.selected_logit_token_ids.is_empty(),
            "selected_logit_token_ids must not be empty"
        );
        let unique: BTreeSet<u32> = fixture.selected_logit_token_ids.iter().copied().collect();
        assert_eq!(
            unique.len(),
            fixture.selected_logit_token_ids.len(),
            "selected_logit_token_ids must be unique"
        );
        assert!(
            fixture
                .selected_logit_token_ids
                .iter()
                .all(|token| *token < 262_144),
            "selected Gemma4 logit IDs must fit the pinned artifact vocab"
        );
        assert!(
            fixture.tolerance.is_finite() && fixture.tolerance > 0.0 && fixture.tolerance <= 0.05,
            "Gemma4 logits tolerance must be finite, positive, and explicit"
        );
        assert!(
            fixture.reference_status.contains("llama.cpp"),
            "fixture must describe the llama.cpp reference state"
        );
        assert_eq!(fixture.llama_cpp_reference.tool, "llama-debug");
        assert!(
            fixture.llama_cpp_reference.revision.contains("856c3ad"),
            "fixture must record the pinned llama.cpp reference revision"
        );
        assert!(
            fixture
                .llama_cpp_reference
                .source
                .contains("examples/debug"),
            "fixture must name llama.cpp examples/debug as the logits source"
        );
        assert!(
            fixture
                .llama_cpp_reference
                .command
                .iter()
                .any(|arg| arg == "--save-logits"),
            "llama.cpp logits reference command must save logits"
        );
        assert!(
            fixture
                .llama_cpp_reference
                .command
                .iter()
                .any(|arg| arg == "--logits-output-dir"),
            "llama.cpp logits reference command must choose an output directory"
        );
        assert!(
            fixture
                .ocelotl_reference
                .command
                .contains("cargo test -p ocelotl"),
            "fixture must name the Ocelotl ignored parity command"
        );
        assert!(
            fixture.ocelotl_reference.notes.contains("multimodal"),
            "fixture notes must document the text-only multimodal projection"
        );
        assert!(
            fixture.regeneration.contains("docs/validation/parity.md"),
            "fixture regeneration notes must name the parity docs"
        );
    }

    #[test]
    fn llama_tokenize_ids_parser_accepts_python_style_id_list() {
        assert_eq!(
            parse_llama_tokenize_ids(b" [2, 9259]\r\n").expect("ID list should parse"),
            vec![2, 9259]
        );
    }

    #[test]
    fn llama_tokenize_ids_parser_rejects_non_list_output() {
        let err = parse_llama_tokenize_ids(b"Total number of tokens: 2")
            .expect_err("non-list llama-tokenize output must reject");

        assert!(err.contains("missing '['"), "unexpected parse error: {err}");
    }

    #[test]
    fn llama_debug_logits_parser_extracts_selected_token_values() {
        let parsed =
            parse_llama_debug_selected_logits_text("0: -1.25\n1: 0.5\n9259: 3.75\n", &[9259, 0])
                .expect("selected llama-debug logits should parse");

        assert_eq!(parsed.get(&0), Some(&-1.25));
        assert_eq!(parsed.get(&9259), Some(&3.75));
    }

    #[test]
    fn llama_debug_logits_parser_rejects_missing_selected_token() {
        let err = parse_llama_debug_selected_logits_text("0: -1.25\n", &[0, 2])
            .expect_err("missing selected token must reject");

        assert!(
            err.contains("missing selected logit token id 2"),
            "unexpected parse error: {err}"
        );
    }

    #[test]
    fn llama_debug_logits_parser_rejects_non_contiguous_full_vector() {
        let err = parse_llama_debug_logits_text("0: -1.25\n2: 0.5\n")
            .expect_err("non-contiguous full llama-debug logits must reject");

        assert!(
            err.contains("expected token id 1"),
            "unexpected parse error: {err}"
        );
    }

    #[test]
    fn llama_debug_tensor_summary_parser_extracts_shape_and_sum() {
        let raw = "\
common_debug_cb_eval:              result_norm = (f32)    RMS_NORM(inp{4, 1, 1, 1}, }) = {4, 1, 1, 1}
    [
        [
            [      0.2500,       0.5000,       0.7500,       1.0000  ],
        ],
    ]
    sum = 2.500000
common_debug_cb_eval:            result_output = (f32)    MUL_MAT(out{4, 8, 1, 1}, result_norm{4, 1, 1, 1}) = {8, 1, 1, 1}
    [
        [
            [      1.0000,       2.0000,       3.0000,    ...,       8.0000  ],
        ],
    ]
    sum = 36.000000
";

        let parsed = parse_llama_debug_tensor_summaries(raw, &["result_norm", "result_output"])
            .expect("llama-debug tensor summaries must parse");

        assert_eq!(
            parsed.get("result_norm"),
            Some(&LlamaDebugTensorSummary {
                shape: vec![4, 1, 1, 1],
                sum: 2.5,
            })
        );
        assert_eq!(
            parsed.get("result_output"),
            Some(&LlamaDebugTensorSummary {
                shape: vec![8, 1, 1, 1],
                sum: 36.0,
            })
        );
    }

    #[test]
    fn llama_debug_tensor_summary_parser_rejects_missing_tensor() {
        let raw =
            "common_debug_cb_eval: result_norm = (f32) OP(a{1}, }) = {1, 1, 1, 1}\n    sum = 1.0\n";

        let err = parse_llama_debug_tensor_summaries(raw, &["result_norm", "result_output"])
            .expect_err("missing tensor summary must reject");

        assert!(
            err.contains("missing llama-debug tensor summary for result_output"),
            "unexpected parse error: {err}"
        );
    }

    #[test]
    fn llama_debug_tensor_summary_parser_keeps_last_duplicate_tensor() {
        let raw = "\
common_debug_cb_eval: result_norm = (f32) OP(a{1}, }) = {1, 1, 1, 1}
    sum = 1.0
common_debug_cb_eval: result_norm = (f32) OP(a{1}, }) = {1, 1, 1, 1}
    sum = 2.0
";

        let parsed = parse_llama_debug_tensor_summaries(raw, &["result_norm"])
            .expect("duplicate llama-debug tensor summaries should keep the last value");

        assert_eq!(
            parsed.get("result_norm"),
            Some(&LlamaDebugTensorSummary {
                shape: vec![1, 1, 1, 1],
                sum: 2.0,
            })
        );
    }

    #[test]
    fn gemma4_gguf_tokenizer_spec_maps_loader_metadata_without_crate_cycle() {
        let spec = gemma4_gguf_tokenizer_spec_from_metadata(&tiny_metadata())
            .expect("tiny Gemma4 tokenizer metadata should map");

        assert_eq!(spec.model.as_deref(), Some("gemma4"));
        assert_eq!(spec.special_tokens.bos_token_id, Some(TokenId(2)));
        assert_eq!(spec.special_tokens.unknown_token_id, Some(TokenId(3)));
        assert_eq!(spec.token_types[3], GgufTokenType::Unknown);
        assert_eq!(spec.token_types[13], GgufTokenType::Control);
        assert_eq!(spec.token_types[14], GgufTokenType::UserDefined);
        assert!(spec.add_bos_token);
        assert!(!spec.add_space_prefix);
    }

    #[test]
    fn gemma4_gguf_tokenizer_builds_from_loader_metadata() {
        let spec = gemma4_gguf_tokenizer_spec_from_metadata(&tiny_metadata())
            .expect("tiny Gemma4 tokenizer metadata should map");
        let tokenizer =
            GgufBpeTokenizer::from_spec(spec).expect("mapped Gemma4 tokenizer should build");

        assert_eq!(tokenizer.encode("Hello").unwrap(), vec![TokenId(12)]);
        assert_eq!(
            tokenizer.encode_with_configured_bos("Hello").unwrap(),
            vec![TokenId(2), TokenId(12)]
        );
        assert_eq!(
            tokenizer
                .decode(&[TokenId(13), TokenId(12), TokenId(13)])
                .unwrap(),
            "Hello"
        );
        assert_eq!(tokenizer.encode("<tool_call>").unwrap(), vec![TokenId(14)]);
    }

    #[test]
    fn gemma4_gguf_tokenizer_spec_rejects_non_gemma4_metadata() {
        let mut metadata = tiny_metadata();
        metadata.model = Some("llama".to_string());

        let err = gemma4_gguf_tokenizer_spec_from_metadata(&metadata)
            .expect_err("wrong tokenizer model must reject");

        match err {
            OcelotlError::Tokenizer(tokenizer) => assert!(
                tokenizer.message.contains("model = \"gemma4\""),
                "unexpected tokenizer error: {tokenizer}"
            ),
            other => panic!("expected tokenizer error, got {other:?}"),
        }
    }

    #[test]
    fn gemma4_gguf_tokenizer_spec_rejects_unknown_token_type_values() {
        let mut metadata = tiny_metadata();
        metadata.token_types[5] = 99;

        let err = gemma4_gguf_tokenizer_spec_from_metadata(&metadata)
            .expect_err("unknown token_type must reject");

        match err {
            OcelotlError::Tokenizer(tokenizer) => assert!(
                tokenizer.message.contains("token_type value 99"),
                "unexpected tokenizer error: {tokenizer}"
            ),
            other => panic!("expected tokenizer error, got {other:?}"),
        }
    }

    #[test]
    #[ignore = "requires local-artifacts/gemma4_e4b_it_q4_k_m/google_gemma-4-E4B-it-Q4_K_M.gguf or OCELOTL_GEMMA4_GGUF_PATH"]
    fn local_gemma4_q4_k_m_gguf_tokenizer_builds_from_embedded_metadata() {
        let path = local_gemma4_gguf_path();
        assert!(
            path.exists(),
            "missing Gemma4 GGUF at {}; see docs/artifact-preparation.md",
            path.display()
        );

        let fixture = load_gemma4_tokenizer_fixture();
        let tokenizer =
            load_gemma4_gguf_tokenizer(&path).expect("local Gemma4 GGUF tokenizer must build");
        assert_eq!(tokenizer.special_tokens().bos_token_id, Some(TokenId(2)));
        assert_eq!(tokenizer.special_tokens().eos_token_id, Some(TokenId(1)));
        assert!(tokenizer.add_bos_token());
        assert!(!tokenizer.add_space_prefix());

        let plain = tokenizer
            .encode(&fixture.input)
            .expect("local Gemma4 GGUF tokenizer should encode a basic prompt");
        let plain_raw: Vec<u32> = plain.iter().map(|token| token.0).collect();
        assert_eq!(
            plain_raw, fixture.expected_token_ids,
            "local Gemma4 GGUF tokenizer IDs drifted from fixture"
        );
        let with_bos = tokenizer
            .encode_with_configured_bos(&fixture.input)
            .expect("configured BOS encode should succeed");
        let with_bos_raw: Vec<u32> = with_bos.iter().map(|token| token.0).collect();
        assert_eq!(
            with_bos_raw, fixture.expected_with_bos_token_ids,
            "local Gemma4 GGUF configured-BOS IDs drifted from fixture"
        );
        let decoded = tokenizer
            .decode(&plain)
            .expect("local Gemma4 GGUF tokenizer should decode fixture IDs");
        assert_eq!(decoded, fixture.decoded);
    }

    #[test]
    #[ignore = "requires Gemma4 GGUF plus OCELOTL_LLAMA_TOKENIZE_PATH or local-artifacts/llama_cpp/llama-tokenize.exe; see docs/artifact-preparation.md"]
    fn local_gemma4_q4_k_m_gguf_tokenizer_matches_llama_cpp_tokenize_reference() {
        let model_path = local_gemma4_gguf_path();
        assert!(
            model_path.exists(),
            "missing Gemma4 GGUF at {}; see docs/artifact-preparation.md",
            model_path.display()
        );
        let llama_tokenize = local_llama_tokenize_path();
        assert!(
            llama_tokenize.exists(),
            "missing llama-tokenize at {}; set OCELOTL_LLAMA_TOKENIZE_PATH or see docs/artifact-preparation.md",
            llama_tokenize.display()
        );

        let fixture = load_gemma4_tokenizer_fixture();
        let tokenizer = load_gemma4_gguf_tokenizer(&model_path)
            .expect("local Gemma4 GGUF tokenizer must build");

        let ocelotl_plain: Vec<u32> = tokenizer
            .encode(&fixture.input)
            .expect("Ocelotl Gemma4 GGUF tokenizer should encode fixture input")
            .iter()
            .map(|token| token.0)
            .collect();
        let llama_plain =
            run_llama_tokenize_ids(&llama_tokenize, &model_path, &fixture.input, false);
        assert_eq!(
            llama_plain, fixture.expected_token_ids,
            "fixture must match llama.cpp BOS-free tokenizer output"
        );
        assert_eq!(
            ocelotl_plain, llama_plain,
            "Ocelotl Gemma4 BOS-free tokenizer output must match llama.cpp"
        );

        let ocelotl_with_bos: Vec<u32> = tokenizer
            .encode_with_configured_bos(&fixture.input)
            .expect("Ocelotl Gemma4 GGUF tokenizer should encode fixture input with configured BOS")
            .iter()
            .map(|token| token.0)
            .collect();
        let llama_with_bos =
            run_llama_tokenize_ids(&llama_tokenize, &model_path, &fixture.input, true);
        assert_eq!(
            llama_with_bos, fixture.expected_with_bos_token_ids,
            "fixture must match llama.cpp configured-BOS tokenizer output"
        );
        assert_eq!(
            ocelotl_with_bos, llama_with_bos,
            "Ocelotl Gemma4 configured-BOS tokenizer output must match llama.cpp"
        );
    }

    #[test]
    #[ignore = "requires Gemma4 GGUF plus OCELOTL_LLAMA_DEBUG_PATH or local-artifacts/llama_cpp/llama-debug.exe; see docs/artifact-preparation.md"]
    fn local_gemma4_q4_k_m_prefill_logits_match_llama_cpp_debug() {
        let model_path = local_gemma4_gguf_path();
        assert!(
            model_path.exists(),
            "missing Gemma4 GGUF at {}; see docs/artifact-preparation.md",
            model_path.display()
        );
        let llama_debug = local_llama_debug_path();
        assert!(
            llama_debug.exists(),
            "missing llama-debug at {}; set OCELOTL_LLAMA_DEBUG_PATH or see docs/artifact-preparation.md",
            llama_debug.display()
        );

        let fixture = load_gemma4_logits_fixture();
        let tokenizer = load_gemma4_gguf_tokenizer(&model_path)
            .expect("local Gemma4 GGUF tokenizer must build");
        let tokens = tokenizer
            .encode_with_configured_bos(&fixture.input)
            .expect("Gemma4 GGUF tokenizer must encode logits fixture prompt");
        let token_ids: Vec<u32> = tokens.iter().map(|token| token.0).collect();
        assert_eq!(
            token_ids, fixture.token_ids,
            "Ocelotl tokenization must match the logits fixture before comparing logits"
        );

        let output_dir = unique_llama_debug_output_dir();
        let llama_logits =
            run_llama_debug_logits(&llama_debug, &model_path, &fixture.input, &output_dir);
        let _ = fs::remove_dir_all(&output_dir);

        let (mut config, tensors) = load_gemma4_dequantized_tensors_from_gguf(&model_path)
            .expect("local Gemma4 GGUF tensors must load through explicit F32 dequantization");
        assert!(
            config.multimodal,
            "selected real Gemma4 artifact should still be recorded as multimodal"
        );
        config.multimodal = false;
        let weights = Gemma4TextWeights::from_loaded_tensors(&config, tensors)
            .expect("dequantized local Gemma4 tensors must map into text weights");
        let model = Gemma4TextModel::new(config, weights)
            .expect("text-projected dequantized Gemma4 model must build");
        let ocelotl_logits = prefill(&model, &tokens).expect("Gemma4 text prefill must run");
        assert_eq!(
            ocelotl_logits.len(),
            model.config().tokenizer_token_count,
            "Gemma4 prefill must return one final-position logit per token"
        );
        assert_eq!(
            llama_logits.len(),
            ocelotl_logits.len(),
            "llama.cpp and Ocelotl must expose the same Gemma4 final-logit vector length"
        );
        for token_id in &fixture.selected_logit_token_ids {
            assert!(
                (*token_id as usize) < llama_logits.len(),
                "selected fixture token id {token_id} must fit llama.cpp logits"
            );
        }

        let mut max_diff = 0.0_f32;
        let mut max_diff_token_id = 0_usize;
        for (token_id, (got, want)) in ocelotl_logits.iter().zip(llama_logits.iter()).enumerate() {
            let diff = (got - want).abs();
            if diff > max_diff {
                max_diff = diff;
                max_diff_token_id = token_id;
            }
            assert!(
                diff <= fixture.tolerance,
                "Gemma4 logit token {token_id}: got {got}, llama.cpp {want}, diff {diff} exceeds tolerance {}; max diff so far token {max_diff_token_id} diff {max_diff}",
                fixture.tolerance
            );
        }
    }

    #[test]
    #[ignore = "requires Gemma4 GGUF plus OCELOTL_LLAMA_DEBUG_PATH or local-artifacts/llama_cpp/llama-debug.exe; see docs/artifact-preparation.md"]
    fn local_gemma4_q4_k_m_late_tensor_summaries_match_llama_cpp_debug() {
        let model_path = local_gemma4_gguf_path();
        assert!(
            model_path.exists(),
            "missing Gemma4 GGUF at {}; see docs/artifact-preparation.md",
            model_path.display()
        );
        let llama_debug = local_llama_debug_path();
        assert!(
            llama_debug.exists(),
            "missing llama-debug at {}; set OCELOTL_LLAMA_DEBUG_PATH or see docs/artifact-preparation.md",
            llama_debug.display()
        );

        let fixture = load_gemma4_logits_fixture();
        let tokenizer = load_gemma4_gguf_tokenizer(&model_path)
            .expect("local Gemma4 GGUF tokenizer must build");
        let tokens = tokenizer
            .encode_with_configured_bos(&fixture.input)
            .expect("Gemma4 GGUF tokenizer must encode logits fixture prompt");
        let token_ids: Vec<u32> = tokens.iter().map(|token| token.0).collect();
        assert_eq!(
            token_ids, fixture.token_ids,
            "Ocelotl tokenization must match the logits fixture before comparing tensors"
        );

        let llama_tensors = run_llama_debug_tensor_summaries(
            &llama_debug,
            &model_path,
            &fixture.input,
            &["result_norm", "result_output"],
        );

        let (mut config, tensors) = load_gemma4_dequantized_tensors_from_gguf(&model_path)
            .expect("local Gemma4 GGUF tensors must load through explicit F32 dequantization");
        assert!(
            config.multimodal,
            "selected real Gemma4 artifact should still be recorded as multimodal"
        );
        config.multimodal = false;
        let weights = Gemma4TextWeights::from_loaded_tensors(&config, tensors)
            .expect("dequantized local Gemma4 tensors must map into text weights");
        let model = Gemma4TextModel::new(config.clone(), weights)
            .expect("text-projected dequantized Gemma4 model must build");
        let trace = model
            .prefill_with_trace(&tokens)
            .expect("Gemma4 text prefill trace must run");

        let llama_norm = llama_tensors
            .get("result_norm")
            .expect("llama-debug result_norm summary should exist");
        let ocelotl_norm = ocelotl_slice_for_llama_shape(
            "result_norm",
            &llama_norm.shape,
            &trace.result_norm,
            config.embedding_length,
        );
        assert_sum_close(
            "result_norm",
            sum_f32(ocelotl_norm),
            llama_norm.sum,
            fixture.tolerance,
        );

        let llama_output = llama_tensors
            .get("result_output")
            .expect("llama-debug result_output summary should exist");
        let output_element_count = shape_element_count(&llama_output.shape)
            .expect("llama-debug result_output shape element count must fit usize");
        assert_eq!(
            output_element_count,
            trace.result_output.len(),
            "llama-debug result_output shape {:?} must match Ocelotl result_output length {}",
            llama_output.shape,
            trace.result_output.len()
        );
        assert_sum_close(
            "result_output",
            sum_f32(&trace.result_output),
            llama_output.sum,
            fixture.tolerance,
        );
    }

    #[test]
    #[ignore = "requires Gemma4 GGUF plus OCELOTL_LLAMA_DEBUG_PATH or local-artifacts/llama_cpp/llama-debug.exe; see docs/artifact-preparation.md"]
    fn local_gemma4_q4_k_m_layer_output_summaries_match_llama_cpp_debug() {
        let model_path = local_gemma4_gguf_path();
        assert!(
            model_path.exists(),
            "missing Gemma4 GGUF at {}; see docs/artifact-preparation.md",
            model_path.display()
        );
        let llama_debug = local_llama_debug_path();
        assert!(
            llama_debug.exists(),
            "missing llama-debug at {}; set OCELOTL_LLAMA_DEBUG_PATH or see docs/artifact-preparation.md",
            llama_debug.display()
        );

        let fixture = load_gemma4_logits_fixture();
        let tokenizer = load_gemma4_gguf_tokenizer(&model_path)
            .expect("local Gemma4 GGUF tokenizer must build");
        let tokens = tokenizer
            .encode_with_configured_bos(&fixture.input)
            .expect("Gemma4 GGUF tokenizer must encode logits fixture prompt");
        let token_ids: Vec<u32> = tokens.iter().map(|token| token.0).collect();
        assert_eq!(
            token_ids, fixture.token_ids,
            "Ocelotl tokenization must match the logits fixture before comparing tensors"
        );

        let (mut config, tensors) = load_gemma4_dequantized_tensors_from_gguf(&model_path)
            .expect("local Gemma4 GGUF tensors must load through explicit F32 dequantization");
        assert!(
            config.multimodal,
            "selected real Gemma4 artifact should still be recorded as multimodal"
        );
        config.multimodal = false;

        let layer_names: Vec<String> = (0..config.block_count)
            .map(|layer_idx| format!("l_out-{layer_idx}"))
            .collect();
        let layer_name_refs: Vec<&str> = layer_names.iter().map(String::as_str).collect();
        let llama_layers = run_llama_debug_tensor_summaries(
            &llama_debug,
            &model_path,
            &fixture.input,
            &layer_name_refs,
        );

        let weights = Gemma4TextWeights::from_loaded_tensors(&config, tensors)
            .expect("dequantized local Gemma4 tensors must map into text weights");
        let model = Gemma4TextModel::new(config.clone(), weights)
            .expect("text-projected dequantized Gemma4 model must build");
        let trace = model
            .prefill_with_trace(&tokens)
            .expect("Gemma4 text prefill trace must run");
        assert_eq!(
            trace.layer_outputs.len(),
            config.block_count,
            "Gemma4 trace must include one layer output per block"
        );

        for (layer_idx, layer_output) in trace.layer_outputs.iter().enumerate() {
            let name = &layer_names[layer_idx];
            let llama = llama_layers
                .get(name)
                .unwrap_or_else(|| panic!("llama-debug summary for {name} should exist"));
            let ocelotl = ocelotl_slice_for_llama_shape(
                name,
                &llama.shape,
                layer_output,
                config.embedding_length,
            );
            let got = sum_f32(ocelotl);
            let diff = (got - llama.sum).abs();
            assert!(
                diff <= fixture.tolerance,
                "Gemma4 tensor {name} sum: got {got}, llama.cpp {}, diff {diff} exceeds tolerance {}; llama shape {:?}",
                llama.sum,
                fixture.tolerance,
                llama.shape
            );
        }
    }

    #[test]
    #[ignore = "requires Gemma4 GGUF plus OCELOTL_LLAMA_DEBUG_PATH or local-artifacts/llama_cpp/llama-debug.exe; see docs/artifact-preparation.md"]
    fn local_gemma4_q4_k_m_layer0_substep_summaries_match_llama_cpp_debug() {
        let model_path = local_gemma4_gguf_path();
        assert!(
            model_path.exists(),
            "missing Gemma4 GGUF at {}; see docs/artifact-preparation.md",
            model_path.display()
        );
        let llama_debug = local_llama_debug_path();
        assert!(
            llama_debug.exists(),
            "missing llama-debug at {}; set OCELOTL_LLAMA_DEBUG_PATH or see docs/artifact-preparation.md",
            llama_debug.display()
        );

        let fixture = load_gemma4_logits_fixture();
        let tokenizer = load_gemma4_gguf_tokenizer(&model_path)
            .expect("local Gemma4 GGUF tokenizer must build");
        let tokens = tokenizer
            .encode_with_configured_bos(&fixture.input)
            .expect("Gemma4 GGUF tokenizer must encode logits fixture prompt");
        let token_ids: Vec<u32> = tokens.iter().map(|token| token.0).collect();
        assert_eq!(
            token_ids, fixture.token_ids,
            "Ocelotl tokenization must match the logits fixture before comparing tensors"
        );

        let names = [
            "inp_scaled",
            "attn_norm-0",
            "Qcur-0",
            "Qcur_normed-0",
            "Qcur_pos-0",
            "Kcur-0",
            "Vcur-0",
            "Kcur_normed-0",
            "Vcur_normed-0",
            "Kcur_pos-0",
            "attn_post_norm-0",
            "attn_out-0",
            "ffn_norm-0",
            "ffn_out-0",
            "ffn_post_norm-0",
            "pe_in-0",
            "per_layer_embd_out-0",
            "l_out-0",
        ];
        let llama_tensors =
            run_llama_debug_tensor_summaries(&llama_debug, &model_path, &fixture.input, &names);

        let (mut config, tensors) = load_gemma4_dequantized_tensors_from_gguf(&model_path)
            .expect("local Gemma4 GGUF tensors must load through explicit F32 dequantization");
        assert!(
            config.multimodal,
            "selected real Gemma4 artifact should still be recorded as multimodal"
        );
        config.multimodal = false;
        let weights = Gemma4TextWeights::from_loaded_tensors(&config, tensors)
            .expect("dequantized local Gemma4 tensors must map into text weights");
        let model = Gemma4TextModel::new(config, weights)
            .expect("text-projected dequantized Gemma4 model must build");
        let trace = model
            .prefill_with_trace(&tokens)
            .expect("Gemma4 text prefill trace must run");

        for name in names {
            let llama = llama_tensors
                .get(name)
                .unwrap_or_else(|| panic!("llama-debug summary for {name} should exist"));
            let ocelotl = trace
                .named_tensors
                .get(name)
                .unwrap_or_else(|| panic!("Ocelotl trace tensor {name} should exist"));
            let element_count = shape_element_count(&llama.shape).unwrap_or_else(|| {
                panic!(
                    "llama-debug tensor {name} shape {:?} must fit usize",
                    llama.shape
                )
            });
            assert_eq!(
                element_count,
                ocelotl.len(),
                "llama-debug tensor {name} shape {:?} must match Ocelotl tensor length {}",
                llama.shape,
                ocelotl.len()
            );
            let got = sum_f32(ocelotl);
            let diff = (got - llama.sum).abs();
            assert!(
                diff <= fixture.tolerance,
                "Gemma4 tensor {name} sum: got {got}, llama.cpp {}, diff {diff} exceeds tolerance {}; llama shape {:?}",
                llama.sum,
                fixture.tolerance,
                llama.shape
            );
        }
    }

    #[test]
    #[ignore = "requires Gemma4 GGUF plus OCELOTL_LLAMA_DEBUG_PATH or local-artifacts/llama_cpp/llama-debug.exe; see docs/artifact-preparation.md"]
    fn local_gemma4_q4_k_m_native_kquant_q8k_qcur_summary_matches_llama_cpp_debug() {
        let model_path = local_gemma4_gguf_path();
        assert!(
            model_path.exists(),
            "missing Gemma4 GGUF at {}; see docs/artifact-preparation.md",
            model_path.display()
        );
        let llama_debug = local_llama_debug_path();
        assert!(
            llama_debug.exists(),
            "missing llama-debug at {}; set OCELOTL_LLAMA_DEBUG_PATH or see docs/artifact-preparation.md",
            llama_debug.display()
        );

        let fixture = load_gemma4_logits_fixture();
        let tokenizer = load_gemma4_gguf_tokenizer(&model_path)
            .expect("local Gemma4 GGUF tokenizer must build");
        let tokens = tokenizer
            .encode_with_configured_bos(&fixture.input)
            .expect("Gemma4 GGUF tokenizer must encode logits fixture prompt");
        let token_ids: Vec<u32> = tokens.iter().map(|token| token.0).collect();
        assert_eq!(
            token_ids, fixture.token_ids,
            "Ocelotl tokenization must match the logits fixture before comparing tensors"
        );

        let llama_tensors = run_llama_debug_tensor_summaries(
            &llama_debug,
            &model_path,
            &fixture.input,
            &["Qcur-0"],
        );
        let llama_qcur = llama_tensors
            .get("Qcur-0")
            .expect("llama-debug Qcur-0 summary should exist");

        let (mut config, tensors) = load_gemma4_dequantized_tensors_from_gguf(&model_path)
            .expect("local Gemma4 GGUF tensors must load through explicit F32 dequantization");
        assert!(
            config.multimodal,
            "selected real Gemma4 artifact should still be recorded as multimodal"
        );
        config.multimodal = false;
        let weights = Gemma4TextWeights::from_loaded_tensors(&config, tensors)
            .expect("dequantized local Gemma4 tensors must map into text weights");
        let model = Gemma4TextModel::new(config.clone(), weights)
            .expect("text-projected dequantized Gemma4 model must build");
        let trace = model
            .prefill_with_trace(&tokens)
            .expect("Gemma4 text prefill trace must run");
        let attn_norm = trace
            .named_tensors
            .get("attn_norm-0")
            .expect("Ocelotl trace must include attn_norm-0");

        let raw_attn_q = load_raw_gguf_tensor_bytes(&model_path, "blk.0.attn_q.weight");
        assert_eq!(
            raw_attn_q.tensor_type,
            GgmlTensorType::Q6K,
            "selected Gemma4 Q4_K_M artifact stores blk.0.attn_q.weight as Q6_K"
        );
        let native_sum = k_quant_q8_k_projection_sum(
            &raw_attn_q.data,
            raw_attn_q.tensor_type,
            &raw_attn_q.shape,
            attn_norm,
            tokens.len(),
            config.embedding_length,
            *raw_attn_q
                .shape
                .get(1)
                .expect("Gemma4 attn_q GGUF tensor must include output dimension"),
        );

        assert_sum_close(
            "native K-quant x Q8_K Qcur-0",
            native_sum,
            llama_qcur.sum,
            fixture.tolerance,
        );
    }

    fn local_gemma4_gguf_path() -> PathBuf {
        if let Ok(path) = std::env::var("OCELOTL_GEMMA4_GGUF_PATH") {
            return PathBuf::from(path);
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("local-artifacts")
            .join("gemma4_e4b_it_q4_k_m")
            .join("google_gemma-4-E4B-it-Q4_K_M.gguf")
    }

    fn local_llama_tokenize_path() -> PathBuf {
        if let Ok(path) = std::env::var("OCELOTL_LLAMA_TOKENIZE_PATH") {
            return PathBuf::from(path);
        }
        let executable = if cfg!(windows) {
            "llama-tokenize.exe"
        } else {
            "llama-tokenize"
        };
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("local-artifacts")
            .join("llama_cpp")
            .join(executable)
    }

    fn local_llama_debug_path() -> PathBuf {
        if let Ok(path) = std::env::var("OCELOTL_LLAMA_DEBUG_PATH") {
            return PathBuf::from(path);
        }
        let executable = if cfg!(windows) {
            "llama-debug.exe"
        } else {
            "llama-debug"
        };
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("local-artifacts")
            .join("llama_cpp")
            .join(executable)
    }

    fn run_llama_tokenize_ids(
        llama_tokenize: &Path,
        model_path: &Path,
        input: &str,
        add_bos: bool,
    ) -> Vec<u32> {
        let mut command = Command::new(llama_tokenize);
        command
            .arg("--model")
            .arg(model_path)
            .arg("--prompt")
            .arg(input)
            .arg("--ids")
            .arg("--no-escape")
            .arg("--log-disable");
        if !add_bos {
            command.arg("--no-bos");
        }

        let output = command.output().unwrap_or_else(|err| {
            panic!(
                "failed to run llama-tokenize at {}: {err}",
                llama_tokenize.display()
            )
        });
        assert!(
            output.status.success(),
            "llama-tokenize failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        parse_llama_tokenize_ids(&output.stdout).unwrap_or_else(|err| {
            panic!(
                "failed to parse llama-tokenize stdout: {err}\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        })
    }

    fn parse_llama_tokenize_ids(stdout: &[u8]) -> std::result::Result<Vec<u32>, String> {
        let rendered = std::str::from_utf8(stdout)
            .map_err(|err| format!("llama-tokenize stdout was not UTF-8: {err}"))?;
        let start = rendered
            .find('[')
            .ok_or_else(|| format!("missing '[' in llama-tokenize stdout: {rendered:?}"))?;
        let tail = &rendered[start + 1..];
        let end = tail
            .find(']')
            .ok_or_else(|| format!("missing ']' in llama-tokenize stdout: {rendered:?}"))?;
        let ids = tail[..end].trim();
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        ids.split(',')
            .map(|part| {
                let trimmed = part.trim();
                trimmed
                    .parse::<u32>()
                    .map_err(|err| format!("invalid llama-tokenize token id {trimmed:?}: {err}"))
            })
            .collect()
    }

    fn unique_llama_debug_output_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "ocelotl-gemma4-llama-debug-{}-{nanos}",
            std::process::id()
        ))
    }

    fn run_llama_debug_logits(
        llama_debug: &Path,
        model_path: &Path,
        input: &str,
        output_dir: &Path,
    ) -> Vec<f32> {
        fs::create_dir_all(output_dir).unwrap_or_else(|err| {
            panic!(
                "failed to create llama-debug output dir {}: {err}",
                output_dir.display()
            )
        });

        let output = Command::new(llama_debug)
            .arg("--model")
            .arg(model_path)
            .arg("--prompt")
            .arg(input)
            .arg("--no-escape")
            .arg("--save-logits")
            .arg("--logits-output-dir")
            .arg(output_dir)
            .output()
            .unwrap_or_else(|err| {
                panic!(
                    "failed to run llama-debug at {}: {err}",
                    llama_debug.display()
                )
            });
        assert!(
            output.status.success(),
            "llama-debug failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let logits_path = llama_debug_logits_text_path(output_dir);
        let raw = fs::read_to_string(&logits_path).unwrap_or_else(|err| {
            panic!(
                "failed to read llama-debug logits text at {}: {err}",
                logits_path.display()
            )
        });
        parse_llama_debug_logits_text(&raw).unwrap_or_else(|err| {
            panic!(
                "failed to parse llama-debug logits text at {}: {err}",
                logits_path.display()
            )
        })
    }

    fn run_llama_debug_tensor_summaries(
        llama_debug: &Path,
        model_path: &Path,
        input: &str,
        tensor_names: &[&str],
    ) -> BTreeMap<String, LlamaDebugTensorSummary> {
        let mut command = Command::new(llama_debug);
        command
            .arg("--model")
            .arg(model_path)
            .arg("--prompt")
            .arg(input)
            .arg("--no-escape")
            .arg("--no-warmup")
            .arg("--verbose");
        if !tensor_names.is_empty() {
            command
                .arg("--tensor-filter")
                .arg(format!("({})$", tensor_names.join("|")));
        }

        let output = command.output().unwrap_or_else(|err| {
            panic!(
                "failed to run llama-debug at {}: {err}",
                llama_debug.display()
            )
        });
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "llama-debug tensor run failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            stdout,
            stderr
        );

        let raw = format!("{stdout}\n{stderr}");
        parse_llama_debug_tensor_summaries(&raw, tensor_names).unwrap_or_else(|err| {
            panic!(
                "failed to parse llama-debug tensor summaries: {err}\nstdout:\n{}\nstderr:\n{}",
                stdout, stderr
            )
        })
    }

    #[derive(Debug, Clone)]
    struct RawGgufTensorBytes {
        tensor_type: GgmlTensorType,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    fn load_raw_gguf_tensor_bytes(path: &Path, tensor_name: &str) -> RawGgufTensorBytes {
        let manifest = inspect_gguf(path).expect("local Gemma4 GGUF manifest must inspect");
        let tensor = manifest
            .tensors
            .iter()
            .find(|tensor| tensor.name == tensor_name)
            .unwrap_or_else(|| panic!("GGUF tensor {tensor_name} must exist"));
        let byte_len = tensor
            .byte_len
            .unwrap_or_else(|| panic!("GGUF tensor {tensor_name} must have known byte length"));
        let byte_len: usize = byte_len
            .try_into()
            .expect("GGUF tensor byte length must fit usize");
        let mut file = fs::File::open(path).unwrap_or_else(|err| {
            panic!(
                "failed to open local Gemma4 GGUF at {}: {err}",
                path.display()
            )
        });
        file.seek(SeekFrom::Start(tensor.file_offset))
            .unwrap_or_else(|err| panic!("failed to seek to tensor {tensor_name}: {err}"));
        let mut data = vec![0_u8; byte_len];
        file.read_exact(&mut data)
            .unwrap_or_else(|err| panic!("failed to read tensor {tensor_name}: {err}"));
        RawGgufTensorBytes {
            tensor_type: tensor.tensor_type,
            shape: tensor.shape.clone(),
            data,
        }
    }

    #[derive(Debug, Clone)]
    struct Q8KBlock {
        d: f32,
        qs: [i8; QK_K],
        bsums: [i16; QK_K / 16],
    }

    fn k_quant_q8_k_projection_sum(
        raw_weight: &[u8],
        tensor_type: GgmlTensorType,
        weight_shape: &[usize],
        input: &[f32],
        seq: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> f32 {
        assert_eq!(
            weight_shape,
            [input_dim, output_dim],
            "Gemma4 GGUF K-quant projection shape must be [input_dim, output_dim]"
        );
        assert_eq!(
            input.len(),
            seq * input_dim,
            "Gemma4 K-quant projection input length must match seq * input_dim"
        );
        assert_eq!(
            input_dim % QK_K,
            0,
            "K-quant/Q8_K projection input dim must be block-aligned"
        );
        let blocks_per_row = input_dim / QK_K;
        let block_bytes = match tensor_type {
            GgmlTensorType::Q4K => Q4_K_BLOCK_BYTES,
            GgmlTensorType::Q6K => Q6_K_BLOCK_BYTES,
            other => panic!("unsupported native K-quant diagnostic type {other:?}"),
        };
        assert_eq!(
            raw_weight.len(),
            output_dim * blocks_per_row * block_bytes,
            "Gemma4 raw K-quant projection byte length must match shape"
        );

        let mut total = 0.0_f32;
        for token_idx in 0..seq {
            let input_start = token_idx * input_dim;
            let q8_blocks = quantize_row_q8_k(&input[input_start..input_start + input_dim]);
            for output_idx in 0..output_dim {
                let row_start = output_idx * blocks_per_row * block_bytes;
                let row = &raw_weight[row_start..row_start + blocks_per_row * block_bytes];
                total += match tensor_type {
                    GgmlTensorType::Q4K => vec_dot_q4_k_q8_k(row, &q8_blocks),
                    GgmlTensorType::Q6K => vec_dot_q6_k_q8_k(row, &q8_blocks),
                    other => panic!("unsupported native K-quant diagnostic type {other:?}"),
                };
            }
        }
        total
    }

    fn quantize_row_q8_k(input: &[f32]) -> Vec<Q8KBlock> {
        assert_eq!(
            input.len() % QK_K,
            0,
            "Q8_K input length must be block-aligned"
        );
        let mut blocks = Vec::with_capacity(input.len() / QK_K);
        for block_input in input.chunks_exact(QK_K) {
            let mut max = 0.0_f32;
            let mut amax = 0.0_f32;
            for value in block_input {
                let abs = value.abs();
                if abs > amax {
                    amax = abs;
                    max = *value;
                }
            }

            let mut block = Q8KBlock {
                d: 0.0,
                qs: [0; QK_K],
                bsums: [0; QK_K / 16],
            };
            if amax == 0.0 {
                blocks.push(block);
                continue;
            }

            let iscale = -127.0 / max;
            for (idx, value) in block_input.iter().enumerate() {
                let quant = nearest_int(iscale * *value).min(127);
                assert!(
                    (-128..=127).contains(&quant),
                    "Q8_K quantized value must fit i8"
                );
                block.qs[idx] = quant as i8;
            }
            for group in 0..QK_K / 16 {
                let start = group * 16;
                block.bsums[group] = block.qs[start..start + 16]
                    .iter()
                    .map(|value| i16::from(*value))
                    .sum();
            }
            block.d = 1.0 / iscale;
            blocks.push(block);
        }
        blocks
    }

    fn vec_dot_q4_k_q8_k(raw_q4_k_row: &[u8], q8_blocks: &[Q8KBlock]) -> f32 {
        assert_eq!(
            raw_q4_k_row.len(),
            q8_blocks.len() * Q4_K_BLOCK_BYTES,
            "Q4_K row bytes must align with Q8_K block count"
        );
        let mut lane_sums = [0.0_f32; 8];
        let mut sumf = 0.0_f32;

        for (raw_block, q8) in raw_q4_k_row
            .chunks_exact(Q4_K_BLOCK_BYTES)
            .zip(q8_blocks.iter())
        {
            let q4 = &raw_block[16..144];
            let mut aux8 = [0_i8; QK_K];
            for group64 in 0..QK_K / 64 {
                let q4_start = group64 * 32;
                let value_start = group64 * 64;
                for l in 0..32 {
                    aux8[value_start + l] = (q4[q4_start + l] & 0x0f) as i8;
                }
                for l in 0..32 {
                    aux8[value_start + 32 + l] = (q4[q4_start + l] >> 4) as i8;
                }
            }

            let mut aux32 = [0_i32; 8];
            for group32 in 0..QK_K / 32 {
                let (scale, _) = get_scale_min_k4(group32, &raw_block[4..16]);
                let value_start = group32 * 32;
                for l in 0..32 {
                    let lane = l % 8;
                    aux32[lane] += i32::from(scale)
                        * i32::from(q8.qs[value_start + l])
                        * i32::from(aux8[value_start + l]);
                }
            }

            let mut min_sum = 0_i32;
            for group16 in 0..QK_K / 16 {
                let (_, min) = get_scale_min_k4(group16 / 2, &raw_block[4..16]);
                min_sum += i32::from(q8.bsums[group16]) * i32::from(min);
            }

            let d = f16_le_at(raw_block, 0) * q8.d;
            for (lane, aux) in aux32.iter().enumerate() {
                lane_sums[lane] += d * *aux as f32;
            }
            let dmin = f16_le_at(raw_block, 2) * q8.d;
            sumf -= dmin * min_sum as f32;
        }

        sumf + lane_sums.iter().copied().sum::<f32>()
    }

    fn vec_dot_q6_k_q8_k(raw_q6_k_row: &[u8], q8_blocks: &[Q8KBlock]) -> f32 {
        assert_eq!(
            raw_q6_k_row.len(),
            q8_blocks.len() * Q6_K_BLOCK_BYTES,
            "Q6_K row bytes must align with Q8_K block count"
        );
        let mut lane_sums = [0.0_f32; 8];

        for (raw_block, q8) in raw_q6_k_row
            .chunks_exact(Q6_K_BLOCK_BYTES)
            .zip(q8_blocks.iter())
        {
            let ql = &raw_block[0..128];
            let qh = &raw_block[128..192];
            let scales = &raw_block[192..208];
            let mut aux8 = [0_i8; QK_K];

            for super_group in 0..2 {
                let ql_base = super_group * 64;
                let qh_base = super_group * 32;
                let value_base = super_group * 128;
                for l in 0..32 {
                    let high = qh[qh_base + l];
                    aux8[value_base + l] =
                        (((ql[ql_base + l] & 0x0f) | ((high & 0x03) << 4)) as i8) - 32;
                    aux8[value_base + l + 32] =
                        (((ql[ql_base + l + 32] & 0x0f) | (((high >> 2) & 0x03) << 4)) as i8) - 32;
                    aux8[value_base + l + 64] =
                        (((ql[ql_base + l] >> 4) | (((high >> 4) & 0x03) << 4)) as i8) - 32;
                    aux8[value_base + l + 96] =
                        (((ql[ql_base + l + 32] >> 4) | (((high >> 6) & 0x03) << 4)) as i8) - 32;
                }
            }

            let mut aux32 = [0_i32; 8];
            for (group16, scale_byte) in scales.iter().enumerate().take(QK_K / 16) {
                let scale = i8::from_ne_bytes([*scale_byte]);
                let value_start = group16 * 16;
                for l in 0..16 {
                    let lane = l % 8;
                    aux32[lane] += i32::from(scale)
                        * i32::from(q8.qs[value_start + l])
                        * i32::from(aux8[value_start + l]);
                }
            }

            let d = f16_le_at(raw_block, 208) * q8.d;
            for (lane, aux) in aux32.iter().enumerate() {
                lane_sums[lane] += d * *aux as f32;
            }
        }

        lane_sums.iter().copied().sum()
    }

    fn get_scale_min_k4(index: usize, scales: &[u8]) -> (u8, u8) {
        assert_eq!(scales.len(), 12);
        assert!(index < 8);
        if index < 4 {
            (scales[index] & 0x3f, scales[index + 4] & 0x3f)
        } else {
            (
                (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
                (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
            )
        }
    }

    fn nearest_int(value: f32) -> i32 {
        assert!(value.abs() <= 4_194_303.0);
        let bits = (value + 12_582_912.0).to_bits();
        ((bits & 0x007f_ffff) as i32) - 0x0040_0000
    }

    fn f16_le_at(data: &[u8], offset: usize) -> f32 {
        let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        f16_bits_to_f32(bits)
    }

    fn f16_bits_to_f32(bits: u16) -> f32 {
        let sign = ((bits & 0x8000) as u32) << 16;
        let exp = (bits >> 10) & 0x1f;
        let frac = (bits & 0x03ff) as u32;

        let f32_bits = match exp {
            0 => {
                if frac == 0 {
                    sign
                } else {
                    let mut frac_norm = frac;
                    let mut exp_unbiased = -14_i32;
                    while (frac_norm & 0x0400) == 0 {
                        frac_norm <<= 1;
                        exp_unbiased -= 1;
                    }
                    frac_norm &= 0x03ff;
                    let exp32 = (exp_unbiased + 127) as u32;
                    sign | (exp32 << 23) | (frac_norm << 13)
                }
            }
            0x1f => sign | 0x7f80_0000 | (frac << 13),
            _ => {
                let exp32 = u32::from(exp) + (127 - 15);
                sign | (exp32 << 23) | (frac << 13)
            }
        };

        f32::from_bits(f32_bits)
    }

    fn llama_debug_logits_text_path(output_dir: &Path) -> PathBuf {
        let mut candidates = Vec::new();
        for entry in fs::read_dir(output_dir).unwrap_or_else(|err| {
            panic!(
                "failed to read llama-debug output dir {}: {err}",
                output_dir.display()
            )
        }) {
            let path = entry
                .unwrap_or_else(|err| panic!("failed to read llama-debug output entry: {err}"))
                .path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("txt") {
                continue;
            }
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.ends_with("-prompt.txt") {
                continue;
            }
            candidates.push(path);
        }

        assert_eq!(
            candidates.len(),
            1,
            "expected exactly one llama-debug logits .txt file in {}, got {:?}",
            output_dir.display(),
            candidates
        );
        candidates.remove(0)
    }

    fn parse_llama_debug_selected_logits_text(
        raw: &str,
        selected_token_ids: &[u32],
    ) -> std::result::Result<BTreeMap<u32, f32>, String> {
        let mut selected = BTreeMap::new();
        for token_id in selected_token_ids {
            if selected.insert(*token_id, None).is_some() {
                return Err(format!("duplicate selected logit token id {token_id}"));
            }
        }

        for (line_idx, line) in raw.lines().enumerate() {
            let Some((left, right)) = line.split_once(':') else {
                continue;
            };
            let token_id = left.trim().parse::<u32>().map_err(|err| {
                format!(
                    "invalid llama-debug logit token id {:?} on line {}: {err}",
                    left.trim(),
                    line_idx + 1
                )
            })?;
            let Some(slot) = selected.get_mut(&token_id) else {
                continue;
            };
            let value = right.trim().parse::<f32>().map_err(|err| {
                format!(
                    "invalid llama-debug logit value {:?} for token id {} on line {}: {err}",
                    right.trim(),
                    token_id,
                    line_idx + 1
                )
            })?;
            if !value.is_finite() {
                return Err(format!(
                    "non-finite llama-debug logit value {value} for token id {token_id}"
                ));
            }
            *slot = Some(value);
        }

        let mut parsed = BTreeMap::new();
        for (token_id, value) in selected {
            let value =
                value.ok_or_else(|| format!("missing selected logit token id {token_id}"))?;
            parsed.insert(token_id, value);
        }
        Ok(parsed)
    }

    fn parse_llama_debug_logits_text(raw: &str) -> std::result::Result<Vec<f32>, String> {
        let mut logits = Vec::new();
        for (line_idx, line) in raw.lines().enumerate() {
            let Some((left, right)) = line.split_once(':') else {
                continue;
            };
            let token_id = left.trim().parse::<usize>().map_err(|err| {
                format!(
                    "invalid llama-debug logit token id {:?} on line {}: {err}",
                    left.trim(),
                    line_idx + 1
                )
            })?;
            if token_id != logits.len() {
                return Err(format!(
                    "expected token id {}, got {token_id} on line {}",
                    logits.len(),
                    line_idx + 1
                ));
            }
            let value = right.trim().parse::<f32>().map_err(|err| {
                format!(
                    "invalid llama-debug logit value {:?} for token id {} on line {}: {err}",
                    right.trim(),
                    token_id,
                    line_idx + 1
                )
            })?;
            if !value.is_finite() {
                return Err(format!(
                    "non-finite llama-debug logit value {value} for token id {token_id}"
                ));
            }
            logits.push(value);
        }
        if logits.is_empty() {
            return Err("llama-debug logits text did not contain any `id: value` rows".to_string());
        }
        Ok(logits)
    }

    fn parse_llama_debug_tensor_summaries(
        raw: &str,
        tensor_names: &[&str],
    ) -> std::result::Result<BTreeMap<String, LlamaDebugTensorSummary>, String> {
        let expected: BTreeSet<&str> = tensor_names.iter().copied().collect();
        if expected.len() != tensor_names.len() {
            return Err("duplicate llama-debug tensor names requested".to_string());
        }

        let mut summaries = BTreeMap::new();
        let mut pending: Option<(String, Vec<usize>)> = None;

        for line in raw.lines() {
            if let Some((name, shape)) = parse_llama_debug_tensor_header(line, &expected)? {
                if pending.is_some() {
                    let pending_name = pending.as_ref().map(|(name, _)| name.as_str()).unwrap();
                    return Err(format!(
                        "missing sum line for llama-debug tensor {pending_name}"
                    ));
                }
                pending = Some((name, shape));
                continue;
            }

            let trimmed = line.trim();
            if let Some(sum_text) = trimmed.strip_prefix("sum =") {
                let Some((name, shape)) = pending.take() else {
                    continue;
                };
                let sum = sum_text.trim().parse::<f32>().map_err(|err| {
                    format!(
                        "invalid llama-debug tensor sum {:?} for {name}: {err}",
                        sum_text.trim()
                    )
                })?;
                if !sum.is_finite() {
                    return Err(format!(
                        "non-finite llama-debug tensor sum {sum} for {name}"
                    ));
                }
                summaries.insert(name, LlamaDebugTensorSummary { shape, sum });
            }
        }

        if let Some((name, _)) = pending {
            return Err(format!("missing sum line for llama-debug tensor {name}"));
        }

        for tensor_name in tensor_names {
            if !summaries.contains_key(*tensor_name) {
                return Err(format!(
                    "missing llama-debug tensor summary for {tensor_name}"
                ));
            }
        }

        Ok(summaries)
    }

    fn parse_llama_debug_tensor_header(
        line: &str,
        expected: &BTreeSet<&str>,
    ) -> std::result::Result<Option<(String, Vec<usize>)>, String> {
        let Some((_, right)) = line.split_once("common_debug_cb_eval:") else {
            return Ok(None);
        };
        let Some(name) = right.split_whitespace().next() else {
            return Ok(None);
        };
        if !expected.contains(name) {
            return Ok(None);
        }
        let Some(shape_start) = line.rfind("= {") else {
            return Err(format!(
                "missing trailing shape for llama-debug tensor {name}: {line:?}"
            ));
        };
        let shape_text = &line[shape_start + 3..];
        let Some(shape_end) = shape_text.find('}') else {
            return Err(format!(
                "unterminated trailing shape for llama-debug tensor {name}: {line:?}"
            ));
        };
        let dims = shape_text[..shape_end]
            .split(',')
            .map(|part| {
                let trimmed = part.trim();
                trimmed.parse::<usize>().map_err(|err| {
                    format!("invalid llama-debug tensor dimension {trimmed:?} for {name}: {err}")
                })
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if dims.is_empty() {
            return Err(format!("empty llama-debug tensor shape for {name}"));
        }

        Ok(Some((name.to_string(), dims)))
    }

    fn shape_element_count(shape: &[usize]) -> Option<usize> {
        shape
            .iter()
            .try_fold(1_usize, |acc, dim| acc.checked_mul(*dim))
    }

    fn ocelotl_slice_for_llama_shape<'a>(
        name: &str,
        llama_shape: &[usize],
        values: &'a [f32],
        final_token_width: usize,
    ) -> &'a [f32] {
        let element_count = shape_element_count(llama_shape).unwrap_or_else(|| {
            panic!("llama-debug tensor {name} shape {llama_shape:?} element count must fit usize")
        });
        if element_count == values.len() {
            return values;
        }
        if element_count == final_token_width {
            let start = values.len() - final_token_width;
            return &values[start..start + final_token_width];
        }
        panic!(
            "llama-debug tensor {name} shape {llama_shape:?} has {element_count} elements, expected either Ocelotl full length {} or final-token length {final_token_width}",
            values.len()
        );
    }

    fn sum_f32(values: &[f32]) -> f32 {
        values.iter().copied().sum()
    }

    fn assert_sum_close(name: &str, got: f32, want: f32, tolerance: f32) {
        let diff = (got - want).abs();
        assert!(
            diff <= tolerance,
            "Gemma4 tensor {name} sum: got {got}, llama.cpp {want}, diff {diff} exceeds tolerance {tolerance}"
        );
    }
}
