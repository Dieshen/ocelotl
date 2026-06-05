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
        path::{Path, PathBuf},
        process::Command,
    };

    use ocelotl_tokenizer::Tokenizer;
    use serde::Deserialize;

    use super::*;

    const GEMMA4_GGUF_REVISION: &str = "c04cb322fd63e347db759a08b6249b867488ccf8";

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

    fn load_gemma4_tokenizer_fixture() -> Gemma4TokenizerFixture {
        let path = gemma4_tokenizer_fixture_path();
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
}
