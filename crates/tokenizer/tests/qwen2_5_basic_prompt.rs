//! M2.3 — exact token-ID round trip against the pinned Qwen2.5-0.5B-Instruct
//! tokenizer.
//!
//! This file pairs two test surfaces:
//!
//! 1. A default-on test that validates the small committed fixture
//!    (`fixtures/tokenizer/qwen2_5_basic_prompt.json`) is well-formed and
//!    has been populated with real IDs. This runs offline and does NOT
//!    require local artifacts; it guards the fixture itself against
//!    accidental regressions back to the placeholder `[]` shape.
//!
//! 2. An `#[ignore]`-by-default opt-in test that loads the real Qwen2.5
//!    `tokenizer.json` from `local-artifacts/qwen2_5_0_5b_instruct/` (per
//!    `docs/artifact-preparation.md`), encodes the fixture's `input` via
//!    the Ocelotl `JsonTokenizer`, and asserts the IDs match the fixture
//!    AND that decode round-trips back to the original text. This is the
//!    test that proves the `expected_token_ids` were not just typed in by
//!    a human — they came from the wrapper itself against the pinned SHA.
//!
//! Run the opt-in test with:
//!
//! ```text
//! cargo test -p ocelotl-tokenizer --test qwen2_5_basic_prompt -- --ignored
//! ```

use std::path::{Path, PathBuf};

use ocelotl_tokenizer::{JsonTokenizer, TokenId, Tokenizer};
use serde::Deserialize;

const LOCAL_ARTIFACT_DIR: &str = "local-artifacts/qwen2_5_0_5b_instruct";
const PINNED_REVISION: &str = "7ae557604adf67be50417f59c2c2f167def9a775";

/// Typed view of `fixtures/tokenizer/qwen2_5_basic_prompt.json`. Only the
/// fields the test asserts on are deserialized — extra fields in the JSON
/// (e.g. `purpose`, `notes`) are tolerated but ignored.
#[derive(Debug, Deserialize)]
struct BasicPromptFixture {
    fixture_version: u32,
    name: String,
    source: String,
    input: String,
    expected_token_ids: Vec<u32>,
    #[serde(default)]
    decoded: Option<String>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("tokenizer")
        .join("qwen2_5_basic_prompt.json")
}

fn load_fixture() -> BasicPromptFixture {
    let path = fixture_path();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture at {} — {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| {
        panic!(
            "failed to parse fixture at {} as JSON — {e}",
            path.display()
        )
    })
}

fn artifact_path(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(LOCAL_ARTIFACT_DIR)
        .join(file)
}

#[test]
fn fixture_is_well_formed_and_populated() {
    let fixture = load_fixture();

    assert_eq!(fixture.fixture_version, 1, "fixture_version must be 1");
    assert_eq!(
        fixture.name, "qwen2_5_basic_prompt",
        "fixture name pinned by docs/validation/fixtures.md"
    );
    assert!(
        fixture.source.contains(PINNED_REVISION),
        "fixture source must reference pinned Qwen2.5 SHA `{PINNED_REVISION}`; \
         got source = {:?}",
        fixture.source
    );
    assert!(!fixture.input.is_empty(), "fixture input must be non-empty");
    assert!(
        !fixture.expected_token_ids.is_empty(),
        "expected_token_ids is still empty — M2.3 requires real IDs from the \
         pinned Qwen2.5 tokenizer; populate via the #[ignore]'d test or by \
         hand from a known reference and pin them here"
    );
    assert!(
        fixture.decoded.is_some(),
        "fixture must declare its decoded form so round-trip semantics are \
         explicit (whitespace handling, special-token stripping, etc.)"
    );
}

#[test]
#[ignore = "requires local-artifacts/qwen2_5_0_5b_instruct/tokenizer.json — see docs/artifact-preparation.md"]
fn json_tokenizer_round_trips_qwen2_5_basic_prompt() {
    let tok_path = artifact_path("tokenizer.json");
    assert!(
        tok_path.exists(),
        "missing artifact at {} — see docs/artifact-preparation.md for fetch \
         instructions (huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
         --revision {PINNED_REVISION} ...)",
        tok_path.display(),
    );

    let fixture = load_fixture();
    let tokenizer = JsonTokenizer::from_json_path(&tok_path)
        .expect("JsonTokenizer must load the pinned Qwen2.5 tokenizer.json");

    // Encode-side: the wrapper's IDs must equal the pinned fixture IDs
    // exactly. No tolerance — different IDs means either the tokenizer
    // changed under the pin (a violation; bump model-target.md) or the
    // fixture is wrong.
    let encoded: Vec<TokenId> = tokenizer
        .encode(&fixture.input)
        .expect("encode must succeed against the pinned tokenizer");
    let encoded_raw: Vec<u32> = encoded.iter().map(|t| t.0).collect();
    assert_eq!(
        encoded_raw, fixture.expected_token_ids,
        "wrapper-produced IDs disagree with fixture; either re-pin the \
         fixture (ran against a different SHA?) or the tokenizer.json on disk \
         is not the pinned revision"
    );

    // Decode-side: the wrapper's text must equal the fixture's declared
    // decoded form. For the basic ASCII prompt `Hello`, this should be the
    // original input verbatim — but we assert against the fixture's
    // `decoded` field rather than `input` so the fixture remains the
    // single source of truth for whitespace/special-token semantics.
    let decoded = tokenizer
        .decode(&encoded)
        .expect("decode must succeed against the pinned tokenizer");
    let expected_decoded = fixture
        .decoded
        .as_deref()
        .expect("decoded field guarded by fixture_is_well_formed_and_populated");
    assert_eq!(
        decoded, expected_decoded,
        "decoded text disagrees with fixture; check decoder configuration on \
         the pinned tokenizer.json"
    );
}

/// Sanity helper kept around to make it cheap to capture IDs when re-pinning
/// after an upstream tokenizer change. Not part of the M2.3 contract — the
/// real assertion lives in `json_tokenizer_round_trips_qwen2_5_basic_prompt`.
///
/// To re-pin: edit the fixture, re-run `cargo test -p ocelotl-tokenizer
/// --test qwen2_5_basic_prompt -- --ignored`.
#[allow(dead_code)]
fn _capture_ids_for_repinning(input: &str) -> (Vec<u32>, String) {
    let tok_path: &Path = &artifact_path("tokenizer.json");
    let tokenizer = JsonTokenizer::from_json_path(tok_path).unwrap();
    let ids: Vec<u32> = tokenizer
        .encode(input)
        .unwrap()
        .into_iter()
        .map(|t| t.0)
        .collect();
    let decoded = tokenizer
        .decode(&ids.iter().copied().map(TokenId).collect::<Vec<_>>())
        .unwrap();
    (ids, decoded)
}
