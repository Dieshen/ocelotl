//! M2.4 — exact chat-template render against the pinned Qwen2.5-0.5B-Instruct
//! chat template.
//!
//! Pairs two test surfaces (per James's M2.3 two-test pattern, applied here
//! as a second occurrence — flag for promotion):
//!
//! 1. A default-on test that compiles the inline chat_template stored in
//!    `fixtures/chat/qwen2_5_basic_chat.json` and renders the fixture's
//!    `messages` through it. Asserts the output is byte-identical to
//!    `expected_rendered`. Runs offline; does NOT require local artifacts.
//!    Guards against accidental regressions in the wrapper's render path.
//!
//! 2. An `#[ignore]`-by-default opt-in test that loads the real
//!    `tokenizer_config.json` from `local-artifacts/qwen2_5_0_5b_instruct/`
//!    (per `docs/artifact-preparation.md`), extracts the `chat_template`
//!    field, and asserts it is byte-identical to the inline copy in the
//!    fixture. This is what proves the inline template was lifted from the
//!    pinned upstream and not hand-edited; without it, the inline copy
//!    could silently drift from the real model.
//!
//! Run the opt-in test with:
//!
//! ```text
//! cargo test -p ocelotl-tokenizer --test qwen2_5_chat_template -- --ignored
//! ```

use std::path::PathBuf;

use ocelotl_tokenizer::{ChatMessage, ChatTemplate};
use serde::Deserialize;

const LOCAL_ARTIFACT_DIR: &str = "local-artifacts/qwen2_5_0_5b_instruct";
const PINNED_REVISION: &str = "7ae557604adf67be50417f59c2c2f167def9a775";

/// Typed view of `fixtures/chat/qwen2_5_basic_chat.json`. Only the fields
/// the tests assert on are deserialized; extra fields are tolerated.
#[derive(Debug, Deserialize)]
struct BasicChatFixture {
    fixture_version: u32,
    name: String,
    source: String,
    chat_template: String,
    messages: Vec<ChatMessage>,
    add_generation_prompt: bool,
    expected_rendered: String,
    expected_rendered_byte_length: usize,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("chat")
        .join("qwen2_5_basic_chat.json")
}

fn load_fixture() -> BasicChatFixture {
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
fn fixture_is_well_formed_and_pinned_to_qwen2_5_revision() {
    let f = load_fixture();
    assert_eq!(f.fixture_version, 1);
    assert_eq!(f.name, "qwen2_5_basic_chat");
    assert!(
        f.source.contains(PINNED_REVISION),
        "fixture source must reference pinned Qwen2.5 SHA `{PINNED_REVISION}`; got {:?}",
        f.source
    );
    assert!(
        !f.chat_template.is_empty(),
        "chat_template must be the inline copy of the upstream Jinja"
    );
    assert!(
        !f.messages.is_empty(),
        "fixture must define at least one message"
    );
    assert_eq!(
        f.expected_rendered.len(),
        f.expected_rendered_byte_length,
        "expected_rendered_byte_length must match the byte length of expected_rendered \
         exactly — guards against subtle whitespace edits to expected_rendered that \
         silently break the byte count contract"
    );
}

#[test]
fn chat_template_renders_inline_fixture_to_expected_bytes() {
    // Default-on contract: the wrapper's render output must be byte-identical
    // to expected_rendered for the fixture inputs. This is the M2.4
    // determinism criterion stated as a single assertion.
    let f = load_fixture();
    let tmpl = ChatTemplate::from_jinja(&f.chat_template)
        .expect("inline chat_template must compile via the Ocelotl wrapper");

    let rendered = tmpl
        .apply(&f.messages, f.add_generation_prompt)
        .expect("render must succeed");

    assert_eq!(
        rendered, f.expected_rendered,
        "rendered output disagrees with fixture expected_rendered — either the \
         wrapper's render path changed semantics or expected_rendered is stale; \
         re-run the #[ignore]'d semantic test to confirm the inline template \
         still matches upstream"
    );
}

#[test]
fn chat_template_render_is_byte_for_byte_deterministic_across_repeated_calls() {
    // Same fixture, three renders, must all be byte-identical. Pins
    // determinism as a property test on top of the exact-bytes assertion.
    let f = load_fixture();
    let tmpl = ChatTemplate::from_jinja(&f.chat_template).expect("template compiles");

    let first = tmpl
        .apply(&f.messages, f.add_generation_prompt)
        .expect("first render");
    let second = tmpl
        .apply(&f.messages, f.add_generation_prompt)
        .expect("second render");
    let third = tmpl
        .apply(&f.messages, f.add_generation_prompt)
        .expect("third render");

    assert_eq!(first, second, "render 1 vs 2 differed");
    assert_eq!(second, third, "render 2 vs 3 differed");
}

#[test]
#[ignore = "requires local-artifacts/qwen2_5_0_5b_instruct/tokenizer_config.json — see docs/artifact-preparation.md"]
fn inline_chat_template_matches_upstream_pinned_tokenizer_config() {
    // The semantic gate: prove the inline chat_template baked into
    // fixtures/chat/qwen2_5_basic_chat.json is exactly what the pinned
    // upstream tokenizer_config.json carries. Without this, the inline
    // copy could silently drift from the real model and the default-on
    // test would still pass (against itself) but mean nothing.
    let cfg_path = artifact_path("tokenizer_config.json");
    assert!(
        cfg_path.exists(),
        "missing artifact at {} — see docs/artifact-preparation.md for fetch \
         instructions (huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
         --revision {PINNED_REVISION} --include tokenizer_config.json ...)",
        cfg_path.display(),
    );

    let raw = std::fs::read_to_string(&cfg_path)
        .unwrap_or_else(|e| panic!("failed to read {} — {e}", cfg_path.display()));
    let cfg: serde_json::Value = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("failed to parse tokenizer_config.json — {e}"));

    let upstream_template = cfg
        .get("chat_template")
        .and_then(|v| v.as_str())
        .expect("tokenizer_config.json missing string `chat_template` field");

    let f = load_fixture();
    assert_eq!(
        f.chat_template, upstream_template,
        "fixture inline chat_template has drifted from upstream tokenizer_config.json. \
         If this is intentional (upstream re-pin under a new SHA), copy the new field \
         into the fixture and update fixtures/manifest/qwen2_5_0_5b_instruct.json + \
         docs/model-target.md in the same dedicated commit per the regeneration discipline."
    );

    // Bonus: render upstream too and assert it produces the same bytes.
    // If the templates are identical (asserted above) and the renderer is
    // deterministic (asserted by the default-on tests), this is implied —
    // but cheap to verify and the failure mode is more explicit.
    let upstream_compiled =
        ChatTemplate::from_jinja(upstream_template).expect("upstream template compiles");
    let upstream_rendered = upstream_compiled
        .apply(&f.messages, f.add_generation_prompt)
        .expect("upstream render");
    assert_eq!(
        upstream_rendered, f.expected_rendered,
        "upstream-template render disagrees with fixture expected_rendered"
    );
}
