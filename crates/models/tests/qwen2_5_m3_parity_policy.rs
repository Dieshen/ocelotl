//! M3.9 parity-policy tripwire.
//!
//! This test does not recompute model outputs. The output parity tests do that
//! directly. This test makes the policy surface auditable: if the M3 prefill
//! fixture tolerance or decode pin changes, `docs/validation/parity.md` must
//! change in the same patch.

const PARITY_DOC: &str = "../../docs/validation/parity.md";
const PREFILL_FIXTURE: &str = "../../fixtures/logits/qwen2_5_tiny_synthetic_prefill.json";

#[test]
fn parity_doc_names_m3_sources_and_tolerances() {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR for tests");
    let manifest_dir = std::path::Path::new(&manifest_dir);

    let parity_doc_path = manifest_dir.join(PARITY_DOC);
    let parity_doc = std::fs::read_to_string(&parity_doc_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", parity_doc_path.display()));

    let fixture_path = manifest_dir.join(PREFILL_FIXTURE);
    let fixture_json = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", fixture_path.display()));
    let fixture: serde_json::Value = serde_json::from_str(&fixture_json)
        .unwrap_or_else(|e| panic!("parse {}: {e}", fixture_path.display()));
    let tolerance = fixture
        .get("tolerance")
        .and_then(serde_json::Value::as_f64)
        .expect("prefill fixture must carry numeric tolerance");

    assert!(
        (tolerance - 0.0001).abs() < f64::EPSILON,
        "fixture tolerance changed from M3's documented 1e-4; update \
         docs/validation/parity.md and this tripwire in the same patch"
    );

    let required_snippets = [
        "fixtures/logits/qwen2_5_tiny_synthetic_prefill.json",
        "crates/models/tests/qwen2_5_tiny_synthetic_prefill.rs",
        "`1e-4`",
        "crates/runtime/tests/qwen2_5_tiny_synthetic_decode.rs",
        "`TokenId(16)`",
        "exact token equality",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "`1e-3`",
    ];

    for snippet in required_snippets {
        assert!(
            parity_doc.contains(snippet),
            "docs/validation/parity.md must mention {snippet:?} for M3 parity"
        );
    }
}
