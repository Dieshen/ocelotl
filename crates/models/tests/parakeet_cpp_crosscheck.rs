//! Cross-check ocelotl's Parakeet against `parakeet.cpp`, an independent
//! hand-written C++/ggml implementation.
//!
//! # Why a second reference at all
//!
//! Every other Parakeet test in this tree diffs against the ONNX export. That is
//! a strong oracle for *tensors*, but it shares a lineage with the checkpoint:
//! if the export and ocelotl both misread the same convention, the diff agrees
//! and says nothing. `parakeet.cpp` is an independent reimplementation from the
//! same weights via a different converter and a different runtime, so agreement
//! with it is evidence the *interpretation* is right, not just the arithmetic.
//!
//! It is also the only comparison that speaks to speed against a real
//! implementation rather than against onnxruntime.
//!
//! # Setup (not automated: it needs a clone, a build, and a 1.4 GB download)
//!
//! ```sh
//! git clone https://github.com/mudler/parakeet.cpp && cd parakeet.cpp
//! git checkout 1da853421          # pinned, as llama.cpp is elsewhere in this repo
//! git submodule update --init --recursive --depth 1
//! cmake -B build -DCMAKE_BUILD_TYPE=Release -DPARAKEET_BUILD_SERVER=OFF -DGGML_NATIVE=ON
//! cmake --build build -j
//! # f16 only — quantization noise would contaminate a token-exact diff
//! curl -sL -o tdt-0.6b-v3-f16.gguf \
//!   https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt-0.6b-v3-f16.gguf
//! ```
//!
//! Then set:
//!   `OCELOTL_PARAKEET_CPP`       path to `parakeet-cli`
//!   `OCELOTL_PARAKEET_CPP_GGUF`  path to `tdt-0.6b-v3-f16.gguf`
//!   `OCELOTL_PARAKEET_REF_DIR`   dir holding `decode/` (ocelotl's golden decode)

use std::path::PathBuf;
use std::process::Command;

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key).ok().map(PathBuf::from)
}

fn read_i32(path: &PathBuf) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Pull `"id":N` and `"t":X` out of the CLI's `--json` token array.
///
/// A hand-rolled scan rather than a JSON dependency: this test exists to compare
/// two numbers, and adding a parser to the dev-dependency graph for one optional
/// test that most runs skip is not worth it.
fn parse_tokens(json: &str) -> (Vec<i32>, Vec<usize>) {
    let start = json
        .find("\"tokens\":")
        .expect("no tokens array in CLI output");
    let mut ids = Vec::new();
    let mut frames = Vec::new();
    for entry in json[start..].split("{\"id\":").skip(1) {
        let id: i32 = entry
            .split(&[',', '}'][..])
            .next()
            .and_then(|s| s.trim().parse().ok())
            .expect("token id");
        let t: f64 = entry
            .split("\"t\":")
            .nth(1)
            .and_then(|s| s.split(&[',', '}'][..]).next())
            .and_then(|s| s.trim().parse().ok())
            .expect("token time");
        ids.push(id);
        // 80 ms per encoder frame (8x subsampling of a 10 ms hop).
        frames.push((t / 0.08).round() as usize);
    }
    (ids, frames)
}

/// The independent implementation must agree token-for-token and frame-for-frame.
///
/// Exact, not approximate. Both implementations decode greedily from the same
/// weights, so any disagreement is a real difference in interpretation — of the
/// duration table, the blank index, the state-commit rule, or the time axis —
/// and none of those degrade gracefully.
#[test]
#[ignore = "requires a built parakeet.cpp and its f16 GGUF; see module docs"]
fn parakeet_cpp_agrees_token_and_frame_exactly() {
    let (Some(cli), Some(gguf), Some(ref_dir)) = (
        env_path("OCELOTL_PARAKEET_CPP"),
        env_path("OCELOTL_PARAKEET_CPP_GGUF"),
        env_path("OCELOTL_PARAKEET_REF_DIR"),
    ) else {
        eprintln!(
            "skipping: set OCELOTL_PARAKEET_CPP, OCELOTL_PARAKEET_CPP_GGUF, \
             OCELOTL_PARAKEET_REF_DIR"
        );
        return;
    };
    let wav = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/asr/parity_jfk.wav")
        .canonicalize()
        .expect("fixture wav — run fixtures/asr/README.md step 1 first");

    let out = Command::new(&cli)
        .args(["transcribe", "--model"])
        .arg(&gguf)
        .arg("--input")
        .arg(&wav)
        .args(["--decoder", "tdt", "--threads", "1", "--json"])
        .output()
        .expect("run parakeet-cli");
    assert!(
        out.status.success(),
        "parakeet-cli failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json = String::from_utf8_lossy(&out.stdout);
    let (cpp_tokens, cpp_frames) = parse_tokens(&json);

    let decode = ref_dir.join("decode");
    let want_tokens = read_i32(&decode.join("tokens.i32"));
    let want_frames: Vec<usize> = read_i32(&decode.join("frames.i32"))
        .into_iter()
        .map(|v| v as usize)
        .collect();

    eprintln!(
        "PARAKEET_CPP tokens={} ocelotl={}",
        cpp_tokens.len(),
        want_tokens.len()
    );
    assert!(
        !cpp_tokens.is_empty(),
        "parsed no tokens — the CLI's --json shape probably changed"
    );
    assert_eq!(
        cpp_tokens, want_tokens,
        "parakeet.cpp and ocelotl disagree on tokens"
    );
    assert_eq!(
        cpp_frames, want_frames,
        "parakeet.cpp and ocelotl disagree on frame indices despite matching tokens"
    );
}
