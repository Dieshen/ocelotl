//! Chat-template rendering boundary.
//!
//! Hugging Face models ship a Jinja2 chat template in `tokenizer_config.json`
//! (`chat_template` field). This module renders structured messages through
//! that template using `minijinja` — kept private to this module so the
//! rest of the workspace never touches `minijinja` types.
//!
//! Design choices (M2.4):
//!
//! - **Standalone, not a `Tokenizer` trait method.** The design doc
//!   (`docs/design/tokenizer.md` § Chat Templates) says "Chat templates
//!   are model behavior. They should be explicit inputs to the tokenizer
//!   layer or normalized by the loader when present in the artifact." A
//!   separate `ChatTemplate` value composes with a `Tokenizer` rather
//!   than burdening the `Tokenizer` trait with an Optional capability
//!   that most callers don't need. Keeps the trait minimal (M2.2's
//!   discipline) and keeps the chat-template type usable independently
//!   (e.g. when someone wants the rendered string but is encoding via
//!   their own tokenizer for a test).
//!
//! - **MiniJinja over `tokenizers`-builtin or hand-rolled.** Verified
//!   2026-05-03 that `tokenizers = "0.23.1"` exposes no chat-template
//!   API (rg `chat_template|apply_chat|ChatTemplate` in the crate
//!   source returns zero hits). Hand-rolling Jinja2 for the Qwen2.5
//!   template is a rabbit hole — the template uses `for`, `if`,
//!   `loop.first`, `loop.index0`, `loop.last`, `tojson`, `is defined`,
//!   `set`, and nested branches. MiniJinja covers all of this and is
//!   sandboxed by default (no filesystem access, no arbitrary code
//!   execution). One new foreign-crate boundary, applied via the
//!   external-crate-boundary pattern.

use std::collections::BTreeMap;

use minijinja::{Environment, ErrorKind, context};
use ocelotl_core::{OcelotlError, Result, TokenizerError, UnsupportedError};
use serde::{Deserialize, Serialize};

/// A structured chat message as understood by the chat-template layer.
///
/// `role` values are model-defined ("system", "user", "assistant", "tool").
/// The chat template is responsible for rejecting unknown roles per its
/// own logic; we don't gate them here because different model families
/// accept different role sets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// A compiled chat template ready to render messages into a model-prompt
/// string.
///
/// Wraps a `minijinja::Environment` privately. Construction parses the
/// Jinja source once; subsequent `apply` calls are pure renders. Errors
/// from minijinja are translated to typed `OcelotlError` at every public
/// boundary so callers never see `minijinja::Error`.
pub struct ChatTemplate {
    // Owns the template source string so the Environment can borrow from
    // a stable address; minijinja's `add_template` requires a borrowed
    // `&str` so we pin the source on the heap.
    env: Environment<'static>,
    // Kept around so callers can re-inspect the template they compiled
    // (useful for fixture round-trip tests). Not exposed publicly.
    _source: Box<str>,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Don't leak the inner Environment / source through Debug — the
        // source can be several KB and is rarely useful in error logs.
        f.debug_struct("ChatTemplate").finish_non_exhaustive()
    }
}

const TEMPLATE_NAME: &str = "chat";

impl ChatTemplate {
    /// Compile a chat template from a Jinja2 source string. Sandboxes
    /// the template (no `include`, no `import`, no filesystem) by
    /// virtue of never registering a loader on the underlying
    /// `Environment`.
    ///
    /// Returns `OcelotlError::Tokenizer` with the parse error preserved
    /// as the source if the template is syntactically invalid.
    pub fn from_jinja(source: &str) -> Result<Self> {
        // Heap-pin the source so the Environment can hold a `'static`
        // borrow into it.
        let source: Box<str> = Box::from(source);
        // SAFETY-ish: we leak the Box to obtain a `'static` reference
        // that the Environment can hold. The Box is then reconstituted
        // and stored alongside the Environment so the lifetime balance
        // works out — the leaked reference and the reconstituted Box
        // refer to the same allocation, which lives as long as `Self`.
        // We never expose the leaked reference.
        let source_ptr: *const str = Box::into_raw(source);
        // SAFETY: source_ptr is non-null and points at a live allocation
        // we just heap-allocated. We immediately reconstruct the Box
        // below to keep ownership; the &'static borrow we hand to the
        // Environment is valid for as long as the Box is alive (the
        // Box is stored on the same struct).
        let source_ref: &'static str = unsafe { &*source_ptr };
        let owned: Box<str> = unsafe { Box::from_raw(source_ptr as *mut str) };

        let mut env = Environment::new();
        // Lenient undefined matches Jinja2's default and is what
        // upstream Hugging Face chat templates are authored against.
        // The Qwen2.5 template (and many others) does
        // `{% if message.tool_calls %}` against messages that don't
        // carry a `tool_calls` field — strict mode would error here
        // even though the upstream behavior is "treat undefined as
        // falsy." We therefore deliberately accept the trade-off that
        // template typos may render to empty strings; the determinism
        // tests catch render *changes* across versions, and the
        // fixture-pinned expected output catches accidental drift in
        // semantics.
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);
        env.add_template(TEMPLATE_NAME, source_ref)
            .map_err(translate_compile_error)?;

        Ok(Self {
            env,
            _source: owned,
        })
    }

    /// Render the template against `messages` plus the standard Hugging
    /// Face chat-template variables.
    ///
    /// `add_generation_prompt`: most chat templates emit a trailing
    /// `<|im_start|>assistant\n` (or family equivalent) when this is
    /// `true`, signalling the model where to start generating.
    ///
    /// Errors (all mapped to typed `OcelotlError`):
    ///
    /// - Unsupported feature in the template (e.g. it tried to
    ///   `{% include %}` another file, or used a filter we don't
    ///   provide) → `OcelotlError::Unsupported` with the requested
    ///   feature recorded.
    /// - Generic render failure (unknown variable in strict mode,
    ///   bad type, etc.) → `OcelotlError::Tokenizer` with the
    ///   minijinja error preserved as `source`.
    pub fn apply(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        // `tools` is referenced by the Qwen2.5 template via `{%- if tools %}`.
        // Strict undefined would error if we omit it, so we pass an empty
        // list — matches the Hugging Face convention where tools defaults
        // to `[]` when not provided.
        let tools: Vec<BTreeMap<String, String>> = Vec::new();

        let template = self.env.get_template(TEMPLATE_NAME).map_err(|e| {
            OcelotlError::Tokenizer(TokenizerError {
                message: format!("internal: chat template not registered: {e}"),
                source: Some(Box::new(e)),
            })
        })?;

        template
            .render(context! {
                messages => messages,
                add_generation_prompt => add_generation_prompt,
                tools => tools,
            })
            .map_err(translate_render_error)
    }
}

/// Translate a parse-time `minijinja::Error` into a typed Ocelotl error.
///
/// MiniJinja rejects unsupported statements (`{% include %}`, `{% import %}`,
/// `{% extends %}` when no loader is registered) with `SyntaxError` carrying
/// "unknown statement <name>" in the detail. We surface those as
/// `Unsupported` so the caller can react with "this model template uses a
/// Jinja feature outside the Ocelotl chat-template subset" — distinct from
/// genuine syntactic mistakes in the template, which remain `Tokenizer` to
/// preserve the existing parse-error semantics from M2.2.
fn translate_compile_error(e: minijinja::Error) -> OcelotlError {
    let detail = format!("{e}");
    let is_unknown_statement =
        matches!(e.kind(), ErrorKind::SyntaxError) && detail.contains("unknown statement");

    if is_unknown_statement {
        OcelotlError::Unsupported(UnsupportedError {
            feature: "chat_template_jinja_feature".to_string(),
            requested: Some(detail),
            supported: SUPPORTED_JINJA_FEATURES
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        })
    } else {
        OcelotlError::Tokenizer(TokenizerError {
            message: format!("failed to parse chat template: {e}"),
            source: Some(Box::new(e)),
        })
    }
}

/// Translate a render-time `minijinja::Error` into a typed Ocelotl error.
///
/// `InvalidOperation`, `BadInclude`, and `UnknownFunction` mean the
/// template tried to do something the renderer does not support at
/// render time (e.g. an undefined custom filter). These map to
/// `Unsupported`. Everything else (undefined variable in strict mode,
/// type mismatch, etc.) is a render-time tokenizer error and maps to
/// `Tokenizer` with the underlying error preserved as `source`.
fn translate_render_error(e: minijinja::Error) -> OcelotlError {
    match e.kind() {
        ErrorKind::InvalidOperation
        | ErrorKind::BadInclude
        | ErrorKind::UnknownFunction
        | ErrorKind::UnknownFilter
        | ErrorKind::UnknownTest
        | ErrorKind::UnknownMethod => OcelotlError::Unsupported(UnsupportedError {
            feature: "chat_template_jinja_feature".to_string(),
            requested: Some(format!("{:?}: {}", e.kind(), e)),
            supported: SUPPORTED_JINJA_FEATURES
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        }),
        _ => OcelotlError::Tokenizer(TokenizerError {
            message: format!("chat template render failed: {e}"),
            source: Some(Box::new(e)),
        }),
    }
}

/// Names of Jinja features the Ocelotl chat-template subset exposes, for
/// inclusion in `Unsupported` errors. Not load-bearing for execution; only
/// surfaces in error diagnostics so users can read what they DO have.
const SUPPORTED_JINJA_FEATURES: &[&str] = &[
    "for-loops with loop variables (loop.first/last/index0)",
    "if/elif/else conditionals",
    "set assignments",
    "string concatenation with +",
    "tojson filter",
    "is defined / is not defined tests",
    "method calls on context objects",
];

// ---------------------------------------------------------------------------
// Tests — unit-level behavior of the renderer itself. The fixture-driven
// integration test (inline-template + #[ignore]'d real-template pair) lives
// in `crates/tokenizer/tests/qwen2_5_chat_template.rs`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_renders_trivial_template_deterministically() {
        // A single render of a tiny template with the message list — the
        // smallest end-to-end proof that messages flow through the
        // template context and out as a string.
        let tmpl = ChatTemplate::from_jinja(
            "{%- for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}",
        )
        .expect("trivial template must compile");

        let out = tmpl
            .apply(
                &[
                    ChatMessage {
                        role: "user".to_string(),
                        content: "hi".to_string(),
                    },
                    ChatMessage {
                        role: "assistant".to_string(),
                        content: "hello".to_string(),
                    },
                ],
                false,
            )
            .expect("trivial render must succeed");

        assert_eq!(out, "user:hi;assistant:hello;");
    }

    #[test]
    fn apply_is_deterministic_across_repeated_calls() {
        // Same template, same inputs, must produce byte-identical output
        // across N calls. This is the "deterministic" half of M2.4's
        // done-when condition stated as a property test.
        let tmpl = ChatTemplate::from_jinja(
            "{%- for m in messages %}[{{ loop.index0 }}]{{ m.role }}:{{ m.content }}\n{% endfor %}{% if add_generation_prompt %}>>>{% endif %}",
        )
        .expect("template must compile");

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "you are helpful".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            },
        ];

        let first = tmpl
            .apply(&messages, true)
            .expect("first render must succeed");
        for i in 1..5 {
            let nth = tmpl
                .apply(&messages, true)
                .expect("subsequent render must succeed");
            assert_eq!(
                first, nth,
                "render {i} disagreed with render 0 — chat-template renderer is not deterministic"
            );
        }
    }

    #[test]
    fn apply_treats_undefined_as_lenient_to_match_upstream_jinja2_semantics() {
        // Hugging Face chat templates (Qwen, Llama, Mistral, ...) are
        // authored against Jinja2's default lenient-undefined behavior:
        // a `{% if message.tool_calls %}` against a message lacking that
        // field must evaluate falsy, not error. We deliberately mirror
        // that here. This test pins the choice — flipping to strict
        // mode would break upstream-authored templates.
        let tmpl = ChatTemplate::from_jinja("{%- if missing_var %}YES{% else %}NO{% endif %}")
            .expect("template must compile");

        let out = tmpl
            .apply(&[], false)
            .expect("undefined variable in conditional must be lenient, not error");

        assert_eq!(
            out, "NO",
            "lenient undefined semantics: missing_var evaluates falsy in `if`"
        );
    }

    #[test]
    fn apply_surfaces_unknown_filter_via_typed_unsupported_error() {
        // An unknown filter is a render-time failure that lenient
        // undefined cannot mask: minijinja raises `UnknownFilter`
        // (a sub-kind of `InvalidOperation`) rather than silently
        // producing empty output. This proves at least one render-time
        // failure path produces a typed error and reaches the
        // `translate_render_error` Unsupported branch.
        let tmpl =
            ChatTemplate::from_jinja(r#"{{ "hi" | totally_not_a_real_filter }}"#).expect("compile");

        let err = tmpl
            .apply(&[], false)
            .expect_err("unknown filter must produce a typed Unsupported error");

        match err {
            OcelotlError::Unsupported(u) => {
                assert_eq!(u.feature, "chat_template_jinja_feature");
                let requested = u.requested.expect("requested feature description present");
                assert!(
                    requested.contains("totally_not_a_real_filter") || requested.contains("filter"),
                    "expected unknown-filter detail in error, got {requested:?}"
                );
            }
            other => panic!("expected OcelotlError::Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn from_jinja_rejects_unsupported_statement_with_typed_unsupported_error() {
        // `{% include %}` is a Jinja statement minijinja-without-loader
        // refuses to recognize at parse time (returns `SyntaxError:
        // unknown statement include`). We surface this as
        // `OcelotlError::Unsupported` so a caller building tooling
        // around chat templates can distinguish "this model uses an
        // include and we don't support that yet" from "this model's
        // template has a typo." The same path applies to `{% import %}`
        // and `{% extends %}` — all of which would require us to register
        // a loader, which we deliberately do not.
        let err = ChatTemplate::from_jinja(r#"{% include "other.jinja" %}"#)
            .expect_err("include must be rejected at parse time as Unsupported");

        match err {
            OcelotlError::Unsupported(u) => {
                assert_eq!(u.feature, "chat_template_jinja_feature");
                let requested = u.requested.expect("requested feature must be populated");
                assert!(
                    requested.contains("unknown statement include"),
                    "expected `unknown statement include` in requested, got {requested:?}"
                );
                assert!(
                    !u.supported.is_empty(),
                    "supported list must enumerate the Ocelotl chat-template subset"
                );
            }
            other => panic!("expected OcelotlError::Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn from_jinja_distinguishes_genuine_syntax_error_from_unsupported_feature() {
        // A genuine syntactic mistake (mismatched `{% if %}` with no
        // `{% endif %}`) must remain `Tokenizer`, not `Unsupported`.
        // This guards the translate_compile_error split: we don't want
        // every parse failure to look like an unsupported feature, only
        // the "unknown statement" subset.
        let err = ChatTemplate::from_jinja("{% if foo %}oops")
            .expect_err("genuine syntax error must produce a typed Tokenizer error");

        match err {
            OcelotlError::Tokenizer(t) => {
                assert!(
                    format!("{t}").contains("failed to parse chat template"),
                    "expected parse-error wording, got {t:?}"
                );
            }
            other => panic!("expected OcelotlError::Tokenizer, got {other:?}"),
        }
    }
}
