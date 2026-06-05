use std::collections::HashSet;

use ocelotl_core::{OcelotlError, Result, TokenId, TokenizerError};
use tokenizers::{
    AddedToken,
    models::bpe::{BPE, Vocab},
    pre_tokenizers::byte_level::ByteLevel,
};

use crate::Tokenizer;

/// Special-token IDs carried by GGUF tokenizer metadata.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GgufSpecialTokens {
    pub bos_token_id: Option<TokenId>,
    pub eos_token_id: Option<TokenId>,
    pub unknown_token_id: Option<TokenId>,
    pub padding_token_id: Option<TokenId>,
    pub mask_token_id: Option<TokenId>,
}

/// GGUF/llama.cpp token type values from `tokenizer.ggml.token_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTokenType {
    Undefined,
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl GgufTokenType {
    pub fn from_gguf_i32(raw: i32) -> Result<Self> {
        match raw {
            0 => Ok(Self::Undefined),
            1 => Ok(Self::Normal),
            2 => Ok(Self::Unknown),
            3 => Ok(Self::Control),
            4 => Ok(Self::UserDefined),
            5 => Ok(Self::Unused),
            6 => Ok(Self::Byte),
            _ => Err(tokenizer_error(format!(
                "unknown GGUF tokenizer token_type value {raw}"
            ))),
        }
    }
}

/// Tokenizer-owned projection of GGUF byte-level BPE metadata.
///
/// `ocelotl-loader` owns GGUF file parsing. This spec intentionally contains
/// only the in-memory parts needed to construct tokenizer behavior so the
/// tokenizer crate does not depend on loader-owned artifact types.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufBpeTokenizerSpec {
    pub model: Option<String>,
    pub tokens: Vec<String>,
    pub token_types: Vec<GgufTokenType>,
    pub merges: Vec<String>,
    pub special_tokens: GgufSpecialTokens,
    pub add_bos_token: bool,
    pub add_space_prefix: bool,
    pub byte_fallback: bool,
}

impl GgufBpeTokenizerSpec {
    pub fn new(tokens: Vec<String>, merges: Vec<String>) -> Self {
        Self {
            model: None,
            tokens,
            token_types: vec![],
            merges,
            special_tokens: GgufSpecialTokens::default(),
            add_bos_token: false,
            add_space_prefix: false,
            byte_fallback: false,
        }
    }
}

/// Concrete tokenizer backed by GGUF embedded byte-level BPE metadata.
pub struct GgufBpeTokenizer {
    inner: tokenizers::Tokenizer,
    special_tokens: GgufSpecialTokens,
    add_bos_token: bool,
    add_space_prefix: bool,
}

impl std::fmt::Debug for GgufBpeTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufBpeTokenizer")
            .field("special_tokens", &self.special_tokens)
            .field("add_bos_token", &self.add_bos_token)
            .field("add_space_prefix", &self.add_space_prefix)
            .finish_non_exhaustive()
    }
}

impl GgufBpeTokenizer {
    pub fn from_spec(spec: GgufBpeTokenizerSpec) -> Result<Self> {
        validate_spec(&spec)?;

        let unknown_token = spec
            .special_tokens
            .unknown_token_id
            .map(|id| token_for_id(&spec.tokens, id, "unknown_token_id"))
            .transpose()?;
        let vocab = vocab_from_tokens(&spec.tokens)?;
        let merges = parse_merges(&spec.merges)?;

        let mut builder = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .byte_fallback(spec.byte_fallback);
        if let Some(unknown_token) = unknown_token {
            builder = builder.unk_token(unknown_token);
        }

        let model = builder.build().map_err(|source| {
            tokenizer_error_with_source("failed to build GGUF BPE tokenizer model", source)
        })?;
        let byte_level = ByteLevel::default().add_prefix_space(spec.add_space_prefix);
        let mut inner = tokenizers::Tokenizer::new(model);
        inner.with_pre_tokenizer(Some(byte_level));
        inner.with_post_processor(Some(ByteLevel::default()));
        inner.with_decoder(Some(ByteLevel::default()));

        let added =
            added_tokens_from_metadata(&spec.tokens, &spec.token_types, spec.special_tokens)?;
        if !added.normal_tokens.is_empty() {
            inner.add_tokens(added.normal_tokens).map_err(|source| {
                tokenizer_error_with_source("failed to register GGUF added tokens", source)
            })?;
        }
        let added_specials = added.special_tokens;
        if !added_specials.is_empty() {
            inner.add_special_tokens(added_specials).map_err(|source| {
                tokenizer_error_with_source("failed to register GGUF special tokens", source)
            })?;
        }

        Ok(Self {
            inner,
            special_tokens: spec.special_tokens,
            add_bos_token: spec.add_bos_token,
            add_space_prefix: spec.add_space_prefix,
        })
    }

    pub fn special_tokens(&self) -> GgufSpecialTokens {
        self.special_tokens
    }

    pub fn add_bos_token(&self) -> bool {
        self.add_bos_token
    }

    pub fn add_space_prefix(&self) -> bool {
        self.add_space_prefix
    }

    /// Encode text and prepend BOS only when the GGUF metadata requested it.
    ///
    /// The plain `Tokenizer::encode` implementation does not add BOS. That
    /// keeps it aligned with `JsonTokenizer` and avoids double-BOS when a
    /// Gemma4 chat template already renders `bos_token`.
    pub fn encode_with_configured_bos(&self, text: &str) -> Result<Vec<TokenId>> {
        let mut ids = self.encode(text)?;
        if self.add_bos_token {
            let bos = self.special_tokens.bos_token_id.ok_or_else(|| {
                tokenizer_error(
                    "GGUF tokenizer requested add_bos_token but has no bos_token_id metadata",
                )
            })?;
            ids.insert(0, bos);
        }
        Ok(ids)
    }
}

impl Tokenizer for GgufBpeTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|source| tokenizer_error_with_source("GGUF BPE encode failed", source))?;
        Ok(encoding.get_ids().iter().copied().map(TokenId).collect())
    }

    fn decode(&self, tokens: &[TokenId]) -> Result<String> {
        let raw: Vec<u32> = tokens.iter().map(|t| t.0).collect();
        self.inner
            .decode(&raw, true)
            .map_err(|source| tokenizer_error_with_source("GGUF BPE decode failed", source))
    }
}

fn validate_spec(spec: &GgufBpeTokenizerSpec) -> Result<()> {
    if spec.tokens.is_empty() {
        return Err(tokenizer_error(
            "GGUF BPE tokenizer requires at least one token",
        ));
    }
    if spec.tokens.len() > u32::MAX as usize {
        return Err(tokenizer_error(
            "GGUF BPE tokenizer token count exceeds u32 token-id range",
        ));
    }
    if !spec.token_types.is_empty() && spec.token_types.len() != spec.tokens.len() {
        return Err(tokenizer_error(format!(
            "GGUF BPE tokenizer token_type length {} does not match token length {}",
            spec.token_types.len(),
            spec.tokens.len()
        )));
    }
    Ok(())
}

fn vocab_from_tokens(tokens: &[String]) -> Result<Vocab> {
    let mut vocab = Vocab::with_capacity(tokens.len());
    for (index, token) in tokens.iter().enumerate() {
        if vocab.insert(token.clone(), index as u32).is_some() {
            return Err(tokenizer_error(format!(
                "GGUF BPE tokenizer contains duplicate token {token:?}"
            )));
        }
    }
    Ok(vocab)
}

fn parse_merges(raw_merges: &[String]) -> Result<Vec<(String, String)>> {
    raw_merges
        .iter()
        .enumerate()
        .map(|(index, raw)| {
            let (left, right) = raw.split_once(' ').ok_or_else(|| {
                tokenizer_error(format!(
                    "GGUF BPE merge at index {index} is not two tokens separated by a space"
                ))
            })?;
            if left.is_empty() || right.is_empty() {
                return Err(tokenizer_error(format!(
                    "GGUF BPE merge at index {index} contains an empty token"
                )));
            }
            Ok((left.to_string(), right.to_string()))
        })
        .collect()
}

struct AddedTokensFromMetadata {
    normal_tokens: Vec<AddedToken>,
    special_tokens: Vec<AddedToken>,
}

fn added_tokens_from_metadata(
    tokens: &[String],
    token_types: &[GgufTokenType],
    special_tokens: GgufSpecialTokens,
) -> Result<AddedTokensFromMetadata> {
    let mut seen = HashSet::new();
    let mut added_special = Vec::new();
    for (label, id) in [
        ("bos_token_id", special_tokens.bos_token_id),
        ("eos_token_id", special_tokens.eos_token_id),
        ("unknown_token_id", special_tokens.unknown_token_id),
        ("padding_token_id", special_tokens.padding_token_id),
        ("mask_token_id", special_tokens.mask_token_id),
    ] {
        if let Some(id) = id {
            let token = token_for_id(tokens, id, label)?;
            if seen.insert(id) {
                added_special.push(AddedToken::from(token, true));
            }
        }
    }

    let mut added_normal = Vec::new();
    for (index, token_type) in token_types.iter().enumerate() {
        let id = TokenId(index as u32);
        if seen.contains(&id) {
            continue;
        }
        match token_type {
            GgufTokenType::Unknown | GgufTokenType::Control => {
                seen.insert(id);
                added_special.push(AddedToken::from(tokens[index].clone(), true));
            }
            GgufTokenType::UserDefined => {
                seen.insert(id);
                added_normal.push(AddedToken::from(tokens[index].clone(), false));
            }
            GgufTokenType::Undefined
            | GgufTokenType::Normal
            | GgufTokenType::Unused
            | GgufTokenType::Byte => {}
        }
    }

    Ok(AddedTokensFromMetadata {
        normal_tokens: added_normal,
        special_tokens: added_special,
    })
}

fn token_for_id(tokens: &[String], id: TokenId, label: &str) -> Result<String> {
    tokens
        .get(id.0 as usize)
        .cloned()
        .ok_or_else(|| tokenizer_error(format!("GGUF BPE {label}={} is outside vocab", id.0)))
}

fn tokenizer_error(message: impl Into<String>) -> OcelotlError {
    OcelotlError::Tokenizer(TokenizerError {
        message: message.into(),
        source: None,
    })
}

fn tokenizer_error_with_source(
    message: impl Into<String>,
    source: Box<dyn std::error::Error + Send + Sync>,
) -> OcelotlError {
    OcelotlError::Tokenizer(TokenizerError {
        message: message.into(),
        source: Some(source),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_spec(add_space_prefix: bool, add_bos_token: bool) -> GgufBpeTokenizerSpec {
        let mut spec = GgufBpeTokenizerSpec::new(
            [
                "<pad>", "</s>", "<s>", "<unk>", "<mask>", "H", "e", "l", "o", "He", "Hel", "Hell",
                "Hello", "Ġ", "ĠHello",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            ["H e", "He l", "Hel l", "Hell o", "Ġ Hello"]
                .into_iter()
                .map(String::from)
                .collect(),
        );
        spec.model = Some("gemma4".to_string());
        spec.special_tokens = GgufSpecialTokens {
            bos_token_id: Some(TokenId(2)),
            eos_token_id: Some(TokenId(1)),
            unknown_token_id: Some(TokenId(3)),
            padding_token_id: Some(TokenId(0)),
            mask_token_id: Some(TokenId(4)),
        };
        spec.add_space_prefix = add_space_prefix;
        spec.add_bos_token = add_bos_token;
        spec
    }

    #[test]
    fn gguf_bpe_tokenizer_encodes_byte_level_merges_without_loader_dependency() {
        let tokenizer =
            GgufBpeTokenizer::from_spec(tiny_spec(false, false)).expect("tiny spec must build");

        assert_eq!(tokenizer.encode("Hello").unwrap(), vec![TokenId(12)]);
        assert_eq!(tokenizer.decode(&[TokenId(12)]).unwrap(), "Hello");
        assert!(!tokenizer.add_space_prefix());
        assert!(!tokenizer.add_bos_token());
    }

    #[test]
    fn gguf_bpe_tokenizer_honors_add_space_prefix_metadata() {
        let tokenizer =
            GgufBpeTokenizer::from_spec(tiny_spec(true, false)).expect("tiny spec must build");

        assert_eq!(tokenizer.encode("Hello").unwrap(), vec![TokenId(14)]);
        assert_eq!(tokenizer.decode(&[TokenId(14)]).unwrap(), " Hello");
        assert!(tokenizer.add_space_prefix());
    }

    #[test]
    fn gguf_bpe_tokenizer_tracks_bos_without_adding_it_to_plain_encode() {
        let tokenizer =
            GgufBpeTokenizer::from_spec(tiny_spec(false, true)).expect("tiny spec must build");

        assert_eq!(tokenizer.encode("Hello").unwrap(), vec![TokenId(12)]);
        assert_eq!(
            tokenizer.encode_with_configured_bos("Hello").unwrap(),
            vec![TokenId(2), TokenId(12)]
        );
        assert_eq!(tokenizer.special_tokens().bos_token_id, Some(TokenId(2)));
    }

    #[test]
    fn gguf_bpe_tokenizer_uses_token_types_for_control_decode_skipping() {
        let mut spec = tiny_spec(false, false);
        spec.tokens.push("<control>".to_string());
        spec.token_types = vec![GgufTokenType::Normal; spec.tokens.len()];
        spec.token_types[15] = GgufTokenType::Control;

        let tokenizer = GgufBpeTokenizer::from_spec(spec).expect("control spec must build");

        assert_eq!(
            tokenizer
                .decode(&[TokenId(15), TokenId(12), TokenId(15)])
                .unwrap(),
            "Hello"
        );
    }

    #[test]
    fn gguf_bpe_tokenizer_uses_token_types_for_user_defined_literals() {
        let mut spec = tiny_spec(false, false);
        spec.tokens.push("<tool_call>".to_string());
        spec.token_types = vec![GgufTokenType::Normal; spec.tokens.len()];
        spec.token_types[15] = GgufTokenType::UserDefined;

        let tokenizer = GgufBpeTokenizer::from_spec(spec).expect("user-defined spec must build");

        assert_eq!(tokenizer.encode("<tool_call>").unwrap(), vec![TokenId(15)]);
        assert_eq!(tokenizer.decode(&[TokenId(15)]).unwrap(), "<tool_call>");
    }

    #[test]
    fn gguf_bpe_tokenizer_skips_registered_special_tokens_on_decode() {
        let tokenizer =
            GgufBpeTokenizer::from_spec(tiny_spec(false, false)).expect("tiny spec must build");

        assert_eq!(
            tokenizer
                .decode(&[TokenId(2), TokenId(12), TokenId(1)])
                .unwrap(),
            "Hello"
        );
    }

    #[test]
    fn gguf_bpe_tokenizer_rejects_mismatched_token_type_lengths() {
        let mut spec = tiny_spec(false, false);
        spec.token_types = vec![GgufTokenType::Normal];

        let err = GgufBpeTokenizer::from_spec(spec)
            .expect_err("token_type length mismatch must be rejected");

        match err {
            OcelotlError::Tokenizer(tokenizer) => assert!(
                tokenizer.message.contains("token_type length"),
                "unexpected tokenizer error: {tokenizer}"
            ),
            other => panic!("expected tokenizer error, got {other:?}"),
        }
    }

    #[test]
    fn gguf_token_type_rejects_unknown_raw_values() {
        let err = GgufTokenType::from_gguf_i32(99).expect_err("unknown type must reject");

        match err {
            OcelotlError::Tokenizer(tokenizer) => assert!(
                tokenizer.message.contains("token_type value 99"),
                "unexpected tokenizer error: {tokenizer}"
            ),
            other => panic!("expected tokenizer error, got {other:?}"),
        }
    }

    #[test]
    fn gguf_bpe_tokenizer_rejects_duplicate_tokens_before_hashmap_overwrite() {
        let spec = GgufBpeTokenizerSpec::new(
            vec!["<unk>".to_string(), "H".to_string(), "H".to_string()],
            vec![],
        );

        let err =
            GgufBpeTokenizer::from_spec(spec).expect_err("duplicate vocab token must be rejected");

        match err {
            OcelotlError::Tokenizer(tokenizer) => assert!(
                tokenizer.message.contains("duplicate token"),
                "unexpected tokenizer error: {tokenizer}"
            ),
            other => panic!("expected tokenizer error, got {other:?}"),
        }
    }

    #[test]
    fn gguf_bpe_tokenizer_rejects_malformed_merge_strings() {
        let spec = GgufBpeTokenizerSpec::new(
            vec!["<unk>".to_string(), "H".to_string(), "e".to_string()],
            vec!["He".to_string()],
        );

        let err = GgufBpeTokenizer::from_spec(spec).expect_err("malformed merge must be rejected");

        match err {
            OcelotlError::Tokenizer(tokenizer) => assert!(
                tokenizer.message.contains("not two tokens"),
                "unexpected tokenizer error: {tokenizer}"
            ),
            other => panic!("expected tokenizer error, got {other:?}"),
        }
    }

    #[test]
    fn gguf_bpe_tokenizer_rejects_special_token_id_outside_vocab() {
        let mut spec = GgufBpeTokenizerSpec::new(vec!["<unk>".to_string()], vec![]);
        spec.special_tokens.unknown_token_id = Some(TokenId(9));

        let err = GgufBpeTokenizer::from_spec(spec).expect_err("bad special ID must be rejected");

        match err {
            OcelotlError::Tokenizer(tokenizer) => assert!(
                tokenizer
                    .message
                    .contains("unknown_token_id=9 is outside vocab"),
                "unexpected tokenizer error: {tokenizer}"
            ),
            other => panic!("expected tokenizer error, got {other:?}"),
        }
    }
}
