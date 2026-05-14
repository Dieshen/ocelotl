//! Whisper-family model semantics.

pub mod audio;
pub mod config;
pub mod real;
pub mod tensors;
pub mod wer;

pub use config::{WhisperConfig, parse_whisper_config_json};
pub use real::{
    WhisperAudioEncodeTimings, WhisperDecoderState, WhisperEncodedAudio, WhisperModel,
    WhisperWeights,
};
pub use tensors::{required_whisper_tensor_names, validate_whisper_tensors};
pub use wer::{
    WerCorpusCase, WerCorpusCaseScore, WerCorpusReport, WerEditCounts, WerScore,
    normalize_transcript, score_wer_corpus, wer_edit_counts, wer_score,
};
