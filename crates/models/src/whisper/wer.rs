//! Transcript-only word error rate scoring for Whisper ASR harnesses.

use ocelotl_core::{InvalidRequestError, OcelotlError, Result};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WerEditCounts {
    pub substitutions: usize,
    pub insertions: usize,
    pub deletions: usize,
    pub reference_words: usize,
}

impl WerEditCounts {
    pub fn errors(self) -> usize {
        self.substitutions + self.insertions + self.deletions
    }

    pub fn word_error_rate(self) -> Option<f32> {
        if self.reference_words == 0 {
            None
        } else {
            Some(self.errors() as f32 / self.reference_words as f32)
        }
    }

    fn add_assign(&mut self, other: Self) {
        self.substitutions += other.substitutions;
        self.insertions += other.insertions;
        self.deletions += other.deletions;
        self.reference_words += other.reference_words;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WerScore {
    pub counts: WerEditCounts,
    pub wer: f32,
}

impl WerScore {
    fn from_counts(counts: WerEditCounts) -> Self {
        let wer = counts
            .word_error_rate()
            .expect("WER score construction requires at least one reference word");
        Self { counts, wer }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WerCorpusCase<'a> {
    pub id: &'a str,
    pub expected_transcript: &'a str,
    pub recognized_transcript: &'a str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WerCorpusCaseScore {
    pub id: String,
    pub score: WerScore,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WerCorpusReport {
    pub cases: Vec<WerCorpusCaseScore>,
    pub aggregate: WerScore,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct EditPath {
    substitutions: usize,
    insertions: usize,
    deletions: usize,
}

impl EditPath {
    fn errors(self) -> usize {
        self.substitutions + self.insertions + self.deletions
    }

    fn with_substitution(mut self) -> Self {
        self.substitutions += 1;
        self
    }

    fn with_insertion(mut self) -> Self {
        self.insertions += 1;
        self
    }

    fn with_deletion(mut self) -> Self {
        self.deletions += 1;
        self
    }

    fn into_counts(self, reference_words: usize) -> WerEditCounts {
        WerEditCounts {
            substitutions: self.substitutions,
            insertions: self.insertions,
            deletions: self.deletions,
            reference_words,
        }
    }
}

pub fn normalize_transcript(transcript: &str) -> String {
    let mut normalized = String::new();
    let mut pending_space = false;

    for ch in transcript.chars() {
        for lower in ch.to_lowercase() {
            if lower.is_alphanumeric() {
                if pending_space && !normalized.is_empty() {
                    normalized.push(' ');
                }
                normalized.push(lower);
                pending_space = false;
            } else {
                pending_space = true;
            }
        }
    }

    normalized
}

pub fn wer_edit_counts(
    expected_transcript: &str,
    recognized_transcript: &str,
) -> Result<WerEditCounts> {
    let reference_words = normalized_words(expected_transcript);
    if reference_words.is_empty() {
        return Err(invalid_request(
            "reference_transcript",
            "WER reference transcript must contain at least one word after normalization",
        ));
    }

    let hypothesis_words = normalized_words(recognized_transcript);
    Ok(edit_path(&reference_words, &hypothesis_words).into_counts(reference_words.len()))
}

pub fn wer_score(expected_transcript: &str, recognized_transcript: &str) -> Result<WerScore> {
    wer_edit_counts(expected_transcript, recognized_transcript).map(WerScore::from_counts)
}

pub fn score_wer_corpus(cases: &[WerCorpusCase<'_>]) -> Result<WerCorpusReport> {
    if cases.is_empty() {
        return Err(invalid_request(
            "wer_corpus.cases",
            "WER corpus must contain at least one case",
        ));
    }

    let mut case_scores = Vec::with_capacity(cases.len());
    let mut aggregate_counts = WerEditCounts::default();

    for case in cases {
        if case.id.trim().is_empty() {
            return Err(invalid_request(
                "wer_corpus.cases[].id",
                "WER corpus case id must be non-empty",
            ));
        }

        let score = wer_score(case.expected_transcript, case.recognized_transcript)?;
        aggregate_counts.add_assign(score.counts);
        case_scores.push(WerCorpusCaseScore {
            id: case.id.to_string(),
            score,
        });
    }

    Ok(WerCorpusReport {
        cases: case_scores,
        aggregate: WerScore::from_counts(aggregate_counts),
    })
}

fn normalized_words(transcript: &str) -> Vec<String> {
    normalize_transcript(transcript)
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

fn edit_path(reference_words: &[String], hypothesis_words: &[String]) -> EditPath {
    let columns = hypothesis_words.len() + 1;
    let mut cells = vec![EditPath::default(); (reference_words.len() + 1) * columns];

    for i in 1..=reference_words.len() {
        let previous = cells[(i - 1) * columns];
        cells[i * columns] = previous.with_deletion();
    }

    for j in 1..=hypothesis_words.len() {
        let previous = cells[j - 1];
        cells[j] = previous.with_insertion();
    }

    for i in 1..=reference_words.len() {
        for j in 1..=hypothesis_words.len() {
            let idx = i * columns + j;
            if reference_words[i - 1] == hypothesis_words[j - 1] {
                cells[idx] = cells[(i - 1) * columns + (j - 1)];
                continue;
            }

            let substitution = cells[(i - 1) * columns + (j - 1)].with_substitution();
            let insertion = cells[i * columns + (j - 1)].with_insertion();
            let deletion = cells[(i - 1) * columns + j].with_deletion();
            cells[idx] = best_path(substitution, best_path(insertion, deletion));
        }
    }

    cells[reference_words.len() * columns + hypothesis_words.len()]
}

fn best_path(a: EditPath, b: EditPath) -> EditPath {
    if edit_sort_key(a) <= edit_sort_key(b) {
        a
    } else {
        b
    }
}

fn edit_sort_key(path: EditPath) -> (usize, usize, usize, usize, usize) {
    (
        path.errors(),
        path.insertions + path.deletions,
        path.substitutions,
        path.insertions,
        path.deletions,
    )
}

fn invalid_request(field: &str, message: &str) -> OcelotlError {
    OcelotlError::InvalidRequest(InvalidRequestError {
        field: field.to_string(),
        message: message.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ocelotl_core::OcelotlError;

    #[test]
    fn normalize_transcript_lowercases_strips_punctuation_and_folds_whitespace() {
        let normalized = normalize_transcript("  Hello, WORLD!\nThis\tis... Ocelotl.  ");

        assert_eq!(normalized, "hello world this is ocelotl");
    }

    #[test]
    fn edit_counts_cover_exact_match_substitution_insertion_and_deletion() {
        assert_eq!(
            wer_edit_counts("hello world", "Hello, world!").expect("exact match should score"),
            WerEditCounts {
                substitutions: 0,
                insertions: 0,
                deletions: 0,
                reference_words: 2,
            }
        );

        assert_eq!(
            wer_edit_counts("hello world", "hello there").expect("substitution should score"),
            WerEditCounts {
                substitutions: 1,
                insertions: 0,
                deletions: 0,
                reference_words: 2,
            }
        );

        assert_eq!(
            wer_edit_counts("hello world", "hello tiny world").expect("insertion should score"),
            WerEditCounts {
                substitutions: 0,
                insertions: 1,
                deletions: 0,
                reference_words: 2,
            }
        );

        assert_eq!(
            wer_edit_counts("hello tiny world", "hello world").expect("deletion should score"),
            WerEditCounts {
                substitutions: 0,
                insertions: 0,
                deletions: 1,
                reference_words: 3,
            }
        );
    }

    #[test]
    fn corpus_report_aggregates_case_counts_without_thresholds() {
        let cases = [
            WerCorpusCase {
                id: "exact",
                expected_transcript: "The quick brown fox.",
                recognized_transcript: "the quick brown fox",
            },
            WerCorpusCase {
                id: "substitution",
                expected_transcript: "hello world",
                recognized_transcript: "hello there",
            },
            WerCorpusCase {
                id: "insertion",
                expected_transcript: "a short sample",
                recognized_transcript: "a very short sample",
            },
            WerCorpusCase {
                id: "deletion",
                expected_transcript: "one two three four",
                recognized_transcript: "one two four",
            },
        ];

        let report = score_wer_corpus(&cases).expect("tiny transcript corpus should score");

        assert_eq!(report.cases.len(), 4);
        assert_eq!(report.aggregate.counts.substitutions, 1);
        assert_eq!(report.aggregate.counts.insertions, 1);
        assert_eq!(report.aggregate.counts.deletions, 1);
        assert_eq!(report.aggregate.counts.reference_words, 13);
        assert!((report.aggregate.wer - (3.0 / 13.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn wer_rejects_reference_that_normalizes_to_empty() {
        let err = wer_edit_counts(" -- !!! ", "hello")
            .expect_err("empty normalized reference should be rejected");

        match err {
            OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "reference_transcript");
                assert!(
                    invalid.message.contains("at least one word"),
                    "expected actionable empty-reference message, got {:?}",
                    invalid.message
                );
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn corpus_report_rejects_empty_case_list() {
        let err = score_wer_corpus(&[]).expect_err("empty corpus should be explicit");

        match err {
            OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "wer_corpus.cases");
                assert!(
                    invalid.message.contains("at least one case"),
                    "expected actionable empty-corpus message, got {:?}",
                    invalid.message
                );
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }
}
