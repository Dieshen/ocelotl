//! Chunk planning for long-form Parakeet transcription.
//!
//! # Why chunking is needed, and what it actually fixes
//!
//! The encoder runs full self-attention (`att_context_size = [-1, -1]`) with no
//! KV cache, so its cost is quadratic in the subsampled frame count.
//!
//! Fitting `cost = a*T + b*T^2` to two measured points (138 frames in 5.23 s,
//! 1513 frames in 482.7 s) gives `a = 9.7e-3 s/frame`, `b = 2.0e-4 s/frame^2`:
//!
//! | audio | frames | linear | quadratic | quadratic share | RTF |
//! |------:|-------:|-------:|----------:|----------------:|----:|
//! |  11 s |    138 |  1.3 s |     3.9 s |           74.5% | 0.5 |
//! |  60 s |    751 |  7.3 s |   115.3 s |           94.1% | 2.0 |
//! | 121 s |  1 513 | 14.7 s |   468.1 s |           97.0% | 4.0 |
//! | 600 s |  7 501 | 72.6 s | 11504.0 s |           99.4% |19.3 |
//!
//! The binding constraint is **time, not memory**: the largest transient is the
//! `[T, 2T-1]` score matrix, only ~450 MB even at ten minutes. RTF crosses 1.0
//! between 11 s and 60 s of audio, far earlier than a naive reading suggests.
//!
//! > ⚠️ **Most of that quadratic term is an unvectorized kernel, not intrinsic
//! > cost.** Attention is only **2.3% of the encoder's multiply-accumulates** at
//! > 138 frames but **74.5% of its time**, because `encoder::self_attention`
//! > computes `QK^T`, `QP^T` and `score·V` in hand-written scalar loops while
//! > every projection and FFN goes through the AVX2 microkernel — roughly a 32x
//! > per-MAC penalty. Chunking bounds the quadratic term; **vectorizing
//! > attention would shrink it**, and is the larger win of the two. Chunking is
//! > still worth having (it makes cost linear in length regardless of kernel
//! > quality) but it is not the fix for this.
//!
//! Chunking replaces the quadratic term with a constant one, so RTF stays at
//! whatever the chunk size costs no matter how long the audio is. Measured on
//! the 121 s fixture at the defaults below:
//!
//! | | wall clock | RTF | tokens |
//! |---|---:|---:|---:|
//! | one window | 493.7 s | 4.08 | 407 |
//! | chunked (375 / 62) | 191.4 s | 1.58 | 413 |
//!
//! **2.58x faster for a 4.2% token error rate** against the unchunked decode.
//! That fixture is deliberately hostile — `parity_jfk` tiled eleven times at
//! -30 dBFS, i.e. quiet and highly repetitive, which is the worst case for a
//! transducer's alignment — so treat 4.2% as a pessimistic bound rather than a
//! typical cost.
//!
//! # Overlap-and-trim, then a single decode
//!
//! Each chunk is encoded with extra **context** frames on both sides, and those
//! context frames are then thrown away.
//!
//! **Chunking a full-attention encoder is lossy everywhere, not just at the
//! seams.** This is worth stating plainly because the intuition from *local*
//! attention does not carry over and is actively misleading. With a limited
//! context window, a frame far enough inside a chunk sees exactly what it would
//! have seen in a full-length encode, so trimming the edges recovers the exact
//! answer. This model has `att_context_size = [-1, -1]`: every frame attends to
//! every other frame, so shortening the window truncates the receptive field of
//! **every frame in it**, interior ones included.
//!
//! Measured on `parity_jfk` (138 frames, 48-frame bodies), the deviation from an
//! unchunked encode is essentially flat across the chunk rather than concentrated
//! at its boundary:
//!
//! | context frames | max deviation | at seams | interior |
//! |---------------:|--------------:|---------:|---------:|
//! |              0 |      2.2e-1   |  2.2e-1  |  1.5e-1  |
//! |             12 |      1.0e-1   |  1.0e-1  |  9.9e-2  |
//! |             25 |      9.3e-2   |  7.9e-2  |  9.3e-2  |
//! |             50 |      7.3e-2   |  7.3e-2  |  7.2e-2  |
//!
//! (On a +/-0.15 output scale. The interior column is the point: it tracks the
//! seam column instead of collapsing.) More context monotonically helps and
//! nothing makes it exact short of encoding the whole utterance.
//!
//! So chunking trades accuracy for time, and the honest gate is bounded
//! degradation rather than token equality. Zero context is where it visibly
//! breaks — the decode loses tokens outright.
//!
//! Crucially the **decode is not chunked**. Encoder outputs are stitched into one
//! sequence and a single greedy TDT pass runs over the whole thing, so the
//! prediction network's recurrent state is continuous across boundaries.
//! Decoding per chunk and concatenating the text would reset that state at every
//! seam and corrupt the tokens either side of it.
//!
//! # Chunk the FEATURES, not the waveform
//!
//! The mel frontend's per-feature normalization reduces mean and standard
//! deviation over **every frame of its input**, so slicing the waveform would
//! give each window its own statistics — a second, avoidable source of
//! divergence layered on top of the attention truncation above. Running the
//! frontend once over the whole utterance and slicing its normalized output puts
//! every chunk on the statistics the unchunked path used. It is also cheaper:
//! the mel is computed once rather than once per overlapping window.
//!
//! Worth recording that this was **not** the fix for the interior error, which
//! is what it was originally implemented to be. Switching from waveform slicing
//! to feature slicing left the interior deviation essentially unchanged
//! (7.6e-2 to 9.3e-2 at 25 context frames — if anything marginally worse on this
//! clip). The attention truncation above is the dominant term; normalization was
//! a plausible-sounding second-order effect that measurement demoted. Keeping
//! the change anyway is justified on cost and on removing a confound, not on the
//! accuracy claim it was proposed under.
//!
//! # The alignment rule that makes stitching exact
//!
//! One encoder frame is 8 mel frames (80 ms), i.e. 12.5 frames per second.
//! **Chunk boundaries are placed on multiples of [`MEL_FRAMES_PER_FRAME`]**,
//! which makes the mapping between a slice's local frame index and its global
//! one an exact integer offset rather than a rounding problem: a window starting
//! at mel frame `8f` has its local frame 0 at global frame `f`.
//!
//! Without that rule the subsampler's three stride-2 stages turn every boundary
//! into an off-by-one risk — and a time-axis off-by-one is precisely the failure
//! that survives a passing value diff and desynchronizes the transducer.

use ocelotl_core::{OcelotlError, Result, RuntimeError};

/// Mel frames per subsampled encoder frame (the subsampler's 8x reduction).
/// Chunk boundaries are placed on multiples of this.
pub const MEL_FRAMES_PER_FRAME: usize = 8;

/// Audio samples per subsampled encoder frame: 8x subsampling over a 160-sample
/// hop. Informational — chunking slices mel frames, not samples.
pub const SAMPLES_PER_FRAME: usize = MEL_FRAMES_PER_FRAME * 160;

/// Encoder frames per second (`16000 / SAMPLES_PER_FRAME`).
pub const FRAMES_PER_SECOND: usize = 12;

/// Default body length per chunk, in frames (~30 s).
///
/// Chosen where the attention share is still small (6% of MACs at 30 s) but the
/// per-chunk fixed costs are amortized. Larger chunks pay the quadratic term;
/// smaller ones pay proportionally more context overhead.
pub const DEFAULT_CHUNK_FRAMES: usize = 375;

/// Default context frames encoded on each side and then discarded (~5 s).
///
/// This is the knob that trades compute for boundary accuracy. Too small and the
/// frames at a seam are computed with truncated attention; too large and every
/// chunk re-encodes audio it will throw away. At the default the overhead is
/// `2 * 62 / 375` — about 33% extra encoder work.
pub const DEFAULT_CONTEXT_FRAMES: usize = 62;

fn rt<S: Into<String>>(m: S) -> OcelotlError {
    OcelotlError::Runtime(RuntimeError { message: m.into() })
}

/// One planned chunk, in **global** subsampled-frame coordinates.
///
/// `ctx_*` is what gets encoded; `body_*` is what is kept. The two coincide at
/// the true start and end of the utterance, where there is no context to trim
/// because there is no audio beyond the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkWindow {
    /// First frame of the encoded window.
    pub ctx_start: usize,
    /// One past the last frame of the encoded window.
    pub ctx_end: usize,
    /// First frame kept from this window's output.
    pub body_start: usize,
    /// One past the last frame kept.
    pub body_end: usize,
}

impl ChunkWindow {
    /// Frames encoded for this window.
    pub fn encoded_frames(&self) -> usize {
        self.ctx_end - self.ctx_start
    }

    /// Frames kept from this window.
    pub fn kept_frames(&self) -> usize {
        self.body_end - self.body_start
    }

    /// Range of the encoded output to keep, in **local** frame coordinates.
    pub fn keep_range(&self) -> std::ops::Range<usize> {
        (self.body_start - self.ctx_start)..(self.body_end - self.ctx_start)
    }

    /// Mel-frame range to slice out of the **already-normalized** features.
    ///
    /// The subsampler reduces 8x with `padding = 1, kernel = 3, stride = 2` at
    /// each of three stages, so encoder frame `f` is centred on mel frame `8f`
    /// and the counts line up as `ceil(mel / 8)`. Slicing on multiples of
    /// [`MEL_FRAMES_PER_FRAME`] therefore makes the local-to-global frame
    /// mapping an exact integer offset.
    ///
    /// The slice's own edges get zero-padding instead of real neighbouring
    /// audio, which is precisely why the context frames are trimmed afterwards.
    pub fn mel_range(&self, total_mel_frames: usize) -> std::ops::Range<usize> {
        let start = (self.ctx_start * MEL_FRAMES_PER_FRAME).min(total_mel_frames);
        let end = (self.ctx_end * MEL_FRAMES_PER_FRAME).min(total_mel_frames);
        start..end.max(start)
    }
}

/// Plan the chunk windows covering `total_frames`.
///
/// The bodies tile `[0, total_frames)` exactly — no gaps, no overlap — so the
/// stitched output has precisely `total_frames` frames in their original order.
/// Overlap exists only in the *encoded* windows, and is always discarded.
pub fn plan_chunks(
    total_frames: usize,
    chunk_frames: usize,
    context_frames: usize,
) -> Result<Vec<ChunkWindow>> {
    if chunk_frames == 0 {
        return Err(rt("chunk_frames must be non-zero"));
    }
    if total_frames == 0 {
        return Ok(Vec::new());
    }
    let mut windows = Vec::new();
    let mut body_start = 0usize;
    while body_start < total_frames {
        let body_end = (body_start + chunk_frames).min(total_frames);
        windows.push(ChunkWindow {
            ctx_start: body_start.saturating_sub(context_frames),
            ctx_end: (body_end + context_frames).min(total_frames),
            body_start,
            body_end,
        });
        body_start = body_end;
    }
    Ok(windows)
}

/// Total encoder frames for `samples` of 16 kHz audio.
///
/// Mirrors the frontend and subsampler closed forms rather than re-deriving
/// them, so the planner cannot disagree with what the encoder actually emits.
pub fn total_frames_for_samples(samples: usize) -> usize {
    let mel = crate::parakeet::audio::frame_count(samples);
    crate::parakeet::subsample::subsampled_frames(mel)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property everything else depends on: bodies tile the frame range
    /// exactly. A gap drops audio silently; an overlap duplicates tokens.
    #[test]
    fn chunk_bodies_tile_the_frame_range_without_gaps_or_overlap() {
        for total in [1usize, 7, 137, 138, 375, 376, 1513] {
            for chunk in [1usize, 8, 48, 375] {
                let plan = plan_chunks(total, chunk, 62).expect("plan");
                let mut expected = 0;
                for w in &plan {
                    assert_eq!(w.body_start, expected, "gap or overlap at {w:?}");
                    assert!(w.body_end > w.body_start, "empty body {w:?}");
                    expected = w.body_end;
                }
                assert_eq!(expected, total, "bodies do not cover total={total}");
                let kept: usize = plan.iter().map(|w| w.kept_frames()).sum();
                assert_eq!(kept, total, "kept frames != total for {total}/{chunk}");
            }
        }
    }

    /// Context must never reach outside the real audio: there is nothing there,
    /// and a window that claims frames past the end would desynchronize the
    /// local-to-global mapping for every later chunk.
    #[test]
    fn context_is_clamped_to_the_available_audio() {
        let plan = plan_chunks(100, 40, 30).expect("plan");
        assert_eq!(plan[0].ctx_start, 0, "no context before the start");
        assert_eq!(plan.last().unwrap().ctx_end, 100, "no context past the end");
        for w in &plan {
            assert!(w.ctx_start <= w.body_start && w.body_end <= w.ctx_end);
            assert!(w.ctx_end <= 100);
        }
    }

    /// The local keep-range must land inside what the window actually encodes.
    #[test]
    fn keep_range_lies_within_the_encoded_window() {
        for (total, chunk, ctx) in [(138, 48, 25), (1513, 375, 62), (10, 3, 100)] {
            for w in plan_chunks(total, chunk, ctx).expect("plan") {
                let r = w.keep_range();
                assert!(
                    r.end <= w.encoded_frames(),
                    "keep {r:?} exceeds encoded {} in {w:?}",
                    w.encoded_frames()
                );
                assert_eq!(r.len(), w.kept_frames());
            }
        }
    }

    /// A single chunk large enough to hold everything must be a no-op plan —
    /// one window, no trimming. This is what keeps short audio on exactly the
    /// same code path it had before chunking existed.
    #[test]
    fn audio_shorter_than_one_chunk_produces_a_single_untrimmed_window() {
        let plan = plan_chunks(138, 375, 62).expect("plan");
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].keep_range(), 0..138);
        assert_eq!(plan[0].ctx_start, 0);
        assert_eq!(plan[0].ctx_end, 138);
    }

    #[test]
    fn empty_audio_plans_nothing_and_zero_chunk_size_is_rejected() {
        assert!(plan_chunks(0, 375, 62).expect("plan").is_empty());
        assert!(plan_chunks(100, 0, 62).is_err());
    }

    /// Mel ranges must start exactly on a frame boundary — that alignment is
    /// what makes the local-to-global frame mapping an integer offset instead of
    /// a rounding problem.
    #[test]
    fn mel_ranges_start_on_frame_boundaries_and_tile_the_input() {
        let total_mel = 1101usize;
        let total = crate::parakeet::subsample::subsampled_frames(total_mel);
        let plan = plan_chunks(total, 48, 25).expect("plan");
        for w in &plan {
            let r = w.mel_range(total_mel);
            assert_eq!(
                r.start % MEL_FRAMES_PER_FRAME,
                0,
                "unaligned mel start in {w:?}"
            );
            assert_eq!(r.start, w.ctx_start * MEL_FRAMES_PER_FRAME);
            assert!(r.end <= total_mel);
            // The slice must be able to produce every frame the window keeps.
            let produced = crate::parakeet::subsample::subsampled_frames(r.len());
            assert!(
                produced >= w.keep_range().end,
                "{w:?}: mel slice of {} yields {produced} frames, need {}",
                r.len(),
                w.keep_range().end
            );
        }
        // The final window must reach the very last mel frame, or the tail of
        // the utterance is silently dropped.
        assert_eq!(plan.last().unwrap().mel_range(total_mel).end, total_mel);
    }
}
