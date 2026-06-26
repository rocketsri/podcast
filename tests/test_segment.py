"""Tests for pipeline/segment.py: pure interval math turning diarization
turns + VAD speech mask into candidate clips. Synthetic fixtures only, no
real audio, no db, no network -- exactly what segment.py's own docstring
says it should be testable with."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipeline import segment
from pipeline.diarize import SpeakerTurn
from pipeline.vad import SpeechSegment, WINDOW_SIZE_SAMPLES, SAMPLE_RATE

FRAME_SECONDS = WINDOW_SIZE_SAMPLES / SAMPLE_RATE  # 0.032s


def make_cfg(**overrides):
    """Mirrors config/pipeline.yaml's `segmentation:` section defaults,
    as a simple attribute-accessible namespace (cfg.target_bucket_ratios.under_10s
    is accessed dotted, so that field is itself a namespace)."""
    defaults = dict(
        min_clip_duration_seconds=1.0,
        max_clip_duration_seconds=30.0,
        short_clip_target_seconds=10.0,
        pause_search_window_seconds=2.0,
        pause_probability_floor=0.5,
        target_bucket_ratios=SimpleNamespace(under_10s=0.70, from_10_to_20s=0.25, from_20_to_30s=0.05),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def flat_frame_probs(duration_seconds: float, prob: float = 0.9) -> np.ndarray:
    """An all-speech (no detectable pause) frame-probability array covering
    [0, duration_seconds)."""
    n_frames = int(np.ceil(duration_seconds / FRAME_SECONDS)) + 5
    return np.full(n_frames, prob, dtype=np.float32)


def frame_probs_with_dip(duration_seconds: float, dip_at: float, base: float = 0.9, dip: float = 0.05) -> np.ndarray:
    """All-speech frame probabilities except a clean low-probability dip
    centered at `dip_at` seconds -- a detectable pause point."""
    probs = flat_frame_probs(duration_seconds, prob=base)
    center_frame = int(dip_at / FRAME_SECONDS)
    for offset in range(-2, 3):
        idx = center_frame + offset
        if 0 <= idx < len(probs):
            probs[idx] = dip
    return probs


# --- overlap exclusion ---------------------------------------------------------


def test_overlap_regions_excluded_from_candidates():
    """Two speakers talking over each other from 5s-7s must not appear in
    either speaker's candidate intervals."""
    turns = [
        SpeakerTurn("spk_a", 0.0, 10.0),
        SpeakerTurn("spk_b", 5.0, 15.0),
    ]
    vad_segments = [SpeechSegment(0.0, 15.0, confidence=0.95)]
    frame_probs = flat_frame_probs(15.0)
    cfg = make_cfg(min_clip_duration_seconds=0.0)  # disable too_short filtering for this check

    clips = segment.build_candidate_clips(
        "ep1", "pod1", turns, vad_segments, frame_probs, {"spk_a": "spk_a", "spk_b": "spk_b"}, cfg
    )

    for clip in clips:
        # No candidate clip may intersect the [5, 7) overlap window... wait,
        # actual overlap is [5.0, 10.0) since spk_a runs 0-10 and spk_b 5-15.
        assert not (clip.start_seconds < 10.0 and clip.end_seconds > 5.0), (
            f"clip {clip.start_seconds}-{clip.end_seconds} intersects the overlap region"
        )

    # spk_a should only contribute [0, 5), spk_b only [10, 15).
    spk_a_clips = [c for c in clips if c.speaker_id == "spk_a"]
    spk_b_clips = [c for c in clips if c.speaker_id == "spk_b"]
    assert all(c.end_seconds <= 5.0 for c in spk_a_clips)
    assert all(c.start_seconds >= 10.0 for c in spk_b_clips)


def test_no_overlap_when_single_speaker():
    turns = [SpeakerTurn("spk_a", 0.0, 8.0)]
    vad_segments = [SpeechSegment(0.0, 8.0, confidence=0.95)]
    frame_probs = flat_frame_probs(8.0)
    cfg = make_cfg()

    clips = segment.build_candidate_clips("ep1", "pod1", turns, vad_segments, frame_probs, {"spk_a": "spk_a"}, cfg)
    assert len(clips) == 1
    assert clips[0].start_seconds == 0.0
    assert clips[0].end_seconds == 8.0
    assert clips[0].discard_reason is None


# --- too_short ------------------------------------------------------------------


def test_short_single_speaker_interval_discarded_as_too_short():
    """A single-speaker interval under min_clip_duration_seconds (1.0s
    default) must be flagged too_short, not silently dropped."""
    turns = [SpeakerTurn("spk_a", 0.0, 0.5)]
    vad_segments = [SpeechSegment(0.0, 0.5, confidence=0.9)]
    frame_probs = flat_frame_probs(0.5)
    cfg = make_cfg()

    clips = segment.build_candidate_clips("ep1", "pod1", turns, vad_segments, frame_probs, {"spk_a": "spk_a"}, cfg)
    assert len(clips) == 1
    assert clips[0].discard_reason == "too_short"
    assert clips[0].start_seconds == 0.0
    assert clips[0].end_seconds == 0.5


def test_interval_at_or_above_floor_not_marked_too_short():
    turns = [SpeakerTurn("spk_a", 0.0, 1.0)]
    vad_segments = [SpeechSegment(0.0, 1.0, confidence=0.9)]
    frame_probs = flat_frame_probs(1.0)
    cfg = make_cfg()

    clips = segment.build_candidate_clips("ep1", "pod1", turns, vad_segments, frame_probs, {"spk_a": "spk_a"}, cfg)
    assert len(clips) == 1
    assert clips[0].discard_reason is None


# --- 10-30s split at detected pause ----------------------------------------------


def test_interval_10_to_30s_splits_at_detected_pause_point():
    """A 20s single-speaker interval with a clean pause near the midpoint
    (10s) should split into two clips at that pause, provided the
    under_10s bucket ratio target isn't already met (it starts at 0 for a
    fresh bucket_counter, so the first split always qualifies)."""
    duration = 20.0
    turns = [SpeakerTurn("spk_a", 0.0, duration)]
    vad_segments = [SpeechSegment(0.0, duration, confidence=0.9)]
    frame_probs = frame_probs_with_dip(duration, dip_at=10.0)
    cfg = make_cfg()

    clips = segment.build_candidate_clips("ep1", "pod1", turns, vad_segments, frame_probs, {"spk_a": "spk_a"}, cfg)

    assert len(clips) == 2
    clips_sorted = sorted(clips, key=lambda c: c.start_seconds)
    assert clips_sorted[0].start_seconds == 0.0
    assert clips_sorted[1].end_seconds == duration
    # The two clips must be contiguous and split near the dip (10s).
    assert clips_sorted[0].end_seconds == clips_sorted[1].start_seconds
    assert 8.0 < clips_sorted[0].end_seconds < 12.0
    for clip in clips:
        assert clip.discard_reason is None


def test_interval_10_to_30s_with_no_pause_stays_whole():
    """No detectable pause anywhere in the search window -> the long-tail
    clip is kept whole rather than split blindly."""
    duration = 20.0
    turns = [SpeakerTurn("spk_a", 0.0, duration)]
    vad_segments = [SpeechSegment(0.0, duration, confidence=0.9)]
    frame_probs = flat_frame_probs(duration, prob=0.9)  # no dip anywhere
    cfg = make_cfg()

    clips = segment.build_candidate_clips("ep1", "pod1", turns, vad_segments, frame_probs, {"spk_a": "spk_a"}, cfg)
    assert len(clips) == 1
    assert clips[0].start_seconds == 0.0
    assert clips[0].end_seconds == duration


# --- >30s force-cut at the cap ---------------------------------------------------


def test_long_interval_with_no_pause_force_cut_at_30s_cap():
    """A 65s single-speaker interval with no detectable pause anywhere must
    be force-cut at exactly the 30s cap, repeatedly, until what's left is
    <= 30s."""
    duration = 65.0
    turns = [SpeakerTurn("spk_a", 0.0, duration)]
    vad_segments = [SpeechSegment(0.0, duration, confidence=0.9)]
    frame_probs = flat_frame_probs(duration, prob=0.9)  # no dip -> no clean pause anywhere
    cfg = make_cfg()

    clips = segment.build_candidate_clips("ep1", "pod1", turns, vad_segments, frame_probs, {"spk_a": "spk_a"}, cfg)
    clips_sorted = sorted(clips, key=lambda c: c.start_seconds)

    # Force cuts land at exactly 30.0 and 60.0; final remainder is [60, 65).
    boundaries = [0.0] + [round(c.end_seconds, 3) for c in clips_sorted]
    assert boundaries == [0.0, 30.0, 60.0, 65.0]
    for clip in clips_sorted:
        assert clip.end_seconds - clip.start_seconds <= cfg.max_clip_duration_seconds + 1e-9


def test_long_interval_with_pause_near_cap_cuts_at_pause_not_blind_cap():
    """A clean pause exists within the search window around the 30s target
    -- the cut should land at that pause, not blindly at exactly 30.0."""
    duration = 40.0
    turns = [SpeakerTurn("spk_a", 0.0, duration)]
    vad_segments = [SpeechSegment(0.0, duration, confidence=0.9)]
    # Put a clean dip at 29s, within the default pause_search_window_seconds=2.0 of the 30s target.
    frame_probs = frame_probs_with_dip(duration, dip_at=29.0)
    cfg = make_cfg()

    clips = segment.build_candidate_clips("ep1", "pod1", turns, vad_segments, frame_probs, {"spk_a": "spk_a"}, cfg)
    clips_sorted = sorted(clips, key=lambda c: c.start_seconds)
    first_cut = clips_sorted[0].end_seconds
    assert 27.0 < first_cut < 31.0
    assert clips_sorted[-1].end_seconds == duration


# --- clip duration cap invariant -------------------------------------------------


def test_no_resulting_clip_exceeds_max_clip_duration_seconds():
    """Across a mix of short, medium, and very long intervals (with and
    without pauses), no clip should ever come out longer than
    cfg.max_clip_duration_seconds."""
    cfg = make_cfg()
    rng = np.random.default_rng(42)

    turns = []
    cursor = 0.0
    interval_lengths = [0.5, 5.0, 12.0, 22.0, 47.0, 93.0, 3.0]
    for length in interval_lengths:
        turns.append(SpeakerTurn("spk_a", cursor, cursor + length))
        cursor += length + 1.0  # gap between turns so they don't merge across speakers
    total_duration = cursor

    vad_segments = [SpeechSegment(0.0, total_duration, confidence=0.9)]
    # Mostly-flat probabilities with random noise so some pauses are detected
    # and some aren't, exercising both split paths.
    frame_probs = (0.85 + rng.uniform(-0.1, 0.1, size=int(np.ceil(total_duration / FRAME_SECONDS)) + 5)).astype(
        np.float32
    )

    clips = segment.build_candidate_clips("ep1", "pod1", turns, vad_segments, frame_probs, {"spk_a": "spk_a"}, cfg)
    assert len(clips) > 0
    for clip in clips:
        duration = clip.end_seconds - clip.start_seconds
        assert duration <= cfg.max_clip_duration_seconds + 1e-9, f"clip too long: {duration}"


def test_vad_excludes_non_speech_so_silence_not_candidate():
    """A turn that runs longer than the VAD-detected speech mask should be
    trimmed down to only the overlapping speech portion."""
    turns = [SpeakerTurn("spk_a", 0.0, 10.0)]
    # Only [2, 6) is actually detected as speech by VAD.
    vad_segments = [SpeechSegment(2.0, 6.0, confidence=0.9)]
    frame_probs = flat_frame_probs(10.0)
    cfg = make_cfg(min_clip_duration_seconds=0.0)

    clips = segment.build_candidate_clips("ep1", "pod1", turns, vad_segments, frame_probs, {"spk_a": "spk_a"}, cfg)
    assert len(clips) == 1
    assert clips[0].start_seconds == 2.0
    assert clips[0].end_seconds == 6.0


def test_label_to_speaker_none_for_unmapped_label():
    """label_to_speaker.get(label) for a label with no usable embedding
    (per cluster.ingest_episode_diarization's contract) must come through
    as speaker_id=None on the clip, not raise."""
    turns = [SpeakerTurn("spk_unmapped", 0.0, 5.0)]
    vad_segments = [SpeechSegment(0.0, 5.0, confidence=0.9)]
    frame_probs = flat_frame_probs(5.0)
    cfg = make_cfg()

    clips = segment.build_candidate_clips("ep1", "pod1", turns, vad_segments, frame_probs, {}, cfg)
    assert len(clips) == 1
    assert clips[0].speaker_id is None
