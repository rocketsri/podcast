"""Clip-segmentation: turns one episode's diarization turns + VAD speech mask
into single-speaker candidate clips sized toward the spec's duration mix
(mostly <10s, a long tail to 30s).

Pure interval math only -- no db, no audio I/O -- so it's fully testable with
synthetic turns/VAD-segment fixtures (see tests/test_segment.py). Discard
bookkeeping here is limited to the one structural reason segmentation itself
can determine (`too_short`); every content-based discard reason (silence,
music, overlap, low ASR confidence, ...) is applied later by quality.py as an
UPDATE over already-persisted clip rows, once ASR has run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from pipeline import db, diarize, vad

logger = logging.getLogger(__name__)

FRAME_SECONDS = vad.WINDOW_SIZE_SAMPLES / vad.SAMPLE_RATE  # 0.032s per Silero frame


@dataclass(frozen=True)
class CandidateClip:
    episode_id: str
    podcast_id: str
    clip_id: str
    start_seconds: float
    end_seconds: float
    speaker_id: str | None
    vad_confidence: float | None = None
    discard_reason: str | None = None


class _BucketCounter:
    """Running per-episode duration histogram, used to bias the 10-30s split
    decision toward the configured target_bucket_ratios mix (a local greedy
    heuristic, not a global optimizer -- see plan's segmentation section)."""

    def __init__(self):
        self.counts = {"under_10s": 0, "from_10_to_20s": 0, "from_20_to_30s": 0}

    @staticmethod
    def _bucket_for(duration: float) -> str:
        if duration < 10.0:
            return "under_10s"
        if duration < 20.0:
            return "from_10_to_20s"
        return "from_20_to_30s"

    def record(self, duration: float) -> None:
        self.counts[self._bucket_for(duration)] += 1

    def ratio(self, bucket: str) -> float:
        total = sum(self.counts.values())
        return self.counts[bucket] / total if total else 0.0


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _intersect_interval_lists(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    a = _merge_intervals(a)
    b = _merge_intervals(b)
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if start < end:
            result.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result


def _subtract_interval_lists(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """a minus b, both treated as disjoint interval lists."""
    a = _merge_intervals(a)
    b = _merge_intervals(b)
    result = []
    for start, end in a:
        cur = start
        for bs, be in b:
            if be <= cur or bs >= end:
                continue
            if bs > cur:
                result.append((cur, min(bs, end)))
            cur = max(cur, be)
            if cur >= end:
                break
        if cur < end:
            result.append((cur, end))
    return result


def _weighted_vad_confidence(
    start: float, end: float, vad_segments: list
) -> float | None:
    """Duration-weighted average of `vad.SpeechSegment.confidence` across
    whatever portion of each VAD segment falls inside [start, end) -- a clip
    can span parts of more than one VAD segment, so this is an overlap
    integral, not a lookup."""
    total_weight = 0.0
    weighted_sum = 0.0
    for seg in vad_segments:
        overlap = min(end, seg.end_seconds) - max(start, seg.start_seconds)
        if overlap > 0:
            total_weight += overlap
            weighted_sum += overlap * seg.confidence
    return weighted_sum / total_weight if total_weight > 0 else None


def _turn_intervals_by_label(turns: list) -> dict[str, list[tuple[float, float]]]:
    by_label: dict[str, list[tuple[float, float]]] = {}
    for turn in turns:
        by_label.setdefault(turn.local_label, []).append((turn.start_seconds, turn.end_seconds))
    return {label: _merge_intervals(ivs) for label, ivs in by_label.items()}


def _find_pause_point(
    frame_probs: np.ndarray,
    target_seconds: float,
    interval_start: float,
    interval_end: float,
    cfg,
) -> float | None:
    """Timestamp of the lowest-speech-probability frame within
    `cfg.pause_search_window_seconds` of `target_seconds`, clamped inside the
    interval -- the "natural pause point" candidate cuts are made at, so a
    long turn doesn't get sliced mid-word. None if the window falls outside
    available frames, or if even the best frame found doesn't read as a real
    pause (probability still above pause_probability_floor) -- a flat
    all-speech window has no minimum worth cutting at, just the smallest of a
    bad bunch, so it must not be mistaken for a clean pause."""
    lo = max(interval_start, target_seconds - cfg.pause_search_window_seconds)
    hi = min(interval_end, target_seconds + cfg.pause_search_window_seconds)
    if hi <= lo:
        return None
    start_frame = min(int(lo / FRAME_SECONDS), len(frame_probs))
    end_frame = min(int(np.ceil(hi / FRAME_SECONDS)), len(frame_probs))
    if start_frame >= end_frame:
        return None
    window = frame_probs[start_frame:end_frame]
    best_offset = int(np.argmin(window))
    if window[best_offset] > cfg.pause_probability_floor:
        return None
    return (start_frame + best_offset) * FRAME_SECONDS


def _split_interval(
    start: float,
    end: float,
    label: str,
    frame_probs: np.ndarray,
    bucket_counter: _BucketCounter,
    cfg,
) -> list[tuple[float, float]]:
    duration = end - start

    if duration > cfg.max_clip_duration_seconds:
        target = start + cfg.max_clip_duration_seconds
        cut = _find_pause_point(frame_probs, target, start, end, cfg)
        if cut is None or cut <= start or cut - start > cfg.max_clip_duration_seconds:
            cut = target
            logger.info("segment: forced hard cut at %.2fs (label=%s, no clean pause found)", cut, label)
        bucket_counter.record(cut - start)
        return [(start, cut)] + _split_interval(cut, end, label, frame_probs, bucket_counter, cfg)

    if duration <= cfg.short_clip_target_seconds:
        bucket_counter.record(duration)
        return [(start, end)]

    # 10s < duration <= 30s: only split if doing so would correct an
    # under-10s shortfall against the target mix; otherwise a clean long-tail
    # clip is exactly what the spec wants, so keep it whole.
    midpoint = start + duration / 2.0
    cut = _find_pause_point(frame_probs, midpoint, start, end, cfg)
    if cut is not None:
        first_len, second_len = cut - start, end - cut
        if (
            first_len >= cfg.min_clip_duration_seconds
            and second_len >= cfg.min_clip_duration_seconds
            and bucket_counter.ratio("under_10s") < cfg.target_bucket_ratios.under_10s
        ):
            bucket_counter.record(first_len)
            bucket_counter.record(second_len)
            return [(start, cut), (cut, end)]

    bucket_counter.record(duration)
    return [(start, end)]


def build_candidate_clips(
    episode_id: str,
    podcast_id: str,
    turns: list,
    vad_segments: list,
    frame_probs: np.ndarray,
    label_to_speaker: dict[str, str | None],
    cfg,
) -> list[CandidateClip]:
    """`turns` is diarize.DiarizationResult.turns; `vad_segments` is
    vad.run_vad's output; `frame_probs` is vad.frame_speech_probabilities's
    output for the same audio; `label_to_speaker` is the dict returned by
    cluster.ingest_episode_diarization (local_label -> global speaker_id, or
    None for labels with no usable embedding); `cfg` is config.segmentation."""
    overlap_intervals = diarize.overlap_mask_seconds(turns)
    vad_intervals = [(s.start_seconds, s.end_seconds) for s in vad_segments]

    clean_by_label: dict[str, list[tuple[float, float]]] = {}
    for label, turn_intervals in _turn_intervals_by_label(turns).items():
        speech_only = _intersect_interval_lists(turn_intervals, vad_intervals)
        clean_by_label[label] = _subtract_interval_lists(speech_only, overlap_intervals)

    # Chronological order across all speakers, not grouped by label, so the
    # running bucket histogram reflects real episode order.
    all_intervals = sorted(
        ((s, e, label) for label, ivs in clean_by_label.items() for s, e in ivs),
        key=lambda item: item[0],
    )

    bucket_counter = _BucketCounter()
    clips: list[CandidateClip] = []
    for index, (start, end, label) in enumerate(all_intervals):
        for clip_index, (clip_start, clip_end) in enumerate(
            _split_interval(start, end, label, frame_probs, bucket_counter, cfg)
        ):
            duration = clip_end - clip_start
            discard_reason = "too_short" if duration < cfg.min_clip_duration_seconds else None
            clips.append(
                CandidateClip(
                    episode_id=episode_id,
                    podcast_id=podcast_id,
                    clip_id=f"{episode_id}_clip_{index:04d}_{clip_index:02d}",
                    start_seconds=clip_start,
                    end_seconds=clip_end,
                    speaker_id=label_to_speaker.get(label),
                    vad_confidence=_weighted_vad_confidence(clip_start, clip_end, vad_segments),
                    discard_reason=discard_reason,
                )
            )
    return clips


def persist_candidate_clips(conn, clips: list[CandidateClip]) -> None:
    for clip in clips:
        db.insert_clip(
            conn, clip.clip_id, clip.episode_id, clip.podcast_id,
            clip.start_seconds, clip.end_seconds,
            speaker_id=clip.speaker_id, discard_reason=clip.discard_reason,
            vad_confidence=clip.vad_confidence,
        )
