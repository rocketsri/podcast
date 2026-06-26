"""Content-based discard taxonomy: applied as an UPDATE over clip rows that
segment.py already persisted, once ASR has filled in `no_speech_prob`/
`avg_logprob`/`utterance` for the episode. Structural discards (`too_short`)
are decided by segment.py itself, before any audio signal is computed --
everything here is a judgment call about clip *content*, layered on after the
fact so segmentation stays pure interval math (see segment.py's docstring).

Filters run in priority order inside `evaluate_clip_discard_reason` and stop
at the first match -- a clip only ever gets one discard_reason, the earliest
(and most certain) one that applies:

| discard_reason       | method                                                          | automated vs heuristic            |
|-----------------------|------------------------------------------------------------------|------------------------------------|
| vad_low_confidence    | clip's own duration-weighted average Silero confidence vs floor  | automated                          |
| silence_or_low_energy | RMS dBFS floor (catches VAD false positives); WADA-SNR-style    | heuristic                          |
| overlap_detected      | distance from nearest crosstalk region vs edge-trim tolerance    | automated mask + heuristic trim    |
| low_asr_confidence    | whisper no_speech_prob AND avg_logprob both bad (neither alone   | automated signal, heuristic combo  |
|                       | is trusted as a sole trigger) OR avg_logprob alone past a far     |                                     |
|                       | more extreme floor (catches wrong-language hallucination, where  |                                     |
|                       | no_speech_prob stays low because real speech IS present -- see   |                                     |
|                       | PROBLEMS.md #19)                                                  |                                     |
| music_detected        | spectral flatness corroborated by whisper no_speech_prob          | heuristic, weakest filter          |
| intro_outro_position  | first/last N seconds of episode + spectral/no_speech corroboration| heuristic                          |
| ad_segment_heuristic  | weaker spectral-flatness + no_speech_prob combo, anywhere in ep   | heuristic only, least reliable     |

`repeated_boilerplate` (cross-episode near-duplicate fingerprinting) is a
stretch goal named in the plan and is NOT implemented in this build -- see
LIMITATIONS.md.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from pipeline import audio, db, diarize


def _rms_dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if rms <= 0.0:
        return -120.0
    return 20.0 * float(np.log10(rms))


def _spectral_flatness(samples: np.ndarray) -> float:
    """Geometric-mean / arithmetic-mean of the magnitude spectrum (Wiener
    entropy), DC bin dropped. Higher = more broadband/noise-like energy --
    layered/produced audio (music beds, jingles, ad stingers) tends to read
    higher than a single clean voice's harmonic-peak-dominated spectrum, which
    is the (heuristic, imperfect) basis for using this as a music/production
    signal rather than a true classifier -- see LIMITATIONS.md."""
    if len(samples) < 4:
        return 0.0
    windowed = samples.astype(np.float64) * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed))[1:] + 1e-10
    geometric_mean = float(np.exp(np.mean(np.log(spectrum))))
    arithmetic_mean = float(np.mean(spectrum))
    return geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0


def _overlap_edge_distance(start: float, end: float, overlap_intervals: list[tuple[float, float]]) -> float:
    """Distance from [start, end) to the nearest crosstalk interval -- 0.0 if
    actually overlapping (shouldn't normally happen, segment.py already
    excludes overlap time ranges from candidates), otherwise the gap to the
    closest edge, so clips sitting right up against a crosstalk boundary
    (mic bleed / reverb tail past the diarization-detected edge) still get
    caught within `overlap_edge_trim_seconds`."""
    if not overlap_intervals:
        return float("inf")
    best = float("inf")
    for ov_start, ov_end in overlap_intervals:
        if end <= ov_start:
            dist = ov_start - end
        elif start >= ov_end:
            dist = start - ov_end
        else:
            dist = 0.0
        best = min(best, dist)
    return best


def evaluate_clip_discard_reason(
    clip_row: sqlite3.Row,
    samples: np.ndarray,
    sample_rate: int,
    episode_duration_seconds: float,
    overlap_intervals: list[tuple[float, float]],
    cfg,
) -> str | None:
    start, end = clip_row["start_seconds"], clip_row["end_seconds"]
    vad_confidence = clip_row["vad_confidence"]
    no_speech_prob = clip_row["no_speech_prob"]
    avg_logprob = clip_row["avg_logprob"]

    if vad_confidence is not None and vad_confidence < cfg.vad_low_confidence_floor:
        return "vad_low_confidence"

    clip_samples = audio.slice_samples(samples, sample_rate, start, end)

    if _rms_dbfs(clip_samples) < cfg.rms_silence_floor_db:
        return "silence_or_low_energy"

    if _overlap_edge_distance(start, end, overlap_intervals) < cfg.overlap_edge_trim_seconds:
        return "overlap_detected"

    if avg_logprob is not None and (
        (
            no_speech_prob is not None
            and no_speech_prob > cfg.low_asr_confidence_no_speech_prob
            and avg_logprob < cfg.low_asr_confidence_avg_logprob
        )
        or avg_logprob < cfg.catastrophic_avg_logprob_floor
    ):
        return "low_asr_confidence"

    flatness = _spectral_flatness(clip_samples)

    if (
        flatness > cfg.music_spectral_flatness_floor
        and no_speech_prob is not None
        and no_speech_prob > cfg.low_asr_confidence_no_speech_prob
    ):
        return "music_detected"

    near_episode_edge = (
        start < cfg.intro_outro_window_seconds
        or (episode_duration_seconds - end) < cfg.intro_outro_window_seconds
    )
    if near_episode_edge and (
        flatness > cfg.ad_heuristic_spectral_flatness_floor
        or (no_speech_prob is not None and no_speech_prob > cfg.ad_heuristic_no_speech_prob_floor)
    ):
        return "intro_outro_position"

    if (
        flatness > cfg.ad_heuristic_spectral_flatness_floor
        and no_speech_prob is not None
        and no_speech_prob > cfg.ad_heuristic_no_speech_prob_floor
    ):
        return "ad_segment_heuristic"

    return None


def apply_quality_filters(
    conn: sqlite3.Connection,
    episode_id: str,
    samples: np.ndarray,
    sample_rate: int,
    cfg,
) -> int:
    """Runs once per episode, after ASR -- evaluates every not-yet-discarded
    clip against the content-based taxonomy above and UPDATEs discard_reason
    in place. `samples`/`sample_rate` are the same full episode-level wav this
    episode's VAD/diarization already ran against (read once via
    audio.read_wav), not per-clip files -- clip-level audio is only ever
    extracted to flac later, for clips that survive this filter. Returns the
    number of clips newly discarded."""
    segments = db.get_local_speaker_segments_for_episode(conn, episode_id)
    turns = [diarize.SpeakerTurn(s["local_label"], s["start_seconds"], s["end_seconds"]) for s in segments]
    overlap_intervals = diarize.overlap_mask_seconds(turns)
    episode_duration_seconds = len(samples) / sample_rate

    discarded = 0
    for clip in db.get_clips_for_episode(conn, episode_id):
        if clip["discard_reason"] is not None:
            continue
        reason = evaluate_clip_discard_reason(
            clip, samples, sample_rate, episode_duration_seconds, overlap_intervals, cfg
        )
        if reason is not None:
            db.update_clip_fields(conn, clip["clip_id"], discard_reason=reason)
            discarded += 1
    return discarded
