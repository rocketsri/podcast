"""Tests for pipeline/quality.py: content-based discard-reason heuristics.
Exercises `evaluate_clip_discard_reason` (the pure per-clip decision
function) directly with synthetic clip rows + synthetic sample arrays --
no real audio files, no db needed for the row (a sqlite3.Row-like mapping
suffices since the function only ever indexes it by column name)."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline import quality


def make_cfg(**overrides):
    """Mirrors config/pipeline.yaml's `quality:` section defaults."""
    defaults = dict(
        vad_low_confidence_floor=0.6,
        rms_silence_floor_db=-45.0,
        overlap_edge_trim_seconds=0.2,
        music_spectral_flatness_floor=0.35,
        low_asr_confidence_no_speech_prob=0.6,
        low_asr_confidence_avg_logprob=-1.2,
        ad_heuristic_spectral_flatness_floor=0.20,
        ad_heuristic_no_speech_prob_floor=0.35,
        intro_outro_window_seconds=30.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_clip_row(
    start_seconds=0.0,
    end_seconds=1.0,
    vad_confidence=0.9,
    no_speech_prob=0.05,
    avg_logprob=-0.2,
):
    """A real sqlite3.Row built off an actual connection, so attribute access
    via clip_row["col"] behaves identically to what evaluate_clip_discard_reason
    receives in production (db.get_clips_for_episode rows)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE clips (start_seconds REAL, end_seconds REAL, vad_confidence REAL, "
        "no_speech_prob REAL, avg_logprob REAL)"
    )
    conn.execute(
        "INSERT INTO clips VALUES (?, ?, ?, ?, ?)",
        (start_seconds, end_seconds, vad_confidence, no_speech_prob, avg_logprob),
    )
    row = conn.execute("SELECT * FROM clips").fetchone()
    conn.close()
    return row


def speech_like_samples(n_samples=16000, amplitude=0.3, seed=0):
    """A synthetic, non-flat, energetic signal that should read as normal
    speech: a dominant fundamental with fast-decaying harmonics (a
    harmonic-peak-dominated spectrum, well below the music/ad spectral
    flatness floors -- see module docstring's rationale for flatness as a
    music/production signal) plus light noise, well above the RMS silence
    floor."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / 16000.0
    signal = (
        amplitude * np.sin(2 * np.pi * 150 * t)
        + amplitude * 0.15 * np.sin(2 * np.pi * 300 * t)
        + amplitude * 0.05 * np.sin(2 * np.pi * 450 * t)
    )
    signal += rng.normal(0, 0.001, size=n_samples)
    return signal.astype(np.float32)


def silent_samples(n_samples=16000):
    return np.zeros(n_samples, dtype=np.float32)


def white_noise_samples(n_samples=16000, amplitude=0.3, seed=1):
    """Flat-spectrum white noise: high spectral flatness, used as a
    music/production-signal stand-in (per quality.py's own heuristic
    rationale -- broadband/noise-like energy reads as non-speech)."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(-amplitude, amplitude, size=n_samples)).astype(np.float32)


SAMPLE_RATE = 16000


# --- vad_low_confidence -----------------------------------------------------------


def test_vad_low_confidence_triggers_below_floor():
    cfg = make_cfg()
    row = make_clip_row(vad_confidence=0.5)  # below 0.6 floor
    reason = quality.evaluate_clip_discard_reason(
        row, speech_like_samples(), SAMPLE_RATE, 1000.0, [], cfg
    )
    assert reason == "vad_low_confidence"


def test_vad_low_confidence_does_not_trigger_above_floor():
    cfg = make_cfg()
    row = make_clip_row(vad_confidence=0.95)
    reason = quality.evaluate_clip_discard_reason(
        row, speech_like_samples(), SAMPLE_RATE, 1000.0, [], cfg
    )
    assert reason is None


def test_vad_low_confidence_none_value_does_not_trigger():
    """vad_confidence=None (e.g. clip had no overlapping VAD segment) must
    not crash or trigger the floor check -- it's skipped."""
    cfg = make_cfg()
    row = make_clip_row(vad_confidence=None)
    reason = quality.evaluate_clip_discard_reason(
        row, speech_like_samples(), SAMPLE_RATE, 1000.0, [], cfg
    )
    assert reason is None


# --- silence_or_low_energy --------------------------------------------------------


def test_silence_or_low_energy_triggers_on_silence():
    cfg = make_cfg()
    row = make_clip_row(vad_confidence=0.9)
    reason = quality.evaluate_clip_discard_reason(
        row, silent_samples(), SAMPLE_RATE, 1000.0, [], cfg
    )
    assert reason == "silence_or_low_energy"


def test_silence_or_low_energy_does_not_trigger_on_normal_speech():
    cfg = make_cfg()
    row = make_clip_row(vad_confidence=0.9)
    reason = quality.evaluate_clip_discard_reason(
        row, speech_like_samples(amplitude=0.3), SAMPLE_RATE, 1000.0, [], cfg
    )
    assert reason is None


def test_rms_dbfs_helper_floor_value_for_silence():
    assert quality._rms_dbfs(silent_samples()) == -120.0


def test_rms_dbfs_helper_above_floor_for_loud_signal():
    loud = np.full(1000, 0.9, dtype=np.float32)
    assert quality._rms_dbfs(loud) > -45.0


# --- overlap_detected --------------------------------------------------------------


def test_overlap_detected_triggers_within_edge_trim_distance():
    cfg = make_cfg()
    # Clip is [100, 105); overlap region [105.1, 110) is only 0.1s past the
    # clip's end -- inside the 0.2s overlap_edge_trim_seconds tolerance.
    row = make_clip_row(start_seconds=100.0, end_seconds=105.0, vad_confidence=0.9)
    overlap_intervals = [(105.1, 110.0)]
    samples = speech_like_samples(n_samples=110 * SAMPLE_RATE)
    reason = quality.evaluate_clip_discard_reason(
        row, samples, SAMPLE_RATE, 1000.0, overlap_intervals, cfg
    )
    assert reason == "overlap_detected"


def test_overlap_detected_does_not_trigger_when_far_from_overlap():
    cfg = make_cfg()
    row = make_clip_row(start_seconds=100.0, end_seconds=105.0, vad_confidence=0.9)
    overlap_intervals = [(200.0, 210.0)]  # far away
    samples = speech_like_samples(n_samples=210 * SAMPLE_RATE)
    reason = quality.evaluate_clip_discard_reason(
        row, samples, SAMPLE_RATE, 1000.0, overlap_intervals, cfg
    )
    assert reason is None


def test_overlap_edge_distance_zero_when_actually_overlapping():
    assert quality._overlap_edge_distance(100.0, 105.0, [(102.0, 108.0)]) == 0.0


def test_overlap_edge_distance_infinite_when_no_overlap_intervals():
    assert quality._overlap_edge_distance(100.0, 105.0, []) == float("inf")


# --- low_asr_confidence -------------------------------------------------------------


def test_low_asr_confidence_triggers_when_both_signals_bad():
    cfg = make_cfg()
    row = make_clip_row(vad_confidence=0.9, no_speech_prob=0.8, avg_logprob=-2.0)
    reason = quality.evaluate_clip_discard_reason(
        row, speech_like_samples(), SAMPLE_RATE, 1000.0, [], cfg
    )
    assert reason == "low_asr_confidence"


def test_low_asr_confidence_does_not_trigger_when_only_no_speech_prob_bad():
    """Neither signal alone is trusted as a sole trigger (per the module
    docstring) -- a bad no_speech_prob with a fine avg_logprob must not
    discard."""
    cfg = make_cfg()
    row = make_clip_row(vad_confidence=0.9, no_speech_prob=0.9, avg_logprob=-0.1)
    reason = quality.evaluate_clip_discard_reason(
        row, speech_like_samples(), SAMPLE_RATE, 1000.0, [], cfg
    )
    assert reason != "low_asr_confidence"


def test_low_asr_confidence_does_not_trigger_when_only_avg_logprob_bad():
    cfg = make_cfg()
    row = make_clip_row(vad_confidence=0.9, no_speech_prob=0.05, avg_logprob=-3.0)
    reason = quality.evaluate_clip_discard_reason(
        row, speech_like_samples(), SAMPLE_RATE, 1000.0, [], cfg
    )
    assert reason != "low_asr_confidence"


def test_low_asr_confidence_none_values_skip_check():
    cfg = make_cfg()
    row = make_clip_row(vad_confidence=0.9, no_speech_prob=None, avg_logprob=None)
    reason = quality.evaluate_clip_discard_reason(
        row, speech_like_samples(), SAMPLE_RATE, 1000.0, [], cfg
    )
    assert reason is None


# --- music_detected -------------------------------------------------------------------


def test_music_detected_triggers_on_flat_spectrum_and_high_no_speech_prob():
    cfg = make_cfg()
    row = make_clip_row(vad_confidence=0.9, no_speech_prob=0.9, avg_logprob=-0.1)
    reason = quality.evaluate_clip_discard_reason(
        row, white_noise_samples(amplitude=0.5), SAMPLE_RATE, 1000.0, [], cfg
    )
    assert reason == "music_detected"


def test_music_detected_does_not_trigger_with_low_no_speech_prob():
    """Flat spectrum alone isn't enough -- must be corroborated by a high
    no_speech_prob (per quality.py's priority-ordered taxonomy)."""
    cfg = make_cfg()
    row = make_clip_row(vad_confidence=0.9, no_speech_prob=0.05, avg_logprob=-0.1)
    reason = quality.evaluate_clip_discard_reason(
        row, white_noise_samples(amplitude=0.5), SAMPLE_RATE, 1000.0, [], cfg
    )
    assert reason != "music_detected"


def test_spectral_flatness_higher_for_noise_than_for_tonal_signal():
    noise_flatness = quality._spectral_flatness(white_noise_samples(amplitude=0.5))
    tonal_flatness = quality._spectral_flatness(speech_like_samples(amplitude=0.5))
    assert noise_flatness > tonal_flatness


def test_spectral_flatness_zero_for_too_few_samples():
    assert quality._spectral_flatness(np.array([0.1, 0.2], dtype=np.float32)) == 0.0


# --- intro_outro_position -------------------------------------------------------------


def test_intro_outro_position_triggers_near_episode_start():
    cfg = make_cfg()
    # Clip starts at 5s, well within the 30s intro_outro_window_seconds.
    row = make_clip_row(start_seconds=5.0, end_seconds=10.0, vad_confidence=0.9, no_speech_prob=0.5, avg_logprob=-0.1)
    samples = white_noise_samples(n_samples=1000 * SAMPLE_RATE, amplitude=0.5)
    reason = quality.evaluate_clip_discard_reason(
        row, samples, SAMPLE_RATE, 1000.0, [], cfg
    )
    assert reason == "intro_outro_position"


def test_intro_outro_position_triggers_near_episode_end():
    cfg = make_cfg()
    episode_duration = 1000.0
    # Clip ends at 995s, 5s from the end -- within the 30s window.
    row = make_clip_row(start_seconds=990.0, end_seconds=995.0, vad_confidence=0.9, no_speech_prob=0.5, avg_logprob=-0.1)
    samples = white_noise_samples(n_samples=1000 * SAMPLE_RATE, amplitude=0.5)
    reason = quality.evaluate_clip_discard_reason(
        row, samples, SAMPLE_RATE, episode_duration, [], cfg
    )
    assert reason == "intro_outro_position"


def test_intro_outro_position_does_not_trigger_mid_episode():
    """Same audio signal/ASR signals as the triggering case, but the clip is
    in the middle of the episode -- should fall through to
    ad_segment_heuristic instead (still flatness+no_speech-driven, just a
    different position-based label) since the no_speech_prob (0.5) clears
    ad_heuristic_no_speech_prob_floor (0.35) even though it doesn't clear
    low_asr_confidence_no_speech_prob (0.6)."""
    cfg = make_cfg()
    episode_duration = 1000.0
    row = make_clip_row(start_seconds=500.0, end_seconds=505.0, vad_confidence=0.9, no_speech_prob=0.5, avg_logprob=-0.1)
    samples = white_noise_samples(n_samples=1000 * SAMPLE_RATE, amplitude=0.5)
    reason = quality.evaluate_clip_discard_reason(
        row, samples, SAMPLE_RATE, episode_duration, [], cfg
    )
    assert reason != "intro_outro_position"
    assert reason == "ad_segment_heuristic"


# --- ad_segment_heuristic -------------------------------------------------------------


def test_ad_segment_heuristic_triggers_mid_episode_with_weaker_thresholds():
    cfg = make_cfg()
    episode_duration = 1000.0
    row = make_clip_row(start_seconds=500.0, end_seconds=505.0, vad_confidence=0.9, no_speech_prob=0.4, avg_logprob=-0.1)
    samples = white_noise_samples(n_samples=1000 * SAMPLE_RATE, amplitude=0.5)
    reason = quality.evaluate_clip_discard_reason(
        row, samples, SAMPLE_RATE, episode_duration, [], cfg
    )
    assert reason == "ad_segment_heuristic"


def test_ad_segment_heuristic_does_not_trigger_for_clean_speech_mid_episode():
    cfg = make_cfg()
    episode_duration = 1000.0
    row = make_clip_row(start_seconds=500.0, end_seconds=505.0, vad_confidence=0.9, no_speech_prob=0.05, avg_logprob=-0.1)
    samples = speech_like_samples(n_samples=1000 * SAMPLE_RATE, amplitude=0.3)
    reason = quality.evaluate_clip_discard_reason(
        row, samples, SAMPLE_RATE, episode_duration, [], cfg
    )
    assert reason is None


# --- priority order: first match wins -------------------------------------------------


def test_priority_order_vad_low_confidence_beats_everything_else():
    """A clip that would also qualify for silence_or_low_energy must still
    report vad_low_confidence since that check runs first."""
    cfg = make_cfg()
    row = make_clip_row(vad_confidence=0.1, no_speech_prob=0.9, avg_logprob=-3.0)
    reason = quality.evaluate_clip_discard_reason(
        row, silent_samples(), SAMPLE_RATE, 1000.0, [(0.0, 0.0)], cfg
    )
    assert reason == "vad_low_confidence"


def test_priority_order_silence_beats_overlap_and_asr():
    cfg = make_cfg()
    row = make_clip_row(
        start_seconds=100.0, end_seconds=105.0, vad_confidence=0.9, no_speech_prob=0.9, avg_logprob=-3.0
    )
    overlap_intervals = [(105.1, 110.0)]
    reason = quality.evaluate_clip_discard_reason(
        row, silent_samples(), SAMPLE_RATE, 1000.0, overlap_intervals, cfg
    )
    assert reason == "silence_or_low_energy"


# --- apply_quality_filters (db-integration smoke test) --------------------------------


def test_apply_quality_filters_updates_only_undiscarded_clips():
    from pipeline import db

    conn = db.connect(":memory:")
    db.init_db(conn)
    db.insert_podcast(conn, "pod1", "feed1", "P", "https://x/feed.xml")
    db.insert_episode(conn, "ep1", "pod1", "pi-1", "E1", "https://x/e1.mp3")

    # Episode is 100s total so clip3, sitting in the middle, is far from
    # both edges (intro_outro_window_seconds=30.0 default) -- otherwise its
    # near-episode-edge position alone could trip an unrelated heuristic.
    # Clip 1: silent -> should get discarded by this pass.
    db.insert_clip(conn, "clip1", "ep1", "pod1", 0.0, 1.0, vad_confidence=0.9)
    # Clip 2: already discarded upstream (too_short) -> must be left alone.
    db.insert_clip(conn, "clip2", "ep1", "pod1", 1.0, 1.5, vad_confidence=0.9, discard_reason="too_short")
    # Clip 3: clean speech, mid-episode -> should remain undiscarded.
    db.insert_clip(conn, "clip3", "ep1", "pod1", 50.0, 51.0, vad_confidence=0.9)

    cfg = make_cfg()
    samples = np.zeros(100 * SAMPLE_RATE, dtype=np.float32)
    samples[: SAMPLE_RATE] = silent_samples(SAMPLE_RATE)  # [0, 1) covers clip1
    samples[50 * SAMPLE_RATE : 51 * SAMPLE_RATE] = speech_like_samples(SAMPLE_RATE, amplitude=0.3)  # clip3

    discarded_count = quality.apply_quality_filters(conn, "ep1", samples, SAMPLE_RATE, cfg)

    clips = db.get_clips_for_episode(conn, "ep1")
    by_id = {c["clip_id"]: c for c in clips}
    assert by_id["clip1"]["discard_reason"] == "silence_or_low_energy"
    assert by_id["clip2"]["discard_reason"] == "too_short"  # untouched
    assert by_id["clip3"]["discard_reason"] is None
    assert discarded_count == 1
