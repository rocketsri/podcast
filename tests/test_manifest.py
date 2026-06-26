"""Tests for pipeline/manifest.py: JSONL manifest row construction +
schema validation + file output. Uses a real in-memory sqlite db so clip/
episode/podcast rows behave exactly like production (sqlite3.Row column
access), no real audio files."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pipeline import db, manifest


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    db.init_db(connection)
    yield connection
    connection.close()


def _seed_podcast_episode(conn, podcast_id="pod1", episode_id="ep1"):
    db.insert_podcast(conn, podcast_id, "feed1", "My Podcast", "https://example.com/feed.xml")
    db.insert_episode(conn, episode_id, podcast_id, "pi-1", "Episode One", "https://example.com/ep1.mp3")


# --- build_manifest_row -----------------------------------------------------------


def _seed_speaker(conn, podcast_id="pod1", episode_id="ep1", speaker_id="pod1_speaker_000"):
    """clips.speaker_id has a FK against speakers(speaker_id) -- a real row
    must exist there before a clip can reference it."""
    db.upsert_speaker(
        conn, speaker_id, podcast_id, local_label_seq=0,
        centroid_embedding=np.zeros(4, dtype=np.float32),
        embedding_count=1, total_speech_seconds=10.0, episode_id=episode_id,
    )


def test_build_manifest_row_has_exactly_the_required_top_level_keys(conn):
    _seed_podcast_episode(conn)
    _seed_speaker(conn)
    db.insert_clip(
        conn, "clip1", "ep1", "pod1", 1.0, 4.5,
        speaker_id="pod1_speaker_000", vad_confidence=0.93,
        overlap_detected=0, music_detected=0, audio_path="/r2/clip1.flac",
    )
    db.update_clip_fields(conn, "clip1", utterance="hello world")

    clip_row = db.get_clips_for_episode(conn, "ep1")[0]
    episode_row = db.get_episode(conn, "ep1")
    podcast_row = db.get_podcast(conn, "pod1")

    row = manifest.build_manifest_row(clip_row, episode_row, podcast_row)

    assert set(row.keys()) == set(manifest.REQUIRED_TOP_LEVEL_KEYS)
    assert set(row["quality_flags"].keys()) == set(manifest.REQUIRED_QUALITY_FLAG_KEYS)


def test_build_manifest_row_field_values(conn):
    _seed_podcast_episode(conn)
    _seed_speaker(conn)
    db.insert_clip(
        conn, "clip1", "ep1", "pod1", 1.0, 4.5,
        speaker_id="pod1_speaker_000", vad_confidence=0.93,
        overlap_detected=1, music_detected=0, audio_path="/r2/clip1.flac",
    )
    db.update_clip_fields(conn, "clip1", utterance="hello world")

    clip_row = db.get_clips_for_episode(conn, "ep1")[0]
    episode_row = db.get_episode(conn, "ep1")
    podcast_row = db.get_podcast(conn, "pod1")
    row = manifest.build_manifest_row(clip_row, episode_row, podcast_row)

    assert row["clip_id"] == "clip1"
    assert row["podcast_id"] == "pod1"
    assert row["podcast_title"] == "My Podcast"
    assert row["episode_id"] == "ep1"
    assert row["episode_title"] == "Episode One"
    assert row["source_url"] == "https://example.com/ep1.mp3"
    assert row["start_seconds"] == 1.0
    assert row["end_seconds"] == 4.5
    assert row["duration_seconds"] == 3.5
    assert row["speaker_id"] == "pod1_speaker_000"
    assert row["utterance"] == "hello world"
    assert row["audio_path"] == "/r2/clip1.flac"
    assert row["quality_flags"]["vad_confidence"] == 0.93
    assert row["quality_flags"]["overlap_detected"] is True
    assert row["quality_flags"]["music_detected"] is False
    assert row["quality_flags"]["discard_reason"] is None


# --- validate_manifest_row --------------------------------------------------------


def _valid_row(**overrides):
    row = {
        "clip_id": "clip1",
        "podcast_id": "pod1",
        "podcast_title": "My Podcast",
        "episode_id": "ep1",
        "episode_title": "Episode One",
        "source_url": "https://example.com/ep1.mp3",
        "start_seconds": 1.0,
        "end_seconds": 4.5,
        "duration_seconds": 3.5,
        "speaker_id": "pod1_speaker_000",
        "utterance": "hello world",
        "audio_path": "/r2/clip1.flac",
        "quality_flags": {
            "vad_confidence": 0.93,
            "overlap_detected": False,
            "music_detected": False,
            "discard_reason": None,
        },
    }
    row.update(overrides)
    return row


def test_validate_manifest_row_valid_row_returns_empty_list():
    assert manifest.validate_manifest_row(_valid_row()) == []


def test_validate_manifest_row_missing_required_key():
    row = _valid_row()
    del row["audio_path"]
    errors = manifest.validate_manifest_row(row)
    assert errors == ["missing key: audio_path"]


def test_validate_manifest_row_missing_multiple_keys_reports_all():
    row = _valid_row()
    del row["audio_path"]
    del row["clip_id"]
    errors = manifest.validate_manifest_row(row)
    assert "missing key: audio_path" in errors
    assert "missing key: clip_id" in errors


def test_validate_manifest_row_wrong_type_clip_id():
    row = _valid_row(clip_id=123)
    errors = manifest.validate_manifest_row(row)
    assert any("clip_id" in e for e in errors)


def test_validate_manifest_row_empty_string_clip_id_invalid():
    row = _valid_row(clip_id="")
    errors = manifest.validate_manifest_row(row)
    assert any("clip_id" in e for e in errors)


def test_validate_manifest_row_wrong_type_start_seconds():
    row = _valid_row(start_seconds="1.0")
    errors = manifest.validate_manifest_row(row)
    assert any("start_seconds" in e for e in errors)


def test_validate_manifest_row_bool_not_accepted_as_numeric():
    """bool is technically an int subclass in Python -- validator must
    explicitly reject it as a numeric value."""
    row = _valid_row(start_seconds=True)
    errors = manifest.validate_manifest_row(row)
    assert any("start_seconds" in e for e in errors)


def test_validate_manifest_row_end_seconds_equal_to_start_seconds_invalid():
    row = _valid_row(start_seconds=5.0, end_seconds=5.0)
    errors = manifest.validate_manifest_row(row)
    assert "end_seconds must be greater than start_seconds" in errors


def test_validate_manifest_row_end_seconds_less_than_start_seconds_invalid():
    row = _valid_row(start_seconds=5.0, end_seconds=2.0)
    errors = manifest.validate_manifest_row(row)
    assert "end_seconds must be greater than start_seconds" in errors


def test_validate_manifest_row_end_seconds_greater_than_start_seconds_valid():
    row = _valid_row(start_seconds=2.0, end_seconds=5.0, duration_seconds=3.0)
    assert manifest.validate_manifest_row(row) == []


def test_validate_manifest_row_speaker_id_none_is_valid():
    row = _valid_row(speaker_id=None)
    assert manifest.validate_manifest_row(row) == []


def test_validate_manifest_row_speaker_id_wrong_type_invalid():
    row = _valid_row(speaker_id=42)
    errors = manifest.validate_manifest_row(row)
    assert any("speaker_id" in e for e in errors)


def test_validate_manifest_row_audio_path_empty_invalid():
    row = _valid_row(audio_path="")
    errors = manifest.validate_manifest_row(row)
    assert any("audio_path" in e for e in errors)


def test_validate_manifest_row_quality_flags_not_dict_invalid():
    row = _valid_row(quality_flags="not a dict")
    errors = manifest.validate_manifest_row(row)
    assert "quality_flags must be an object" in errors


def test_validate_manifest_row_quality_flags_missing_key():
    row = _valid_row()
    del row["quality_flags"]["discard_reason"]
    errors = manifest.validate_manifest_row(row)
    assert "quality_flags missing key: discard_reason" in errors


def test_validate_manifest_row_quality_flags_wrong_type_overlap_detected():
    row = _valid_row()
    row["quality_flags"]["overlap_detected"] = "yes"
    errors = manifest.validate_manifest_row(row)
    assert any("overlap_detected" in e for e in errors)


def test_validate_manifest_row_quality_flags_vad_confidence_none_is_valid():
    row = _valid_row()
    row["quality_flags"]["vad_confidence"] = None
    assert manifest.validate_manifest_row(row) == []


def test_validate_manifest_row_quality_flags_discard_reason_string_is_valid():
    row = _valid_row()
    row["quality_flags"]["discard_reason"] = "too_short"
    assert manifest.validate_manifest_row(row) == []


# --- iter_manifest_rows ------------------------------------------------------------


def test_iter_manifest_rows_skips_clips_with_discard_reason(conn):
    _seed_podcast_episode(conn)
    db.insert_clip(conn, "clip_keep", "ep1", "pod1", 0.0, 5.0, audio_path="/r2/keep.flac")
    db.insert_clip(conn, "clip_discarded", "ep1", "pod1", 5.0, 10.0, audio_path="/r2/discarded.flac",
                   discard_reason="too_short")

    rows = list(manifest.iter_manifest_rows(conn))
    clip_ids = {r["clip_id"] for r in rows}
    assert clip_ids == {"clip_keep"}


def test_iter_manifest_rows_skips_clips_with_audio_path_none(conn):
    _seed_podcast_episode(conn)
    db.insert_clip(conn, "clip_uploaded", "ep1", "pod1", 0.0, 5.0, audio_path="/r2/keep.flac")
    db.insert_clip(conn, "clip_not_uploaded", "ep1", "pod1", 5.0, 10.0)  # audio_path defaults to NULL

    rows = list(manifest.iter_manifest_rows(conn))
    clip_ids = {r["clip_id"] for r in rows}
    assert clip_ids == {"clip_uploaded"}


def test_iter_manifest_rows_filters_by_podcast_id(conn):
    _seed_podcast_episode(conn, podcast_id="pod1", episode_id="ep1")
    _seed_podcast_episode(conn, podcast_id="pod2", episode_id="ep2")
    db.insert_clip(conn, "clip_pod1", "ep1", "pod1", 0.0, 5.0, audio_path="/r2/p1.flac")
    db.insert_clip(conn, "clip_pod2", "ep2", "pod2", 0.0, 5.0, audio_path="/r2/p2.flac")

    rows = list(manifest.iter_manifest_rows(conn, podcast_id="pod1"))
    clip_ids = {r["clip_id"] for r in rows}
    assert clip_ids == {"clip_pod1"}


def test_iter_manifest_rows_empty_db_yields_nothing(conn):
    assert list(manifest.iter_manifest_rows(conn)) == []


# --- write_manifest -----------------------------------------------------------------


def test_write_manifest_writes_one_json_line_per_surviving_clip(conn, tmp_path):
    _seed_podcast_episode(conn)
    db.insert_clip(conn, "clip1", "ep1", "pod1", 0.0, 5.0, audio_path="/r2/clip1.flac")
    db.insert_clip(conn, "clip2", "ep1", "pod1", 5.0, 10.0, audio_path="/r2/clip2.flac")
    db.insert_clip(conn, "clip_discarded", "ep1", "pod1", 10.0, 15.0, discard_reason="too_short")

    output_path = tmp_path / "manifest.jsonl"
    count = manifest.write_manifest(conn, output_path)

    assert count == 2
    lines = output_path.read_text().strip().split("\n")
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    clip_ids = {row["clip_id"] for row in parsed}
    assert clip_ids == {"clip1", "clip2"}
    for row in parsed:
        assert manifest.validate_manifest_row(row) == []


def test_write_manifest_is_idempotent_on_rerun(conn, tmp_path):
    """Rewriting the manifest must regenerate wholesale, not append --
    rerunning must produce the same row count and content, not duplicates."""
    _seed_podcast_episode(conn)
    db.insert_clip(conn, "clip1", "ep1", "pod1", 0.0, 5.0, audio_path="/r2/clip1.flac")

    output_path = tmp_path / "manifest.jsonl"
    count1 = manifest.write_manifest(conn, output_path)
    content1 = output_path.read_text()

    count2 = manifest.write_manifest(conn, output_path)
    content2 = output_path.read_text()

    assert count1 == count2 == 1
    assert content1 == content2
    lines = content2.strip().split("\n")
    assert len(lines) == 1


def test_write_manifest_creates_parent_directories(conn, tmp_path):
    _seed_podcast_episode(conn)
    db.insert_clip(conn, "clip1", "ep1", "pod1", 0.0, 5.0, audio_path="/r2/clip1.flac")

    output_path = tmp_path / "nested" / "dir" / "manifest.jsonl"
    count = manifest.write_manifest(conn, output_path)
    assert count == 1
    assert output_path.exists()


def test_write_manifest_empty_db_writes_empty_file(conn, tmp_path):
    output_path = tmp_path / "manifest.jsonl"
    count = manifest.write_manifest(conn, output_path)
    assert count == 0
    assert output_path.read_text() == ""
