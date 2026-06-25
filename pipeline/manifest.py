"""JSONL manifest construction + schema validation, per the spec's exact row
shape (FILE_3000.pdf, Output Format section). One row per surviving, uploaded
clip -- a clip with discard_reason set was never exported to flac (quality.py/
audio.py only ever encode clips that pass every filter), and a clip with no
audio_path yet hasn't actually been uploaded by storage.py, so neither belongs
in the dataset manifest. Discard-reason counts/processing totals are
PROCESSING_SUMMARY.md's job (scripts/report.py), not this file's.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from pipeline import db

REQUIRED_TOP_LEVEL_KEYS = (
    "clip_id", "podcast_id", "podcast_title", "episode_id", "episode_title",
    "source_url", "start_seconds", "end_seconds", "duration_seconds",
    "speaker_id", "utterance", "audio_path", "quality_flags",
)
REQUIRED_QUALITY_FLAG_KEYS = ("vad_confidence", "overlap_detected", "music_detected", "discard_reason")


def build_manifest_row(clip_row: sqlite3.Row, episode_row: sqlite3.Row, podcast_row: sqlite3.Row) -> dict:
    return {
        "clip_id": clip_row["clip_id"],
        "podcast_id": clip_row["podcast_id"],
        "podcast_title": podcast_row["title"],
        "episode_id": clip_row["episode_id"],
        "episode_title": episode_row["title"],
        "source_url": episode_row["source_url"],
        "start_seconds": clip_row["start_seconds"],
        "end_seconds": clip_row["end_seconds"],
        "duration_seconds": clip_row["duration_seconds"],
        "speaker_id": clip_row["speaker_id"],
        "utterance": clip_row["utterance"],
        "audio_path": clip_row["audio_path"],
        "quality_flags": {
            "vad_confidence": clip_row["vad_confidence"],
            "overlap_detected": bool(clip_row["overlap_detected"]),
            "music_detected": bool(clip_row["music_detected"]),
            "discard_reason": clip_row["discard_reason"],
        },
    }


def validate_manifest_row(row: dict) -> list[str]:
    """Structural/type validation against the spec's example shape -- empty
    list means valid. Doesn't second-guess pipeline correctness properties
    (e.g. duration <= 30s); that belongs to tests/test_segment.py, not the
    manifest schema."""
    errors = []
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in row:
            errors.append(f"missing key: {key}")
    if errors:
        return errors

    def _is_number(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if not isinstance(row["clip_id"], str) or not row["clip_id"]:
        errors.append("clip_id must be a non-empty string")
    if not isinstance(row["podcast_id"], str) or not row["podcast_id"]:
        errors.append("podcast_id must be a non-empty string")
    if not isinstance(row["episode_id"], str) or not row["episode_id"]:
        errors.append("episode_id must be a non-empty string")
    if not isinstance(row["podcast_title"], str):
        errors.append("podcast_title must be a string")
    if not isinstance(row["episode_title"], str):
        errors.append("episode_title must be a string")
    if not isinstance(row["source_url"], str) or not row["source_url"]:
        errors.append("source_url must be a non-empty string")

    for key in ("start_seconds", "end_seconds", "duration_seconds"):
        if not _is_number(row[key]):
            errors.append(f"{key} must be numeric")
    if _is_number(row["start_seconds"]) and _is_number(row["end_seconds"]) and row["end_seconds"] <= row["start_seconds"]:
        errors.append("end_seconds must be greater than start_seconds")

    if row["speaker_id"] is not None and not isinstance(row["speaker_id"], str):
        errors.append("speaker_id must be a string or null")
    if row["utterance"] is not None and not isinstance(row["utterance"], str):
        errors.append("utterance must be a string or null")
    if not isinstance(row["audio_path"], str) or not row["audio_path"]:
        errors.append("audio_path must be a non-empty string")

    quality_flags = row["quality_flags"]
    if not isinstance(quality_flags, dict):
        errors.append("quality_flags must be an object")
        return errors

    for key in REQUIRED_QUALITY_FLAG_KEYS:
        if key not in quality_flags:
            errors.append(f"quality_flags missing key: {key}")
    if "vad_confidence" in quality_flags and quality_flags["vad_confidence"] is not None \
            and not _is_number(quality_flags["vad_confidence"]):
        errors.append("quality_flags.vad_confidence must be numeric or null")
    for key in ("overlap_detected", "music_detected"):
        if key in quality_flags and not isinstance(quality_flags[key], bool):
            errors.append(f"quality_flags.{key} must be a boolean")
    if "discard_reason" in quality_flags and quality_flags["discard_reason"] is not None \
            and not isinstance(quality_flags["discard_reason"], str):
        errors.append("quality_flags.discard_reason must be a string or null")

    return errors


def iter_manifest_rows(conn: sqlite3.Connection, podcast_id: str | None = None):
    """Yields one manifest dict per surviving, uploaded clip. `podcast_id`
    narrows to one podcast's clips (e.g. for a sample-manifest export);
    omitted, it walks every clip in the db."""
    if podcast_id is not None:
        clips = db.get_clips_for_podcast(conn, podcast_id)
    else:
        clips = conn.execute("SELECT * FROM clips ORDER BY podcast_id, episode_id, start_seconds").fetchall()

    episode_cache: dict[str, sqlite3.Row] = {}
    podcast_cache: dict[str, sqlite3.Row] = {}
    for clip in clips:
        if clip["discard_reason"] is not None or clip["audio_path"] is None:
            continue
        ep_id, pod_id = clip["episode_id"], clip["podcast_id"]
        if ep_id not in episode_cache:
            episode_cache[ep_id] = db.get_episode(conn, ep_id)
        if pod_id not in podcast_cache:
            podcast_cache[pod_id] = db.get_podcast(conn, pod_id)
        yield build_manifest_row(clip, episode_cache[ep_id], podcast_cache[pod_id])


def write_manifest(conn: sqlite3.Connection, output_path: str | Path, podcast_id: str | None = None) -> int:
    """Regenerates `output_path` wholesale from current db state (never
    appended to, so a rerun can't leave stale/duplicate rows behind). Returns
    the number of rows written."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w") as f:
        for row in iter_manifest_rows(conn, podcast_id=podcast_id):
            f.write(json.dumps(row) + "\n")
            count += 1
    return count
