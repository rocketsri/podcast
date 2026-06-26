"""SQLite schema + state-machine helpers for resumable pipeline checkpointing.

Single-writer, WAL-mode database at work/pipeline.db. Every episode moves
through an ordered `stage` state machine; `is_at_or_past` lets each
pipeline_runner.run_<stage>() function no-op on rerun instead of redoing
work, and `mark_stage_failed`/`resume_stage` let a retry pick up from the
failure point rather than from `queued`.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path

import numpy as np

DEFAULT_DB_PATH = Path("work/pipeline.db")

# Ordered per-episode state machine (see plan's Database schema section).
EPISODE_STAGES = (
    "queued",
    "downloading",
    "downloaded",
    "transcoding",
    "transcoded",
    "vad_running",
    "vad_done",
    "diarizing",
    "diarized",
    "clustering_done",
    "segmenting",
    "segmented",
    "asr_running",
    "asr_done",
    "quality_filtering",
    "exporting",
    "uploading",
    "done",
)
_STAGE_INDEX = {stage: i for i, stage in enumerate(EPISODE_STAGES)}
FAILED_STAGE = "failed"

COST_CATEGORIES = ("gpu_compute", "r2_storage", "r2_class_a_ops", "r2_class_b_ops", "egress", "other")


class StageError(ValueError):
    pass


def stage_index(stage: str) -> int:
    try:
        return _STAGE_INDEX[stage]
    except KeyError:
        raise StageError(f"unknown stage: {stage}") from None


def is_at_or_past(current_stage: str, target_stage: str) -> bool:
    """True once `current_stage` has reached or passed `target_stage`.

    A `failed` episode is never considered past anything — callers should
    resume from `resume_stage(episode_row)` instead, not from `stage` itself.
    """
    if current_stage == FAILED_STAGE:
        return False
    return stage_index(current_stage) >= stage_index(target_stage)


def resume_stage(episode_row: sqlite3.Row) -> str:
    """The stage a retry should resume from: `failed_stage` if the episode
    failed, otherwise its current `stage`."""
    if episode_row["stage"] == FAILED_STAGE:
        return episode_row["failed_stage"]
    return episode_row["stage"]


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def pack_embedding(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def unpack_embedding(blob: bytes, dim: int | None = None) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float32)
    if dim is not None:
        arr = arr.reshape(dim)
    return arr


_SCHEMA = """
CREATE TABLE IF NOT EXISTS podcasts (
    podcast_id TEXT PRIMARY KEY,
    feed_url TEXT NOT NULL,
    title TEXT,
    source_url TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    podcast_id TEXT NOT NULL REFERENCES podcasts(podcast_id),
    podcastindex_episode_id TEXT,
    title TEXT,
    source_url TEXT NOT NULL,
    duration_seconds_reported REAL,
    duration_seconds_actual REAL,
    raw_seconds REAL,
    usable_seconds REAL,
    stage TEXT NOT NULL DEFAULT 'queued',
    failed_stage TEXT,
    last_error TEXT,
    assigned_shard INTEGER,
    local_raw_path TEXT,
    local_wav_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_queue ON episodes(assigned_shard, stage);

CREATE TABLE IF NOT EXISTS speakers (
    speaker_id TEXT PRIMARY KEY,
    podcast_id TEXT NOT NULL REFERENCES podcasts(podcast_id),
    centroid_embedding BLOB,
    embedding_dim INTEGER,
    num_segments INTEGER NOT NULL DEFAULT 0,
    total_seconds REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS local_speaker_segments (
    segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
    local_label TEXT NOT NULL,
    start_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL,
    embedding BLOB,
    embedding_dim INTEGER,
    resolved_speaker_id TEXT
);

CREATE TABLE IF NOT EXISTS clips (
    clip_id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
    podcast_id TEXT NOT NULL REFERENCES podcasts(podcast_id),
    speaker_id TEXT,
    start_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    transcript TEXT,
    asr_avg_logprob REAL,
    discard_reason TEXT,
    local_flac_path TEXT,
    audio_path TEXT,
    uploaded INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clips_upload_queue ON clips(uploaded, discard_reason);

CREATE TABLE IF NOT EXISTS cost_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    amount_usd REAL NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def insert_podcast(
    conn: sqlite3.Connection, podcast_id: str, feed_url: str, title: str | None = None, source_url: str | None = None,
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO podcasts (podcast_id, feed_url, title, source_url, created_at) VALUES (?, ?, ?, ?, ?)",
        (podcast_id, feed_url, title, source_url, now_iso()),
    )
    conn.commit()


def get_podcast(conn: sqlite3.Connection, podcast_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM podcasts WHERE podcast_id = ?", (podcast_id,)).fetchone()


def insert_episode(
    conn: sqlite3.Connection,
    episode_id: str,
    podcast_id: str,
    podcastindex_episode_id: str | None,
    title: str | None,
    source_url: str,
    duration_seconds_reported: float | None = None,
) -> None:
    now = now_iso()
    conn.execute(
        "INSERT OR IGNORE INTO episodes"
        " (episode_id, podcast_id, podcastindex_episode_id, title, source_url, duration_seconds_reported,"
        "  stage, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?)",
        (episode_id, podcast_id, podcastindex_episode_id, title, source_url, duration_seconds_reported, now, now),
    )
    conn.commit()


def get_episode(conn: sqlite3.Connection, episode_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)).fetchone()


def list_queued_episodes(conn: sqlite3.Connection, shard: int | None = None) -> list[sqlite3.Row]:
    """Episodes ready to claim, shortest-reported-duration first (NULLs
    last) rather than insertion order. select_podcasts_free.py picks
    podcasts longest-average-duration-first to hit the raw-hours target
    efficiently, which would otherwise make a pod's very first episode one
    of the longest available -- the worst case for getting a fast
    correctness signal and some usable hours banked early in a 24h run.
    Sorting the claim order (not the selection set) gets short episodes
    processed first without changing which episodes get queued.
    `shard=None` means the single-pod/Stage-1 case (assigned_shard IS
    NULL); otherwise filters to that pod's own shard."""
    if shard is None:
        query = (
            "SELECT * FROM episodes WHERE stage = 'queued' AND assigned_shard IS NULL "
            "ORDER BY duration_seconds_reported IS NULL, duration_seconds_reported ASC"
        )
        return conn.execute(query).fetchall()
    return conn.execute(
        "SELECT * FROM episodes WHERE stage = 'queued' AND assigned_shard = ? "
        "ORDER BY duration_seconds_reported IS NULL, duration_seconds_reported ASC",
        (shard,),
    ).fetchall()


def list_failed_episodes(conn: sqlite3.Connection, shard: int | None = None) -> list[sqlite3.Row]:
    if shard is None:
        return conn.execute(
            "SELECT * FROM episodes WHERE stage = 'failed' AND assigned_shard IS NULL ORDER BY episode_id"
        ).fetchall()
    return conn.execute(
        "SELECT * FROM episodes WHERE stage = 'failed' AND assigned_shard = ? ORDER BY episode_id",
        (shard,),
    ).fetchall()


def list_stalled_episodes(conn: sqlite3.Connection, shard: int | None = None) -> list[sqlite3.Row]:
    """Episodes whose process was killed mid-stage (pod crash/restart, OOM,
    SIGKILL) with no Python exception ever raised -- so mark_stage_failed
    never ran and the episode is stuck at an arbitrary intermediate stage
    that neither list_queued_episodes nor list_failed_episodes will match.
    process_episode's per-stage skip-guards (is_at_or_past/resume_stage) are
    already safe to resume from any of these stages; the gap was purely in
    what run_queue claimed at startup."""
    incomplete_stages = [s for s in EPISODE_STAGES if s not in ("queued", "done")]
    placeholders = ",".join("?" for _ in incomplete_stages)
    if shard is None:
        query = f"SELECT * FROM episodes WHERE stage IN ({placeholders}) AND assigned_shard IS NULL ORDER BY episode_id"
        return conn.execute(query, incomplete_stages).fetchall()
    query = f"SELECT * FROM episodes WHERE stage IN ({placeholders}) AND assigned_shard = ? ORDER BY episode_id"
    return conn.execute(query, (*incomplete_stages, shard)).fetchall()


def set_assigned_shard(conn: sqlite3.Connection, episode_id: str, shard: int) -> None:
    conn.execute(
        "UPDATE episodes SET assigned_shard = ?, updated_at = ? WHERE episode_id = ?",
        (shard, now_iso(), episode_id),
    )
    conn.commit()


def advance_stage(conn: sqlite3.Connection, episode_id: str, new_stage: str, **fields) -> None:
    stage_index(new_stage)  # validates
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values())
    if set_clause:
        conn.execute(
            f"UPDATE episodes SET stage = ?, updated_at = ?, {set_clause} WHERE episode_id = ?",
            (new_stage, now_iso(), *values, episode_id),
        )
    else:
        conn.execute(
            "UPDATE episodes SET stage = ?, updated_at = ? WHERE episode_id = ?",
            (new_stage, now_iso(), episode_id),
        )
    conn.commit()


def mark_stage_failed(conn: sqlite3.Connection, episode_id: str, failed_stage: str, error: str) -> None:
    stage_index(failed_stage)  # validates
    conn.execute(
        "UPDATE episodes SET stage = ?, failed_stage = ?, last_error = ?, updated_at = ? WHERE episode_id = ?",
        (FAILED_STAGE, failed_stage, error, now_iso(), episode_id),
    )
    conn.commit()


def insert_local_speaker_segment(
    conn: sqlite3.Connection,
    episode_id: str,
    local_label: str,
    start_seconds: float,
    end_seconds: float,
    embedding: np.ndarray | None = None,
    resolved_speaker_id: str | None = None,
) -> int:
    embedding_blob = pack_embedding(embedding) if embedding is not None else None
    embedding_dim = int(embedding.shape[0]) if embedding is not None else None
    cur = conn.execute(
        "INSERT INTO local_speaker_segments"
        " (episode_id, local_label, start_seconds, end_seconds, embedding, embedding_dim, resolved_speaker_id)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (episode_id, local_label, start_seconds, end_seconds, embedding_blob, embedding_dim, resolved_speaker_id),
    )
    conn.commit()
    return cur.lastrowid


def get_local_speaker_segments_for_podcast(conn: sqlite3.Connection, podcast_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT lss.* FROM local_speaker_segments lss"
        " JOIN episodes e ON e.episode_id = lss.episode_id"
        " WHERE e.podcast_id = ?"
        " ORDER BY lss.segment_id",
        (podcast_id,),
    ).fetchall()


def get_local_speaker_segments_for_episode(conn: sqlite3.Connection, episode_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM local_speaker_segments WHERE episode_id = ? ORDER BY segment_id",
        (episode_id,),
    ).fetchall()


def set_segment_resolved_speaker(conn: sqlite3.Connection, segment_id: int, speaker_id: str | None) -> None:
    conn.execute(
        "UPDATE local_speaker_segments SET resolved_speaker_id = ? WHERE segment_id = ?",
        (speaker_id, segment_id),
    )
    conn.commit()


def upsert_speaker(
    conn: sqlite3.Connection,
    speaker_id: str,
    podcast_id: str,
    centroid_embedding: np.ndarray,
    num_segments: int,
    total_seconds: float,
) -> None:
    embedding_blob = pack_embedding(centroid_embedding)
    embedding_dim = int(centroid_embedding.shape[0])
    now = now_iso()
    conn.execute(
        "INSERT INTO speakers (speaker_id, podcast_id, centroid_embedding, embedding_dim, num_segments, total_seconds, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(speaker_id) DO UPDATE SET"
        "   centroid_embedding = excluded.centroid_embedding,"
        "   embedding_dim = excluded.embedding_dim,"
        "   num_segments = excluded.num_segments,"
        "   total_seconds = excluded.total_seconds,"
        "   updated_at = excluded.updated_at",
        (speaker_id, podcast_id, embedding_blob, embedding_dim, num_segments, total_seconds, now, now),
    )
    conn.commit()


def get_speakers_for_podcast(conn: sqlite3.Connection, podcast_id: str) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM speakers WHERE podcast_id = ? ORDER BY speaker_id", (podcast_id,)).fetchall()


def delete_speakers_for_podcast(conn: sqlite3.Connection, podcast_id: str) -> None:
    conn.execute("DELETE FROM speakers WHERE podcast_id = ?", (podcast_id,))
    conn.commit()


def insert_clip(
    conn: sqlite3.Connection,
    clip_id: str,
    episode_id: str,
    podcast_id: str,
    start_seconds: float,
    end_seconds: float,
    speaker_id: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO clips (clip_id, episode_id, podcast_id, speaker_id, start_seconds, end_seconds, duration_seconds, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (clip_id, episode_id, podcast_id, speaker_id, start_seconds, end_seconds, end_seconds - start_seconds, now_iso()),
    )
    conn.commit()


def get_clips_for_podcast(conn: sqlite3.Connection, podcast_id: str) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM clips WHERE podcast_id = ? ORDER BY clip_id", (podcast_id,)).fetchall()


def get_clips_for_episode(conn: sqlite3.Connection, episode_id: str) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM clips WHERE episode_id = ? ORDER BY clip_id", (episode_id,)).fetchall()


def get_clips_pending_upload(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM clips WHERE uploaded = 0 AND discard_reason IS NULL ORDER BY clip_id"
    ).fetchall()


def mark_clip_uploaded(conn: sqlite3.Connection, clip_id: str, audio_path: str) -> None:
    conn.execute("UPDATE clips SET uploaded = 1, audio_path = ? WHERE clip_id = ?", (audio_path, clip_id))
    conn.commit()


def update_clip_speaker(conn: sqlite3.Connection, clip_id: str, speaker_id: str | None) -> None:
    conn.execute("UPDATE clips SET speaker_id = ? WHERE clip_id = ?", (speaker_id, clip_id))
    conn.commit()


def update_clip_fields(conn: sqlite3.Connection, clip_id: str, **fields) -> None:
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(
        f"UPDATE clips SET {set_clause} WHERE clip_id = ?",
        (*fields.values(), clip_id),
    )
    conn.commit()


def record_cost_event(conn: sqlite3.Connection, category: str, amount_usd: float, description: str | None = None) -> None:
    if category not in COST_CATEGORIES:
        raise ValueError(f"unknown cost category: {category}")
    conn.execute(
        "INSERT INTO cost_events (category, amount_usd, description, created_at) VALUES (?, ?, ?, ?)",
        (category, amount_usd, description, now_iso()),
    )
    conn.commit()


def total_cost(conn: sqlite3.Connection) -> float:
    row = conn.execute("SELECT SUM(amount_usd) AS total FROM cost_events").fetchone()
    return row["total"] or 0.0


def set_run_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO run_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()


def get_run_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM run_meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row is not None else default
