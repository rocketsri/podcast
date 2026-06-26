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
    return arr.reshape(dim) if dim else arr


SCHEMA = """
CREATE TABLE IF NOT EXISTS podcasts (
    podcast_id TEXT PRIMARY KEY,
    feed_id TEXT NOT NULL,
    title TEXT NOT NULL,
    feed_url TEXT NOT NULL,
    language TEXT,
    episode_count_total INTEGER,
    selected_at TEXT NOT NULL,
    selection_reason TEXT
);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    podcast_id TEXT NOT NULL REFERENCES podcasts(podcast_id),
    pi_episode_id TEXT NOT NULL,
    title TEXT NOT NULL,
    source_url TEXT NOT NULL,
    published_at TEXT,
    duration_seconds_reported REAL,
    duration_seconds_actual REAL,
    assigned_shard INTEGER,
    local_raw_path TEXT,
    local_wav_path TEXT,
    stage TEXT NOT NULL DEFAULT 'queued',
    failed_stage TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    raw_seconds REAL,
    usable_seconds REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_queue ON episodes(assigned_shard, stage);
CREATE INDEX IF NOT EXISTS idx_episodes_podcast ON episodes(podcast_id);

CREATE TABLE IF NOT EXISTS speakers (
    speaker_id TEXT PRIMARY KEY,
    podcast_id TEXT NOT NULL REFERENCES podcasts(podcast_id),
    local_label_seq INTEGER NOT NULL,
    centroid_embedding BLOB,
    centroid_dim INTEGER,
    embedding_count INTEGER NOT NULL DEFAULT 0,
    total_speech_seconds REAL NOT NULL DEFAULT 0,
    first_seen_episode TEXT,
    last_seen_episode TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_speakers_podcast ON speakers(podcast_id);

-- Durable, shard-safe diarization output: globally-unique episode_id keys
-- mean rows from different pods never collide, so this table is always a
-- clean substrate to (re)build speakers/clips.speaker_id from, even after
-- merging multiple pods' databases (see cluster.recluster_podcast_from_scratch).
CREATE TABLE IF NOT EXISTS local_speaker_segments (
    segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
    local_label TEXT NOT NULL,
    start_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL,
    embedding BLOB,
    resolved_speaker_id TEXT REFERENCES speakers(speaker_id)
);
CREATE INDEX IF NOT EXISTS idx_local_segments_episode ON local_speaker_segments(episode_id);

CREATE TABLE IF NOT EXISTS clips (
    clip_id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
    podcast_id TEXT NOT NULL REFERENCES podcasts(podcast_id),
    start_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    speaker_id TEXT REFERENCES speakers(speaker_id),
    utterance TEXT,
    vad_confidence REAL,
    overlap_detected INTEGER NOT NULL DEFAULT 0,
    music_detected INTEGER NOT NULL DEFAULT 0,
    no_speech_prob REAL,
    avg_logprob REAL,
    discard_reason TEXT,
    audio_path TEXT,
    local_flac_path TEXT,
    uploaded INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clips_episode ON clips(episode_id);
CREATE INDEX IF NOT EXISTS idx_clips_upload_queue ON clips(uploaded);
CREATE INDEX IF NOT EXISTS idx_clips_discard ON clips(discard_reason);

CREATE TABLE IF NOT EXISTS cost_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT,
    amount_usd REAL NOT NULL,
    related_episode_id TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS run_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def connect(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


# --- podcasts ---------------------------------------------------------------

def insert_podcast(
    conn: sqlite3.Connection,
    podcast_id: str,
    feed_id: str,
    title: str,
    feed_url: str,
    language: str | None = None,
    episode_count_total: int | None = None,
    selection_reason: str | None = None,
) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO podcasts
           (podcast_id, feed_id, title, feed_url, language, episode_count_total,
            selected_at, selection_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (podcast_id, feed_id, title, feed_url, language, episode_count_total, now_iso(), selection_reason),
    )
    conn.commit()


def get_podcast(conn: sqlite3.Connection, podcast_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM podcasts WHERE podcast_id = ?", (podcast_id,)).fetchone()


# --- episodes ----------------------------------------------------------------

def insert_episode(
    conn: sqlite3.Connection,
    episode_id: str,
    podcast_id: str,
    pi_episode_id: str,
    title: str,
    source_url: str,
    published_at: str | None = None,
    duration_seconds_reported: float | None = None,
) -> None:
    ts = now_iso()
    conn.execute(
        """INSERT OR IGNORE INTO episodes
           (episode_id, podcast_id, pi_episode_id, title, source_url, published_at,
            duration_seconds_reported, stage, attempt_count, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', 0, ?, ?)""",
        (episode_id, podcast_id, pi_episode_id, title, source_url, published_at,
         duration_seconds_reported, ts, ts),
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


def set_assigned_shard(conn: sqlite3.Connection, episode_id: str, shard: int) -> None:
    conn.execute(
        "UPDATE episodes SET assigned_shard = ?, updated_at = ? WHERE episode_id = ?",
        (shard, now_iso(), episode_id),
    )
    conn.commit()


def advance_stage(conn: sqlite3.Connection, episode_id: str, new_stage: str, **fields) -> None:
    """Move an episode to `new_stage`, clearing any prior failure, optionally
    setting other columns (e.g. raw_seconds=, local_wav_path=) in the same update."""
    if new_stage not in _STAGE_INDEX:
        raise StageError(f"unknown stage: {new_stage}")
    set_clauses = ["stage = ?", "failed_stage = NULL", "last_error = NULL", "updated_at = ?"]
    params: list = [new_stage, now_iso()]
    for key, value in fields.items():
        set_clauses.append(f"{key} = ?")
        params.append(value)
    params.append(episode_id)
    conn.execute(f"UPDATE episodes SET {', '.join(set_clauses)} WHERE episode_id = ?", params)
    conn.commit()


def mark_stage_failed(conn: sqlite3.Connection, episode_id: str, failed_stage: str, error: str) -> None:
    conn.execute(
        """UPDATE episodes
           SET stage = 'failed', failed_stage = ?, last_error = ?,
               attempt_count = attempt_count + 1, updated_at = ?
           WHERE episode_id = ?""",
        (failed_stage, error, now_iso(), episode_id),
    )
    conn.commit()


# --- speakers / local_speaker_segments ---------------------------------------

def insert_local_speaker_segment(
    conn: sqlite3.Connection,
    episode_id: str,
    local_label: str,
    start_seconds: float,
    end_seconds: float,
    embedding: np.ndarray | None,
    resolved_speaker_id: str | None = None,
) -> int:
    blob = pack_embedding(embedding) if embedding is not None else None
    cur = conn.execute(
        """INSERT INTO local_speaker_segments
           (episode_id, local_label, start_seconds, end_seconds, embedding, resolved_speaker_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (episode_id, local_label, start_seconds, end_seconds, blob, resolved_speaker_id),
    )
    conn.commit()
    return cur.lastrowid


def get_local_speaker_segments_for_podcast(conn: sqlite3.Connection, podcast_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT lss.* FROM local_speaker_segments lss
           JOIN episodes e ON e.episode_id = lss.episode_id
           WHERE e.podcast_id = ?
           ORDER BY lss.episode_id, lss.start_seconds""",
        (podcast_id,),
    ).fetchall()


def get_local_speaker_segments_for_episode(conn: sqlite3.Connection, episode_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM local_speaker_segments WHERE episode_id = ? ORDER BY start_seconds", (episode_id,)
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
    local_label_seq: int,
    centroid_embedding: np.ndarray,
    embedding_count: int,
    total_speech_seconds: float,
    episode_id: str,
) -> None:
    ts = now_iso()
    blob = pack_embedding(centroid_embedding)
    conn.execute(
        """INSERT INTO speakers
               (speaker_id, podcast_id, local_label_seq, centroid_embedding, centroid_dim,
                embedding_count, total_speech_seconds, first_seen_episode, last_seen_episode,
                created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(speaker_id) DO UPDATE SET
               centroid_embedding = excluded.centroid_embedding,
               centroid_dim = excluded.centroid_dim,
               embedding_count = excluded.embedding_count,
               total_speech_seconds = excluded.total_speech_seconds,
               last_seen_episode = excluded.last_seen_episode,
               updated_at = excluded.updated_at""",
        (speaker_id, podcast_id, local_label_seq, blob, centroid_embedding.shape[0],
         embedding_count, total_speech_seconds, episode_id, episode_id, ts, ts),
    )
    conn.commit()


def get_speakers_for_podcast(conn: sqlite3.Connection, podcast_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM speakers WHERE podcast_id = ? ORDER BY local_label_seq", (podcast_id,)
    ).fetchall()


def delete_speakers_for_podcast(conn: sqlite3.Connection, podcast_id: str) -> None:
    """Used only by recluster_podcast_from_scratch: wipes provisional/stale
    global speaker rows for a podcast before rebuilding them from merged
    local_speaker_segments. Nulls out the FK references that point at those
    speaker rows first (clips.speaker_id, local_speaker_segments.resolved_speaker_id)
    — the caller is expected to reassign both right after, from the freshly
    recomputed clusters."""
    conn.execute(
        """UPDATE local_speaker_segments SET resolved_speaker_id = NULL
           WHERE episode_id IN (SELECT episode_id FROM episodes WHERE podcast_id = ?)""",
        (podcast_id,),
    )
    conn.execute("UPDATE clips SET speaker_id = NULL WHERE podcast_id = ?", (podcast_id,))
    conn.execute("DELETE FROM speakers WHERE podcast_id = ?", (podcast_id,))
    conn.commit()


# --- clips -------------------------------------------------------------------

def insert_clip(
    conn: sqlite3.Connection,
    clip_id: str,
    episode_id: str,
    podcast_id: str,
    start_seconds: float,
    end_seconds: float,
    speaker_id: str | None = None,
    discard_reason: str | None = None,
    **fields,
) -> None:
    duration_seconds = end_seconds - start_seconds
    columns = [
        "clip_id", "episode_id", "podcast_id", "start_seconds", "end_seconds",
        "duration_seconds", "speaker_id", "discard_reason", "created_at",
    ]
    values = [clip_id, episode_id, podcast_id, start_seconds, end_seconds,
              duration_seconds, speaker_id, discard_reason, now_iso()]
    for key, value in fields.items():
        columns.append(key)
        values.append(value)
    placeholders = ", ".join("?" for _ in values)
    # OR IGNORE (same convention as insert_podcast/insert_episode): segment.py's
    # persist_candidate_clips loop must be safely re-callable after a crash
    # partway through an episode's clips without raising on the clip_ids that
    # already made it to disk before the crash.
    conn.execute(f"INSERT OR IGNORE INTO clips ({', '.join(columns)}) VALUES ({placeholders})", values)
    conn.commit()


def get_clips_for_podcast(conn: sqlite3.Connection, podcast_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM clips WHERE podcast_id = ? ORDER BY episode_id, start_seconds", (podcast_id,)
    ).fetchall()


def get_clips_for_episode(conn: sqlite3.Connection, episode_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM clips WHERE episode_id = ? ORDER BY start_seconds", (episode_id,)
    ).fetchall()


def get_clips_pending_upload(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM clips WHERE uploaded = 0 AND discard_reason IS NULL").fetchall()


def mark_clip_uploaded(conn: sqlite3.Connection, clip_id: str, audio_path: str) -> None:
    conn.execute(
        "UPDATE clips SET uploaded = 1, audio_path = ? WHERE clip_id = ?", (audio_path, clip_id)
    )
    conn.commit()


def update_clip_speaker(conn: sqlite3.Connection, clip_id: str, speaker_id: str | None) -> None:
    conn.execute("UPDATE clips SET speaker_id = ? WHERE clip_id = ?", (speaker_id, clip_id))
    conn.commit()


def update_clip_fields(conn: sqlite3.Connection, clip_id: str, **fields) -> None:
    """Generic column-set update for already-persisted clip rows -- used by
    quality.py/asr.py to layer content-based discard reasons and ASR signals
    onto clips that segment.py already inserted. `fields` keys are always
    internal column names from our own code, never user input."""
    if not fields:
        return
    assignments = ", ".join(f"{key} = ?" for key in fields)
    conn.execute(f"UPDATE clips SET {assignments} WHERE clip_id = ?", (*fields.values(), clip_id))
    conn.commit()


# --- cost_events --------------------------------------------------------------

def record_cost_event(
    conn: sqlite3.Connection,
    category: str,
    amount_usd: float,
    description: str = "",
    related_episode_id: str | None = None,
    metadata: dict | None = None,
) -> None:
    if category not in COST_CATEGORIES:
        raise ValueError(f"unknown cost category: {category}")
    conn.execute(
        """INSERT INTO cost_events (ts, category, description, amount_usd, related_episode_id, metadata_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (now_iso(), category, description, amount_usd, related_episode_id,
         json.dumps(metadata) if metadata else None),
    )
    conn.commit()


def total_cost(conn: sqlite3.Connection) -> float:
    row = conn.execute("SELECT COALESCE(SUM(amount_usd), 0) AS total FROM cost_events").fetchone()
    return float(row["total"])


# --- run_meta ------------------------------------------------------------------

def set_run_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO run_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )
    conn.commit()


def get_run_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM run_meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default
