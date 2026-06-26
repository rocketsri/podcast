"""Recovery logic for the live-fleet upload bug: re-uploads clips whose
local_flac_path file still exists on disk, ignoring the stale `uploaded`
flag set by the old, unverified code path. See scripts/backfill_uploads.py
(CLI entrypoint for an already-downloaded db snapshot) and run_pipeline.py
(calls this automatically on every pod boot, since a pod's dockerStartCmd is
a frozen shell-script snapshot from creation time that a code-only fix can't
reach -- but run_pipeline.py itself is freshly re-imported from the latest
git checkout on every restart, so wiring the fix in here is what actually
makes a plain `restart_pod()` pick it up on the already-running fleet).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from pipeline import costs, storage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackfillResult:
    candidates: int
    missing: int
    already_present: int
    reuploaded: int
    failed: int


def backfill_uploaded_clips(conn: sqlite3.Connection, client, bucket: str) -> BackfillResult:
    rows = conn.execute(
        "SELECT clip_id, episode_id, podcast_id, local_flac_path FROM clips"
        " WHERE uploaded = 1 AND local_flac_path IS NOT NULL"
    ).fetchall()

    candidates = [r for r in rows if Path(r["local_flac_path"]).exists()]
    missing = len(rows) - len(candidates)

    already_present = 0
    reuploaded = 0
    failed = 0
    for row in candidates:
        key = storage.clip_key(row["podcast_id"], row["episode_id"], row["clip_id"])
        try:
            if storage.object_exists(client, bucket, key):
                already_present += 1
                continue
            storage.upload_clip(client, bucket, row["local_flac_path"], row["podcast_id"], row["episode_id"], row["clip_id"])
            if not storage.object_exists(client, bucket, key):
                raise RuntimeError(f"upload_file reported success for {key} but object_exists is False after")
            reuploaded += 1
        except Exception as exc:  # noqa: BLE001 - keep going across the whole backlog
            failed += 1
            logger.error("clip %s (episode %s) backfill upload failed: %s", row["clip_id"], row["episode_id"], exc)

    if reuploaded:
        costs.record_r2_class_a_ops(conn, reuploaded, description=f"backfill: {reuploaded} clip PutObject calls")

    return BackfillResult(
        candidates=len(candidates), missing=missing,
        already_present=already_present, reuploaded=reuploaded, failed=failed,
    )
