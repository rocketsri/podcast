"""Status/heartbeat reporting: a JSON status dict pushed to R2 after every
episode (or on a timer), plus the same dict served live over a local HTTP
port. Dual channel by design -- RunPod's API exposes no container-log
retrieval, so a self-reported heartbeat (R2 object + HTTP endpoint) is the
only operational visibility into a running pod, and a status check
shouldn't depend solely on one write path succeeding.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from pipeline import db


def build_status(
    conn: sqlite3.Connection,
    pod_id: str,
    shard_id: int | None,
    started_at: str,
    extra: dict | None = None,
) -> dict:
    """Snapshots current DB state into the heartbeat shape: stage, episodes
    done, clips produced, running cost, last error -- everything
    poll_status.py needs to judge a pod's progress without SSH or log access."""
    stage_counts = {
        row["stage"]: row["n"]
        for row in conn.execute("SELECT stage, COUNT(*) AS n FROM episodes GROUP BY stage")
    }
    done_count = stage_counts.get("done", 0)
    failed_count = stage_counts.get(db.FAILED_STAGE, 0)
    total_episodes = sum(stage_counts.values())

    clip_row = conn.execute(
        "SELECT COUNT(*) AS total,"
        " SUM(CASE WHEN uploaded = 1 THEN 1 ELSE 0 END) AS uploaded,"
        " SUM(CASE WHEN discard_reason IS NOT NULL THEN 1 ELSE 0 END) AS discarded"
        " FROM clips"
    ).fetchone()

    seconds_row = conn.execute(
        "SELECT SUM(raw_seconds) AS raw_total, SUM(usable_seconds) AS usable_total FROM episodes"
    ).fetchone()

    last_error_row = conn.execute(
        "SELECT episode_id, failed_stage, last_error, updated_at FROM episodes"
        " WHERE stage = ? ORDER BY updated_at DESC LIMIT 1",
        (db.FAILED_STAGE,),
    ).fetchone()

    status = {
        "pod_id": pod_id,
        "shard_id": shard_id,
        "started_at": started_at,
        "updated_at": db.now_iso(),
        "episodes": {
            "total": total_episodes,
            "done": done_count,
            "failed": failed_count,
            "in_progress": total_episodes - done_count - failed_count,
            "stage_counts": stage_counts,
        },
        "clips": {
            "total": clip_row["total"] or 0,
            "uploaded": clip_row["uploaded"] or 0,
            "discarded": clip_row["discarded"] or 0,
        },
        "raw_seconds_total": seconds_row["raw_total"] or 0.0,
        "usable_seconds_total": seconds_row["usable_total"] or 0.0,
        "total_cost_usd": db.total_cost(conn),
        "last_error": (
            {
                "episode_id": last_error_row["episode_id"],
                "failed_stage": last_error_row["failed_stage"],
                "error": last_error_row["last_error"],
                "at": last_error_row["updated_at"],
            }
            if last_error_row is not None
            else None
        ),
    }
    if extra:
        status.update(extra)
    return status


def push_status_to_r2(client, bucket: str, pod_id: str, status: dict, key_prefix: str = "") -> None:
    from pipeline import storage  # local import: keeps heartbeat importable with no boto3 present

    storage.put_json(client, bucket, storage.status_key(pod_id, key_prefix), status)


class _StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = json.dumps(self.server.status_provider()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:
        pass  # structured logging (logging_utils) covers this; silence default stderr access logs


class StatusServer:
    """Serves the latest status dict over HTTP on a background thread, on
    RunPod's exposed proxy port -- redundant insurance alongside the R2
    push, since a status check shouldn't depend solely on R2 write success."""

    def __init__(self, status_provider, port: int = 8080):
        self._server = HTTPServer(("0.0.0.0", port), _StatusHandler)
        self._server.status_provider = status_provider
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
