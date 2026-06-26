"""Polls health of running RunPod GPU pods for the credentialed path.

RunPod's API exposes no container-log retrieval, so the only operational
visibility into a pod is what it self-reports: pipeline/heartbeat.py pushes
a status JSON to R2 (storage.status_key(pod_id)) after every episode, and
also serves the same dict over HTTP on the port bootstrap_pod.py exposes
via RunPod's proxy (config.monitoring.status_http_port, default 8080 --
reachable at https://<runpod-pod-id>-<port>.proxy.runpod.net/). The R2
write and the HTTP serve are independent (see PROBLEMS.md #18: an R2-write
failure left every shard's R2 heartbeat permanently missing while the HTTP
endpoint reported real progress the whole time) -- this script used to
read only R2, which made that whole class of failure indistinguishable
from "still bootstrapping". It now falls back to the HTTP endpoint
whenever R2 has nothing, instead of treating "no R2 object" as the only
signal.

This script cross-references RunPod's own pod lifecycle state (list_pods,
keyed by the `name` bootstrap_pod.py set to each pod's heartbeat pod_id
label) against the latest heartbeat each pod has reported by either
channel, and flags:
  - pods RunPod reports running but with no heartbeat from either channel
    (still installing deps / downloading the code tarball / loading models)
  - pods whose last heartbeat is older than config.monitoring's
    stale_heartbeat_minutes (likely crashed or hung)
  - any pod whose latest heartbeat carries a last_error or a
    heartbeat_r2_push_error (the R2 write itself is failing even though
    the HTTP fallback got through)

Exits non-zero if any pod is stale or RunPod reports it EXITED/terminated
unexpectedly, so this can be used as a cheap watch-loop condition.

Run with: python3 scripts/poll_status.py
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline import config, db, heartbeat, logging_utils, storage  # noqa: E402
from pipeline.runpod_client import RunPodClient, RunPodError  # noqa: E402

logger = logging_utils.get_logger()

HTTP_STATUS_TIMEOUT = 8.0


def proxy_status_url(runpod_pod_id: str, port: int) -> str:
    return f"https://{runpod_pod_id}-{port}.proxy.runpod.net/"


def parse_iso(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def fetch_heartbeat(client, bucket: str, pod_id_label: str) -> dict | None:
    try:
        return storage.get_json(client, bucket, storage.status_key(pod_id_label))
    except Exception:  # noqa: BLE001 - missing object, network blip, etc. -- just means "no heartbeat yet"
        return None


def fetch_heartbeat_via_http(runpod_pod_id: str, port: int) -> dict | None:
    """Fallback for when the R2 object is missing -- hits the same status
    dict directly over RunPod's exposed proxy port (see PROBLEMS.md #18)."""
    try:
        resp = requests.get(proxy_status_url(runpod_pod_id, port), timeout=HTTP_STATUS_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except (requests.exceptions.RequestException, ValueError):
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to pipeline.yaml (for monitoring.stale_heartbeat_minutes)")
    parser.add_argument("--pod-name-prefix", default="podcast-shard", help="only consider RunPod pods whose name starts with this")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = config.load_config(args.config)
    secrets = config.EnvSecrets.from_env()

    missing = [name for name, value in (("RUNPOD_API_KEY", secrets.runpod_api_key), ("R2_BUCKET_NAME", secrets.r2_bucket_name)) if not value]
    if missing:
        logger.error("missing required environment variables: %s", ", ".join(missing))
        return 1

    runpod_client = RunPodClient(secrets.runpod_api_key)
    try:
        pods = [p for p in runpod_client.list_pods() if p.get("name", "").startswith(args.pod_name_prefix)]
    except RunPodError as exc:
        logger.error("failed to list RunPod pods: %s", exc)
        return 1

    if not pods:
        print(f"no RunPod pods found with name prefix {args.pod_name_prefix!r}")
        return 0

    r2_client = storage.build_client(secrets)
    stale_threshold = timedelta(minutes=cfg.monitoring.stale_heartbeat_minutes)
    now = datetime.now(timezone.utc)

    any_problem = False
    for pod in pods:
        pod_id_label = pod.get("name", "?")
        runpod_id = pod.get("id", "?")
        runpod_status = pod.get("desiredStatus") or pod.get("status") or "?"
        hb = fetch_heartbeat(r2_client, secrets.r2_bucket_name, pod_id_label)
        via_http = False
        if hb is None and runpod_id != "?":
            hb = fetch_heartbeat_via_http(runpod_id, cfg.monitoring.status_http_port)
            via_http = hb is not None

        print(f"--- {pod_id_label} (runpod id={runpod_id}, runpod status={runpod_status}) ---")
        if hb is None:
            print("  no heartbeat in R2 or via HTTP fallback (still bootstrapping, loading models, or never started)")
            if runpod_status not in ("RUNNING", "PENDING", "?"):
                any_problem = True
            continue
        if via_http:
            print("  [via HTTP fallback -- R2 object missing, see PROBLEMS.md #18]")

        age = now - parse_iso(hb["updated_at"])
        stale = age > stale_threshold
        any_problem = any_problem or stale or hb.get("last_error") is not None or hb.get("heartbeat_r2_push_error") is not None

        eps = hb["episodes"]
        print(f"  heartbeat age: {age.total_seconds() / 60:.1f}m{'  [STALE]' if stale else ''}")
        print(f"  episodes: {eps['done']} done / {eps['failed']} failed / {eps['in_progress']} in-progress (of {eps['total']})")
        print(f"  clips: {hb['clips']['total']} total, {hb['clips']['uploaded']} uploaded, {hb['clips']['discarded']} discarded")
        print(f"  usable hours: {hb['usable_seconds_total'] / 3600.0:.2f}h  cost so far: ${hb['total_cost_usd']:.2f}")
        if hb.get("last_error"):
            print(f"  last_error: {hb['last_error']}")
        if hb.get("heartbeat_r2_push_error"):
            print(f"  heartbeat_r2_push_error: {hb['heartbeat_r2_push_error']}")

    print()
    print("OK -- all pods reporting" if not any_problem else "PROBLEM -- see [STALE]/last_error/unexpected-status lines above")
    return 1 if any_problem else 0


if __name__ == "__main__":
    sys.exit(main())
