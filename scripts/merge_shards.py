"""Downloads each pod's pipeline.db snapshot from R2 (or accepts local paths
directly), merges every shard into one fresh output db, re-runs cluster.py's
clustering from scratch per podcast (shard-local speaker_id strings are only
provisional -- see cluster.py's module docstring), tags cost_events with
their source pod_id so scripts/report.py's per-pod cost attribution can
activate, and writes the final manifest.jsonl.

podcast_id/episode_id/clip_id are deterministic hashes of feed_url/guid (see
discovery.py), so merging is collision-safe in the common case (independent
per-pod discovery, see infra/bootstrap.sh) via plain INSERT OR IGNORE.
local_speaker_segments/cost_events have no natural key -- their PK is a
per-shard AUTOINCREMENT int with no cross-shard meaning, so this script
drops that column on copy and lets the merged db mint fresh ids rather than
risk silently dropping rows on a spurious PK collision. clips.speaker_id is
always nulled out on copy (not copied verbatim) because no `speakers` rows
are copied at all -- copying a stale per-shard speaker_id would violate the
merged db's foreign key, and recluster_podcast_from_scratch reassigns every
clip's speaker_id from scratch immediately after the merge anyway.

The one case this script does not fully reconcile -- two pods independently
discovering the very same real episode (possible since each pod runs its
own iTunes Search query slice, see infra/bootstrap.sh's own comment on this)
-- is detected and flagged loudly rather than silently resolved: the losing
shard's episode/clip rows are dropped via INSERT OR IGNORE (safe, since
segment.py's clip_id is deterministic from episode_id+index, so a genuine
duplicate collides on conflict rather than double-inserting), but its
local_speaker_segments rows are NOT deduplicated (that table has no natural
key by design) and so could leave two independent diarization runs' segments
coexisting for one episode_id. Rare enough in practice (per the per-shard
disjoint discovery-query design) that this is called out for manual review
rather than engineered around.

Run with:
  python3 scripts/merge_shards.py --pod-id podcast-shard-0 --pod-id podcast-shard-1 --output-db work/merged.db
  python3 scripts/merge_shards.py --auto-discover --output-db work/merged.db
  python3 scripts/merge_shards.py --local-db work/shard0.db --local-db work/shard1.db --output-db work/merged.db --no-upload-manifest
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline import cluster, config, db, logging_utils, manifest, storage  # noqa: E402

logger = logging_utils.get_logger()

# local_speaker_segments / cost_events deliberately excluded here -- see
# module docstring on why their PK column is dropped on copy instead.
_TABLE_COLUMNS = {
    "podcasts": ["podcast_id", "feed_id", "title", "feed_url", "language", "episode_count_total", "selected_at", "selection_reason"],
    "episodes": [
        "episode_id", "podcast_id", "pi_episode_id", "title", "source_url", "published_at",
        "duration_seconds_reported", "duration_seconds_actual", "assigned_shard", "local_raw_path",
        "local_wav_path", "stage", "failed_stage", "attempt_count", "last_error", "raw_seconds",
        "usable_seconds", "created_at", "updated_at",
    ],
    "clips": [
        "clip_id", "episode_id", "podcast_id", "start_seconds", "end_seconds", "duration_seconds",
        "speaker_id", "utterance", "vad_confidence", "overlap_detected", "music_detected",
        "no_speech_prob", "avg_logprob", "discard_reason", "audio_path", "local_flac_path",
        "uploaded", "created_at",
    ],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pod-id", action="append", default=None, help="pod_id label to download db_snapshots/<pod_id>/pipeline.db from R2 for (repeatable)")
    parser.add_argument("--auto-discover", action="store_true", help="also merge every db_snapshots/*/pipeline.db object found in R2, instead of only the --pod-id values given explicitly")
    parser.add_argument("--local-db", action="append", default=None, help="local sqlite path to merge directly, e.g. an already-downloaded snapshot (repeatable)")
    parser.add_argument("--download-dir", default="work/shard_downloads", help="where R2 snapshots get downloaded to before merging")
    parser.add_argument("--output-db", default="work/merged.db", help="path to write the fresh merged db to")
    parser.add_argument("--manifest-out", default="work/manifest.jsonl", help="path to write the merged manifest.jsonl to")
    parser.add_argument("--config", default=None, help="path to pipeline.yaml")
    parser.add_argument("--force", action="store_true", help="overwrite --output-db if it already exists")
    parser.add_argument("--no-upload-manifest", action="store_true", help="skip uploading the merged manifest.jsonl to R2 (manifest/manifest.jsonl)")
    return parser.parse_args(argv)


def _discover_pod_ids(client, bucket: str) -> list[str]:
    keys = storage.list_keys(client, bucket, prefix="db_snapshots/")
    return sorted({key.split("/")[1] for key in keys if key.endswith("/pipeline.db")})


def _download_snapshots(client, bucket: str, pod_ids: list[str], download_dir: Path) -> list[tuple[str, Path]]:
    out = []
    for pod_id in pod_ids:
        dest = download_dir / pod_id / "pipeline.db"
        storage.download_file(client, bucket, storage.db_snapshot_key(pod_id), dest)
        out.append((pod_id, dest))
        logger.info("downloaded %s snapshot -> %s", pod_id, dest)
    return out


def _copy_natural_key_table(src: sqlite3.Connection, dst: sqlite3.Connection, table: str) -> int:
    columns = _TABLE_COLUMNS[table]
    rows = src.execute(f"SELECT {', '.join(columns)} FROM {table}").fetchall()
    if not rows:
        return 0
    placeholders = ", ".join("?" for _ in columns)
    dst.executemany(
        f"INSERT OR IGNORE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        [tuple(row[c] for c in columns) for row in rows],
    )
    dst.commit()
    return len(rows)


def _copy_clips(src: sqlite3.Connection, dst: sqlite3.Connection) -> int:
    columns = _TABLE_COLUMNS["clips"]
    speaker_idx = columns.index("speaker_id")
    rows = src.execute(f"SELECT {', '.join(columns)} FROM clips").fetchall()
    if not rows:
        return 0
    values = []
    for row in rows:
        vals = [row[c] for c in columns]
        vals[speaker_idx] = None  # stale per-shard label; recluster_podcast_from_scratch reassigns every clip's speaker_id right after merge (see module docstring)
        values.append(tuple(vals))
    placeholders = ", ".join("?" for _ in columns)
    dst.executemany(f"INSERT OR IGNORE INTO clips ({', '.join(columns)}) VALUES ({placeholders})", values)
    dst.commit()
    return len(rows)


def _copy_local_speaker_segments(src: sqlite3.Connection, dst: sqlite3.Connection) -> int:
    columns = ["episode_id", "local_label", "start_seconds", "end_seconds", "embedding"]
    rows = src.execute(f"SELECT {', '.join(columns)} FROM local_speaker_segments").fetchall()
    if not rows:
        return 0
    dst.executemany(
        "INSERT INTO local_speaker_segments (episode_id, local_label, start_seconds, end_seconds, embedding, resolved_speaker_id) "
        "VALUES (?, ?, ?, ?, ?, NULL)",
        [tuple(row[c] for c in columns) for row in rows],
    )
    dst.commit()
    return len(rows)


def _copy_cost_events(src: sqlite3.Connection, dst: sqlite3.Connection, pod_label: str) -> float:
    columns = ["ts", "category", "description", "amount_usd", "related_episode_id", "metadata_json"]
    rows = src.execute(f"SELECT {', '.join(columns)} FROM cost_events").fetchall()
    total = 0.0
    for row in rows:
        meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        meta["pod_id"] = pod_label  # lets report.py's per-pod cost attribution activate
        dst.execute(
            "INSERT INTO cost_events (ts, category, description, amount_usd, related_episode_id, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (row["ts"], row["category"], row["description"], row["amount_usd"], row["related_episode_id"], json.dumps(meta)),
        )
        total += row["amount_usd"]
    dst.commit()
    return total


def _copy_run_meta(src: sqlite3.Connection, dst: sqlite3.Connection, pod_label: str) -> None:
    """Namespaced by pod_label rather than merged key-for-key -- most
    run_meta keys (last_cost_checkpoint_at, recluster_count_*, ...) are
    per-pod driver-loop bookkeeping with no cross-pod meaning; keeping them
    around namespaced is cheap provenance for debugging, not a live config."""
    for row in src.execute("SELECT key, value FROM run_meta").fetchall():
        db.set_run_meta(dst, f"{pod_label}__{row['key']}", row["value"])


def merge_shard(src_path: Path, dst_conn: sqlite3.Connection, pod_label: str) -> dict:
    src_conn = sqlite3.connect(src_path)
    src_conn.row_factory = sqlite3.Row
    try:
        n_podcasts = _copy_natural_key_table(src_conn, dst_conn, "podcasts")
        n_episodes = _copy_natural_key_table(src_conn, dst_conn, "episodes")
        n_clips = _copy_clips(src_conn, dst_conn)
        n_segments = _copy_local_speaker_segments(src_conn, dst_conn)
        shard_cost = _copy_cost_events(src_conn, dst_conn, pod_label)
        _copy_run_meta(src_conn, dst_conn, pod_label)
    finally:
        src_conn.close()
    return {
        "pod_label": pod_label, "podcasts": n_podcasts, "episodes": n_episodes,
        "clips": n_clips, "segments": n_segments, "cost_usd": shard_cost,
    }


def _detect_cross_shard_episode_collisions(shard_sources: list[tuple[str, Path]]) -> dict[str, list[str]]:
    seen: dict[str, list[str]] = {}
    for pod_label, path in shard_sources:
        conn = sqlite3.connect(path)
        try:
            for row in conn.execute("SELECT episode_id FROM episodes"):
                seen.setdefault(row[0], []).append(pod_label)
        finally:
            conn.close()
    return {episode_id: labels for episode_id, labels in seen.items() if len(labels) > 1}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = config.load_config(args.config)

    output_db = Path(args.output_db)
    manifest_out = Path(args.manifest_out)
    if output_db.exists() and not args.force:
        logger.error("%s already exists -- pass --force to overwrite (this script always starts from a fresh db, see module docstring)", output_db)
        return 1
    output_db.parent.mkdir(parents=True, exist_ok=True)
    output_db.unlink(missing_ok=True)

    shard_sources: list[tuple[str, Path]] = [(f"local:{Path(p).stem}", Path(p)) for p in (args.local_db or [])]

    pod_ids = list(args.pod_id or [])
    if args.auto_discover or pod_ids:
        secrets = config.EnvSecrets.from_env()
        try:
            secrets.require_r2()
        except config.ConfigError as exc:
            logger.error("R2 credentials required for --pod-id/--auto-discover (%s)", exc)
            return 1
        client = storage.build_client(secrets)
        if args.auto_discover:
            discovered = _discover_pod_ids(client, secrets.r2_bucket_name)
            logger.info("auto-discovered %d pod snapshot(s) in R2: %s", len(discovered), discovered)
            pod_ids = sorted(set(pod_ids) | set(discovered))
        shard_sources.extend(_download_snapshots(client, secrets.r2_bucket_name, pod_ids, Path(args.download_dir)))

    if not shard_sources:
        logger.error("no shard databases to merge -- pass --pod-id, --auto-discover, and/or --local-db")
        return 1

    collisions = _detect_cross_shard_episode_collisions(shard_sources)
    if collisions:
        logger.warning(
            "%d episode_id(s) were independently discovered by more than one pod -- only one "
            "shard's episode/clip rows survive the merge (INSERT OR IGNORE); see module docstring "
            "for why local_speaker_segments duplication for these episode_ids needs manual review: %s",
            len(collisions), collisions,
        )

    dst_conn = db.connect(output_db)
    db.init_db(dst_conn)

    for pod_label, path in shard_sources:
        summary = merge_shard(path, dst_conn, pod_label)
        logger.info(
            "merged %s: %d podcasts, %d episodes, %d clips, %d segments, $%.4f cost",
            pod_label, summary["podcasts"], summary["episodes"], summary["clips"], summary["segments"], summary["cost_usd"],
        )

    podcast_ids = [row["podcast_id"] for row in dst_conn.execute("SELECT podcast_id FROM podcasts")]
    for podcast_id in podcast_ids:
        result = cluster.recluster_podcast_from_scratch(dst_conn, podcast_id, match_threshold=cfg.clustering.match_threshold)
        logger.info("reclustered %s: %d speakers, %d clips corrected", podcast_id, result.num_speakers, result.num_clips_corrected)

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_count = manifest.write_manifest(dst_conn, manifest_out)
    logger.info("wrote %d manifest rows to %s", manifest_count, manifest_out)

    if not args.no_upload_manifest:
        secrets = config.EnvSecrets.from_env()
        try:
            secrets.require_r2()
            client = storage.build_client(secrets)
            storage.upload_file(client, secrets.r2_bucket_name, manifest_out, storage.manifest_key())
            logger.info("uploaded manifest to R2 at %s", storage.manifest_key())
        except config.ConfigError as exc:
            logger.warning("R2 credentials incomplete (%s) -- manifest stays local-only", exc)

    total_cost = db.total_cost(dst_conn)
    total_episodes = dst_conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    total_clips = dst_conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
    dst_conn.close()

    print(f"\nmerged {len(shard_sources)} shard(s) -> {output_db}")
    print(f"  {len(podcast_ids)} podcasts, {total_episodes} episodes, {total_clips} clips, ${total_cost:.4f} total cost")
    print(f"  manifest: {manifest_count} rows -> {manifest_out}")
    if collisions:
        print(f"  WARNING: {len(collisions)} cross-shard episode_id collision(s) detected -- see log above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
