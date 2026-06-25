"""Bin-packs queued episodes (assigned_shard IS NULL) across N pods by
reported duration, using a Longest-Processing-Time-first greedy heuristic --
the standard simple approximation for makespan-balanced multiprocessor
scheduling, good enough for load-balancing disjoint episode shards across
pods without needing exact bin-packing.

Run once, after scripts/select_podcasts.py has registered the candidate
episode pool and the Stage 1 throughput calibration has determined how many
pods to launch (the plan's pods_needed sizing checkpoint), before any
Stage 2 pod starts. Idempotent to rerun against the same unassigned pool,
but rerunning after pods have already claimed work would orphan their
in-flight episodes' shard assignment -- only run this once per batch.

Run with: python3 scripts/partition_episodes.py --shards N
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline import db, logging_utils  # noqa: E402

logger = logging_utils.get_logger()

# Used only when an episode -- and every other queued episode -- lacks a
# PodcastIndex-reported duration; a flat guess is fine since it'd only ever
# bias load balance, never correctness.
DEFAULT_DURATION_SECONDS_FALLBACK = 1800.0


def assign_shards(episodes: list, num_shards: int) -> tuple[dict[str, int], list[float]]:
    """Longest-Processing-Time-first greedy: sort episodes by duration
    descending, repeatedly place the next-longest episode on whichever
    shard currently has the smallest total assigned duration. Returns the
    episode_id -> shard_id assignment and each shard's total seconds."""
    known_durations = [e["duration_seconds_reported"] for e in episodes if e["duration_seconds_reported"]]
    fallback = statistics.mean(known_durations) if known_durations else DEFAULT_DURATION_SECONDS_FALLBACK

    durations = {e["episode_id"]: e["duration_seconds_reported"] or fallback for e in episodes}
    ordered = sorted(episodes, key=lambda e: durations[e["episode_id"]], reverse=True)

    shard_totals = [0.0] * num_shards
    assignment: dict[str, int] = {}
    for e in ordered:
        shard = min(range(num_shards), key=lambda s: shard_totals[s])
        assignment[e["episode_id"]] = shard
        shard_totals[shard] += durations[e["episode_id"]]
    return assignment, shard_totals


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, required=True, help="number of pods/shards to partition across")
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH))
    parser.add_argument("--dry-run", action="store_true", help="print the planned assignment; don't write assigned_shard")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.shards < 1:
        logger.error("--shards must be >= 1")
        return 1

    conn = db.connect(args.db)
    db.init_db(conn)

    episodes = db.list_queued_episodes(conn, shard=None)
    if not episodes:
        logger.warning("no unassigned queued episodes found -- nothing to partition")
        return 0

    assignment, shard_totals = assign_shards(episodes, args.shards)

    for shard_id, total_seconds in enumerate(shard_totals):
        count = sum(1 for s in assignment.values() if s == shard_id)
        logger.info("shard %d: %d episodes, ~%.1fh", shard_id, count, total_seconds / 3600.0)

    if args.dry_run:
        logger.info("dry run: not writing assigned_shard")
        return 0

    for episode_id, shard_id in assignment.items():
        db.set_assigned_shard(conn, episode_id, shard_id)
    logger.info("partitioned %d episodes across %d shards", len(assignment), args.shards)
    return 0


if __name__ == "__main__":
    sys.exit(main())
