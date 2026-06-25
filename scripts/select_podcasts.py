"""Selects interview/talk-style podcasts from PodcastIndex whose combined
catalog can supply the configured raw-hour candidate pool (config/pipeline.yaml's
target_corpus section), writes config/podcasts.json for reproducibility, and
registers each selected podcast + its sampled episodes into the local SQLite db
as queued rows.

This is the only script that talks to PodcastIndex, so it does double duty as
the step that populates `episodes` with `queued` rows -- the precondition
scripts/partition_episodes.py and Stage 1's single-pod run both depend on.

PodcastIndex's API has no paid tier for this usage (search/episode lookups are
free), so this script needs no spend confirmation.

Run with: python3 scripts/select_podcasts.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline import config, db, ingest, logging_utils  # noqa: E402
from pipeline.podcastindex_client import PodcastIndexClient, PodcastIndexError  # noqa: E402

logger = logging_utils.get_logger()

DEFAULT_QUERIES = [
    "interview",
    "conversation",
    "talk show",
    "long form interview",
    "in depth conversation",
]

DEFAULT_OUTPUT = REPO_ROOT / "config" / "podcasts.json"


@dataclass
class Candidate:
    feed_id: str
    title: str
    feed_url: str
    language: str | None
    episode_count_reported: int | None
    sampled_episodes: list[dict] = field(default_factory=list)


def avg_episode_duration_seconds(candidate: Candidate) -> float | None:
    durations = [e["duration"] for e in candidate.sampled_episodes if e.get("duration")]
    return statistics.mean(durations) if durations else None


def estimated_hours_contributed(candidate: Candidate, episodes_per_podcast_max: int) -> float:
    """Modeled raw-hour contribution if up to episodes_per_podcast_max of the
    sampled episodes get registered -- the basis for pool-sizing decisions."""
    avg = avg_episode_duration_seconds(candidate)
    if not avg:
        return 0.0
    n = min(len(candidate.sampled_episodes), episodes_per_podcast_max)
    return avg * n / 3600.0


def select_candidates(
    candidates: list[Candidate],
    *,
    raw_hours_pool_min: float,
    raw_hours_pool_max: float,
    podcast_count_min: int,
    podcast_count_max: int,
    episodes_per_podcast_min: int,
    episodes_per_podcast_max: int,
) -> list[Candidate]:
    """Greedily selects candidates, longest-average-episode-duration first --
    per the plan, longer episodes amortize fixed intro/outro/ad overhead
    better -- stopping once the raw-hour pool target is met (with at least
    podcast_count_min shows) or the podcast-count ceiling is hit. A
    best-effort fallback (not a hard guarantee): if viable candidates run out
    before the minimum pool/count is reached, this returns everything viable
    rather than failing -- callers should check the returned total against
    the target and surface a warning."""
    viable = [
        c
        for c in candidates
        if avg_episode_duration_seconds(c) is not None
        and (c.episode_count_reported is None or c.episode_count_reported >= episodes_per_podcast_min)
        and len(c.sampled_episodes) >= episodes_per_podcast_min
    ]
    viable.sort(key=lambda c: avg_episode_duration_seconds(c), reverse=True)

    selected: list[Candidate] = []
    total_hours = 0.0
    for c in viable:
        if len(selected) >= podcast_count_max:
            break
        selected.append(c)
        total_hours += estimated_hours_contributed(c, episodes_per_podcast_max)
        if total_hours >= raw_hours_pool_min and len(selected) >= podcast_count_min:
            break
    return selected


def fetch_candidates(
    client: PodcastIndexClient,
    queries: list[str],
    candidates_per_query: int,
    sample_episodes: int,
    language_prefix: str,
) -> list[Candidate]:
    seen: dict[str, Candidate] = {}
    for query in queries:
        for feed in client.search_podcasts(query, max_results=candidates_per_query):
            feed_id = feed.get("id")
            if feed_id is None or str(feed_id) in seen:
                continue
            language = feed.get("language")
            if language_prefix and language and not language.lower().startswith(language_prefix.lower()):
                continue
            url = feed.get("url")
            if not url:
                continue
            seen[str(feed_id)] = Candidate(
                feed_id=str(feed_id),
                title=feed.get("title") or str(feed_id),
                feed_url=url,
                language=language,
                episode_count_reported=feed.get("episodeCount"),
            )
        logger.info("query %r: %d distinct candidates so far", query, len(seen))

    for candidate in seen.values():
        try:
            candidate.sampled_episodes = client.get_episodes_by_feed_id(
                int(candidate.feed_id), max_results=sample_episodes
            )
        except PodcastIndexError as exc:
            logger.warning("episode fetch failed for feed %s (%s): %s", candidate.feed_id, candidate.title, exc)
    return list(seen.values())


def podcast_id_for_feed(feed_id: str) -> str:
    return f"pi_{feed_id}"


def register_selection(conn, selected: list[Candidate], episodes_per_podcast_max: int) -> None:
    for candidate in selected:
        podcast_id = podcast_id_for_feed(candidate.feed_id)
        db.insert_podcast(
            conn,
            podcast_id,
            feed_id=candidate.feed_id,
            title=candidate.title,
            feed_url=candidate.feed_url,
            language=candidate.language,
            episode_count_total=candidate.episode_count_reported,
            selection_reason="select_podcasts.py: interview/talk-style, sized to target_corpus raw-hour pool",
        )
        for pi_episode in candidate.sampled_episodes[:episodes_per_podcast_max]:
            ingest.register_episode_from_podcastindex(conn, podcast_id, pi_episode)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to pipeline.yaml")
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH), help="sqlite db path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="where to write podcasts.json")
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES, help="PodcastIndex search terms")
    parser.add_argument("--candidates-per-query", type=int, default=25)
    parser.add_argument("--language-prefix", default="en", help="keep feeds whose language starts with this (empty disables the filter)")
    parser.add_argument("--no-register", action="store_true", help="write podcasts.json only; skip db registration")
    parser.add_argument("--dry-run", action="store_true", help="fetch + select + print summary; write nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = config.load_config(args.config)
    secrets = config.EnvSecrets.from_env()
    if not secrets.podcastindex_api_key or not secrets.podcastindex_api_secret:
        logger.error("PODCASTINDEX_API_KEY / PODCASTINDEX_API_SECRET not set; see .env.example")
        return 1

    client = PodcastIndexClient(secrets.podcastindex_api_key, secrets.podcastindex_api_secret)
    tc = cfg.target_corpus

    logger.info("searching PodcastIndex for candidates (%d queries)...", len(args.queries))
    candidates = fetch_candidates(
        client,
        args.queries,
        args.candidates_per_query,
        sample_episodes=tc.episodes_per_podcast_max,
        language_prefix=args.language_prefix,
    )
    logger.info("fetched %d distinct candidates with episode samples", len(candidates))

    selected = select_candidates(
        candidates,
        raw_hours_pool_min=tc.raw_hours_pool_min,
        raw_hours_pool_max=tc.raw_hours_pool_max,
        podcast_count_min=tc.podcast_count_min,
        podcast_count_max=tc.podcast_count_max,
        episodes_per_podcast_min=tc.episodes_per_podcast_min,
        episodes_per_podcast_max=tc.episodes_per_podcast_max,
    )
    total_hours = sum(estimated_hours_contributed(c, tc.episodes_per_podcast_max) for c in selected)
    logger.info(
        "selected %d podcasts, estimated raw-hour pool ~%.1fh (target %.0f-%.0fh)",
        len(selected), total_hours, tc.raw_hours_pool_min, tc.raw_hours_pool_max,
    )
    if total_hours < tc.raw_hours_pool_min:
        logger.warning(
            "selected pool (~%.1fh) is below the configured minimum (%.0fh) -- "
            "ran out of viable candidates from the given queries; consider widening --queries",
            total_hours, tc.raw_hours_pool_min,
        )

    rows = [
        {
            "podcast_id": podcast_id_for_feed(c.feed_id),
            "feed_id": c.feed_id,
            "title": c.title,
            "feed_url": c.feed_url,
            "language": c.language,
            "episode_count_reported": c.episode_count_reported,
            "avg_episode_duration_seconds": avg_episode_duration_seconds(c),
            "episodes_registered": min(len(c.sampled_episodes), tc.episodes_per_podcast_max),
            "estimated_hours_contributed": round(estimated_hours_contributed(c, tc.episodes_per_podcast_max), 2),
        }
        for c in selected
    ]
    for row in rows:
        print(json.dumps(row))

    if args.dry_run:
        logger.info("dry run: not writing %s or registering in db", args.output)
        return 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2) + "\n")
    logger.info("wrote %s", output_path)

    if not args.no_register:
        conn = db.connect(args.db)
        db.init_db(conn)
        register_selection(conn, selected, tc.episodes_per_podcast_max)
        logger.info("registered %d podcasts and their episodes in %s", len(selected), args.db)

    return 0


if __name__ == "__main__":
    sys.exit(main())
