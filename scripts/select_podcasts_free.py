"""Selects interview/talk-style podcasts via the free, no-signup iTunes
Search API + direct RSS parsing (pipeline/discovery.py) whose combined
catalog can supply the configured raw-hour candidate pool
(config/pipeline.yaml's target_corpus section), writes
config/podcasts_free.json for reproducibility, and registers each selected
podcast + its sampled episodes into the local SQLite db as queued rows.

RSS/iTunes sibling of scripts/select_podcasts.py (which talks to
PodcastIndex and is left untouched for later credentialed use). Both
scripts write into the same `podcasts`/`episodes` tables under different
podcast_id namespaces (`pi_*` vs `free_*`), so the two discovery paths never
collide.

Needs no credentials and no spend confirmation: iTunes Search + RSS are both
free, anonymous endpoints.

Run with: python3 scripts/select_podcasts_free.py
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

from pipeline import config, db, discovery, ingest, logging_utils  # noqa: E402

logger = logging_utils.get_logger()

DEFAULT_QUERIES = [
    "interview",
    "conversation",
    "talk show",
    "long form interview",
    "in depth conversation",
]

DEFAULT_OUTPUT = REPO_ROOT / "config" / "podcasts_free.json"
# ffprobe calls per candidate for episodes RSS doesn't report a duration for; each is a
# real network round trip, so this stays small rather than probing every sampled episode.
DEFAULT_DURATION_PROBE_SAMPLE = 3


@dataclass
class Candidate:
    feed_url: str
    title: str
    language: str | None
    episode_count_reported: int | None
    sampled_episodes: list[discovery.RssEpisode] = field(default_factory=list)


def avg_episode_duration_seconds(candidate: Candidate) -> float | None:
    durations = [e.duration_seconds for e in candidate.sampled_episodes if e.duration_seconds]
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
    """Greedy, longest-average-episode-duration-first selection, stopping
    once the raw-hour pool target is met -- same algorithm and rationale as
    scripts/select_podcasts.py's select_candidates (longer episodes amortize
    fixed intro/outro/ad overhead better)."""
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


def _fill_missing_durations(candidate: Candidate, sample_size: int) -> None:
    """RSS-reported durations are used as-is; for episodes missing one (e.g.
    Hacker Public Radio's feed never sets <itunes:duration>), probe up to
    `sample_size` of them via ffprobe so avg_episode_duration_seconds still
    has data to work with."""
    probed = 0
    for ep in candidate.sampled_episodes:
        if ep.duration_seconds is not None:
            continue
        if probed >= sample_size:
            break
        ep.duration_seconds = discovery.probe_duration_seconds(ep.enclosure_url)
        probed += 1


def fetch_candidates(
    queries: list[str],
    candidates_per_query: int,
    sample_episodes: int,
    language_prefix: str,
    duration_probe_sample: int,
) -> list[Candidate]:
    seen: dict[str, Candidate] = {}
    for query in queries:
        try:
            results = discovery.itunes_search_podcasts(query, limit=candidates_per_query)
        except discovery.DiscoveryError as exc:
            logger.warning("iTunes search failed for %r: %s", query, exc)
            continue
        for result in results:
            feed_url = result.get("feedUrl")
            if not feed_url or feed_url in seen:
                continue
            seen[feed_url] = Candidate(
                feed_url=feed_url,
                title=result.get("collectionName") or feed_url,
                language=None,  # iTunes search results don't expose RSS <language>; filled in from the feed itself below
                episode_count_reported=result.get("trackCount"),
            )
        logger.info("query %r: %d distinct candidates so far", query, len(seen))

    for feed_url in list(seen.keys()):
        candidate = seen[feed_url]
        try:
            feed = discovery.fetch_rss_feed(feed_url, max_episodes=sample_episodes)
        except discovery.DiscoveryError as exc:
            logger.warning("RSS fetch failed for %s (%s): %s", feed_url, candidate.title, exc)
            del seen[feed_url]
            continue
        candidate.title = feed.title or candidate.title
        candidate.language = feed.language
        candidate.sampled_episodes = feed.episodes
        if language_prefix and candidate.language and not candidate.language.lower().startswith(language_prefix.lower()):
            del seen[feed_url]
            continue
        _fill_missing_durations(candidate, duration_probe_sample)

    return list(seen.values())


def register_selection(conn, selected: list[Candidate], episodes_per_podcast_max: int) -> None:
    for candidate in selected:
        podcast_id = discovery.podcast_id_for_feed_url(candidate.feed_url)
        db.insert_podcast(
            conn,
            podcast_id,
            feed_id=podcast_id,
            title=candidate.title,
            feed_url=candidate.feed_url,
            language=candidate.language,
            episode_count_total=candidate.episode_count_reported,
            selection_reason="select_podcasts_free.py: interview/talk-style, sized to target_corpus raw-hour pool (iTunes Search + RSS, no signup)",
        )
        for ep in candidate.sampled_episodes[:episodes_per_podcast_max]:
            ingest.register_episode_from_rss(conn, podcast_id, ep)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to pipeline.yaml")
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH), help="sqlite db path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="where to write podcasts_free.json")
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES, help="iTunes search terms")
    parser.add_argument("--candidates-per-query", type=int, default=25)
    parser.add_argument("--language-prefix", default="en", help="keep feeds whose RSS <language> starts with this (empty disables the filter)")
    parser.add_argument("--duration-probe-sample", type=int, default=DEFAULT_DURATION_PROBE_SAMPLE, help="max ffprobe calls per candidate for episodes missing RSS duration")
    parser.add_argument("--no-register", action="store_true", help="write podcasts_free.json only; skip db registration")
    parser.add_argument("--dry-run", action="store_true", help="fetch + select + print summary; write nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = config.load_config(args.config)
    tc = cfg.target_corpus

    logger.info("searching iTunes for candidates (%d queries)...", len(args.queries))
    candidates = fetch_candidates(
        args.queries,
        args.candidates_per_query,
        sample_episodes=tc.episodes_per_podcast_max,
        language_prefix=args.language_prefix,
        duration_probe_sample=args.duration_probe_sample,
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
            "podcast_id": discovery.podcast_id_for_feed_url(c.feed_url),
            "feed_url": c.feed_url,
            "title": c.title,
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
