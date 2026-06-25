"""Episode ingestion I/O: downloads raw episode audio over HTTP and registers
PodcastIndex episode metadata as queued rows. Pure I/O, no stage-machine
awareness -- like audio.py/podcastindex_client.py, the state-machine logic
(advance_stage/mark_stage_failed around these calls) lives in
pipeline_runner.py, not here.
"""

from __future__ import annotations

from pathlib import Path

import requests

from pipeline import db

DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1MB


class IngestError(RuntimeError):
    pass


def register_episode_from_podcastindex(conn, podcast_id: str, pi_episode: dict) -> str:
    """Inserts one PodcastIndex episode dict (one item from
    PodcastIndexClient.get_episodes_by_feed_id) as a queued episode row --
    idempotent via db.insert_episode's INSERT OR IGNORE. Returns episode_id."""
    pi_episode_id = str(pi_episode["id"])
    episode_id = f"{podcast_id}_ep_{pi_episode_id}"
    published_at = pi_episode.get("datePublished")
    duration = pi_episode.get("duration")
    db.insert_episode(
        conn, episode_id, podcast_id, pi_episode_id,
        title=pi_episode.get("title") or episode_id,
        source_url=pi_episode["enclosureUrl"],
        published_at=str(published_at) if published_at else None,
        duration_seconds_reported=float(duration) if duration else None,
    )
    return episode_id


def source_file_suffix(url: str) -> str:
    """A short, filesystem-safe extension guess from the enclosure URL --
    falls back to .mp3 (the overwhelmingly common podcast enclosure format)
    when the URL has no usable suffix (e.g. a query-string-only tracking
    redirect URL)."""
    suffix = Path(url.split("?")[0]).suffix
    return suffix if suffix and len(suffix) <= 5 else ".mp3"


def download_episode_audio(source_url: str, dest_path: str | Path, timeout: float = 120.0) -> None:
    """Streams the episode's raw audio to disk."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(source_url, stream=True, timeout=timeout) as resp:
        if resp.status_code != 200:
            raise IngestError(f"download failed (HTTP {resp.status_code}) for {source_url}")
        with dest_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                if chunk:
                    f.write(chunk)
