"""No-signup podcast discovery: the iTunes Search API
(https://itunes.apple.com/search) for podcast-level metadata + feed URLs,
and direct RSS parsing (stdlib xml.etree.ElementTree, no extra dependency)
for episode-level enclosure URLs/titles/publish dates. Free, anonymous, no
API key -- the discovery-layer swap for the free/local trial path. See
pipeline/podcastindex_client.py for the credentialed equivalent, left
untouched for later use when PodcastIndex credentials are available.

Episode duration: many RSS feeds simply omit <itunes:duration> (Hacker
Public Radio, for one) -- rather than discard those podcasts as "unviable"
the way scripts/select_podcasts.py's PodcastIndex-reported-duration check
would, probe_duration_seconds() falls back to a real ffprobe(1) call against
the enclosure URL for episodes missing it. ffprobe only reads as much of the
remote stream as it needs to read the container header, not the whole file,
so this stays cheap even sampled across many candidate episodes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime

import requests

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
USER_AGENT = "podcast-speech-dataset-pipeline/1.0 (free-path discovery)"
ITUNES_DURATION_TAG = "{http://www.itunes.com/dtds/podcast-1.0.dtd}duration"


class DiscoveryError(RuntimeError):
    pass


@dataclass
class RssEpisode:
    guid: str
    title: str
    enclosure_url: str
    published_at: str | None
    duration_seconds: float | None


@dataclass
class RssFeed:
    title: str
    language: str | None
    episodes: list[RssEpisode] = field(default_factory=list)


def itunes_search_podcasts(term: str, limit: int = 25, country: str = "US") -> list[dict]:
    """Returns the raw `results` array from the public, no-auth iTunes
    Search API -- podcast-level dicts with feedUrl/collectionName/
    primaryGenreName/trackCount among other fields."""
    resp = requests.get(
        ITUNES_SEARCH_URL,
        params={"term": term, "media": "podcast", "entity": "podcast", "limit": limit, "country": country},
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
    )
    if resp.status_code != 200:
        raise DiscoveryError(f"iTunes search returned HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json().get("results", [])


def _parse_pubdate(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).isoformat()
    except (TypeError, ValueError):
        return None


def _parse_itunes_duration(value: str | None) -> float | None:
    """itunes:duration is documented as HH:MM:SS, MM:SS, or plain seconds --
    all three show up in real feeds."""
    if not value:
        return None
    parts = value.strip().split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def probe_duration_seconds(url: str, timeout: float = 30.0) -> float | None:
    """ffprobe(1) fallback for episodes/feeds that don't report duration in
    RSS. Returns None (never raises) on any probe failure -- this is a
    best-effort estimate for candidate selection, not a pipeline-critical
    value."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", url],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    try:
        duration = json.loads(proc.stdout).get("format", {}).get("duration")
        return float(duration) if duration is not None else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def fetch_rss_feed(feed_url: str, timeout: float = 30.0, max_episodes: int | None = None) -> RssFeed:
    """Parses an RSS 2.0 podcast feed directly. `max_episodes` caps how many
    <item>s are parsed, in feed order (RSS items are conventionally
    newest-first) -- discovery only ever needs a sample, not the full
    back-catalog."""
    resp = requests.get(feed_url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    if resp.status_code != 200:
        raise DiscoveryError(f"RSS fetch failed (HTTP {resp.status_code}) for {feed_url}")
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        raise DiscoveryError(f"RSS parse failed for {feed_url}: {exc}") from exc

    channel = root.find("channel")
    if channel is None:
        raise DiscoveryError(f"no <channel> in RSS feed: {feed_url}")

    title = (channel.findtext("title") or feed_url).strip()
    language = channel.findtext("language")

    episodes: list[RssEpisode] = []
    for item in channel.findall("item"):
        enclosure = item.find("enclosure")
        if enclosure is None or not enclosure.get("url"):
            continue  # no playable audio -- e.g. a text-only bonus item
        guid = (item.findtext("guid") or enclosure.get("url")).strip()
        episodes.append(RssEpisode(
            guid=guid,
            title=(item.findtext("title") or guid).strip(),
            enclosure_url=enclosure.get("url"),
            published_at=_parse_pubdate(item.findtext("pubDate")),
            duration_seconds=_parse_itunes_duration(item.findtext(ITUNES_DURATION_TAG)),
        ))
        if max_episodes is not None and len(episodes) >= max_episodes:
            break

    return RssFeed(title=title, language=language, episodes=episodes)


def podcast_id_for_feed_url(feed_url: str) -> str:
    """Deterministic, filesystem-safe podcast_id derived from the feed URL
    (free-path podcasts have no PodcastIndex numeric feed id to key off)."""
    return f"free_{hashlib.sha1(feed_url.encode('utf-8')).hexdigest()[:12]}"


def episode_slug_for_guid(guid: str) -> str:
    """Deterministic, filesystem-safe slug for an RSS guid -- guids are
    often full URLs (as in Hacker Public Radio's feed), but episode_id gets
    embedded directly into on-disk paths by pipeline_runner.py, so it must
    stay short and safe."""
    return hashlib.sha1(guid.encode("utf-8")).hexdigest()[:16]
