"""Tests for pipeline/discovery.py: the no-signup free-path discovery layer
(iTunes Search + direct RSS parsing). Network calls (requests.get) and the
ffprobe subprocess are faked -- no real HTTP, no real audio file."""

from __future__ import annotations

import json
import subprocess

import pytest

from pipeline import discovery


class FakeResponse:
    def __init__(self, status_code: int = 200, json_data: dict | None = None, content: bytes = b"", text: str = ""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.content = content
        self.text = text

    def json(self):
        return self._json_data


RSS_FEED_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Test Feed</title>
    <language>en-us</language>
    <item>
      <title>Episode One</title>
      <guid>guid-1</guid>
      <pubDate>Mon, 01 Jan 2024 12:00:00 +0000</pubDate>
      <itunes:duration>01:02:03</itunes:duration>
      <enclosure url="https://example.com/ep1.mp3" type="audio/mpeg" />
    </item>
    <item>
      <title>Episode Two (no enclosure)</title>
      <guid>guid-2</guid>
      <pubDate>Tue, 02 Jan 2024 12:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Episode Three</title>
      <guid>guid-3</guid>
      <enclosure url="https://example.com/ep3.mp3" type="audio/mpeg" />
    </item>
  </channel>
</rss>
"""


# -- itunes_search_podcasts --------------------------------------------------

def test_itunes_search_podcasts_returns_results(monkeypatch):
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return FakeResponse(json_data={"results": [{"feedUrl": "https://x.com/feed.xml"}]})

    monkeypatch.setattr(discovery.requests, "get", fake_get)
    results = discovery.itunes_search_podcasts("history", limit=10, country="GB")
    assert results == [{"feedUrl": "https://x.com/feed.xml"}]
    assert captured["url"] == discovery.ITUNES_SEARCH_URL
    assert captured["params"] == {"term": "history", "media": "podcast", "entity": "podcast", "limit": 10, "country": "GB"}


def test_itunes_search_podcasts_raises_on_non_200(monkeypatch):
    monkeypatch.setattr(discovery.requests, "get", lambda *a, **k: FakeResponse(status_code=500, text="boom"))
    with pytest.raises(discovery.DiscoveryError):
        discovery.itunes_search_podcasts("history")


def test_itunes_search_podcasts_missing_results_key_returns_empty_list(monkeypatch):
    monkeypatch.setattr(discovery.requests, "get", lambda *a, **k: FakeResponse(json_data={}))
    assert discovery.itunes_search_podcasts("history") == []


# -- _parse_pubdate -----------------------------------------------------------

def test_parse_pubdate_valid_rfc822():
    result = discovery._parse_pubdate("Mon, 01 Jan 2024 12:00:00 +0000")
    assert result is not None
    assert result.startswith("2024-01-01T12:00:00")


def test_parse_pubdate_none_input():
    assert discovery._parse_pubdate(None) is None


def test_parse_pubdate_unparseable_returns_none():
    assert discovery._parse_pubdate("not a date") is None


# -- _parse_itunes_duration ----------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        ("45", 45.0),
        ("02:30", 150.0),
        ("01:02:03", 3723.0),
        (None, None),
        ("", None),
        ("garbage", None),
        ("1:2:3:4", None),
    ],
)
def test_parse_itunes_duration(value, expected):
    assert discovery._parse_itunes_duration(value) == expected


# -- fetch_rss_feed ------------------------------------------------------------

def test_fetch_rss_feed_parses_episodes_and_skips_missing_enclosure(monkeypatch):
    monkeypatch.setattr(discovery.requests, "get", lambda *a, **k: FakeResponse(content=RSS_FEED_XML))
    feed = discovery.fetch_rss_feed("https://example.com/feed.xml")
    assert feed.title == "Test Feed"
    assert feed.language == "en-us"
    # Episode Two has no enclosure and must be skipped.
    assert [ep.guid for ep in feed.episodes] == ["guid-1", "guid-3"]
    ep1 = feed.episodes[0]
    assert ep1.enclosure_url == "https://example.com/ep1.mp3"
    assert ep1.duration_seconds == 3723.0
    assert ep1.published_at is not None
    ep3 = feed.episodes[1]
    assert ep3.duration_seconds is None
    assert ep3.published_at is None


def test_fetch_rss_feed_respects_max_episodes(monkeypatch):
    monkeypatch.setattr(discovery.requests, "get", lambda *a, **k: FakeResponse(content=RSS_FEED_XML))
    feed = discovery.fetch_rss_feed("https://example.com/feed.xml", max_episodes=1)
    assert len(feed.episodes) == 1
    assert feed.episodes[0].guid == "guid-1"


def test_fetch_rss_feed_raises_on_non_200(monkeypatch):
    monkeypatch.setattr(discovery.requests, "get", lambda *a, **k: FakeResponse(status_code=404))
    with pytest.raises(discovery.DiscoveryError):
        discovery.fetch_rss_feed("https://example.com/feed.xml")


def test_fetch_rss_feed_raises_on_malformed_xml(monkeypatch):
    monkeypatch.setattr(discovery.requests, "get", lambda *a, **k: FakeResponse(content=b"<not valid xml"))
    with pytest.raises(discovery.DiscoveryError):
        discovery.fetch_rss_feed("https://example.com/feed.xml")


def test_fetch_rss_feed_raises_when_no_channel(monkeypatch):
    monkeypatch.setattr(discovery.requests, "get", lambda *a, **k: FakeResponse(content=b"<rss></rss>"))
    with pytest.raises(discovery.DiscoveryError):
        discovery.fetch_rss_feed("https://example.com/feed.xml")


def test_fetch_rss_feed_falls_back_to_enclosure_url_and_title_when_missing(monkeypatch):
    xml = b"""<rss><channel><title>F</title>
      <item><enclosure url="https://example.com/no-guid.mp3" /></item>
    </channel></rss>"""
    monkeypatch.setattr(discovery.requests, "get", lambda *a, **k: FakeResponse(content=xml))
    feed = discovery.fetch_rss_feed("https://example.com/feed.xml")
    assert feed.episodes[0].guid == "https://example.com/no-guid.mp3"
    assert feed.episodes[0].title == "https://example.com/no-guid.mp3"


# -- probe_duration_seconds -----------------------------------------------------

def _fake_completed_process(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_probe_duration_seconds_parses_ffprobe_output(monkeypatch):
    payload = json.dumps({"format": {"duration": "123.45"}})
    monkeypatch.setattr(discovery.subprocess, "run", lambda *a, **k: _fake_completed_process(stdout=payload))
    assert discovery.probe_duration_seconds("https://example.com/ep.mp3") == 123.45


def test_probe_duration_seconds_returns_none_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(discovery.subprocess, "run", lambda *a, **k: _fake_completed_process(returncode=1))
    assert discovery.probe_duration_seconds("https://example.com/ep.mp3") is None


def test_probe_duration_seconds_returns_none_on_timeout(monkeypatch):
    def raise_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)

    monkeypatch.setattr(discovery.subprocess, "run", raise_timeout)
    assert discovery.probe_duration_seconds("https://example.com/ep.mp3") is None


def test_probe_duration_seconds_returns_none_on_malformed_json(monkeypatch):
    monkeypatch.setattr(discovery.subprocess, "run", lambda *a, **k: _fake_completed_process(stdout="not json"))
    assert discovery.probe_duration_seconds("https://example.com/ep.mp3") is None


def test_probe_duration_seconds_returns_none_when_duration_missing(monkeypatch):
    monkeypatch.setattr(discovery.subprocess, "run", lambda *a, **k: _fake_completed_process(stdout=json.dumps({"format": {}})))
    assert discovery.probe_duration_seconds("https://example.com/ep.mp3") is None


# -- id helpers -----------------------------------------------------------------

def test_podcast_id_for_feed_url_is_deterministic_and_formatted():
    id1 = discovery.podcast_id_for_feed_url("https://example.com/feed.xml")
    id2 = discovery.podcast_id_for_feed_url("https://example.com/feed.xml")
    assert id1 == id2
    assert id1.startswith("free_")
    assert len(id1) == len("free_") + 12


def test_podcast_id_for_feed_url_differs_per_url():
    id1 = discovery.podcast_id_for_feed_url("https://example.com/feed-a.xml")
    id2 = discovery.podcast_id_for_feed_url("https://example.com/feed-b.xml")
    assert id1 != id2


def test_episode_slug_for_guid_is_deterministic_and_formatted():
    slug1 = discovery.episode_slug_for_guid("some-guid")
    slug2 = discovery.episode_slug_for_guid("some-guid")
    assert slug1 == slug2
    assert len(slug1) == 16


def test_episode_slug_for_guid_differs_per_guid():
    assert discovery.episode_slug_for_guid("guid-a") != discovery.episode_slug_for_guid("guid-b")
