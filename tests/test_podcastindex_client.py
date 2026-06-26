"""Tests for pipeline/podcastindex_client.py: auth header construction (the
sha1(api_key + api_secret + timestamp) scheme PodcastIndex requires) and the
search/episodes call wrappers, against a fake requests.Session -- no real
network call, no real credentials needed."""

from __future__ import annotations

import hashlib

import pytest

from pipeline.podcastindex_client import API_BASE, PodcastIndexClient, PodcastIndexError


class FakeResponse:
    def __init__(self, status_code: int = 200, json_data: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text or str(json_data)

    def json(self):
        return self._json_data


class FakeSession:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls: list[dict] = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "params": params, "timeout": timeout})
        return self.response


def make_client(response: FakeResponse, **kwargs) -> tuple[PodcastIndexClient, FakeSession]:
    session = FakeSession(response)
    client = PodcastIndexClient("key123", "secret456", session=session, **kwargs)
    return client, session


def test_init_requires_key_and_secret():
    with pytest.raises(PodcastIndexError):
        PodcastIndexClient("", "secret")
    with pytest.raises(PodcastIndexError):
        PodcastIndexClient("key", "")


def test_headers_auth_hash_matches_documented_scheme(monkeypatch):
    client, _ = make_client(FakeResponse())
    monkeypatch.setattr("pipeline.podcastindex_client.time.time", lambda: 1700000000.0)
    headers = client._headers()
    expected_hash = hashlib.sha1(b"key123secret4561700000000").hexdigest()
    assert headers["Authorization"] == expected_hash
    assert headers["X-Auth-Date"] == "1700000000"
    assert headers["X-Auth-Key"] == "key123"
    assert "User-Agent" in headers


def test_headers_timestamp_changes_each_call(monkeypatch):
    client, _ = make_client(FakeResponse())
    times = iter([1000.0, 2000.0])
    monkeypatch.setattr("pipeline.podcastindex_client.time.time", lambda: next(times))
    first = client._headers()
    second = client._headers()
    assert first["X-Auth-Date"] != second["X-Auth-Date"]
    assert first["Authorization"] != second["Authorization"]


def test_get_raises_on_non_200():
    client, _ = make_client(FakeResponse(status_code=503, text="upstream down"))
    with pytest.raises(PodcastIndexError, match="503"):
        client.search_podcasts("test")


def test_search_podcasts_returns_feeds_array():
    feeds = [{"id": 1, "title": "Show A"}, {"id": 2, "title": "Show B"}]
    client, session = make_client(FakeResponse(json_data={"feeds": feeds}))
    result = client.search_podcasts("history", max_results=10)
    assert result == feeds
    call = session.calls[0]
    assert call["url"] == f"{API_BASE}/search/byterm"
    assert call["params"] == {"q": "history", "max": 10}


def test_search_podcasts_missing_feeds_key_returns_empty_list():
    client, _ = make_client(FakeResponse(json_data={}))
    assert client.search_podcasts("nothing") == []


def test_get_podcast_by_feed_id_returns_feed_dict():
    client, session = make_client(FakeResponse(json_data={"feed": {"id": 42, "title": "Show"}}))
    result = client.get_podcast_by_feed_id(42)
    assert result == {"id": 42, "title": "Show"}
    assert session.calls[0]["params"] == {"id": 42}


def test_get_podcast_by_feed_id_missing_feed_returns_none():
    client, _ = make_client(FakeResponse(json_data={}))
    assert client.get_podcast_by_feed_id(42) is None


def test_get_episodes_by_feed_id_omits_since_when_not_given():
    client, session = make_client(FakeResponse(json_data={"items": []}))
    client.get_episodes_by_feed_id(7, max_results=50)
    assert session.calls[0]["params"] == {"id": 7, "max": 50}


def test_get_episodes_by_feed_id_includes_since_when_given():
    client, session = make_client(FakeResponse(json_data={"items": []}))
    client.get_episodes_by_feed_id(7, max_results=50, since=1600000000)
    assert session.calls[0]["params"] == {"id": 7, "max": 50, "since": 1600000000}


def test_get_episodes_by_feed_id_returns_items_array():
    items = [{"id": 1}, {"id": 2}]
    client, _ = make_client(FakeResponse(json_data={"items": items}))
    assert client.get_episodes_by_feed_id(7) == items


def test_check_connectivity_uses_cheap_query():
    client, session = make_client(FakeResponse(json_data={"feeds": []}))
    client.check_connectivity()
    call = session.calls[0]
    assert call["url"] == f"{API_BASE}/search/byterm"
    assert call["params"] == {"q": "test", "max": 1}


def test_check_connectivity_raises_on_failure():
    client, _ = make_client(FakeResponse(status_code=401, text="bad auth"))
    with pytest.raises(PodcastIndexError):
        client.check_connectivity()
