"""PodcastIndex API client: auth header construction + search/episodes calls.

Auth scheme (per api.podcastindex.org docs): every request carries
X-Auth-Date (unix timestamp), X-Auth-Key (api key), and an Authorization
header that is sha1(api_key + api_secret + timestamp) hex-encoded — no
request signing beyond that single hash, and no token to refresh.
"""

from __future__ import annotations

import hashlib
import time

import requests

API_BASE = "https://api.podcastindex.org/api/1.0"
USER_AGENT = "podcast-speech-dataset-pipeline/1.0"


class PodcastIndexError(RuntimeError):
    pass


class PodcastIndexClient:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        session: requests.Session | None = None,
        timeout: float = 30.0,
    ):
        if not api_key or not api_secret:
            raise PodcastIndexError("PodcastIndex API key/secret required")
        self._api_key = api_key
        self._api_secret = api_secret
        self._session = session or requests.Session()
        self._timeout = timeout

    def _headers(self) -> dict:
        timestamp = str(int(time.time()))
        auth_hash = hashlib.sha1(
            (self._api_key + self._api_secret + timestamp).encode("utf-8")
        ).hexdigest()
        return {
            "X-Auth-Date": timestamp,
            "X-Auth-Key": self._api_key,
            "Authorization": auth_hash,
            "User-Agent": USER_AGENT,
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{API_BASE}{path}"
        resp = self._session.get(url, headers=self._headers(), params=params, timeout=self._timeout)
        if resp.status_code != 200:
            raise PodcastIndexError(f"PodcastIndex {path} returned HTTP {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    def search_podcasts(self, query: str, max_results: int = 40) -> list[dict]:
        """Search by term; returns the `feeds` array (podcast-level metadata)."""
        data = self._get("/search/byterm", params={"q": query, "max": max_results})
        return data.get("feeds", [])

    def get_podcast_by_feed_id(self, feed_id: int) -> dict | None:
        data = self._get("/podcasts/byfeedid", params={"id": feed_id})
        return data.get("feed")

    def get_episodes_by_feed_id(
        self, feed_id: int, max_results: int = 100, since: int | None = None
    ) -> list[dict]:
        """`since` is a unix timestamp; when set, only episodes published after
        it are returned — used for incremental episode discovery on reruns."""
        params: dict = {"id": feed_id, "max": max_results}
        if since is not None:
            params["since"] = since
        data = self._get("/episodes/byfeedid", params=params)
        return data.get("items", [])

    def check_connectivity(self) -> None:
        """Cheap auth-validating call for scripts/smoke_test.py; raises PodcastIndexError on failure."""
        self._get("/search/byterm", params={"q": "test", "max": 1})
