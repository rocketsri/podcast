"""Tests for pipeline/db.py: schema creation, episode state machine,
cost ledger, and run_meta key/value store. Pure sqlite, in-memory db,
no audio/network."""

from __future__ import annotations

import sqlite3

import pytest

from pipeline import db


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    db.init_db(connection)
    yield connection
    connection.close()


# --- init_db -----------------------------------------------------------------


def test_init_db_creates_all_tables(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    table_names = {row["name"] for row in rows}
    expected = {
        "podcasts",
        "episodes",
        "speakers",
        "local_speaker_segments",
        "clips",
        "cost_events",
        "run_meta",
    }
    assert expected.issubset(table_names)


def test_init_db_is_idempotent(conn):
    # Calling init_db again on an already-initialized db must not raise.
    db.init_db(conn)
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    assert len(rows) > 0


# --- podcasts / episodes round-trip -------------------------------------------


def test_insert_and_get_podcast_round_trip(conn):
    db.insert_podcast(
        conn,
        podcast_id="pod1",
        feed_id="feed1",
        title="My Podcast",
        feed_url="https://example.com/feed.xml",
        language="en",
        episode_count_total=42,
        selection_reason="longest avg duration",
    )
    row = db.get_podcast(conn, "pod1")
    assert row is not None
    assert row["podcast_id"] == "pod1"
    assert row["feed_id"] == "feed1"
    assert row["title"] == "My Podcast"
    assert row["feed_url"] == "https://example.com/feed.xml"
    assert row["language"] == "en"
    assert row["episode_count_total"] == 42
    assert row["selection_reason"] == "longest avg duration"
    assert row["selected_at"]  # non-empty timestamp


def test_get_podcast_missing_returns_none(conn):
    assert db.get_podcast(conn, "does-not-exist") is None


def test_insert_and_get_episode_round_trip(conn):
    db.insert_podcast(conn, "pod1", "feed1", "My Podcast", "https://example.com/feed.xml")
    db.insert_episode(
        conn,
        episode_id="ep1",
        podcast_id="pod1",
        pi_episode_id="pi-1",
        title="Episode One",
        source_url="https://example.com/ep1.mp3",
        published_at="2024-01-01T00:00:00Z",
        duration_seconds_reported=1800.0,
    )
    row = db.get_episode(conn, "ep1")
    assert row is not None
    assert row["episode_id"] == "ep1"
    assert row["podcast_id"] == "pod1"
    assert row["pi_episode_id"] == "pi-1"
    assert row["title"] == "Episode One"
    assert row["source_url"] == "https://example.com/ep1.mp3"
    assert row["duration_seconds_reported"] == 1800.0
    assert row["stage"] == "queued"
    assert row["attempt_count"] == 0
    assert row["failed_stage"] is None


def test_get_episode_missing_returns_none(conn):
    assert db.get_episode(conn, "nope") is None


def test_insert_episode_or_ignore_does_not_raise_on_duplicate(conn):
    db.insert_podcast(conn, "pod1", "feed1", "My Podcast", "https://example.com/feed.xml")
    db.insert_episode(conn, "ep1", "pod1", "pi-1", "Episode One", "https://example.com/ep1.mp3")
    # Re-inserting the same episode_id must not raise (INSERT OR IGNORE).
    db.insert_episode(conn, "ep1", "pod1", "pi-1", "Episode One (dup)", "https://example.com/ep1.mp3")
    row = db.get_episode(conn, "ep1")
    # Original row wins; the "duplicate" title never overwrote it.
    assert row["title"] == "Episode One"


# --- advance_stage / mark_stage_failed / resume_stage -------------------------


def _make_episode(conn, episode_id="ep1", podcast_id="pod1"):
    db.insert_podcast(conn, podcast_id, "feed1", "My Podcast", "https://example.com/feed.xml")
    db.insert_episode(conn, episode_id, podcast_id, "pi-1", "Episode One", "https://example.com/ep1.mp3")
    return episode_id


def test_advance_stage_moves_stage_forward(conn):
    episode_id = _make_episode(conn)
    db.advance_stage(conn, episode_id, "downloading")
    row = db.get_episode(conn, episode_id)
    assert row["stage"] == "downloading"


def test_advance_stage_clears_failed_stage_and_last_error(conn):
    episode_id = _make_episode(conn)
    db.mark_stage_failed(conn, episode_id, "downloading", "boom")
    row = db.get_episode(conn, episode_id)
    assert row["stage"] == "failed"
    assert row["failed_stage"] == "downloading"
    assert row["last_error"] == "boom"

    db.advance_stage(conn, episode_id, "downloaded")
    row = db.get_episode(conn, episode_id)
    assert row["stage"] == "downloaded"
    assert row["failed_stage"] is None
    assert row["last_error"] is None


def test_advance_stage_can_set_extra_fields(conn):
    episode_id = _make_episode(conn)
    db.advance_stage(conn, episode_id, "downloaded", local_raw_path="/tmp/ep1.mp3", raw_seconds=123.4)
    row = db.get_episode(conn, episode_id)
    assert row["local_raw_path"] == "/tmp/ep1.mp3"
    assert row["raw_seconds"] == 123.4


def test_advance_stage_rejects_unknown_stage(conn):
    episode_id = _make_episode(conn)
    with pytest.raises(db.StageError):
        db.advance_stage(conn, episode_id, "not_a_real_stage")


def test_mark_stage_failed_sets_failed_state_and_increments_attempt_count(conn):
    episode_id = _make_episode(conn)
    db.mark_stage_failed(conn, episode_id, "vad_running", "vad crashed")
    row = db.get_episode(conn, episode_id)
    assert row["stage"] == "failed"
    assert row["failed_stage"] == "vad_running"
    assert row["last_error"] == "vad crashed"
    assert row["attempt_count"] == 1

    db.mark_stage_failed(conn, episode_id, "vad_running", "vad crashed again")
    row = db.get_episode(conn, episode_id)
    assert row["attempt_count"] == 2
    assert row["last_error"] == "vad crashed again"


def test_resume_stage_returns_failed_stage_when_failed(conn):
    episode_id = _make_episode(conn)
    db.mark_stage_failed(conn, episode_id, "diarizing", "oom")
    row = db.get_episode(conn, episode_id)
    assert db.resume_stage(row) == "diarizing"


def test_resume_stage_returns_current_stage_when_not_failed(conn):
    episode_id = _make_episode(conn)
    db.advance_stage(conn, episode_id, "transcoded")
    row = db.get_episode(conn, episode_id)
    assert db.resume_stage(row) == "transcoded"


def test_simulated_mid_stage_failure_then_retry_resumes_from_failure_point(conn):
    """A retry must resume from the stage that actually failed, not from
    `queued` (the episode's original starting point)."""
    episode_id = _make_episode(conn)
    db.advance_stage(conn, episode_id, "downloading")
    db.advance_stage(conn, episode_id, "downloaded")
    db.advance_stage(conn, episode_id, "transcoding")
    db.mark_stage_failed(conn, episode_id, "transcoding", "ffmpeg failed")

    row = db.get_episode(conn, episode_id)
    resume_from = db.resume_stage(row)
    assert resume_from == "transcoding"
    assert resume_from != "queued"

    # Retry succeeds: re-advance to "transcoding" then past it.
    db.advance_stage(conn, episode_id, "transcoding")
    db.advance_stage(conn, episode_id, "transcoded")
    row = db.get_episode(conn, episode_id)
    assert row["stage"] == "transcoded"
    assert row["failed_stage"] is None
    assert row["attempt_count"] == 1  # only incremented by the one mark_stage_failed call


# --- is_at_or_past -------------------------------------------------------------


def test_is_at_or_past_true_when_current_equals_target():
    assert db.is_at_or_past("vad_done", "vad_done") is True


def test_is_at_or_past_true_when_current_is_later():
    assert db.is_at_or_past("diarized", "vad_done") is True


def test_is_at_or_past_false_when_current_is_earlier():
    assert db.is_at_or_past("vad_running", "diarized") is False


def test_is_at_or_past_queued_is_earliest():
    assert db.is_at_or_past("queued", "queued") is True
    assert db.is_at_or_past("queued", "downloading") is False


def test_is_at_or_past_failed_is_never_past_anything():
    """Documented rule: a `failed` episode is never considered past any
    stage, including 'queued' (the earliest stage) and 'failed' itself."""
    assert db.is_at_or_past("failed", "queued") is False
    assert db.is_at_or_past("failed", "downloading") is False
    assert db.is_at_or_past("failed", "done") is False


def test_stage_index_rejects_unknown_stage():
    with pytest.raises(db.StageError):
        db.stage_index("bogus_stage")


def test_is_at_or_past_rejects_unknown_target_stage():
    with pytest.raises(db.StageError):
        db.is_at_or_past("queued", "bogus_stage")


# --- list_queued_episodes / list_failed_episodes ------------------------------


def test_list_queued_episodes_filters_by_shard_none_means_is_null(conn):
    db.insert_podcast(conn, "pod1", "feed1", "P", "https://example.com/feed.xml")
    db.insert_episode(conn, "ep_noshard", "pod1", "pi-1", "E1", "https://example.com/e1.mp3")
    db.insert_episode(conn, "ep_shard0", "pod1", "pi-2", "E2", "https://example.com/e2.mp3")
    db.insert_episode(conn, "ep_shard1", "pod1", "pi-3", "E3", "https://example.com/e3.mp3")
    db.set_assigned_shard(conn, "ep_shard0", 0)
    db.set_assigned_shard(conn, "ep_shard1", 1)

    unsharded = db.list_queued_episodes(conn, shard=None)
    assert {row["episode_id"] for row in unsharded} == {"ep_noshard"}

    shard0 = db.list_queued_episodes(conn, shard=0)
    assert {row["episode_id"] for row in shard0} == {"ep_shard0"}

    shard1 = db.list_queued_episodes(conn, shard=1)
    assert {row["episode_id"] for row in shard1} == {"ep_shard1"}


def test_list_queued_episodes_excludes_non_queued_stage(conn):
    db.insert_podcast(conn, "pod1", "feed1", "P", "https://example.com/feed.xml")
    db.insert_episode(conn, "ep1", "pod1", "pi-1", "E1", "https://example.com/e1.mp3")
    db.advance_stage(conn, "ep1", "downloading")
    assert db.list_queued_episodes(conn, shard=None) == []


def test_list_queued_episodes_orders_shortest_duration_first_nulls_last(conn):
    db.insert_podcast(conn, "pod1", "feed1", "P", "https://example.com/feed.xml")
    db.insert_episode(conn, "ep_long", "pod1", "pi-1", "Long", "https://x/long.mp3", duration_seconds_reported=3600.0)
    db.insert_episode(conn, "ep_null", "pod1", "pi-2", "NullDur", "https://x/null.mp3")
    db.insert_episode(conn, "ep_short", "pod1", "pi-3", "Short", "https://x/short.mp3", duration_seconds_reported=300.0)

    rows = db.list_queued_episodes(conn, shard=None)
    ordered_ids = [row["episode_id"] for row in rows]
    assert ordered_ids == ["ep_short", "ep_long", "ep_null"]


def test_list_failed_episodes_filters_by_shard(conn):
    db.insert_podcast(conn, "pod1", "feed1", "P", "https://example.com/feed.xml")
    db.insert_episode(conn, "ep_a", "pod1", "pi-1", "A", "https://x/a.mp3")
    db.insert_episode(conn, "ep_b", "pod1", "pi-2", "B", "https://x/b.mp3")
    db.set_assigned_shard(conn, "ep_b", 5)
    db.mark_stage_failed(conn, "ep_a", "downloading", "err a")
    db.mark_stage_failed(conn, "ep_b", "vad_running", "err b")

    no_shard_failed = db.list_failed_episodes(conn, shard=None)
    assert {row["episode_id"] for row in no_shard_failed} == {"ep_a"}

    shard5_failed = db.list_failed_episodes(conn, shard=5)
    assert {row["episode_id"] for row in shard5_failed} == {"ep_b"}


def test_list_failed_episodes_excludes_non_failed(conn):
    db.insert_podcast(conn, "pod1", "feed1", "P", "https://example.com/feed.xml")
    db.insert_episode(conn, "ep_a", "pod1", "pi-1", "A", "https://x/a.mp3")
    assert db.list_failed_episodes(conn, shard=None) == []


# --- list_stalled_episodes -----------------------------------------------------


def test_list_stalled_episodes_picks_up_arbitrary_mid_pipeline_stage(conn):
    # Simulates a process killed externally (pod crash/restart) mid-episode:
    # stage sits at some intermediate value with no exception ever raised,
    # so mark_stage_failed never ran -- list_queued/list_failed must both
    # miss it, and list_stalled_episodes must be the one that catches it.
    db.insert_podcast(conn, "pod1", "feed1", "P", "https://example.com/feed.xml")
    db.insert_episode(conn, "ep_stuck", "pod1", "pi-1", "A", "https://x/a.mp3")
    db.advance_stage(conn, "ep_stuck", "asr_running")

    assert db.list_queued_episodes(conn, shard=None) == []
    assert db.list_failed_episodes(conn, shard=None) == []
    stalled = db.list_stalled_episodes(conn, shard=None)
    assert {row["episode_id"] for row in stalled} == {"ep_stuck"}


def test_list_stalled_episodes_excludes_queued_failed_and_done(conn):
    db.insert_podcast(conn, "pod1", "feed1", "P", "https://example.com/feed.xml")
    db.insert_episode(conn, "ep_queued", "pod1", "pi-1", "A", "https://x/a.mp3")
    db.insert_episode(conn, "ep_failed", "pod1", "pi-2", "B", "https://x/b.mp3")
    db.insert_episode(conn, "ep_done", "pod1", "pi-3", "C", "https://x/c.mp3")
    db.mark_stage_failed(conn, "ep_failed", "downloading", "err")
    db.advance_stage(conn, "ep_done", "done")

    assert db.list_stalled_episodes(conn, shard=None) == []


def test_list_stalled_episodes_filters_by_shard(conn):
    db.insert_podcast(conn, "pod1", "feed1", "P", "https://example.com/feed.xml")
    db.insert_episode(conn, "ep_a", "pod1", "pi-1", "A", "https://x/a.mp3")
    db.insert_episode(conn, "ep_b", "pod1", "pi-2", "B", "https://x/b.mp3")
    db.set_assigned_shard(conn, "ep_b", 5)
    db.advance_stage(conn, "ep_a", "diarizing")
    db.advance_stage(conn, "ep_b", "diarizing")

    no_shard = db.list_stalled_episodes(conn, shard=None)
    assert {row["episode_id"] for row in no_shard} == {"ep_a"}

    shard5 = db.list_stalled_episodes(conn, shard=5)
    assert {row["episode_id"] for row in shard5} == {"ep_b"}


# --- cost_events ---------------------------------------------------------------


def test_record_cost_event_and_total_cost(conn):
    assert db.total_cost(conn) == 0.0
    db.record_cost_event(conn, "gpu_compute", 1.5, description="1h gpu")
    db.record_cost_event(conn, "r2_storage", 0.25, description="storage")
    assert db.total_cost(conn) == pytest.approx(1.75)


def test_record_cost_event_rejects_unknown_category(conn):
    with pytest.raises(ValueError):
        db.record_cost_event(conn, "not_a_category", 1.0)


def test_record_cost_event_stores_metadata_json(conn):
    db.record_cost_event(conn, "other", 0.0, metadata={"foo": "bar"})
    row = conn.execute("SELECT metadata_json FROM cost_events").fetchone()
    assert row["metadata_json"] == '{"foo": "bar"}'


# --- run_meta -------------------------------------------------------------------


def test_set_and_get_run_meta_round_trip(conn):
    db.set_run_meta(conn, "gpu_hourly_rate", "0.42")
    assert db.get_run_meta(conn, "gpu_hourly_rate") == "0.42"


def test_get_run_meta_default_when_missing(conn):
    assert db.get_run_meta(conn, "does_not_exist") is None
    assert db.get_run_meta(conn, "does_not_exist", default="fallback") == "fallback"


def test_set_run_meta_upserts_existing_key(conn):
    db.set_run_meta(conn, "k", "v1")
    db.set_run_meta(conn, "k", "v2")
    assert db.get_run_meta(conn, "k") == "v2"
    rows = conn.execute("SELECT COUNT(*) AS c FROM run_meta WHERE key = 'k'").fetchone()
    assert rows["c"] == 1
