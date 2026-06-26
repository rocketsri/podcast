"""Tests for pipeline/cluster.py: cross-episode speaker clustering, against
a real in-memory sqlite db (db.connect(":memory:")). Synthetic embedding
vectors with known cluster structure -- two tight clusters of near-unit
vectors with small noise, clearly separable by cosine distance at the
default 0.75 match_threshold -- stand in for real pyannote/Resemblyzer
centroids. No audio, no network, no GPU."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import cluster, db
from pipeline.diarize import SpeakerTurn


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    db.init_db(connection)
    db.insert_podcast(connection, "pod1", "feed1", "My Podcast", "https://example.com/feed.xml")
    yield connection
    connection.close()


def _make_episode(conn, episode_id, podcast_id="pod1"):
    db.insert_episode(conn, episode_id, podcast_id, f"pi-{episode_id}", episode_id, f"https://x/{episode_id}.mp3")


def unit_vector(base: np.ndarray, noise_scale: float = 0.02, seed: int = 0) -> np.ndarray:
    """A near-duplicate of `base`, perturbed by small noise then NOT
    re-normalized -- cosine_similarity itself is scale-invariant, so this
    is still a clean test of "near-duplicate direction" matching."""
    rng = np.random.default_rng(seed)
    noisy = base + rng.normal(0, noise_scale, size=base.shape)
    return noisy.astype(np.float32)


# Two clearly-separable cluster directions in a small embedding space.
SPEAKER_A_BASE = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
SPEAKER_B_BASE = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cluster.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cluster.cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_returns_zero_not_nan():
    a = np.zeros(3, dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cluster.cosine_similarity(a, b) == 0.0


# --- match_or_create_speaker ---------------------------------------------------------


def test_match_or_create_speaker_assigns_same_id_to_near_duplicate_embedding(conn):
    _make_episode(conn, "ep1")
    _make_episode(conn, "ep2")

    speaker_id_1 = cluster.match_or_create_speaker(
        conn, "pod1", "ep1", unit_vector(SPEAKER_A_BASE, seed=1), speech_seconds=10.0
    )
    speaker_id_2 = cluster.match_or_create_speaker(
        conn, "pod1", "ep2", unit_vector(SPEAKER_A_BASE, seed=2), speech_seconds=8.0
    )

    assert speaker_id_1 == speaker_id_2

    speakers = db.get_speakers_for_podcast(conn, "pod1")
    assert len(speakers) == 1
    # Centroid update folded in both members' speech-seconds.
    assert speakers[0]["embedding_count"] == 2
    assert speakers[0]["total_speech_seconds"] == pytest.approx(18.0)


def test_match_or_create_speaker_creates_new_id_for_clearly_different_embedding(conn):
    _make_episode(conn, "ep1")
    _make_episode(conn, "ep2")

    speaker_id_a = cluster.match_or_create_speaker(
        conn, "pod1", "ep1", unit_vector(SPEAKER_A_BASE, seed=1), speech_seconds=10.0
    )
    speaker_id_b = cluster.match_or_create_speaker(
        conn, "pod1", "ep2", unit_vector(SPEAKER_B_BASE, seed=1), speech_seconds=10.0
    )

    assert speaker_id_a != speaker_id_b
    speakers = db.get_speakers_for_podcast(conn, "pod1")
    assert len(speakers) == 2


def test_match_or_create_speaker_respects_custom_match_threshold(conn):
    _make_episode(conn, "ep1")
    _make_episode(conn, "ep2")

    # Two vectors offset enough that cosine similarity is well below 1.0 but
    # still positive -- a strict threshold should NOT match them, a lenient
    # one should.
    a = np.array([1.0, 0.3, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, -0.3, 0.0, 0.0], dtype=np.float32)
    similarity = cluster.cosine_similarity(a, b)
    assert 0.5 < similarity < 0.95  # sanity check the fixture is in the interesting zone

    speaker_id_1 = cluster.match_or_create_speaker(conn, "pod1", "ep1", a, speech_seconds=5.0, match_threshold=0.99)
    speaker_id_2 = cluster.match_or_create_speaker(conn, "pod1", "ep2", b, speech_seconds=5.0, match_threshold=0.99)
    assert speaker_id_1 != speaker_id_2  # strict threshold rejects the match

    conn2 = db.connect(":memory:")
    db.init_db(conn2)
    db.insert_podcast(conn2, "pod1", "feed1", "P", "https://x/feed.xml")
    _make_episode(conn2, "ep1")
    _make_episode(conn2, "ep2")
    speaker_id_3 = cluster.match_or_create_speaker(conn2, "pod1", "ep1", a, speech_seconds=5.0, match_threshold=float(similarity) - 0.01)
    speaker_id_4 = cluster.match_or_create_speaker(conn2, "pod1", "ep2", b, speech_seconds=5.0, match_threshold=float(similarity) - 0.01)
    assert speaker_id_3 == speaker_id_4  # lenient threshold accepts the match
    conn2.close()


def test_match_or_create_speaker_speaker_id_format(conn):
    _make_episode(conn, "ep1")
    speaker_id = cluster.match_or_create_speaker(conn, "pod1", "ep1", SPEAKER_A_BASE, speech_seconds=5.0)
    assert speaker_id == "pod1_speaker_000"


# --- ingest_episode_diarization -------------------------------------------------------


def test_ingest_episode_diarization_persists_local_speaker_segments(conn):
    _make_episode(conn, "ep1")
    turns = [
        SpeakerTurn("spk_a", 0.0, 5.0),
        SpeakerTurn("spk_a", 10.0, 12.0),
        SpeakerTurn("spk_b", 5.0, 10.0),
    ]
    embeddings = {"spk_a": unit_vector(SPEAKER_A_BASE, seed=1), "spk_b": unit_vector(SPEAKER_B_BASE, seed=1)}

    resolved = cluster.ingest_episode_diarization(conn, "ep1", "pod1", turns, embeddings)

    assert set(resolved.keys()) == {"spk_a", "spk_b"}
    assert resolved["spk_a"] is not None
    assert resolved["spk_b"] is not None
    assert resolved["spk_a"] != resolved["spk_b"]

    segments = db.get_local_speaker_segments_for_episode(conn, "ep1")
    assert len(segments) == 3
    by_label = {}
    for seg in segments:
        by_label.setdefault(seg["local_label"], []).append(seg)
    assert len(by_label["spk_a"]) == 2
    assert len(by_label["spk_b"]) == 1
    for seg in segments:
        assert seg["resolved_speaker_id"] == resolved[seg["local_label"]]


def test_ingest_episode_diarization_label_with_no_embedding_resolves_to_none(conn):
    """A local label that appears in `turns` but never cleared diarize.py's
    speech-duration floor (so it's absent from `embeddings`) must resolve to
    None, not raise or get silently dropped from local_speaker_segments."""
    _make_episode(conn, "ep1")
    turns = [
        SpeakerTurn("spk_a", 0.0, 5.0),
        SpeakerTurn("spk_no_embedding", 5.0, 5.5),  # too short to get an embedding
    ]
    embeddings = {"spk_a": unit_vector(SPEAKER_A_BASE, seed=1)}

    resolved = cluster.ingest_episode_diarization(conn, "ep1", "pod1", turns, embeddings)

    assert resolved["spk_no_embedding"] is None
    assert resolved["spk_a"] is not None

    segments = db.get_local_speaker_segments_for_episode(conn, "ep1")
    no_embedding_segments = [s for s in segments if s["local_label"] == "spk_no_embedding"]
    assert len(no_embedding_segments) == 1
    assert no_embedding_segments[0]["resolved_speaker_id"] is None
    assert no_embedding_segments[0]["embedding"] is None


def test_ingest_episode_diarization_incremental_matches_across_episodes(conn):
    """Running ingest_episode_diarization episode-by-episode (the
    incremental path) should fold a second episode's near-duplicate
    embedding into the same speaker created by the first."""
    _make_episode(conn, "ep1")
    _make_episode(conn, "ep2")

    cluster.ingest_episode_diarization(
        conn, "ep1", "pod1",
        [SpeakerTurn("host", 0.0, 20.0)],
        {"host": unit_vector(SPEAKER_A_BASE, seed=1)},
    )
    resolved2 = cluster.ingest_episode_diarization(
        conn, "ep2", "pod1",
        [SpeakerTurn("speaker_x", 0.0, 15.0)],
        {"speaker_x": unit_vector(SPEAKER_A_BASE, seed=2)},
    )

    speakers = db.get_speakers_for_podcast(conn, "pod1")
    assert len(speakers) == 1
    assert resolved2["speaker_x"] == speakers[0]["speaker_id"]


# --- recluster_podcast_from_scratch -----------------------------------------------------


def _seed_two_speaker_podcast(conn):
    """3 episodes, 2 real speakers (A appears in all 3, B appears in 2),
    via ingest_episode_diarization's incremental path -- mirrors how
    local_speaker_segments would actually accumulate in production before a
    recluster call."""
    _make_episode(conn, "ep1")
    _make_episode(conn, "ep2")
    _make_episode(conn, "ep3")

    cluster.ingest_episode_diarization(
        conn, "ep1", "pod1",
        [SpeakerTurn("l0", 0.0, 20.0), SpeakerTurn("l1", 20.0, 35.0)],
        {"l0": unit_vector(SPEAKER_A_BASE, seed=1), "l1": unit_vector(SPEAKER_B_BASE, seed=1)},
    )
    cluster.ingest_episode_diarization(
        conn, "ep2", "pod1",
        [SpeakerTurn("l0", 0.0, 18.0), SpeakerTurn("l1", 18.0, 30.0)],
        {"l0": unit_vector(SPEAKER_A_BASE, seed=2), "l1": unit_vector(SPEAKER_B_BASE, seed=2)},
    )
    cluster.ingest_episode_diarization(
        conn, "ep3", "pod1",
        [SpeakerTurn("l0", 0.0, 25.0)],
        {"l0": unit_vector(SPEAKER_A_BASE, seed=3)},
    )


def test_recluster_podcast_from_scratch_finds_two_speakers(conn):
    _seed_two_speaker_podcast(conn)
    result = cluster.recluster_podcast_from_scratch(conn, "pod1")
    assert result.num_speakers == 2

    speakers = db.get_speakers_for_podcast(conn, "pod1")
    assert len(speakers) == 2


def _membership_signature(conn, podcast_id="pod1"):
    """Maps each (episode_id, local_label) segment key to the *set* of other
    keys sharing its resolved speaker_id -- a representation of cluster
    membership that's invariant to the actual speaker_id strings minted,
    since those are documented as re-sequenced on every recluster call."""
    segments = db.get_local_speaker_segments_for_podcast(conn, podcast_id)
    by_speaker: dict[str, list[tuple[str, str]]] = {}
    for seg in segments:
        key = (seg["episode_id"], seg["local_label"])
        by_speaker.setdefault(seg["resolved_speaker_id"], []).append(key)
    # frozenset of frozensets: the partition of keys into clusters, order-independent.
    return frozenset(frozenset(group) for group in by_speaker.values())


def test_recluster_podcast_from_scratch_idempotent_in_cluster_membership(conn):
    """Running recluster twice must group the same people together both
    times, even though speaker_id strings get re-sequenced 0..N-1 each call
    (the module's own documented caveat) -- so compare membership
    partitions, not raw speaker_id equality."""
    _seed_two_speaker_podcast(conn)

    cluster.recluster_podcast_from_scratch(conn, "pod1")
    signature_1 = _membership_signature(conn, "pod1")

    cluster.recluster_podcast_from_scratch(conn, "pod1")
    signature_2 = _membership_signature(conn, "pod1")

    assert signature_1 == signature_2
    # Sanity: still exactly 2 distinct clusters, not collapsed/split.
    assert len(signature_1) == 2


def test_recluster_podcast_from_scratch_rebuilds_purely_from_local_speaker_segments(conn):
    """Prior shard-local speaker_id assignments on local_speaker_segments
    and clips must be ignored/overwritten by a recluster call -- the
    rebuild reads only (episode_id, local_label, embedding, seconds) from
    local_speaker_segments, never the previous resolved_speaker_id."""
    _seed_two_speaker_podcast(conn)

    # Corrupt every existing resolved_speaker_id by pointing every segment at
    # the SAME single (real, FK-valid) speaker row -- a stand-in for two
    # different shards both having minted "podcast123_speaker_002" for two
    # different real people (per the module docstring's collision scenario)
    # -- before reclustering, to prove the rebuild ignores this prior
    # assignment entirely rather than trusting/merging it.
    bogus_speaker_id = "pod1_speaker_999"
    db.upsert_speaker(
        conn, bogus_speaker_id, "pod1", local_label_seq=999,
        centroid_embedding=np.zeros(4, dtype=np.float32),
        embedding_count=1, total_speech_seconds=1.0, episode_id="ep1",
    )
    segments = db.get_local_speaker_segments_for_podcast(conn, "pod1")
    for seg in segments:
        db.set_segment_resolved_speaker(conn, seg["segment_id"], bogus_speaker_id)
    refreshed_before = db.get_local_speaker_segments_for_podcast(conn, "pod1")
    assert all(seg["resolved_speaker_id"] == bogus_speaker_id for seg in refreshed_before)

    result = cluster.recluster_podcast_from_scratch(conn, "pod1")

    assert result.num_speakers == 2
    refreshed = db.get_local_speaker_segments_for_podcast(conn, "pod1")
    resolved_ids = {seg["resolved_speaker_id"] for seg in refreshed}
    assert bogus_speaker_id not in resolved_ids
    # delete_speakers_for_podcast wipes every prior speakers row for this
    # podcast (including the bogus one) before rebuilding from scratch.
    speaker_ids = {row["speaker_id"] for row in db.get_speakers_for_podcast(conn, "pod1")}
    assert bogus_speaker_id not in speaker_ids
    # Every segment ended up with one of the freshly minted pod1_speaker_* ids.
    assert all(rid is not None and rid.startswith("pod1_speaker_") for rid in resolved_ids)


def test_recluster_podcast_from_scratch_updates_clip_speaker_assignments(conn):
    """Clips persisted with one (possibly stale) speaker_id should get
    reconciled to the freshly recomputed speaker_id for whichever segment
    they overlap most, via _reconcile_clip_speakers."""
    _seed_two_speaker_podcast(conn)

    # A clip that overlaps ep1's l0 turn (0.0-20.0) most.
    db.insert_clip(conn, "clip1", "ep1", "pod1", 2.0, 8.0)

    result = cluster.recluster_podcast_from_scratch(conn, "pod1")
    assert result.num_clips_corrected >= 1

    clip = db.get_clips_for_episode(conn, "ep1")[0]
    assert clip["speaker_id"] is not None
    assert clip["speaker_id"].startswith("pod1_speaker_")


def test_recluster_podcast_from_scratch_empty_podcast_no_speakers(conn):
    _make_episode(conn, "ep1")  # episode exists but has no local_speaker_segments
    result = cluster.recluster_podcast_from_scratch(conn, "pod1")
    assert result.num_speakers == 0
    assert db.get_speakers_for_podcast(conn, "pod1") == []


def test_recluster_podcast_from_scratch_single_segment_one_speaker(conn):
    """A podcast with exactly one (episode, local_label) embedding-bearing
    segment is the n=1 AgglomerativeClustering edge case -- must not raise
    and must produce exactly one speaker."""
    _make_episode(conn, "ep1")
    db.insert_local_speaker_segment(conn, "ep1", "l0", 0.0, 10.0, unit_vector(SPEAKER_A_BASE, seed=1))

    result = cluster.recluster_podcast_from_scratch(conn, "pod1")
    assert result.num_speakers == 1
    speakers = db.get_speakers_for_podcast(conn, "pod1")
    assert len(speakers) == 1


def test_recluster_podcast_from_scratch_segments_with_no_embedding_get_none_speaker(conn):
    """local_speaker_segments rows with embedding=None (label never cleared
    the speech-duration floor) must not be fed into clustering and must end
    up with resolved_speaker_id=None, not crash the embeddings_matrix stack."""
    _make_episode(conn, "ep1")
    db.insert_local_speaker_segment(conn, "ep1", "l0", 0.0, 10.0, unit_vector(SPEAKER_A_BASE, seed=1))
    db.insert_local_speaker_segment(conn, "ep1", "l1", 10.0, 10.3, None)  # no embedding

    result = cluster.recluster_podcast_from_scratch(conn, "pod1")
    assert result.num_speakers == 1

    segments = db.get_local_speaker_segments_for_episode(conn, "ep1")
    by_label = {s["local_label"]: s for s in segments}
    assert by_label["l0"]["resolved_speaker_id"] is not None
    assert by_label["l1"]["resolved_speaker_id"] is None
