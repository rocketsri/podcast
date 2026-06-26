"""Tests for pipeline/pipeline_runner.py's stage-machine helpers. Mostly
exercises the recluster/label_to_speaker interaction in
_ensure_diarized_and_clustered with a real in-memory sqlite db -- the
diarize_fn and cluster.recluster_podcast_from_scratch are faked so the
behavior under test (does pipeline_runner.py re-read post-recluster state
before returning it to the caller?) is decoupled from real audio, real
embeddings, and AgglomerativeClustering's internals."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import cluster, config, db, pipeline_runner
from pipeline.diarize import DiarizationResult, SpeakerTurn


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    db.init_db(connection)
    db.insert_podcast(connection, "pod1", "feed1", "My Podcast", "https://example.com/feed.xml")
    db.insert_episode(connection, "ep1", "pod1", "pi-ep1", "ep1", "https://x/ep1.mp3")
    yield connection
    connection.close()


def _cfg():
    return config._Section({
        "clustering": {
            "match_threshold": 0.75,
            "dominant_speaker_warn_threshold": 0.99,
            "min_local_speaker_seconds_for_embedding": 0.0,
            "recluster_every_n_episodes": 1,
        },
    })


def _ctx(conn, diarize_fn):
    models = pipeline_runner.Models(vad_model=None, diarize_pipeline=None, asr_model=None)
    return pipeline_runner.RunContext(
        conn=conn, cfg=_cfg(), models=models, work_dir=None, diarize_fn=diarize_fn,
    )


def test_recluster_firing_mid_call_does_not_leave_stale_speaker_ids(conn, monkeypatch):
    """_maybe_recluster_podcast can delete and recreate every speakers row
    for the podcast (including the one ingest_episode_diarization just wrote
    for the episode being processed right now) before
    _ensure_diarized_and_clustered returns. The returned label_to_speaker
    must reflect that post-recluster state, not whatever
    ingest_episode_diarization resolved a moment earlier -- otherwise the
    caller persists clips referencing a speaker_id recluster already
    deleted, which is a FOREIGN KEY violation against the live schema."""
    turns = [SpeakerTurn(local_label="A", start_seconds=0.0, end_seconds=5.0)]
    embeddings = {"A": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}

    def fake_diarize_fn(audio_path, pipeline, min_local_speaker_seconds_for_embedding):
        return DiarizationResult(turns=turns, embeddings=embeddings)

    def fake_recluster(conn, podcast_id, match_threshold):
        # Simulates what a real recluster does: wipe every speakers row for
        # the podcast (nulling the FK references first) and rebuild from
        # local_speaker_segments with brand-new ids, deliberately different
        # from whatever match_or_create_speaker just assigned moments ago.
        segments = db.get_local_speaker_segments_for_episode(conn, "ep1")
        db.delete_speakers_for_podcast(conn, podcast_id)
        new_speaker_id = f"{podcast_id}_speaker_999"
        db.upsert_speaker(
            conn, new_speaker_id, podcast_id, local_label_seq=999,
            centroid_embedding=embeddings["A"], embedding_count=1,
            total_speech_seconds=5.0, episode_id="ep1",
        )
        for seg in segments:
            db.set_segment_resolved_speaker(conn, seg["segment_id"], new_speaker_id)
        return cluster.ReclusterResult(num_speakers=1, num_clips_corrected=0)

    monkeypatch.setattr(cluster, "recluster_podcast_from_scratch", fake_recluster)

    ctx = _ctx(conn, fake_diarize_fn)
    episode_row = db.get_episode(conn, "ep1")
    _turns, label_to_speaker = pipeline_runner._ensure_diarized_and_clustered(ctx, episode_row)

    assert label_to_speaker == {"A": "pod1_speaker_999"}

    speakers = db.get_speakers_for_podcast(conn, "pod1")
    assert {s["speaker_id"] for s in speakers} == {"pod1_speaker_999"}
