"""Cross-episode speaker clustering: turns diarize.py's per-episode local
speaker centroids into globally-consistent speaker_id strings, podcast-scoped.

Two entry points share one core (see plan's "Cross-episode speaker
clustering" section):

  - `ingest_episode_diarization` runs once per episode, right after
    diarization, and does *incremental* matching via `match_or_create_speaker`
    -- O(existing speakers for that podcast), cheap, but can't catch drift
    across many episodes.
  - `recluster_podcast_from_scratch` re-derives every speaker_id for a
    podcast from nothing, over all persisted `local_speaker_segments`. It is
    called periodically (every N episodes, to catch what incremental
    matching misses) and exactly once per podcast after a multi-pod shard
    merge, where per-shard speaker_id strings can collide in *meaning*
    across pods (two pods may each mint "podcast123_speaker_002" for two
    different real people) even though the underlying segment rows never
    collide (globally-unique episode_id keys). Same clustering math, two
    callers -- this is what makes horizontal pod parallelism safe for
    speaker-ID accuracy.

Speaker_id strings are NOT guaranteed stable across repeated calls to
recluster_podcast_from_scratch (clusters are re-sequenced 0..N-1 by sorted
cluster id every call) even though the underlying voice clusters they refer
to are consistent. Downstream code should treat speaker_id as an opaque,
podcast-scoped label, not a durable cross-run identity.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from pipeline import db

DEFAULT_MATCH_THRESHOLD = 0.75


@dataclass(frozen=True)
class ReclusterResult:
    num_speakers: int
    num_clips_corrected: int


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _weighted_update(
    old_centroid: np.ndarray | None,
    old_total_seconds: float,
    old_count: int,
    new_embedding: np.ndarray,
    new_seconds: float,
) -> tuple[np.ndarray, float, int]:
    """Duration-weighted running average of a speaker's centroid as a new
    (episode, local_label) member is folded in."""
    new_embedding = np.asarray(new_embedding, dtype=np.float32)
    if old_centroid is None:
        return new_embedding, float(new_seconds), 1
    new_total = old_total_seconds + new_seconds
    if new_total <= 0.0:
        return np.asarray(old_centroid, dtype=np.float32), old_total_seconds, old_count + 1
    blended = (np.asarray(old_centroid, dtype=np.float32) * old_total_seconds + new_embedding * new_seconds) / new_total
    return blended.astype(np.float32), float(new_total), old_count + 1


def _next_local_label_seq(conn: sqlite3.Connection, podcast_id: str) -> int:
    existing = db.get_speakers_for_podcast(conn, podcast_id)
    if not existing:
        return 0
    return max(row["local_label_seq"] for row in existing) + 1


def match_or_create_speaker(
    conn: sqlite3.Connection,
    podcast_id: str,
    episode_id: str,
    embedding: np.ndarray,
    speech_seconds: float,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> str:
    """Cosine-match `embedding` against every existing global speaker for
    this podcast; above threshold, fold it into that speaker's running
    centroid and return its id, else mint a new speaker_id."""
    candidates = db.get_speakers_for_podcast(conn, podcast_id)
    best_speaker = None
    best_similarity = -1.0
    for candidate in candidates:
        centroid = db.unpack_embedding(candidate["centroid_embedding"], candidate["centroid_dim"])
        similarity = cosine_similarity(embedding, centroid)
        if similarity > best_similarity:
            best_similarity = similarity
            best_speaker = candidate

    if best_speaker is not None and best_similarity >= match_threshold:
        old_centroid = db.unpack_embedding(best_speaker["centroid_embedding"], best_speaker["centroid_dim"])
        new_centroid, new_total, new_count = _weighted_update(
            old_centroid, best_speaker["total_speech_seconds"], best_speaker["embedding_count"],
            embedding, speech_seconds,
        )
        db.upsert_speaker(
            conn, best_speaker["speaker_id"], podcast_id, best_speaker["local_label_seq"],
            new_centroid, new_count, new_total, episode_id,
        )
        return best_speaker["speaker_id"]

    seq = _next_local_label_seq(conn, podcast_id)
    speaker_id = f"{podcast_id}_speaker_{seq:03d}"
    new_centroid, new_total, new_count = _weighted_update(None, 0.0, 0, embedding, speech_seconds)
    db.upsert_speaker(conn, speaker_id, podcast_id, seq, new_centroid, new_count, new_total, episode_id)
    return speaker_id


def ingest_episode_diarization(
    conn: sqlite3.Connection,
    episode_id: str,
    podcast_id: str,
    turns: list,
    embeddings: dict[str, np.ndarray],
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> dict[str, str | None]:
    """Persist one episode's diarize.py output and resolve each local label
    to a global speaker_id (or None, if its centroid never cleared
    diarize.py's speech-duration floor). `turns` is a list of
    diarize.SpeakerTurn; `embeddings` is diarize.DiarizationResult.embeddings.

    Every turn is written to local_speaker_segments regardless of whether
    its label resolved to a speaker_id -- diarization is never re-run, so
    this is the durable substrate recluster_podcast_from_scratch reads back."""
    speech_seconds_by_label: dict[str, float] = {}
    for turn in turns:
        speech_seconds_by_label[turn.local_label] = (
            speech_seconds_by_label.get(turn.local_label, 0.0) + (turn.end_seconds - turn.start_seconds)
        )

    resolved: dict[str, str | None] = {}
    for label, embedding in embeddings.items():
        resolved[label] = match_or_create_speaker(
            conn, podcast_id, episode_id, embedding, speech_seconds_by_label.get(label, 0.0), match_threshold
        )
    for label in speech_seconds_by_label:
        resolved.setdefault(label, None)

    for turn in turns:
        db.insert_local_speaker_segment(
            conn, episode_id, turn.local_label, turn.start_seconds, turn.end_seconds,
            embeddings.get(turn.local_label), resolved.get(turn.local_label),
        )

    return resolved


def _reconcile_clip_speakers(
    conn: sqlite3.Connection,
    podcast_id: str,
    segments: list[sqlite3.Row],
    key_to_speaker: dict[tuple[str, str], str],
) -> int:
    """Re-point each clip's speaker_id at whichever local_speaker_segment in
    the same episode overlaps it most -- segments don't map 1:1 to clips
    (segment.py cuts clips out of diarization turns), so this is a
    best-time-overlap lookup, not a key match."""
    segments_by_episode: dict[str, list[sqlite3.Row]] = {}
    for seg in segments:
        segments_by_episode.setdefault(seg["episode_id"], []).append(seg)

    updated = 0
    for clip in db.get_clips_for_podcast(conn, podcast_id):
        best_overlap = 0.0
        best_key: tuple[str, str] | None = None
        for seg in segments_by_episode.get(clip["episode_id"], []):
            overlap = min(clip["end_seconds"], seg["end_seconds"]) - max(clip["start_seconds"], seg["start_seconds"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = (seg["episode_id"], seg["local_label"])
        new_speaker_id = key_to_speaker.get(best_key) if best_key else None
        if new_speaker_id != clip["speaker_id"]:
            db.update_clip_speaker(conn, clip["clip_id"], new_speaker_id)
            updated += 1
    return updated


def recluster_podcast_from_scratch(
    conn: sqlite3.Connection,
    podcast_id: str,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> ReclusterResult:
    """Rebuild every global speaker_id for a podcast from nothing, over all
    persisted local_speaker_segments -- ignoring any prior speakers rows,
    shard-local or not. Used both for periodic re-clustering and as the
    one-time centralized recompute after a multi-pod shard merge."""
    segments = db.get_local_speaker_segments_for_podcast(conn, podcast_id)

    groups: dict[tuple[str, str], dict] = {}
    for seg in segments:
        key = (seg["episode_id"], seg["local_label"])
        group = groups.setdefault(key, {"seconds": 0.0, "embedding": None})
        group["seconds"] += seg["end_seconds"] - seg["start_seconds"]
        if group["embedding"] is None and seg["embedding"] is not None:
            group["embedding"] = db.unpack_embedding(seg["embedding"])

    # Episode-id order approximates chronological order for first/last_seen
    # bookkeeping in upsert_speaker, which only ever sees one member at a time.
    ordered_keys = sorted((k for k, v in groups.items() if v["embedding"] is not None), key=lambda k: k[0])

    db.delete_speakers_for_podcast(conn, podcast_id)

    key_to_speaker: dict[tuple[str, str], str] = {}
    num_speakers = 0
    if ordered_keys:
        if len(ordered_keys) == 1:
            cluster_ids = np.zeros(1, dtype=int)
        else:
            embeddings_matrix = np.stack([groups[k]["embedding"] for k in ordered_keys])
            clustering = AgglomerativeClustering(
                n_clusters=None, metric="cosine", linkage="average",
                distance_threshold=1.0 - match_threshold,
            )
            cluster_ids = clustering.fit_predict(embeddings_matrix)

        members_by_cluster: dict[int, list[tuple[str, str]]] = {}
        for key, cluster_id in zip(ordered_keys, cluster_ids):
            members_by_cluster.setdefault(int(cluster_id), []).append(key)
        num_speakers = len(members_by_cluster)

        for seq, cluster_id in enumerate(sorted(members_by_cluster.keys())):
            speaker_id = f"{podcast_id}_speaker_{seq:03d}"
            centroid, total_seconds, count = None, 0.0, 0
            for key in members_by_cluster[cluster_id]:
                episode_id, _local_label = key
                centroid, total_seconds, count = _weighted_update(
                    centroid, total_seconds, count, groups[key]["embedding"], groups[key]["seconds"]
                )
                db.upsert_speaker(conn, speaker_id, podcast_id, seq, centroid, count, total_seconds, episode_id)
                key_to_speaker[key] = speaker_id

    for seg in segments:
        key = (seg["episode_id"], seg["local_label"])
        db.set_segment_resolved_speaker(conn, seg["segment_id"], key_to_speaker.get(key))

    num_clips_corrected = _reconcile_clip_speakers(conn, podcast_id, segments, key_to_speaker)

    return ReclusterResult(num_speakers=num_speakers, num_clips_corrected=num_clips_corrected)
