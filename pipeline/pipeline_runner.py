"""Per-episode stage-machine driver: the only module that calls
advance_stage/mark_stage_failed. Every leaf module it calls (audio.py,
ingest.py, vad.py, diarize.py, cluster.py, segment.py, asr.py, quality.py,
storage.py) is pure I/O or pure logic with no stage-machine awareness of its
own -- see their own module docstrings.

Resumability split, by stage cost (see plan's Database schema section):
  - Cheap stages (download/transcode) and the GPU-expensive diarize+cluster
    stage are skip-guarded with db.is_at_or_past against a real "done"
    sentinel stage.
  - VAD has no persistence table at all (vad.py confirms Silero VAD is cheap
    enough -- 30-100x+ realtime even on CPU -- to just recompute every time
    it's needed, never skip-guarded).
  - The ASR/quality/export/upload tail relies on the leaf modules' own
    per-row idempotency (asr skips clips with utterance already set,
    quality skips clips with discard_reason already set, export/upload
    filter locally for clips missing local_flac_path / not yet uploaded)
    rather than a stage-equality skip-guard -- quality_filtering has no
    separate "done" sentinel before exporting begins, so a naive
    is_at_or_past(current, "quality_filtering") guard would wrongly skip a
    retry that failed mid-filter.
"""

from __future__ import annotations

import datetime
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pipeline import audio, cluster, costs, db, diarize, heartbeat, ingest, segment, storage, vad
from pipeline import asr as asr_module
from pipeline import quality as quality_module

logger = logging.getLogger(__name__)


@dataclass
class Models:
    vad_model: object
    diarize_pipeline: object
    asr_model: object


@dataclass
class RunContext:
    conn: sqlite3.Connection
    cfg: object  # pipeline.config._Section
    models: Models
    work_dir: Path
    storage_client: object | None = None
    bucket: str | None = None
    pod_id: str = "local-pod"
    shard_id: int | None = None
    latest_status: dict = field(default_factory=dict)
    diarize_fn: object = None  # (audio_path, pipeline, min_local_speaker_seconds_for_embedding) -> diarize.DiarizationResult; defaults to diarize.diarize


class StageFailure(RuntimeError):
    """Raised after mark_stage_failed has already recorded the error --
    signals process_episode to stop this episode and let the driver loop
    move on, per the plan's "driver moves on, never crashes the run" rule."""


def _fail(conn: sqlite3.Connection, episode_id: str, stage: str, exc: Exception) -> None:
    db.mark_stage_failed(conn, episode_id, stage, str(exc))
    logger.error("episode %s failed at %s: %s", episode_id, stage, exc)


def _raw_path(work_dir: Path, episode_id: str, suffix: str) -> Path:
    return work_dir / "raw" / f"{episode_id}{suffix}"


def _wav_path(work_dir: Path, episode_id: str) -> Path:
    return work_dir / "wav" / f"{episode_id}.wav"


def _clip_flac_path(work_dir: Path, podcast_id: str, episode_id: str, clip_id: str) -> Path:
    return work_dir / "clips" / podcast_id / episode_id / f"{clip_id}.flac"


# --- per-stage helpers -------------------------------------------------------

def _ensure_downloaded(ctx: RunContext, episode_row: sqlite3.Row) -> sqlite3.Row:
    episode_id = episode_row["episode_id"]
    if db.is_at_or_past(db.resume_stage(episode_row), "downloaded"):
        return episode_row
    db.advance_stage(ctx.conn, episode_id, "downloading")
    suffix = ingest.source_file_suffix(episode_row["source_url"])
    dest = _raw_path(ctx.work_dir, episode_id, suffix)
    try:
        ingest.download_episode_audio(episode_row["source_url"], dest)
        duration = audio.probe_duration_seconds(dest)
    except Exception as exc:
        _fail(ctx.conn, episode_id, "downloading", exc)
        raise StageFailure(str(exc)) from exc
    db.advance_stage(
        ctx.conn, episode_id, "downloaded",
        local_raw_path=str(dest), duration_seconds_actual=duration, raw_seconds=duration,
    )
    return db.get_episode(ctx.conn, episode_id)


def _ensure_transcoded(ctx: RunContext, episode_row: sqlite3.Row) -> sqlite3.Row:
    episode_id = episode_row["episode_id"]
    if db.is_at_or_past(db.resume_stage(episode_row), "transcoded"):
        return episode_row
    db.advance_stage(ctx.conn, episode_id, "transcoding")
    wav_path = _wav_path(ctx.work_dir, episode_id)
    try:
        audio.transcode_to_wav(
            episode_row["local_raw_path"], wav_path,
            sample_rate=ctx.cfg.audio.target_sample_rate, channels=ctx.cfg.audio.target_channels,
        )
    except Exception as exc:
        _fail(ctx.conn, episode_id, "transcoding", exc)
        raise StageFailure(str(exc)) from exc
    db.advance_stage(ctx.conn, episode_id, "transcoded", local_wav_path=str(wav_path))
    return db.get_episode(ctx.conn, episode_id)


def _compute_vad(ctx: RunContext, episode_row: sqlite3.Row, samples: np.ndarray) -> tuple[list, np.ndarray]:
    """Always recomputed (see module docstring) -- the stage-column writes
    are observability only and are guarded against moving the stage
    backward when resuming from further along (e.g. resuming at
    "segmented", where vad_segments/frame_probs are still needed in memory
    to reach segmenting but the stage must not regress to vad_running)."""
    episode_id = episode_row["episode_id"]
    current = db.resume_stage(episode_row)
    if not db.is_at_or_past(current, "vad_running"):
        db.advance_stage(ctx.conn, episode_id, "vad_running")
    vad_segments = vad.run_vad(samples, ctx.models.vad_model, min_speech_confidence=ctx.cfg.vad.min_speech_confidence)
    frame_probs = vad.frame_speech_probabilities(samples, ctx.models.vad_model)
    if not db.is_at_or_past(current, "vad_done"):
        db.advance_stage(ctx.conn, episode_id, "vad_done")
    return vad_segments, frame_probs


def _maybe_recluster_podcast(ctx: RunContext, podcast_id: str) -> None:
    """Every N episodes per podcast (config.clustering.recluster_every_n_episodes),
    re-run agglomerative clustering from scratch over all persisted
    embeddings -- catches cross-episode drift the incremental match misses.
    Counter is podcast-scoped run_meta, so multiple podcasts interleave
    independently within one pod's run."""
    key = f"recluster_count_{podcast_id}"
    count = int(db.get_run_meta(ctx.conn, key, "0")) + 1
    db.set_run_meta(ctx.conn, key, str(count))
    every_n = ctx.cfg.clustering.recluster_every_n_episodes
    if every_n > 0 and count % every_n == 0:
        result = cluster.recluster_podcast_from_scratch(
            ctx.conn, podcast_id, match_threshold=ctx.cfg.clustering.match_threshold
        )
        logger.info(
            "periodic recluster for %s: %d speakers, %d clips corrected",
            podcast_id, result.num_speakers, result.num_clips_corrected,
        )


def _ensure_diarized_and_clustered(ctx: RunContext, episode_row: sqlite3.Row) -> tuple[list, dict]:
    episode_id, podcast_id = episode_row["episode_id"], episode_row["podcast_id"]
    current = db.resume_stage(episode_row)
    if db.is_at_or_past(current, "clustering_done"):
        segments = db.get_local_speaker_segments_for_episode(ctx.conn, episode_id)
        turns = [diarize.SpeakerTurn(s["local_label"], s["start_seconds"], s["end_seconds"]) for s in segments]
        label_to_speaker = {s["local_label"]: s["resolved_speaker_id"] for s in segments}
        return turns, label_to_speaker

    db.advance_stage(ctx.conn, episode_id, "diarizing")
    try:
        diarize_fn = ctx.diarize_fn or diarize.diarize
        result = diarize_fn(
            episode_row["local_wav_path"], ctx.models.diarize_pipeline,
            min_local_speaker_seconds_for_embedding=ctx.cfg.clustering.min_local_speaker_seconds_for_embedding,
        )
    except Exception as exc:
        _fail(ctx.conn, episode_id, "diarizing", exc)
        raise StageFailure(str(exc)) from exc
    db.advance_stage(ctx.conn, episode_id, "diarized")

    dominant = diarize.dominant_speaker_share(result.turns)
    if dominant is not None:
        label, share, num_labels = dominant
        if num_labels >= 2 and share >= ctx.cfg.clustering.dominant_speaker_warn_threshold:
            logger.warning(
                "episode %s: diarization looks collapsed -- label %s holds %.1f%% of speech across "
                "%d detected labels (>=%d%% threshold); likely a clustering/threshold problem, not a "
                "genuinely solo episode -- see PROBLEMS.md",
                episode_id, label, share * 100, num_labels, int(ctx.cfg.clustering.dominant_speaker_warn_threshold * 100),
            )
            db.set_run_meta(
                ctx.conn, f"dominant_speaker_warning_{episode_id}", f"{label}:{share:.4f}:{num_labels}",
            )

    try:
        label_to_speaker = cluster.ingest_episode_diarization(
            ctx.conn, episode_id, podcast_id, result.turns, result.embeddings,
            match_threshold=ctx.cfg.clustering.match_threshold,
        )
    except Exception as exc:
        _fail(ctx.conn, episode_id, "diarized", exc)
        raise StageFailure(str(exc)) from exc
    db.advance_stage(ctx.conn, episode_id, "clustering_done")

    _maybe_recluster_podcast(ctx, podcast_id)
    return result.turns, label_to_speaker


def _ensure_segmented(
    ctx: RunContext, episode_row: sqlite3.Row, turns: list, vad_segments: list,
    frame_probs: np.ndarray, label_to_speaker: dict,
) -> None:
    episode_id, podcast_id = episode_row["episode_id"], episode_row["podcast_id"]
    if db.is_at_or_past(db.resume_stage(episode_row), "segmented"):
        return
    db.advance_stage(ctx.conn, episode_id, "segmenting")
    try:
        clips = segment.build_candidate_clips(
            episode_id, podcast_id, turns, vad_segments, frame_probs, label_to_speaker, ctx.cfg.segmentation,
        )
        segment.persist_candidate_clips(ctx.conn, clips)
    except Exception as exc:
        _fail(ctx.conn, episode_id, "segmenting", exc)
        raise StageFailure(str(exc)) from exc
    db.advance_stage(ctx.conn, episode_id, "segmented")


def _ensure_asr(ctx: RunContext, episode_row: sqlite3.Row, samples: np.ndarray, sample_rate: int) -> None:
    episode_id = episode_row["episode_id"]
    db.advance_stage(ctx.conn, episode_id, "asr_running")
    try:
        count = asr_module.transcribe_clips_for_episode(ctx.conn, episode_id, samples, sample_rate, ctx.models.asr_model)
    except Exception as exc:
        _fail(ctx.conn, episode_id, "asr_running", exc)
        raise StageFailure(str(exc)) from exc
    db.advance_stage(ctx.conn, episode_id, "asr_done")
    logger.info("episode %s: transcribed %d clips", episode_id, count)


def _ensure_quality_filtered(ctx: RunContext, episode_row: sqlite3.Row, samples: np.ndarray, sample_rate: int) -> None:
    episode_id = episode_row["episode_id"]
    db.advance_stage(ctx.conn, episode_id, "quality_filtering")
    try:
        discarded = quality_module.apply_quality_filters(ctx.conn, episode_id, samples, sample_rate, ctx.cfg.quality)
    except Exception as exc:
        _fail(ctx.conn, episode_id, "quality_filtering", exc)
        raise StageFailure(str(exc)) from exc
    logger.info("episode %s: quality filters discarded %d clips", episode_id, discarded)


def _ensure_exported_and_uploaded(ctx: RunContext, episode_row: sqlite3.Row) -> None:
    """No stage-equality skip-guard (see module docstring) -- filters
    locally for clips still missing local_flac_path / not yet uploaded, so a
    retry never re-encodes or re-uploads a clip that already made it
    through. Falls back to a local file path as `audio_path` when no R2
    client is configured (dev/smoke-test runs without R2 credentials), so
    the manifest is still buildable locally."""
    episode_id, podcast_id = episode_row["episode_id"], episode_row["podcast_id"]
    wav_path = episode_row["local_wav_path"]
    db.advance_stage(ctx.conn, episode_id, "exporting")

    surviving = [c for c in db.get_clips_for_episode(ctx.conn, episode_id) if c["discard_reason"] is None]
    for clip in surviving:
        if clip["local_flac_path"]:
            continue
        flac_path = _clip_flac_path(ctx.work_dir, podcast_id, episode_id, clip["clip_id"])
        try:
            audio.extract_clip_to_flac(wav_path, flac_path, clip["start_seconds"], clip["end_seconds"])
        except Exception as exc:
            _fail(ctx.conn, episode_id, "exporting", exc)
            raise StageFailure(str(exc)) from exc
        db.update_clip_fields(ctx.conn, clip["clip_id"], local_flac_path=str(flac_path))

    db.advance_stage(ctx.conn, episode_id, "uploading")

    uploaded_count = 0
    for clip in db.get_clips_for_episode(ctx.conn, episode_id):
        if clip["discard_reason"] is not None or clip["uploaded"]:
            continue
        local_flac_path = clip["local_flac_path"]
        if not local_flac_path:
            continue  # exporting loop above already set this for every surviving clip
        try:
            if ctx.storage_client is not None and ctx.bucket is not None:
                audio_path = storage.upload_clip(
                    ctx.storage_client, ctx.bucket, local_flac_path, podcast_id, episode_id, clip["clip_id"]
                )
                uploaded_count += 1
            else:
                audio_path = local_flac_path
        except Exception as exc:
            _fail(ctx.conn, episode_id, "uploading", exc)
            raise StageFailure(str(exc)) from exc
        db.mark_clip_uploaded(ctx.conn, clip["clip_id"], audio_path)

    if uploaded_count:
        costs.record_r2_class_a_ops(ctx.conn, uploaded_count, description=f"{episode_id}: {uploaded_count} clip PutObject calls")

    usable_seconds = sum(c["duration_seconds"] for c in surviving)
    db.advance_stage(ctx.conn, episode_id, "done", usable_seconds=usable_seconds)


# --- per-episode + driver loop ----------------------------------------------

def process_episode(ctx: RunContext, episode_id: str) -> bool:
    """Runs every remaining stage for one episode. Returns True if the
    episode reached "done" (or already had), False if it failed partway --
    the failure itself is already recorded in the db by the stage helper
    that raised, so the caller just moves on to the next episode."""
    episode_row = db.get_episode(ctx.conn, episode_id)
    if episode_row is None:
        raise ValueError(f"unknown episode_id: {episode_id}")
    if db.resume_stage(episode_row) == "done":
        return True

    try:
        episode_row = _ensure_downloaded(ctx, episode_row)
        episode_row = _ensure_transcoded(ctx, episode_row)
        samples, sample_rate = audio.read_wav(episode_row["local_wav_path"])
        vad_segments, frame_probs = _compute_vad(ctx, episode_row, samples)
        turns, label_to_speaker = _ensure_diarized_and_clustered(ctx, episode_row)
        episode_row = db.get_episode(ctx.conn, episode_id)
        _ensure_segmented(ctx, episode_row, turns, vad_segments, frame_probs, label_to_speaker)
        episode_row = db.get_episode(ctx.conn, episode_id)
        _ensure_asr(ctx, episode_row, samples, sample_rate)
        _ensure_quality_filtered(ctx, episode_row, samples, sample_rate)
        _ensure_exported_and_uploaded(ctx, episode_row)
    except StageFailure:
        return False
    return True


def _record_gpu_compute_checkpoint(ctx: RunContext, pod_started_at: datetime.datetime) -> None:
    """gpu_compute cost is wall-clock pod uptime x the booked hourly rate,
    attributed by the driver loop (here) rather than per-episode-stage --
    matches the plan's "each pod ledgers only its own uptime" design. Ticks
    forward from the last checkpoint (run_meta), not from pod start, so
    repeated calls don't double-count already-ledgered time."""
    last_checkpoint_iso = db.get_run_meta(ctx.conn, "last_cost_checkpoint_at")
    last_checkpoint = (
        datetime.datetime.fromisoformat(last_checkpoint_iso) if last_checkpoint_iso else pod_started_at
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    hours = (now - last_checkpoint).total_seconds() / 3600.0
    if hours <= 0:
        return
    hourly_rate = float(db.get_run_meta(ctx.conn, "gpu_hourly_rate_usd", str(ctx.cfg.cost.assumed_gpu_hourly_usd)))
    costs.record_gpu_compute(ctx.conn, hours, hourly_rate, description="driver-loop checkpoint")
    db.set_run_meta(ctx.conn, "last_cost_checkpoint_at", now.isoformat())


def _push_heartbeat(ctx: RunContext) -> None:
    pod_started_at = db.get_run_meta(ctx.conn, "pod_started_at", db.now_iso())
    status = heartbeat.build_status(ctx.conn, ctx.pod_id, ctx.shard_id, pod_started_at)
    ctx.latest_status = status  # StatusServer's status_provider reads this directly
    if ctx.storage_client is not None and ctx.bucket is not None:
        try:
            heartbeat.push_status_to_r2(ctx.storage_client, ctx.bucket, ctx.pod_id, status)
        except Exception:
            logger.exception("heartbeat push to R2 failed -- continuing (local HTTP status channel still serves it)")


def run_queue(ctx: RunContext, shard: int | None = None, max_episodes: int | None = None) -> dict:
    """Main per-pod driver loop: claims this pod's failed-then-queued
    episodes (failed first, since resume_stage picks up from the failure
    point rather than restarting from queued), checks budget/time caps
    before each new episode, and records a cost checkpoint + heartbeat after
    every one. Never raises on a single episode's failure -- see
    process_episode."""
    pod_started_iso = db.get_run_meta(ctx.conn, "pod_started_at")
    if pod_started_iso is None:
        pod_started_iso = db.now_iso()
        db.set_run_meta(ctx.conn, "pod_started_at", pod_started_iso)
    pod_started_at = datetime.datetime.fromisoformat(pod_started_iso)

    costs.record_egress(ctx.conn)  # measured-zero, once per run start (see costs.record_egress)

    episodes = list(db.list_failed_episodes(ctx.conn, shard=shard)) + list(db.list_queued_episodes(ctx.conn, shard=shard))
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    processed = succeeded = failed = 0
    for episode_row in episodes:
        now = datetime.datetime.now(datetime.timezone.utc)
        elapsed_hours = (now - pod_started_at).total_seconds() / 3600.0
        avg_hours_per_episode = (elapsed_hours / processed) if processed else 0.0
        if costs.should_stop_for_time(elapsed_hours, avg_hours_per_episode, ctx.cfg.cost):
            logger.warning("time cap reached (%.2fh elapsed) -- stopping before episode %s", elapsed_hours, episode_row["episode_id"])
            break

        avg_cost_per_episode = (db.total_cost(ctx.conn) / processed) if processed else 0.0
        if costs.should_stop_for_budget(ctx.conn, avg_cost_per_episode, ctx.cfg.cost):
            logger.warning("budget cap reached ($%.2f spent) -- stopping before episode %s", db.total_cost(ctx.conn), episode_row["episode_id"])
            break

        ok = process_episode(ctx, episode_row["episode_id"])
        processed += 1
        succeeded += int(ok)
        failed += int(not ok)

        _record_gpu_compute_checkpoint(ctx, pod_started_at)
        _push_heartbeat(ctx)

    return {"processed": processed, "succeeded": succeeded, "failed": failed, "total_episodes_claimed": len(episodes)}
