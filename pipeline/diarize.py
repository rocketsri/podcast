"""pyannote/speaker-diarization-3.1 wrapper: per-episode local speaker turns
+ one representative embedding per local speaker, for cluster.py's
cross-episode matching.

Uses the pipeline's own `return_embeddings=True` rather than a second pass
through a standalone embedding model. Confirmed directly from the installed
pyannote.audio 3.3.2 source (pyannote/audio/pipelines/speaker_diarization.py,
SpeakerDiarization.apply()), not guessed from docs:

    diarization, embeddings = pipeline(file, return_embeddings=True)
    # embeddings[i] is the centroid embedding for diarization.labels()[i]

This is one centroid per *local speaker for the whole file* (pyannote's own
clustering step already aggregates per-chunk embeddings into a centroid per
detected speaker) -- not one embedding per turn. That centroid is exactly
what cluster.py needs for cross-episode matching, so no separate
`pyannote/embedding` model/pass is run here.

Two documented edge cases in the centroid array (both confirmed from the
same source, not inferred): if no speech is detected at all, embeddings is
an empty `(0, dim)` array; if `speaker_count` ever reconstructs more
diarization labels than clustering produced centroids for, the extra rows
are zero-padded. Zero-padded rows are filtered out below since an
all-zero vector is not a real embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyannote.audio import Pipeline

PIPELINE_CHECKPOINT = "pyannote/speaker-diarization-3.1"


class DiarizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpeakerTurn:
    local_label: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class DiarizationResult:
    turns: list[SpeakerTurn]
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)  # local_label -> centroid, only for labels that cleared the speech-duration floor


def load_pipeline(hf_token: str, device: str = "cuda", embedding_exclude_overlap: bool = True) -> "Pipeline":
    import torch
    from pyannote.audio import Pipeline

    if not hf_token:
        raise DiarizationError("HF_TOKEN required to load gated pyannote pipeline")
    pipeline = Pipeline.from_pretrained(PIPELINE_CHECKPOINT, use_auth_token=hf_token)
    if pipeline is None:
        raise DiarizationError(
            f"Pipeline.from_pretrained returned None for {PIPELINE_CHECKPOINT} "
            "-- usually means the HF gated-model agreement hasn't been accepted"
        )
    pipeline.embedding_exclude_overlap = embedding_exclude_overlap
    pipeline.to(torch.device(device))
    return pipeline


def diarize(
    audio_path: str,
    pipeline: "Pipeline",
    min_local_speaker_seconds_for_embedding: float = 1.5,
) -> DiarizationResult:
    """Run diarization once on a transcoded (16kHz mono wav) episode file."""
    diarization, centroids = pipeline(audio_path, return_embeddings=True)

    labels = diarization.labels()
    turns: list[SpeakerTurn] = []
    speech_seconds_by_label: dict[str, float] = {}
    for segment, _, label in diarization.itertracks(yield_label=True):
        turns.append(SpeakerTurn(local_label=label, start_seconds=segment.start, end_seconds=segment.end))
        speech_seconds_by_label[label] = speech_seconds_by_label.get(label, 0.0) + (segment.end - segment.start)

    embeddings: dict[str, np.ndarray] = {}
    if centroids is not None:
        for label, row in zip(labels, centroids):
            if speech_seconds_by_label.get(label, 0.0) < min_local_speaker_seconds_for_embedding:
                continue
            if not np.any(row):
                continue  # zero-padded placeholder centroid, not a real embedding
            embeddings[label] = np.asarray(row, dtype=np.float32)

    return DiarizationResult(turns=turns, embeddings=embeddings)


def dominant_speaker_share(turns: list[SpeakerTurn]) -> tuple[str, float, int] | None:
    """(label, share, num_labels) for the most-talkative local label by total
    turn-seconds, or None if turns is empty. Cheap, model-agnostic guardrail
    against clustering collapse (see PROBLEMS.md): a real multi-host
    conversation essentially never has one label holding >90%+ of all
    speech, so a high share alongside >=2 detected labels means the
    diarization backend likely merged distinct speakers into one cluster
    rather than that the episode is genuinely solo."""
    totals: dict[str, float] = {}
    for turn in turns:
        totals[turn.local_label] = totals.get(turn.local_label, 0.0) + (turn.end_seconds - turn.start_seconds)
    if not totals:
        return None
    total_seconds = sum(totals.values())
    if total_seconds <= 0:
        return None
    label, seconds = max(totals.items(), key=lambda kv: kv[1])
    return label, seconds / total_seconds, len(totals)


def overlap_mask_seconds(turns: list[SpeakerTurn]) -> list[tuple[float, float]]:
    """Time ranges where >=2 distinct local speakers' turns overlap -- the
    crosstalk exclusion mask segment.py subtracts before clip candidates are
    built. Sweep-line over turn boundaries counting concurrently-active
    distinct labels, since pyannote turns for the same label are already
    non-overlapping but turns across labels can and do overlap."""
    if len(turns) < 2:
        return []

    events = []
    for turn in turns:
        events.append((turn.start_seconds, 1, turn.local_label))
        events.append((turn.end_seconds, -1, turn.local_label))
    events.sort(key=lambda e: (e[0], e[1]))  # ends (-1) before starts (+1) at the same timestamp

    active: dict[str, int] = {}
    overlaps: list[tuple[float, float]] = []
    overlap_start: float | None = None
    for time, delta, label in events:
        was_overlapping = sum(1 for c in active.values() if c > 0) >= 2
        active[label] = active.get(label, 0) + delta
        is_overlapping = sum(1 for c in active.values() if c > 0) >= 2
        if is_overlapping and not was_overlapping:
            overlap_start = time
        elif was_overlapping and not is_overlapping:
            overlaps.append((overlap_start, time))
            overlap_start = None

    return overlaps
