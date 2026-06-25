"""CPU-only, no-signup stand-in for pipeline/diarize.py's pyannote pipeline,
for the free/local trial path: Silero VAD's own speech segments (vad.py,
already a hard dependency everywhere) stand in for pyannote's speech-activity
step, Resemblyzer embeds each segment, and within-episode
AgglomerativeClustering -- the same cosine/average-linkage pattern
cluster.py already uses for cross-episode matching -- assigns local speaker
labels. Produces the exact diarize.SpeakerTurn/DiarizationResult shapes, so
every downstream module (segment.py, quality.py, cluster.py,
pipeline_runner.py) runs unmodified against either backend.

Resemblyzer ships its own pretrained weights with the pip package (no HF
auth, no license click-through) -- the entire reason this module exists
instead of just pointing diarize.py's pyannote pipeline at a CPU device.

This is a coarser approximation than pyannote on two specific axes, both
inherent to swapping a joint segmentation+diarization model for "cluster
fixed VAD segments": a speaker turn can only start/end at a VAD segment
boundary (no mid-segment speaker-change detection), and overlapping speech
is invisible to it (Silero VAD reports one speech/non-speech timeline, not
concurrent per-speaker activity) -- so overlap_mask_seconds() will never
find anything in turns this module produces, unlike pyannote's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering

from pipeline import audio, vad
from pipeline.diarize import DiarizationResult, SpeakerTurn

MIN_SEGMENT_SECONDS_FOR_EMBEDDING = 0.5  # below this, a Resemblyzer embedding is mostly zero-padding


@dataclass
class LocalDiarizationPipeline:
    vad_model: object
    voice_encoder: VoiceEncoder
    match_threshold: float = 0.75


def load_pipeline(device: str = "cpu", match_threshold: float = 0.75) -> LocalDiarizationPipeline:
    return LocalDiarizationPipeline(
        vad_model=vad.load_model(),
        voice_encoder=VoiceEncoder(device=device),
        match_threshold=match_threshold,
    )


def _embed_segment(encoder: VoiceEncoder, samples: np.ndarray, sample_rate: int, start_seconds: float, end_seconds: float) -> np.ndarray | None:
    chunk = samples[int(start_seconds * sample_rate):int(end_seconds * sample_rate)]
    if len(chunk) < int(MIN_SEGMENT_SECONDS_FOR_EMBEDDING * sample_rate):
        return None
    processed = preprocess_wav(chunk)
    if len(processed) == 0:
        return None
    return encoder.embed_utterance(processed)


def diarize(
    audio_path: str,
    pipeline: LocalDiarizationPipeline,
    min_local_speaker_seconds_for_embedding: float = 1.5,
) -> DiarizationResult:
    """Approximates diarize.diarize(): one turn per VAD speech segment,
    labeled by within-episode voice cluster instead of pyannote's joint
    segmentation+diarization. Same return shape, so cluster.py's
    cross-episode matching and segment.py's turn-grouping need no changes."""
    samples, sample_rate = audio.read_wav(audio_path)
    segments = vad.run_vad(samples, pipeline.vad_model)
    if not segments:
        return DiarizationResult(turns=[], embeddings={})

    segment_embeddings = [
        _embed_segment(pipeline.voice_encoder, samples, sample_rate, seg.start_seconds, seg.end_seconds)
        for seg in segments
    ]
    embeddable_indices = [i for i, e in enumerate(segment_embeddings) if e is not None]

    cluster_id_by_index: dict[int, int] = {}
    if len(embeddable_indices) == 1:
        cluster_id_by_index[embeddable_indices[0]] = 0
    elif len(embeddable_indices) > 1:
        matrix = np.stack([segment_embeddings[i] for i in embeddable_indices])
        clustering = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average",
            distance_threshold=1.0 - pipeline.match_threshold,
        )
        labels = clustering.fit_predict(matrix)
        cluster_id_by_index = {i: int(label) for i, label in zip(embeddable_indices, labels)}

    turns: list[SpeakerTurn] = []
    speech_seconds_by_label: dict[str, float] = {}
    embedding_sum_by_label: dict[str, np.ndarray] = {}
    embedding_weight_by_label: dict[str, float] = {}
    for i, seg in enumerate(segments):
        cluster_id = cluster_id_by_index.get(i)
        local_label = f"SPEAKER_{cluster_id:02d}" if cluster_id is not None else f"SPEAKER_UNK_{i:04d}"
        turns.append(SpeakerTurn(local_label=local_label, start_seconds=seg.start_seconds, end_seconds=seg.end_seconds))

        duration = seg.end_seconds - seg.start_seconds
        speech_seconds_by_label[local_label] = speech_seconds_by_label.get(local_label, 0.0) + duration
        if cluster_id is not None:
            embedding = segment_embeddings[i]
            embedding_sum_by_label[local_label] = embedding_sum_by_label.get(local_label, np.zeros_like(embedding)) + embedding * duration
            embedding_weight_by_label[local_label] = embedding_weight_by_label.get(local_label, 0.0) + duration

    embeddings: dict[str, np.ndarray] = {}
    for label, total_seconds in speech_seconds_by_label.items():
        if total_seconds < min_local_speaker_seconds_for_embedding:
            continue
        weight = embedding_weight_by_label.get(label, 0.0)
        if weight <= 0.0:
            continue
        embeddings[label] = (embedding_sum_by_label[label] / weight).astype(np.float32)

    return DiarizationResult(turns=turns, embeddings=embeddings)
