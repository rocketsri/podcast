"""faster-whisper wrapper: per-clip transcript + the two whisper signals
quality.py corroborates its content-based filters with (`no_speech_prob`,
`avg_logprob`). The transcript itself is a spec-required manifest field, not
the pipeline's core deliverable -- accuracy requirements here are looser than
for diarization/speaker-ID.

Each clip is transcribed independently (no cross-clip conditioning) since
clips are disjoint single-speaker spans, not a continuous narrative -- so
`condition_on_previous_text=False` is passed explicitly rather than relying
on faster-whisper's own default (True), which would otherwise let an
unrelated previous clip's text bias decoding of this one. A clip can decode
into more than one internal whisper segment (e.g. a clip-internal pause);
those are joined into one utterance and their two confidence signals
averaged, duration-weighted, into one number per clip -- the same
duration-weighted-average pattern used elsewhere in this codebase
(cluster.py's centroid updates, segment.py's vad_confidence).

`language="en"` is passed explicitly rather than left to Whisper's
per-clip auto-detection: every podcast in the corpus is already filtered to
English at discovery time (`select_podcasts_free.py --language-prefix`,
default "en"), and on short/noisy/ambiguous clips Whisper's own
auto-detection occasionally misfires to a wrong language and decodes fluent
-looking gibberish in it -- a known Whisper failure mode, confirmed live in
this corpus (PROBLEMS.md #20). Pinning the language to what every clip
actually is removes that failure mode at the source rather than just
filtering its output after the fact.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from pipeline import audio, db


@dataclass(frozen=True)
class TranscriptionResult:
    utterance: str
    no_speech_prob: float
    avg_logprob: float


def load_model(model_size: str, device: str = "cuda", compute_type: str = "float16") -> WhisperModel:
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_clip(samples: np.ndarray, sample_rate: int, model: WhisperModel) -> TranscriptionResult:
    """Run ASR over one clip's already-sliced sample array. Empty/no-speech
    output is a valid result (utterance="", no_speech_prob=1.0,
    avg_logprob=0.0), not an error -- quality.py's low_asr_confidence filter
    is what acts on it."""
    if sample_rate != 16000:
        raise ValueError(f"asr.transcribe_clip expects 16kHz audio, got {sample_rate}")

    segments, _info = model.transcribe(samples, language="en", condition_on_previous_text=False)
    segments = list(segments)
    if not segments:
        return TranscriptionResult(utterance="", no_speech_prob=1.0, avg_logprob=0.0)

    total_duration = sum(seg.end - seg.start for seg in segments)
    if total_duration <= 0:
        total_duration = len(segments)  # degenerate zero-length segments; fall back to an unweighted average
        weights = [1.0] * len(segments)
    else:
        weights = [seg.end - seg.start for seg in segments]

    utterance = " ".join(seg.text.strip() for seg in segments).strip()
    no_speech_prob = sum(w * seg.no_speech_prob for w, seg in zip(weights, segments)) / total_duration
    avg_logprob = sum(w * seg.avg_logprob for w, seg in zip(weights, segments)) / total_duration
    return TranscriptionResult(utterance=utterance, no_speech_prob=no_speech_prob, avg_logprob=avg_logprob)


def transcribe_clips_for_episode(
    conn: sqlite3.Connection,
    episode_id: str,
    episode_samples: np.ndarray,
    sample_rate: int,
    model: WhisperModel,
) -> int:
    """Runs once per episode, after segmentation -- transcribes every
    not-yet-discarded, not-yet-transcribed clip and writes the three ASR
    columns back via db.update_clip_fields. Clips already marked
    discard_reason (e.g. too_short) are skipped; ASR signals only matter for
    clips still in contention. Returns the number of clips transcribed."""
    transcribed = 0
    for clip in db.get_clips_for_episode(conn, episode_id):
        if clip["discard_reason"] is not None or clip["utterance"] is not None:
            continue
        clip_samples = audio.slice_samples(episode_samples, sample_rate, clip["start_seconds"], clip["end_seconds"])
        result = transcribe_clip(clip_samples, sample_rate, model)
        db.update_clip_fields(
            conn, clip["clip_id"],
            utterance=result.utterance, no_speech_prob=result.no_speech_prob, avg_logprob=result.avg_logprob,
        )
        transcribed += 1
    return transcribed
