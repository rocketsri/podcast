"""Silero VAD wrapper: speech sub-segments with a per-segment confidence score.

Confidence isn't returned by Silero's own get_speech_timestamps (it only
applies a threshold internally and discards the raw probabilities), so this
module recomputes per-chunk speech probabilities the same way that function
does internally and averages them over each detected segment. The same
frame-probability array is also exposed standalone for segment.py's
pause-point search (cutting at low-probability frames inside a long turn,
not just at the merged segment boundaries).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

SAMPLE_RATE = 16000
WINDOW_SIZE_SAMPLES = 512  # fixed chunk size for the 16kHz Silero VAD model


@dataclass(frozen=True)
class SpeechSegment:
    start_seconds: float
    end_seconds: float
    confidence: float  # mean frame-level speech probability over the segment


def load_model():
    model = load_silero_vad()
    model.eval()
    return model


def frame_speech_probabilities(samples: np.ndarray, model) -> np.ndarray:
    """Per-chunk (32ms) speech probability across the whole signal."""
    model.reset_states()
    audio = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
    probs = []
    with torch.no_grad():
        for start in range(0, len(audio), WINDOW_SIZE_SAMPLES):
            chunk = audio[start : start + WINDOW_SIZE_SAMPLES]
            if len(chunk) < WINDOW_SIZE_SAMPLES:
                chunk = torch.nn.functional.pad(chunk, (0, WINDOW_SIZE_SAMPLES - len(chunk)))
            probs.append(model(chunk, SAMPLE_RATE).item())
    return np.array(probs, dtype=np.float32)


def run_vad(
    samples: np.ndarray,
    model,
    min_speech_confidence: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
) -> list[SpeechSegment]:
    """Detect speech sub-segments and attach a per-segment confidence (mean
    frame-level probability inside that segment), consumed by quality.py's
    `vad_low_confidence` discard filter and by config.vad.min_speech_confidence."""
    audio = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
    timestamps = get_speech_timestamps(
        audio,
        model,
        threshold=min_speech_confidence,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=True,
    )
    if not timestamps:
        return []

    frame_probs = frame_speech_probabilities(samples, model)
    frame_seconds = WINDOW_SIZE_SAMPLES / SAMPLE_RATE

    segments = []
    for ts in timestamps:
        start_frame = int(ts["start"] / frame_seconds)
        end_frame = max(start_frame + 1, int(ts["end"] / frame_seconds))
        window = frame_probs[start_frame:end_frame]
        confidence = float(window.mean()) if len(window) else min_speech_confidence
        segments.append(SpeechSegment(start_seconds=ts["start"], end_seconds=ts["end"], confidence=confidence))
    return segments
