"""ffmpeg/ffprobe wrappers: probe duration, transcode to 16k mono wav, encode clips to flac.

All commands are invoked as argv lists (never via a shell), so there is no
shell-injection surface even though paths/timestamps are interpolated.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioProcessingError(RuntimeError):
    pass


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioProcessingError(f"command failed ({cmd[0]}): {result.stderr[-2000:]}")


def probe_duration_seconds(path: str | Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioProcessingError(f"ffprobe failed on {path}: {result.stderr[-2000:]}")
    data = json.loads(result.stdout)
    try:
        return float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AudioProcessingError(f"ffprobe returned no duration for {path}") from exc


def transcode_to_wav(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> None:
    """Transcode arbitrary input audio (mp3/m4a/etc) to PCM16 mono wav at the
    target sample rate — the common substrate VAD/diarization/ASR all expect."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", str(sample_rate), "-ac", str(channels),
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    _run(cmd)


def extract_clip_to_flac(
    input_wav_path: str | Path,
    output_flac_path: str | Path,
    start_seconds: float,
    end_seconds: float,
) -> None:
    """Cut [start_seconds, end_seconds) out of an already-transcoded wav and
    encode it to flac — the final per-clip artifact uploaded to R2. `-ss`
    before `-i` is sample-accurate here because the input is uncompressed PCM
    (no keyframe-seek inaccuracy), and is much faster than output seeking."""
    if end_seconds <= start_seconds:
        raise AudioProcessingError(f"invalid clip range: {start_seconds}..{end_seconds}")
    output_flac_path = Path(output_flac_path)
    output_flac_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end_seconds - start_seconds
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_seconds:.3f}",
        "-i", str(input_wav_path),
        "-t", f"{duration:.3f}",
        "-c:a", "flac",
        str(output_flac_path),
    ]
    _run(cmd)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a wav file as a float32 mono array + sample rate — used by
    quality.py's RMS-energy/silence check and similar lightweight signal math."""
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, sample_rate


def slice_samples(samples: np.ndarray, sample_rate: int, start_seconds: float, end_seconds: float) -> np.ndarray:
    """Cut [start_seconds, end_seconds) out of an already-loaded episode-level
    sample array -- the in-memory equivalent of extract_clip_to_flac, used by
    quality.py/asr.py so they don't need a per-clip file on disk just to
    compute a signal or run ASR."""
    start_idx = max(0, int(start_seconds * sample_rate))
    end_idx = min(len(samples), int(end_seconds * sample_rate))
    if end_idx <= start_idx:
        return np.zeros(1, dtype=np.float32)
    return samples[start_idx:end_idx]
