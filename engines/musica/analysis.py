"""Objective measurements of a rendered audio file.

Standard library only -- ``wave`` plus ``array``. ``audioop`` was removed in
Python 3.13, so the arithmetic is done here explicitly.

Everything in this module is a *measurement*. Nothing here forms an opinion
about whether the music is good; that is not something these functions could
justify (spec 11).
"""

from __future__ import annotations

import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

_WIDTH_TO_TYPE = {1: "b", 2: "h", 4: "i"}


class AudioUnreadable(Exception):
    """The file is not a WAV this module can measure."""


@dataclass(frozen=True)
class AudioStats:
    path: str
    duration_s: float
    sample_rate: int
    channels: int
    sample_width: int
    peak: float                  # 0.0 - 1.0, absolute peak
    rms: float                   # 0.0 - 1.0
    clipped_samples: int
    leading_silence_s: float
    trailing_silence_s: float

    @property
    def clipping_ratio(self) -> float:
        total = max(1, int(self.duration_s * self.sample_rate * self.channels))
        return self.clipped_samples / total


def _samples(frames: bytes, width: int) -> array:
    if width not in _WIDTH_TO_TYPE:
        if width == 3:  # 24-bit: widen to 32-bit signed
            out = array("i")
            for i in range(0, len(frames) - 2, 3):
                value = int.from_bytes(frames[i:i + 3], "little", signed=True)
                out.append(value)
            return out
        raise AudioUnreadable(f"unsupported sample width: {width} bytes")
    data = array(_WIDTH_TO_TYPE[width])
    data.frombytes(frames[: len(frames) - (len(frames) % data.itemsize)])
    return data


def analyse(path: str | Path, *, silence_threshold: float = 0.001) -> AudioStats:
    """Measure a WAV file. Raises ``AudioUnreadable`` rather than guessing."""
    path = Path(path)
    if not path.exists():
        raise AudioUnreadable(f"no such audio file: {path}")
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            nframes = wav.getnframes()
            frames = wav.readframes(nframes)
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioUnreadable(f"could not read {path}: {exc}") from exc

    if rate <= 0:
        raise AudioUnreadable(f"{path} reports a sample rate of {rate}")

    data = _samples(frames, width)
    if not data:
        return AudioStats(str(path), 0.0, rate, channels, width, 0.0, 0.0, 0, 0.0, 0.0)

    full_scale = float(2 ** (8 * (4 if width == 3 else width) - 1))
    peak_raw = 0
    sum_squares = 0.0
    clipped = 0
    clip_at = full_scale - 2

    for value in data:
        magnitude = -value if value < 0 else value
        if magnitude > peak_raw:
            peak_raw = magnitude
        sum_squares += float(value) * float(value)
        if magnitude >= clip_at:
            clipped += 1

    count = len(data)
    peak = peak_raw / full_scale
    rms = (sum_squares / count) ** 0.5 / full_scale

    threshold_raw = silence_threshold * full_scale
    lead = 0
    for value in data:
        if (value if value >= 0 else -value) > threshold_raw:
            break
        lead += 1
    trail = 0
    for i in range(count - 1, -1, -1):
        value = data[i]
        if (value if value >= 0 else -value) > threshold_raw:
            break
        trail += 1

    per_second = rate * channels
    duration = nframes / rate
    return AudioStats(
        path=str(path),
        duration_s=duration,
        sample_rate=rate,
        channels=channels,
        sample_width=width,
        peak=peak,
        rms=rms,
        clipped_samples=clipped,
        leading_silence_s=lead / per_second,
        trailing_silence_s=(0.0 if trail >= count else trail / per_second),
    )
