"""WAV read/write for backend workers.

Workers cannot rely on torchaudio/soundfile being present in every upstream
environment, so 16-bit PCM WAV via the stdlib ``wave`` module is the common
interchange format. The host converts inputs to WAV with ffmpeg first.
"""

from __future__ import annotations

import array
import sys
import wave
from pathlib import Path
from typing import Any


def write_wav(path: Path, samples: Any, sample_rate: int) -> tuple[int, int]:
    """Write float samples in [-1, 1] shaped (channels, frames). Returns (channels, frames).

    ``samples`` may be a numpy array, a torch tensor, or a list of per-channel lists.
    """
    if hasattr(samples, "detach"):  # torch tensor
        samples = samples.detach().float().cpu().numpy()
    if hasattr(samples, "ndim"):
        import numpy as np

        data = np.asarray(samples, dtype=np.float32)
        if data.ndim == 1:
            data = data[None, :]
        if data.ndim != 2:
            raise ValueError(f"expected (channels, frames) audio, got shape {data.shape}")
        if not np.all(np.isfinite(data)):
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
        pcm = (np.clip(data, -1.0, 1.0) * 32767.0).round().astype("<i2")
        channels, frames = pcm.shape
        payload = pcm.T.reshape(-1).tobytes()
    else:
        chans = [list(c) for c in samples]
        channels, frames = len(chans), len(chans[0]) if chans else 0
        buf = array.array("h")
        for i in range(frames):
            for c in range(channels):
                v = max(-1.0, min(1.0, float(chans[c][i])))
                buf.append(int(round(v * 32767.0)))
        if sys.byteorder == "big":
            buf.byteswap()
        payload = buf.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(payload)
    return channels, frames


def read_wav(path: Path) -> tuple[Any, int]:
    """Read 16-bit PCM WAV into a float32 numpy array (channels, frames)."""
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM (convert with ffmpeg first)")
        channels = wav.getnchannels()
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return data.reshape(-1, channels).T.copy(), rate


def output_name(base: Path, index: int, total: int, suffix: str = ".wav") -> Path:
    """``song.wav`` for a single output, ``song-2.wav`` etc. for batches."""
    stem = base.stem if base.suffix else base.name
    parent = base.parent if base.suffix else base
    name = f"{stem}{suffix}" if total == 1 else f"{stem}-{index + 1}{suffix}"
    return parent / name
