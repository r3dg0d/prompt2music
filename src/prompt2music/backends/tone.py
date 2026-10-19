"""Diagnostic backend: a deterministic sine chord. Not a music model.

It exists so the worker protocol, job daemon, library and MCP server can be
exercised end to end on machines without model weights (including CI).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from prompt2music.backends.audio_io import output_name, write_wav
from prompt2music.backends.base import (
    CancelToken,
    GeneratedAudio,
    GenerationRequest,
    GenerationResult,
    MusicBackend,
    ProgressCallback,
)

SAMPLE_RATE = 44100


class ToneBackend(MusicBackend):
    def load(self, device: str, precision: str) -> None:
        self.device = "cpu"
        self.load_seconds = 0.0

    def _generate(self, request: GenerationRequest, output_dir: Path,
                  progress: ProgressCallback, cancel: CancelToken) -> GenerationResult:
        seed = request.seed if request.seed is not None else 0
        root = 220.0 * (2 ** ((seed % 12) / 12))
        freqs = (root, root * 5 / 4, root * 3 / 2)
        frames = int(request.duration * SAMPLE_RATE)
        delay = float(request.extra.get("seconds_per_step",
                                        self.options.get("seconds_per_step", 0.0)))
        outputs = []
        base = request.output_path or (output_dir / "tone.wav")
        for b in range(request.batch_size):
            left: list[float] = []
            steps = 10
            chunk = max(1, frames // steps)
            for step in range(steps):
                cancel.raise_if_cancelled()
                start, stop = step * chunk, frames if step == steps - 1 else (step + 1) * chunk
                for i in range(start, stop):
                    t = i / SAMPLE_RATE
                    env = min(1.0, t * 20, (frames - i) / SAMPLE_RATE * 20)
                    left.append(0.2 * env * sum(math.sin(2 * math.pi * f * t) for f in freqs))
                progress((b * steps + step + 1) / (steps * request.batch_size),
                         f"synthesising tone {b + 1}/{request.batch_size}")
                if delay:
                    time.sleep(delay)
            path = output_name(base, b, request.batch_size)
            write_wav(path, [left, left], SAMPLE_RATE)
            outputs.append(GeneratedAudio(path, SAMPLE_RATE, 2, frames / SAMPLE_RATE, seed + b))
        return GenerationResult(outputs, self.model_id, 0.0, {"note": "diagnostic tone"})
