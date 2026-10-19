"""YuE2 through ``yue2.YuE2Pipeline`` (upstream README, Sept 2026).

    with YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cuda") as pipe:
        song = pipe(style=..., lyrics=..., cot="full", seed=...)
        song.save_artifacts(out_dir)   # writes audio.flac + score + latents

Weights resolve through the Hugging Face cache, which the host points at the
managed model directory with offline mode forced. The song length follows the
lyrics and the generated plan; upstream documents no duration control.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from prompt2music.backends.audio_io import output_name
from prompt2music.backends.base import (
    CancelToken,
    GeneratedAudio,
    GenerationRequest,
    GenerationResult,
    MusicBackend,
    ProgressCallback,
)
from prompt2music.errors import GenerationError


class YuEBackend(MusicBackend):
    required_modules = ("torch", "yue2")

    def load(self, device: str, precision: str) -> None:
        from yue2 import YuE2Pipeline

        started = time.perf_counter()
        self._ctx = YuE2Pipeline.from_pretrained(self.options.get("repo_id", "m-a-p/YuE2-3B"),
                                                 device=device)
        self.pipe = self._ctx.__enter__() if hasattr(self._ctx, "__enter__") else self._ctx
        self.device = device
        self.load_seconds = time.perf_counter() - started

    def unload(self) -> None:
        ctx = getattr(self, "_ctx", None)
        if ctx is not None and hasattr(ctx, "__exit__"):
            ctx.__exit__(None, None, None)
        self._ctx = None
        self.pipe = None
        super().unload()

    def _generate(self, request: GenerationRequest, output_dir: Path,
                  progress: ProgressCallback, cancel: CancelToken) -> GenerationResult:
        base = request.output_path or (output_dir / "yue2.flac")
        outputs = []
        for b in range(request.batch_size):
            cancel.raise_if_cancelled()
            seed = request.seed + b if request.seed is not None else None
            kwargs: dict[str, Any] = {"style": request.prompt, "lyrics": request.lyrics,
                                      "cot": self.options.get("cot", "full")}
            if seed is not None:
                kwargs["seed"] = seed
            progress(b / request.batch_size, f"planning and rendering song {b + 1}")
            song = self.pipe(**kwargs)
            artifacts = output_dir / f".yue2-{b + 1}"
            song.save_artifacts(str(artifacts))
            produced = artifacts / "audio.flac"
            if not produced.is_file():
                raise GenerationError(f"YuE2 did not produce {produced}")
            path = output_name(base, b, request.batch_size, ".flac")
            shutil.move(str(produced), path)
            info = _flac_info(path)
            outputs.append(GeneratedAudio(path, info[0], info[1], info[2], seed))
            if getattr(song, "truncated", False):
                progress(None, "warning: YuE2 reports the song was truncated")
        return GenerationResult(outputs, self.model_id, 0.0, {"cot": kwargs["cot"]})


def _flac_info(path: Path) -> tuple[int, int, float]:
    """Parse the FLAC STREAMINFO block (stdlib only)."""
    with path.open("rb") as fh:
        head = fh.read(42)
    if head[:4] != b"fLaC":
        return 48000, 2, 0.0
    info = head[8:26]
    rate = (info[10] << 12) | (info[11] << 4) | (info[12] >> 4)
    channels = ((info[12] >> 1) & 0x07) + 1
    total = ((info[13] & 0x0F) << 32) | int.from_bytes(info[14:18], "big")
    return rate, channels, total / rate if rate else 0.0
