"""HeartMuLa through ``heartlib.HeartMuLaGenPipeline``.

Mirrors upstream ``examples/run_music_generation.py``:

    pipe = HeartMuLaGenPipeline.from_pretrained(model_path,
        device={"mula": dev, "codec": dev},
        dtype={"mula": torch.bfloat16, "codec": torch.float32},
        version="3B", lazy_load=True)
    with torch.no_grad():
        pipe({"lyrics": lyrics, "tags": tags}, max_audio_length_ms=..., save_path=...,
             topk=50, temperature=1.0, cfg_scale=3.0)

Upstream exposes no seed or progress hooks; seeding goes through the global
torch RNG, which is the best available and is reported as such.
"""

from __future__ import annotations

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


def _wav_info(path: Path) -> tuple[int, int, float]:
    import wave

    with wave.open(str(path), "rb") as wav:
        rate, channels, frames = wav.getframerate(), wav.getnchannels(), wav.getnframes()
    return rate, channels, frames / rate


class HeartMuLaBackend(MusicBackend):
    required_modules = ("torch", "heartlib")

    def load(self, device: str, precision: str) -> None:
        import torch
        from heartlib import HeartMuLaGenPipeline

        if self.model_dir is None:
            raise GenerationError("HeartMuLa needs a local checkpoint directory")
        started = time.perf_counter()
        self.torch = torch
        mula_dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(precision, torch.bfloat16)
        if device == "cpu":
            mula_dtype = torch.float32
        self.pipe = HeartMuLaGenPipeline.from_pretrained(
            str(self.model_dir),
            device={"mula": torch.device(device), "codec": torch.device(device)},
            dtype={"mula": mula_dtype, "codec": torch.float32},
            version=self.options.get("version", "3B"),
            lazy_load=bool(self.options.get("lazy_load", True)),
        )
        self.device = device
        self.load_seconds = time.perf_counter() - started

    def unload(self) -> None:
        self.pipe = None
        super().unload()

    def _generate(self, request: GenerationRequest, output_dir: Path,
                  progress: ProgressCallback, cancel: CancelToken) -> GenerationResult:
        torch = self.torch
        lyrics = (request.lyrics or "").strip() or "[Instrumental]"
        tags = request.prompt.strip()
        base = request.output_path or (output_dir / "heartmula.wav")
        outputs = []
        for b in range(request.batch_size):
            cancel.raise_if_cancelled()
            seed = None
            if request.seed is not None:
                seed = request.seed + b
                torch.manual_seed(seed)
            path = output_name(base, b, request.batch_size)
            progress(b / request.batch_size, f"generating song {b + 1}/{request.batch_size}")
            kwargs: dict[str, Any] = {
                "max_audio_length_ms": int(request.duration * 1000),
                "save_path": str(path),
                "topk": int(request.extra.get("top_k", 50)),
                "temperature": float(request.extra.get("temperature", 1.0)),
                "cfg_scale": float(request.guidance or 3.0),
            }
            with torch.no_grad():
                self.pipe({"lyrics": lyrics, "tags": tags}, **kwargs)
            if not path.is_file():
                raise GenerationError(f"HeartMuLa did not write {path}")
            rate, channels, seconds = _wav_info(path)
            outputs.append(GeneratedAudio(path, rate, channels, seconds, seed))
        return GenerationResult(outputs, self.model_id, 0.0,
                                {"seeding": "global torch RNG (upstream has no seed input)"})
