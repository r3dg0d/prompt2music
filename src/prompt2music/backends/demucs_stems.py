"""Stem separation with Demucs >= 4.1 (``demucs.api.Separator``).

Demucs 4.1 loads bag-of-models definitions and safetensors weights from the
Hugging Face Hub (``adefossez/HTDemucs`` etc.). The host points HF_HUB_CACHE
at the managed model directory and forces offline mode, so separation never
touches the network once a stem model is downloaded.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from prompt2music.backends.audio_io import read_wav, write_wav
from prompt2music.backends.base import (
    CancelToken,
    GenerationRequest,
    GenerationResult,
    MusicBackend,
    ProgressCallback,
)
from prompt2music.errors import CancelledError, UnsupportedTaskError


class DemucsBackend(MusicBackend):
    required_modules = ("torch", "demucs", "numpy")

    def load(self, device: str, precision: str) -> None:
        import torch
        from demucs.api import Separator

        started = time.perf_counter()
        self.torch = torch
        self.separator: Any = Separator(model=self.options.get("demucs_name", "htdemucs"),
                                   device=device, progress=False)
        self.device = device
        self.load_seconds = time.perf_counter() - started

    def unload(self) -> None:
        self.separator = None
        super().unload()

    def _generate(self, request: GenerationRequest, output_dir: Path,
                  progress: ProgressCallback, cancel: CancelToken) -> GenerationResult:
        raise UnsupportedTaskError("stem models separate audio; they do not generate it")

    def separate(self, input_wav: Path, output_dir: Path, stems: list[str] | None,
                 progress: ProgressCallback, cancel: CancelToken) -> dict[str, Any]:
        torch = self.torch
        audio, rate = read_wav(input_wav)
        sep = self.separator

        def _callback(data: dict[str, Any]) -> None:
            if cancel.cancelled:
                raise CancelledError("separation cancelled")
            length = data.get("audio_length") or 1
            frac = min(1.0, float(data.get("segment_offset", 0)) / float(length))
            progress(frac, f"separating ({data.get('state', 'running')})")

        sep.update_parameter(callback=_callback)
        started = time.perf_counter()
        with torch.inference_mode():
            _orig, separated = sep.separate_tensor(torch.from_numpy(audio), rate)
        output_dir.mkdir(parents=True, exist_ok=True)
        written = {}
        for name, tensor in separated.items():
            if stems and name not in stems:
                continue
            path = output_dir / f"{name}.wav"
            write_wav(path, tensor, sep.samplerate)
            written[name] = str(path)
        return {
            "stems": written,
            "sample_rate": sep.samplerate,
            "model": self.model_id,
            "seconds": round(time.perf_counter() - started, 3),
            "available": list(separated),
        }
