"""Stable Audio 3 through ``stable_audio_3.StableAudioModel``.

API used (upstream README, ``docs/workflows/inference.md`` and ``model.py``):

    model = StableAudioModel.from_pretrained(name, device=..., model_half=...)
    audio = model.generate(prompt=..., negative_prompt=..., duration=..., steps=...,
                           cfg_scale=..., seed=..., batch_size=...,
                           init_audio=(sr, array), init_noise_level=...,
                           inpaint_audio=(sr, array), inpaint_mask_start_seconds=...,
                           inpaint_mask_end_seconds=...)
    # -> float32 tensor (batch, channels, samples) at model.model.sample_rate

Continuation is upstream's documented recipe: inpaint with the mask starting
at the end of the source and a longer total duration.

Weights are resolved by upstream through ``hf_hub_download``; the host points
HF_HUB_CACHE at the managed model directory and forces offline mode.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from prompt2music.backends.audio_io import output_name, read_wav, write_wav
from prompt2music.backends.base import (
    CancelToken,
    GeneratedAudio,
    GenerationRequest,
    GenerationResult,
    MusicBackend,
    ProgressCallback,
    Task,
)
from prompt2music.errors import GenerationError, OutOfMemoryError


class StableAudioBackend(MusicBackend):
    required_modules = ("torch", "stable_audio_3")

    def load(self, device: str, precision: str) -> None:
        import torch
        from stable_audio_3 import StableAudioModel

        started = time.perf_counter()
        self.torch = torch
        self.model = StableAudioModel.from_pretrained(
            self.options["upstream_name"], device=device, model_half=precision != "fp32"
        )
        self.sample_rate = int(self.model.model.sample_rate)
        self.device = device
        self.load_seconds = time.perf_counter() - started

    def unload(self) -> None:
        self.model = None
        super().unload()

    def _generate(self, request: GenerationRequest, output_dir: Path,
                  progress: ProgressCallback, cancel: CancelToken) -> GenerationResult:
        torch = self.torch
        seed = request.seed if request.seed is not None else int(torch.seed() % 99999)
        kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "duration": float(request.duration),
            "seed": seed,
            "batch_size": request.batch_size,
        }
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
        if request.steps:
            kwargs["steps"] = request.steps
        if request.guidance:
            kwargs["cfg_scale"] = request.guidance
        source_seconds = 0.0
        if request.input_audio is not None:
            audio, rate = read_wav(request.input_audio)
            source = (rate, audio)
            source_seconds = audio.shape[-1] / rate
            if request.task is Task.EDIT:
                kwargs["init_audio"] = source
                kwargs["init_noise_level"] = (request.strength
                                              if request.strength is not None else 0.9)
                kwargs["duration"] = source_seconds
            elif request.task is Task.INPAINT:
                kwargs["inpaint_audio"] = source
                kwargs["inpaint_mask_start_seconds"] = float(request.region_start or 0.0)
                kwargs["inpaint_mask_end_seconds"] = float(request.region_end or 0.0)
                kwargs["duration"] = max(source_seconds, float(request.region_end or 0.0))
            elif request.task is Task.CONTINUE:
                kwargs["inpaint_audio"] = source
                kwargs["inpaint_mask_start_seconds"] = source_seconds
                kwargs["inpaint_mask_end_seconds"] = source_seconds + request.duration
                kwargs["duration"] = source_seconds + request.duration
        progress(None, "sampling")
        cancel.raise_if_cancelled()
        try:
            with torch.inference_mode():
                audio = self.model.generate(**kwargs)
        except torch.cuda.OutOfMemoryError as exc:
            raise OutOfMemoryError(f"CUDA out of memory: {exc}") from exc
        cancel.raise_if_cancelled()
        out = audio.detach().float().cpu().numpy()
        if out.ndim == 2:
            out = out[None]
        if out.shape[0] == 0:
            raise GenerationError("Stable Audio returned an empty batch")
        if request.task is Task.CONTINUE and request.extra.get("only_new"):
            out = out[..., int(source_seconds * self.sample_rate):]
        progress(1.0, "writing audio")
        base = request.output_path or (output_dir / "stable-audio.wav")
        outputs = []
        for b in range(out.shape[0]):
            path = output_name(base, b, out.shape[0])
            channels, frames = write_wav(path, out[b], self.sample_rate)
            outputs.append(GeneratedAudio(path, self.sample_rate, channels,
                                          frames / self.sample_rate, seed))
        return GenerationResult(outputs, self.model_id, 0.0, {"seed": seed,
                                "upstream_model": self.options["upstream_name"]})
