"""ACE-Step 1.5 through its documented Python API (``docs/en/INFERENCE.md``).

    AceStepHandler().initialize_service(project_root, config_path, device, ...)
    LLMHandler().initialize(checkpoint_dir, lm_model_path, backend, device, ...)
    generate_music(dit_handler, llm_handler, GenerationParams(...), GenerationConfig(...))

Task mapping: text-to-music and song → ``text2music``; edit → ``cover`` (with
``audio_cover_strength``); inpaint → ``repaint``. Upstream documents no
continuation task, so continuation is refused rather than faked.

Installed layout: ``<model_dir>/checkpoints/<component>``, matching upstream's
``<project_root>/checkpoints`` convention; ACESTEP_CHECKPOINTS_DIR is set too.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from prompt2music.backends.audio_io import output_name, write_wav
from prompt2music.backends.base import (
    CancelToken,
    GeneratedAudio,
    GenerationRequest,
    GenerationResult,
    MusicBackend,
    ProgressCallback,
    Task,
)
from prompt2music.errors import CancelledError, GenerationError, InvalidRequestError

CAPTION_LIMIT = 512
LYRICS_LIMIT = 4096
_TASKS = {
    Task.TEXT_TO_MUSIC: "text2music",
    Task.SONG: "text2music",
    Task.EDIT: "cover",
    Task.INPAINT: "repaint",
}


class AceStepBackend(MusicBackend):
    required_modules = ("torch", "acestep")

    def load(self, device: str, precision: str) -> None:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        if self.model_dir is None:
            raise GenerationError("ACE-Step needs a local checkpoint directory")
        started = time.perf_counter()
        ckpt = self.model_dir / "checkpoints"
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(ckpt)
        offload = bool(self.options.get("cpu_offload", False))
        self.dit = AceStepHandler()
        status, ok = self.dit.initialize_service(
            project_root=str(self.model_dir),
            config_path=self.options["config_path"],
            device=device,
            compile_model=bool(self.options.get("torch_compile", False)),
            offload_to_cpu=offload,
            quantization=self.options.get("quantization"),
        )
        if not ok:
            raise GenerationError(f"ACE-Step DiT failed to initialise: {status}")
        self.llm: Any = None
        if self.options.get("use_lm", True):
            self.llm = LLMHandler()
            status, ok = self.llm.initialize(
                checkpoint_dir=str(ckpt),
                lm_model_path=self.options["lm_model"],
                backend=self.options.get("lm_backend", "pt"),
                device=device,
                offload_to_cpu=offload,
            )
            if not ok:
                raise GenerationError(f"ACE-Step LM failed to initialise: {status}",
                                      hint="set backend option use_lm=false to run without it")
        self.device = device
        self.load_seconds = time.perf_counter() - started

    def unload(self) -> None:
        self.dit = None
        self.llm = None
        super().unload()

    def _generate(self, request: GenerationRequest, output_dir: Path,
                  progress: ProgressCallback, cancel: CancelToken) -> GenerationResult:
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        caption = request.prompt.strip()
        if len(caption) > CAPTION_LIMIT:
            raise InvalidRequestError(f"ACE-Step captions are limited to {CAPTION_LIMIT} "
                                      f"characters (got {len(caption)})")
        lyrics = (request.lyrics or "").strip()
        if request.instrumental and not lyrics:
            lyrics = "[Instrumental]"
        if len(lyrics) > LYRICS_LIMIT:
            raise InvalidRequestError(f"ACE-Step lyrics are limited to {LYRICS_LIMIT} characters")
        params: dict[str, Any] = {
            "task_type": _TASKS[request.task],
            "caption": caption,
            "lyrics": lyrics,
            "instrumental": request.instrumental,
            "duration": float(request.duration),
            "bpm": request.bpm,
            "keyscale": request.key or "",
            "timesignature": request.time_signature or "",
            "vocal_language": request.vocal_language or "unknown",
            "inference_steps": request.steps or int(self.options.get("inference_steps", 8)),
            "seed": request.seed if request.seed is not None else -1,
        }
        if request.input_audio is not None:
            params["src_audio"] = str(request.input_audio)
        if request.reference_audio is not None:
            params["reference_audio"] = str(request.reference_audio)
        if request.task is Task.EDIT and request.strength is not None:
            params["audio_cover_strength"] = 1.0 - request.strength
        if request.task is Task.INPAINT:
            params["repainting_start"] = float(request.region_start or 0.0)
            params["repainting_end"] = float(request.region_end or -1)
        seeds = None
        if request.seed is not None:
            seeds = [request.seed + i for i in range(request.batch_size)]
        config = GenerationConfig(batch_size=request.batch_size,
                                  use_random_seed=request.seed is None,
                                  seeds=seeds, audio_format="wav")

        def _progress(*args: Any, **kwargs: Any) -> None:
            if cancel.cancelled:
                raise CancelledError("generation cancelled")
            value = args[0] if args else kwargs.get("value")
            desc = kwargs.get("desc") or (args[1] if len(args) > 1 else "")
            frac = float(value) if isinstance(value, (int, float)) else None
            progress(frac, str(desc or "generating"))

        scratch = output_dir / ".acestep"
        result = generate_music(self.dit, self.llm, GenerationParams(**params), config,
                                save_dir=str(scratch), progress=_progress)
        cancel.raise_if_cancelled()
        if not getattr(result, "success", False):
            raise GenerationError(f"ACE-Step generation failed: {getattr(result, 'error', '?')}")
        base = request.output_path or (output_dir / "ace-step.wav")
        outputs = []
        audios = list(result.audios)
        for i, item in enumerate(audios):
            path = output_name(base, i, len(audios))
            sr = int(item.get("sample_rate", 48000))
            tensor = item.get("tensor")
            if tensor is not None:
                channels, frames = write_wav(path, tensor, sr)
            else:
                src = Path(item["path"])
                path = path.with_suffix(src.suffix)
                os.replace(src, path)
                channels, frames = 2, int(request.duration * sr)
            seed = (item.get("params") or {}).get("seed", seeds[i] if seeds else None)
            outputs.append(GeneratedAudio(path, sr, channels, frames / sr, seed))
        return GenerationResult(outputs, self.model_id, 0.0,
                                {"task_type": params["task_type"],
                                 "inference_steps": params["inference_steps"]})
