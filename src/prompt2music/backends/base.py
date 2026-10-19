"""Backend contract shared by every music model integration.

Stdlib-only on purpose: this module is imported inside each backend's isolated
virtual environment by :mod:`prompt2music.worker`.

A backend implements only the operations its model genuinely supports.
Capabilities are declared per checkpoint in :mod:`prompt2music.registry`
and checked before a request ever reaches the model; backends re-check with
:meth:`MusicBackend.check_request` so a direct caller cannot bypass it.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from prompt2music.errors import CancelledError, InvalidRequestError, UnsupportedTaskError


class Task(str, Enum):  # noqa: UP042 - workers may run on Python 3.10 (no StrEnum)
    """Operations a model may support. Values are stable wire/CLI names."""

    TEXT_TO_MUSIC = "text-to-music"  # instrumental or prompt-only generation
    SONG = "song"  # lyrics-conditioned vocal music
    CONTINUE = "continue"  # native audio-conditioned continuation
    EDIT = "edit"  # audio-to-audio (cover / init-audio / style transfer)
    INPAINT = "inpaint"  # regenerate a time region of existing audio

    @classmethod
    def parse(cls, value: str | Task) -> Task:
        if isinstance(value, Task):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            valid = ", ".join(t.value for t in cls)
            raise InvalidRequestError(f"unknown task {value!r}; expected one of: {valid}") from exc


@dataclass(frozen=True)
class Capabilities:
    """What a specific checkpoint supports. Absent features must stay False."""

    tasks: frozenset[Task]
    vocals: bool = False
    lyrics: bool = False
    instrumental: bool = True
    negative_prompt: bool = False
    bpm: bool = False
    key: bool = False
    time_signature: bool = False
    seed: bool = True
    reference_audio: bool = False
    melody_conditioning: bool = False
    batch: bool = False
    cooperative_cancel: bool = False
    min_duration: float = 1.0
    max_duration: float = 30.0
    sample_rate: int | None = None
    channels: int | None = None
    lyrics_format: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tasks"] = sorted(t.value for t in self.tasks)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Capabilities:
        values = dict(data)
        values["tasks"] = frozenset(Task.parse(t) for t in values.get("tasks", []))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in values.items() if k in known})


@dataclass
class GenerationRequest:
    """A single generation call. Paths are absolute by the time a worker sees them."""

    prompt: str
    model: str
    duration: float = 30.0
    task: Task = Task.TEXT_TO_MUSIC
    seed: int | None = None
    negative_prompt: str | None = None
    bpm: int | None = None
    key: str | None = None
    time_signature: str | None = None
    instrumental: bool = True
    lyrics: str | None = None
    vocal_language: str | None = None
    input_audio: Path | None = None  # continue / edit / inpaint source
    region_start: float | None = None  # inpaint region, seconds
    region_end: float | None = None
    strength: float | None = None  # edit: 0 = keep source, 1 = ignore source
    reference_audio: Path | None = None
    melody_audio: Path | None = None
    batch_size: int = 1
    steps: int | None = None
    guidance: float | None = None
    output_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.task = Task.parse(self.task)
        for name in ("input_audio", "reference_audio", "melody_audio", "output_path"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                setattr(self, name, Path(value))
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.prompt, str):
            raise InvalidRequestError("prompt must be a string")
        if self.task in (Task.TEXT_TO_MUSIC,) and not self.prompt.strip():
            raise InvalidRequestError("prompt must not be empty")
        if not (self.duration > 0) or self.duration != self.duration:  # rejects NaN
            raise InvalidRequestError("duration must be a positive number of seconds")
        if self.seed is not None and self.seed < 0:
            self.seed = None  # -1 conventionally means random
        if self.bpm is not None and not (20 <= self.bpm <= 400):
            raise InvalidRequestError("bpm must be between 20 and 400")
        if not (1 <= self.batch_size <= 16):
            raise InvalidRequestError("batch_size must be between 1 and 16")
        if self.steps is not None and self.steps < 1:
            raise InvalidRequestError("steps must be >= 1")
        if self.strength is not None and not (0.0 <= self.strength <= 1.0):
            raise InvalidRequestError("strength must be within [0, 1]")
        if self.task in (Task.CONTINUE, Task.EDIT, Task.INPAINT) and self.input_audio is None:
            raise InvalidRequestError(f"task {self.task.value} requires input audio")
        if self.task is Task.SONG and not (self.lyrics and self.lyrics.strip()):
            raise InvalidRequestError("song generation requires lyrics")
        if self.task is Task.INPAINT:
            if self.region_start is None or self.region_end is None:
                raise InvalidRequestError("inpaint requires region start and end")
            if self.region_start < 0 or self.region_end <= self.region_start:
                raise InvalidRequestError("inpaint region must satisfy 0 <= start < end")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["task"] = self.task.value
        for key in ("input_audio", "reference_audio", "melody_audio", "output_path"):
            if data[key] is not None:
                data[key] = str(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationRequest:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class GeneratedAudio:
    path: Path
    sample_rate: int
    channels: int
    duration: float
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration": round(self.duration, 3),
            "seed": self.seed,
        }


@dataclass
class GenerationResult:
    outputs: list[GeneratedAudio]
    model: str
    generation_seconds: float
    info: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "outputs": [o.as_dict() for o in self.outputs],
            "model": self.model,
            "generation_seconds": round(self.generation_seconds, 3),
            "info": self.info,
        }


ProgressCallback = Callable[[float | None, str], None]


class CancelToken:
    """Cooperative cancellation flag checked by backends between steps."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancelledError("generation cancelled")


def _noop_progress(fraction: float | None, message: str) -> None:
    return None


class MusicBackend(ABC):
    """One loaded checkpoint. Instances live inside a worker process."""

    #: modules that must import in the backend environment
    required_modules: tuple[str, ...] = ()

    def __init__(self, model_id: str, model_dir: Path | None, options: dict[str, Any],
                 capabilities: Capabilities) -> None:
        self.model_id = model_id
        self.model_dir = model_dir
        self.options = dict(options)
        self._capabilities = capabilities
        self.device: str | None = None
        self.load_seconds: float | None = None

    # -- lifecycle -----------------------------------------------------
    @abstractmethod
    def load(self, device: str, precision: str) -> None:
        """Load weights onto ``device``. Must be safe to call once."""

    @abstractmethod
    def _generate(self, request: GenerationRequest, output_dir: Path,
                  progress: ProgressCallback, cancel: CancelToken) -> GenerationResult:
        """Run inference and write audio files into ``output_dir``."""

    def unload(self) -> None:
        """Release model memory. Workers also exit, which frees everything."""
        self._release_torch_memory()

    def capabilities(self) -> Capabilities:
        return self._capabilities

    # -- shared behaviour ---------------------------------------------
    def check_request(self, request: GenerationRequest) -> None:
        check_request_against(self.model_id, self._capabilities, request)

    def generate(self, request: GenerationRequest, output_dir: Path,
                 progress: ProgressCallback | None = None,
                 cancel: CancelToken | None = None) -> GenerationResult:
        self.check_request(request)
        output_dir.mkdir(parents=True, exist_ok=True)
        cancel = cancel or CancelToken()
        started = time.perf_counter()
        result = self._generate(request, output_dir, progress or _noop_progress, cancel)
        if not result.outputs:
            raise_generation_error("backend returned no audio")
        for out in result.outputs:
            if not out.path.is_file() or out.path.stat().st_size == 0:
                raise_generation_error(f"backend reported missing or empty output {out.path}")
        result.generation_seconds = time.perf_counter() - started
        return result

    def memory_stats(self) -> dict[str, Any]:
        """Peak accelerator memory since load, when torch exposes it."""
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            return {}
        if not torch.cuda.is_available():
            return {}
        return {
            "cuda_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "cuda_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        }

    @staticmethod
    def _release_torch_memory() -> None:
        try:
            import gc

            import torch  # type: ignore[import-not-found]
        except ImportError:
            return
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def raise_generation_error(message: str) -> None:
    from prompt2music.errors import GenerationError

    raise GenerationError(message)


def check_request_against(model_id: str, caps: Capabilities, request: GenerationRequest) -> None:
    """Refuse anything the checkpoint does not genuinely support."""
    if request.task not in caps.tasks:
        supported = ", ".join(sorted(t.value for t in caps.tasks)) or "none"
        raise UnsupportedTaskError(
            f"{model_id} does not support {request.task.value} (supports: {supported})",
            hint="run `prompt2music models available --task "
            f"{request.task.value}` to find a model that does",
        )
    if not request.instrumental and not caps.vocals:
        raise UnsupportedTaskError(f"{model_id} cannot generate vocals; use --instrumental")
    if request.lyrics and not caps.lyrics:
        raise UnsupportedTaskError(f"{model_id} does not accept lyrics")
    if request.instrumental and not caps.instrumental:
        raise UnsupportedTaskError(f"{model_id} cannot generate instrumental-only music")
    if request.negative_prompt and not caps.negative_prompt:
        raise UnsupportedTaskError(f"{model_id} does not support negative prompts")
    if request.reference_audio is not None and not caps.reference_audio:
        raise UnsupportedTaskError(f"{model_id} does not support reference-audio conditioning")
    if request.melody_audio is not None and not caps.melody_conditioning:
        raise UnsupportedTaskError(f"{model_id} does not support melody conditioning")
    if request.batch_size > 1 and not caps.batch:
        raise UnsupportedTaskError(f"{model_id} does not support batched generation")
    if request.duration > caps.max_duration:
        raise InvalidRequestError(
            f"{model_id} supports at most {caps.max_duration:g}s per request "
            f"(requested {request.duration:g}s)"
        )
    if request.duration < caps.min_duration:
        raise InvalidRequestError(
            f"{model_id} needs at least {caps.min_duration:g}s (requested {request.duration:g}s)"
        )
