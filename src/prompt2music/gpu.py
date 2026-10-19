"""GPU/device resolution, VRAM budgeting and a cross-process GPU lock.

Detection reuses ``prompt2music._shared.gpu`` (nvidia-smi), so it works without
torch in the CLI's own environment. The budget is a policy check before a
model loads; inside the worker the same limit is enforced with
``torch.cuda.set_per_process_memory_fraction``.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from prompt2music import paths
from prompt2music._shared.gpu import GpuInfo, query_gpus
from prompt2music.config import MusicConfig
from prompt2music.errors import InvalidRequestError, ResourceLimitError
from prompt2music.registry import ModelSpec

_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)(?:i?b)?\s*$", re.IGNORECASE)


def parse_size_gb(text: str | float | int) -> float:
    """``20G`` / ``20GB`` / ``20480M`` / ``20`` → gigabytes (binary)."""
    if isinstance(text, (int, float)):
        return float(text)
    match = _SIZE.match(text)
    if not match:
        raise InvalidRequestError(f"cannot parse size {text!r} (examples: 20G, 18.5GB, 16384M)")
    value, unit = float(match.group(1)), match.group(2).lower()
    scale = {"": 1.0, "k": 1 / 1024**2, "m": 1 / 1024, "g": 1.0, "t": 1024.0}[unit]
    return value * scale


def compute_capability(index: int = 0) -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", f"-i={index}"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return float(proc.stdout.strip().splitlines()[0])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def driver_info() -> dict[str, str]:
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5,
                             check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    info = {"driver_version": proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""}
    m = re.search(r"CUDA Version:\s*([\d.]+)", out)
    if m:
        info["driver_cuda_version"] = m.group(1)
    return info


@dataclass(frozen=True)
class DevicePlan:
    device: str  # cpu | cuda:N | mps
    precision: str  # bf16 | fp16 | fp32
    vram_budget_gb: float | None
    gpu: GpuInfo | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "precision": self.precision,
            "vram_budget_gb": None if self.vram_budget_gb is None
            else round(self.vram_budget_gb, 2),
            "gpu": self.gpu.as_dict() if self.gpu else None,
            "reason": self.reason,
        }


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def vram_budget_gb(cfg: MusicConfig, gpu: GpuInfo, override_gb: float | None = None) -> float:
    total = gpu.memory_total_mb / 1024
    limit = override_gb if override_gb is not None else float(cfg["hardware"]["vram_limit_gb"])
    if limit and limit > 0:
        return min(limit, total)
    return max(0.0, total - float(cfg["hardware"]["vram_headroom_gb"]))


def plan_device(cfg: MusicConfig, spec: ModelSpec, *, device: str | None = None,
                precision: str | None = None, vram_limit_gb: float | None = None,
                gpus: list[GpuInfo] | None = None) -> DevicePlan:
    requested = (device or cfg["hardware"]["device"]).lower()
    gpus = query_gpus() if gpus is None else gpus
    gpu: GpuInfo | None = None
    if requested == "auto":
        if gpus:
            gpu = gpus[0]
            dev, why = "cuda:0", f"NVIDIA GPU detected ({gpu.name})"
        elif _is_apple_silicon():
            dev, why = "mps", "Apple Silicon"
        else:
            dev, why = "cpu", "no supported GPU detected"
    elif requested.startswith("cuda"):
        idx = int(requested.split(":", 1)[1]) if ":" in requested else 0
        gpu = next((g for g in gpus if g.index == idx), None)
        if gpu is None and gpus:
            raise ResourceLimitError(f"GPU {idx} not found (have {len(gpus)})")
        dev, why = f"cuda:{idx}", "requested"
    else:
        dev, why = requested, "requested"

    if dev == "cpu" and not spec.cpu_capable and not spec.diagnostic:
        raise ResourceLimitError(
            f"{spec.id} needs a GPU (upstream does not support CPU inference)",
            hint="pick a CPU-capable model (e.g. musicgen-small, stable-audio-3-small-music)",
        )

    budget = None
    if gpu is not None:
        budget = vram_budget_gb(cfg, gpu, vram_limit_gb)
        if spec.min_vram_gb > budget and not cfg["hardware"]["cpu_offload"]:
            raise ResourceLimitError(
                f"{spec.id} needs ~{spec.min_vram_gb:g} GB VRAM; budget is {budget:.1f} GB "
                f"on {gpu.name}",
                hint="raise hardware.vram_limit_gb / --vram-limit, enable cpu_offload, "
                "or choose a smaller model",
            )
        free = gpu.memory_free_mb / 1024
        if spec.min_vram_gb > free + 0.25:
            raise ResourceLimitError(
                f"only {free:.1f} GB VRAM is free on {gpu.name} but {spec.id} needs "
                f"~{spec.min_vram_gb:g} GB (other processes are using the GPU)",
                hint="close other GPU applications or `prompt2music models unload`",
            )

    prec = (precision or cfg["hardware"]["precision"]).lower()
    if prec == "auto":
        if dev.startswith("cuda"):
            cap = compute_capability(gpu.index if gpu else 0)
            prec = "bf16" if cap is None or cap >= 8.0 else "fp16"
        elif dev == "mps":
            prec = "fp16"
        else:
            prec = "fp32"
    return DevicePlan(dev, prec, budget, gpu, why)


class GpuLock:
    """Exclusive advisory lock so two processes never load models onto one GPU."""

    def __init__(self, name: str = "gpu") -> None:
        self.path = paths.ensure(paths.data_dir()) / f"{name}.lock"
        self._fh: Any = None

    def holder(self) -> str | None:
        try:
            return self.path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def acquire(self, timeout: float = 0.0, poll: float = 0.5) -> bool:
        try:
            import fcntl
        except ImportError:  # Windows: best effort, single-user desktop assumption
            return True
        fh = open(self.path, "a+", encoding="utf-8")  # noqa: SIM115 - held while locked
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    fh.close()
                    return False
                time.sleep(poll)
        fh.seek(0)
        fh.truncate()
        fh.write(f"pid={os.getpid()}\n")
        fh.flush()
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            import fcntl

            self._fh.seek(0)
            self._fh.truncate()
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None

    @property
    def held(self) -> bool:
        return self._fh is not None
