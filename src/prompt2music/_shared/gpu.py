"""nvidia-smi parsing and GPU summaries (works without GPU — empty list)."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_pct: int | None
    temperature_c: int | None

    @property
    def memory_total_gb(self) -> float:
        return self.memory_total_mb / 1024.0

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "memory_total_mb": self.memory_total_mb,
            "memory_used_mb": self.memory_used_mb,
            "memory_free_mb": self.memory_free_mb,
            "utilization_pct": self.utilization_pct,
            "temperature_c": self.temperature_c,
        }


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def query_gpus() -> list[GpuInfo]:
    """Parse nvidia-smi CSV. Returns [] if missing or error."""
    if not nvidia_smi_available():
        return []
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0 or not result.stdout.strip():
        return []
    gpus: list[GpuInfo] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            index = int(parts[0])
            name = parts[1]
            total = int(float(parts[2]))
            used = int(float(parts[3]))
            free = int(float(parts[4]))
            util = (
                int(float(parts[5]))
                if len(parts) > 5 and parts[5] not in {"[N/A]", ""}
                else None
            )
            temp = (
                int(float(parts[6]))
                if len(parts) > 6 and parts[6] not in {"[N/A]", ""}
                else None
            )
        except ValueError:
            continue
        gpus.append(
            GpuInfo(
                index=index,
                name=name,
                memory_total_mb=total,
                memory_used_mb=used,
                memory_free_mb=free,
                utilization_pct=util,
                temperature_c=temp,
            )
        )
    return gpus


def primary_gpu() -> GpuInfo | None:
    gpus = query_gpus()
    return gpus[0] if gpus else None
