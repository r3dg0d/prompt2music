"""Host hardware summary (CPU/RAM/GPU) for doctor and monitoring."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

from prompt2music._shared.gpu import GpuInfo, query_gpus


@dataclass(frozen=True)
class HardwareInfo:
    system: str
    machine: str
    processor: str
    python: str
    cpu_count: int | None
    ram_total_mb: int | None
    gpus: tuple[GpuInfo, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "system": self.system,
            "machine": self.machine,
            "processor": self.processor,
            "python": self.python,
            "cpu_count": self.cpu_count,
            "ram_total_mb": self.ram_total_mb,
            "gpus": [g.as_dict() for g in self.gpus],
        }


def _ram_total_mb() -> int | None:
    # Linux: /proc/meminfo
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb // 1024
    except OSError:
        pass
    return None


def probe_hardware() -> HardwareInfo:
    return HardwareInfo(
        system=platform.system(),
        machine=platform.machine(),
        processor=platform.processor() or platform.machine(),
        python=platform.python_version(),
        cpu_count=os.cpu_count(),
        ram_total_mb=_ram_total_mb(),
        gpus=tuple(query_gpus()),
    )
