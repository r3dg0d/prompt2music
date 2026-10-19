"""NixOS runtime helpers shared by backends that run in separate venvs."""

from __future__ import annotations

import os
from pathlib import Path

DRIVER_LIB_DIRS = ("/run/opengl-driver/lib", "/run/opengl-driver-32/lib")


def is_nixos() -> bool:
    return Path("/etc/NIXOS").exists() or Path("/run/current-system/nixos-version").exists()


def nixos_ld_library_path(current: str | None = None) -> str:
    """Prepend NixOS NVIDIA driver libs (when present) to LD_LIBRARY_PATH.

    ``AI_MEDIA_EXTRA_LD_PATH`` adds further directories, e.g. gcc-lib for
    libstdc++, matching the install_*_wrapper scripts.
    """
    current = os.environ.get("LD_LIBRARY_PATH", "") if current is None else current
    parts = [p for p in DRIVER_LIB_DIRS if Path(p).exists()]
    extra = os.environ.get("AI_MEDIA_EXTRA_LD_PATH", "")
    parts.extend(p for p in extra.split(":") if p)
    if current:
        parts.append(current)
    return ":".join(parts)
