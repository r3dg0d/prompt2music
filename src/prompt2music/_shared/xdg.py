"""XDG base directories (same layout as ai-media-cli, so the two share state)."""

from __future__ import annotations

import os
from pathlib import Path


def xdg_config_home() -> Path:
    raw = os.environ.get("XDG_CONFIG_HOME")
    return Path(raw) if raw else Path.home() / ".config"


def xdg_cache_home() -> Path:
    raw = os.environ.get("XDG_CACHE_HOME")
    return Path(raw) if raw else Path.home() / ".cache"


def xdg_data_home() -> Path:
    raw = os.environ.get("XDG_DATA_HOME")
    return Path(raw) if raw else Path.home() / ".local" / "share"
