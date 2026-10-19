"""Filesystem locations for prompt2music (XDG, shared with the rest of ai-media).

Every location can be overridden by environment variable or configuration so
weights can live outside read-only roots such as the Nix store.
"""

from __future__ import annotations

import os
from pathlib import Path

from prompt2music._shared.xdg import xdg_cache_home, xdg_config_home, xdg_data_home


def expand(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()


def config_path() -> Path:
    override = os.environ.get("PROMPT2MUSIC_CONFIG")
    if override:
        return expand(override)
    return xdg_config_home() / "ai-media" / "prompt2music.toml"


def data_dir() -> Path:
    override = os.environ.get("PROMPT2MUSIC_DATA_DIR")
    if override:
        return expand(override)
    return xdg_data_home() / "ai-media" / "music"


def cache_dir() -> Path:
    override = os.environ.get("PROMPT2MUSIC_CACHE_DIR")
    if override:
        return expand(override)
    return xdg_cache_home() / "ai-media" / "music"


def default_models_dir() -> Path:
    override = os.environ.get("PROMPT2MUSIC_MODELS_DIR")
    if override:
        return expand(override)
    return xdg_data_home() / "ai-media" / "models" / "music"


def default_envs_dir() -> Path:
    """Backend virtualenvs live beside the existing qwen/scail/trellis envs."""
    override = os.environ.get("PROMPT2MUSIC_ENVS_DIR")
    if override:
        return expand(override)
    return xdg_data_home() / "ai-media" / "envs"


def default_output_dir() -> Path:
    return Path.home() / "Music" / "AI" / "prompt2music"


def db_path() -> Path:
    return data_dir() / "music.db"


def logs_dir() -> Path:
    return cache_dir() / "logs"


def daemon_dir() -> Path:
    return data_dir() / "daemon"


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
