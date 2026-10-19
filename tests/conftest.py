"""Fixtures for prompt2music tests: isolated XDG dirs, config, cleanup."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from prompt2music import envs


@pytest.fixture()
def xdg_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate XDG dirs and HOME."""
    for name in ("config", "cache", "data", "home"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")


@pytest.fixture()
def music_env(xdg_tmp: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolated prompt2music state; stops any daemon a test started."""
    for var in ("PROMPT2MUSIC_CONFIG", "PROMPT2MUSIC_DATA_DIR", "PROMPT2MUSIC_CACHE_DIR",
                "PROMPT2MUSIC_MODELS_DIR", "PROMPT2MUSIC_ENVS_DIR", "PROMPT2MUSIC_OFFLINE",
                "HF_HUB_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    envs._probe_cache.clear()
    from prompt2music.config import load_music_config

    cfg = load_music_config(create=True)
    cfg.set("general.output_directory", str(xdg_tmp / "home" / "Music"))
    cfg.save()
    yield load_music_config()
    from prompt2music import daemon

    if daemon.status()["running"]:
        daemon.stop(timeout=20)
    envs._probe_cache.clear()


@pytest.fixture()
def cfg(music_env):
    return music_env
