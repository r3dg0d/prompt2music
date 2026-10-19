"""Model registry and configuration."""

from __future__ import annotations

import pytest

from prompt2music import envs, registry
from prompt2music.backends import BACKEND_CLASSES
from prompt2music.backends.base import Task
from prompt2music.config import (
    SCHEMA,
    load_music_config,
    parse_cli_value,
    reset_config,
    validate_value,
)
from prompt2music.errors import ConfigError, UnknownModelError


def test_ids_and_aliases_unique_and_resolvable():
    seen = set()
    for spec in registry.all_models():
        for key in (spec.id, *spec.aliases):
            assert key not in seen
            seen.add(key)
            assert registry.get_model(key) is spec
    assert registry.get_model("ACE-STEP").id == "ace-step-1.5"


def test_unknown_model_has_hint():
    with pytest.raises(UnknownModelError) as ei:
        registry.get_model("suno-v9")
    assert "known models" in (ei.value.hint or "")


@pytest.mark.parametrize("spec", registry.all_models(), ids=lambda s: s.id)
def test_every_spec_is_complete(spec):
    assert spec.backend in BACKEND_CLASSES
    assert spec.backend in envs.BACKEND_ENVS
    assert spec.license and spec.license_url.startswith("http") or spec.diagnostic
    assert spec.commercial_use in ("yes", "no", "conditional")
    caps = spec.capabilities
    if caps.lyrics:
        assert caps.vocals
    if Task.SONG in caps.tasks:
        assert caps.lyrics
    assert caps.max_duration >= caps.min_duration > 0
    if spec.kind == "music" and not spec.diagnostic:
        assert spec.sources and spec.tasks
    if spec.gated:
        assert spec.requires_license_acceptance


def test_api_only_models_not_offered():
    ids = {s.id for s in registry.all_models()}
    assert not any("large" in i and "stable-audio" in i for i in ids)
    assert not any("7b" in i for i in ids)  # HeartMuLa-7B is unreleased


def test_noncommercial_models_flagged():
    for model_id in ("musicgen-small", "yue2-3b"):
        spec = registry.get_model(model_id)
        assert spec.commercial_use == "no"
        assert spec.requires_license_acceptance


def test_continue_only_where_upstream_supports_it():
    assert Task.CONTINUE not in registry.get_model("ace-step-1.5").tasks
    assert Task.CONTINUE in registry.get_model("stable-audio-3-medium").tasks
    assert Task.CONTINUE in registry.get_model("musicgen-small").tasks


def test_search_and_kinds():
    assert {s.id for s in registry.search_models("vocals")} >= {"ace-step-1.5", "yue2-3b"}
    assert all(s.kind == "stems" for s in registry.all_models(kind="stems"))
    assert "test-tone" not in {s.id for s in registry.models_for_task(Task.TEXT_TO_MUSIC)}


def test_config_defaults_written_with_comments(music_env):
    text = music_env.path.read_text()
    assert "[mcp]" in text and "allow_model_downloads = false" in text
    assert "# Model id" in text


def test_config_set_preserves_user_comments(music_env):
    path = music_env.path
    path.write_text(path.read_text() + "\n# my own note\n")
    cfg = load_music_config(path)
    cfg.set("general.default_model", "musicgen-small")
    cfg.save()
    text = path.read_text()
    assert "# my own note" in text
    assert load_music_config(path).get("general.default_model") == "musicgen-small"


@pytest.mark.parametrize("key,value", [
    ("hardware.precision", "fp8"),
    ("mcp.port", 70000),
    ("generation.batch_size", 0),
    ("hardware.device", "gpu"),
    ("mcp.max_queued_jobs", True),  # bool must not pass as int
])
def test_config_rejects_invalid(key, value):
    section, _, name = key.partition(".")
    with pytest.raises(ConfigError):
        validate_value(section, name, value)


def test_config_rejects_unknown_keys_in_file(music_env):
    music_env.path.write_text("[general]\ndefualt_model = 'x'\n")
    with pytest.raises(ConfigError, match="unknown config key"):
        load_music_config(music_env.path)


def test_parse_cli_values():
    assert parse_cli_value("hardware", "vram_limit_gb", "20") == 20
    assert parse_cli_value("hardware", "vram_limit_gb", "18.5") == 18.5
    assert parse_cli_value("general", "offline", "yes") is True
    assert parse_cli_value("mcp", "allowed_models", '["musicgen-small"]') == ["musicgen-small"]
    with pytest.raises(ConfigError):
        parse_cli_value("mcp", "port", "80.5")


def test_reset_keeps_backup(music_env):
    music_env.set("general.quality", "best")
    music_env.save()
    reset_config(music_env.path)
    assert music_env.path.with_suffix(".toml.bak").exists()
    assert load_music_config(music_env.path).get("general.quality") == "balanced"


def test_schema_has_spec_sections():
    assert {"general", "hardware", "models", "generation", "mcp", "logging"} <= set(SCHEMA)


def test_managed_config_is_read_only(music_env, monkeypatch, tmp_path):
    """A Home Manager store symlink (or read-only file) is never rewritten."""
    import os

    monkeypatch.setattr(os, "access", lambda *_a, **_k: False)
    cfg = load_music_config(music_env.path)
    assert cfg.managed
    cfg.set("general.quality", "best")  # in-memory change is fine
    with pytest.raises(ConfigError, match="managed declaratively") as ei:
        cfg.save()
    assert "programs.prompt2music.settings" in (ei.value.hint or "")


def test_home_manager_style_toml_loads(tmp_path):
    path = tmp_path / "prompt2music.toml"
    path.write_text('[general]\ndefault_model = "ace-step-1.5"\n\n[hardware]\nvram_limit_gb = 20\n')
    cfg = load_music_config(path)
    assert cfg.get("hardware.vram_limit_gb") == 20
    assert cfg.get("mcp.allow_model_downloads") is False  # defaults fill the rest
