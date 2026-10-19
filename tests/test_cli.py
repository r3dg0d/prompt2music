"""prompt2music / aimusic / ai-media music command line."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from prompt2music.cli import build_parser, run
from prompt2music.mcp.server import main as mcp_main
from tests.conftest import needs_ffmpeg


def _subparsers(parser):
    for action in parser._actions:
        if action.__class__.__name__ == "_SubParsersAction":
            return action.choices
    return {}


def all_command_paths():
    parser = build_parser()
    paths = []
    for name, sub in _subparsers(parser).items():
        paths.append([name])
        for child in _subparsers(sub):
            paths.append([name, child])
    return paths


def test_spec_commands_exist():
    top = set(_subparsers(build_parser()))
    assert {"doctor", "setup", "generate", "song", "continue", "edit", "inpaint", "enhance",
            "models", "stems", "audio", "jobs", "benchmark", "config", "mcp",
            "library"} <= top
    models = set(_subparsers(_subparsers(build_parser())["models"]))
    assert {"list", "available", "search", "info", "download", "verify", "load", "unload",
            "remove", "update", "cache", "benchmark"} <= models


@pytest.mark.parametrize("path", all_command_paths(), ids=" ".join)
def test_every_command_has_help(path, capsys):
    with pytest.raises(SystemExit) as ei:
        run([*path, "--help"])
    assert ei.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("module,prog", [
    ("prompt2music.cli:main", "prompt2music"),
    ("prompt2music.cli:main_aimusic", "aimusic"),
])
def test_version(module, prog, capsys):
    mod, _, fn = module.partition(":")
    entry = getattr(__import__(mod, fromlist=[fn]), fn)
    with pytest.raises(SystemExit) as ei:
        entry(["--version"])
    assert ei.value.code == 0
    assert capsys.readouterr().out.startswith(prog)


def test_mcp_entry_points(capsys):
    with pytest.raises(SystemExit):
        mcp_main(["--version"])
    assert capsys.readouterr().out.startswith("prompt2music-mcp")
    with pytest.raises(SystemExit):
        mcp_main(["--help"], prog="aimusic-mcp")
    assert "aimusic-mcp" in capsys.readouterr().out


def test_console_scripts_declared():
    text = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    for name in ("prompt2music", "aimusic", "prompt2music-mcp", "aimusic-mcp"):
        assert f'\n{name} = "' in text


def test_config_commands(music_env, capsys):
    assert run(["config", "set", "hardware.vram_limit_gb", "20"]) == 0
    capsys.readouterr()
    assert run(["config", "get", "hardware.vram_limit_gb"]) == 0
    assert capsys.readouterr().out.strip() == "20"
    assert run(["config", "set", "hardware.precision", "fp8"]) == 2
    assert run(["--json", "config", "show"]) == 0
    assert json.loads(capsys.readouterr().out)["hardware"]["vram_limit_gb"] == 20


def test_json_flag_after_subcommand(music_env, capsys):
    assert run(["models", "available", "--json", "--kind", "stems"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {m["id"] for m in data} == {"htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"}


def test_errors_exit_nonzero_with_hint(music_env, capsys):
    assert run(["generate", "x", "--model", "ace-step-1.5", "--duration", "20"]) == 2
    err = capsys.readouterr().err
    assert "not installed" in err and "models download ace-step-1.5" in err
    assert run(["generate", "x", "--model", "nope"]) == 2
    assert run(["models", "remove", "musicgen-small"]) == 2  # not confirmed / not installed


def test_dry_run_shows_resolved_request(music_env, capsys):
    assert run(["--json", "generate", "techno", "--model", "test-tone", "--duration", "2",
                "--bpm", "145", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["model"] == "test-tone"
    assert plan["request"]["prompt"] == "techno, 145 BPM"


@needs_ffmpeg
def test_generate_and_library_json(music_env, capsys):
    out = music_env.output_dir / "cli.wav"
    assert run(["--json", "generate", "cli test", "--model", "test-tone", "--duration", "1",
                "--no-daemon", "-o", str(out)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["outputs"][0]["path"] == str(out) and out.is_file()
    assert run(["--json", "library", "list"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["path"] == str(out)
    assert run(["--json", "jobs", "list"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["status"] == "completed"


@needs_ffmpeg
def test_vocals_flag_is_honoured(music_env, capsys):
    assert run(["generate", "x", "--model", "test-tone", "--vocals", "--no-daemon"]) == 2
    assert "cannot generate vocals" in capsys.readouterr().err


def test_completion_scripts(capsys):
    expected = {"bash": "complete -F _prompt2music prompt2music aimusic",
                "zsh": "bashcompinit", "fish": "__fish_seen_subcommand_from models"}
    for shell, marker in expected.items():
        assert run(["completion", shell]) == 0
        script = capsys.readouterr().out
        assert marker in script and "generate" in script


def test_bash_completion_is_valid_bash(tmp_path, capsys):
    run(["completion", "bash"])
    path = tmp_path / "c.sh"
    path.write_text(capsys.readouterr().out)
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_enhance_and_lyrics_template(music_env, capsys):
    assert run(["--json", "enhance", "sad piano music", "--no-llm"]) == 0
    assert "melancholic" in json.loads(capsys.readouterr().out)["prompt"]
    assert run(["lyrics", "template", "--structure", "short"]) == 0
    assert "[Chorus]" in capsys.readouterr().out
    assert run(["lyrics", "generate", "--prompt", "x"]) == 2  # no LLM configured


def test_module_entry_runs(music_env):
    proc = subprocess.run([sys.executable, "-m", "prompt2music.cli", "--version"],
                          capture_output=True, text=True, check=True)
    assert proc.stdout.startswith("prompt2music")
