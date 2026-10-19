"""MCP server: security policy, schemas, and a real stdio client session."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from prompt2music.errors import InvalidRequestError, SecurityError
from prompt2music.mcp.security import Policy
from prompt2music.mcp.tools import TOOL_SPECS
from tests.conftest import needs_ffmpeg

mcp = pytest.importorskip("mcp")

REQUIRED_TOOLS = {
    "music_generate", "music_generate_instrumental", "music_generate_song", "music_continue",
    "music_edit", "music_inpaint", "music_enhance_prompt", "music_models_list",
    "music_models_search", "music_model_info", "music_model_download", "music_model_load",
    "music_model_unload", "music_model_remove", "music_audio_info", "music_audio_convert",
    "music_audio_normalize", "music_audio_trim", "music_audio_mix", "music_audio_loop",
    "music_audio_waveform", "music_audio_spectrogram", "music_separate_stems",
    "music_system_info", "music_gpu_status", "music_benchmark", "music_job_status",
    "music_job_cancel", "music_jobs_list", "music_job_result",
}


def test_all_specified_tools_exist():
    assert REQUIRED_TOOLS <= set(TOOL_SPECS)
    assert TOOL_SPECS["music_model_remove"]["destructive"]
    assert TOOL_SPECS["music_job_status"]["read_only"]


# -- policy ---------------------------------------------------------------------------
@pytest.fixture()
def policy(cfg, tmp_path):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return Policy(cfg)


def test_output_paths_confined(policy, cfg, tmp_path):
    assert policy.resolve_output("song.wav") == cfg.output_dir.resolve() / "song.wav"
    assert policy.resolve_output(str(cfg.output_dir / "a" / "b.wav")).name == "b.wav"
    for evil in ("../../etc/cron.d/x.wav", "/etc/passwd", "~/.bashrc",
                 "file:///etc/passwd", "a\x00b.wav"):
        with pytest.raises(SecurityError):
            policy.resolve_output(evil)


def test_symlink_escape_blocked(policy, cfg, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.wav").write_bytes(b"RIFF")
    (cfg.output_dir / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SecurityError):
        policy.resolve_output("link/new.wav")
    with pytest.raises(SecurityError):
        policy.resolve_input("link/secret.wav")
    (cfg.output_dir / "file-link.wav").symlink_to(cfg.output_dir / "x.wav")
    with pytest.raises(SecurityError, match="symlink"):
        policy.resolve_output("file-link.wav")


def test_inputs_size_limited_and_allowlisted_dirs(cfg, tmp_path):
    extra = tmp_path / "samples"
    extra.mkdir()
    (extra / "big.wav").write_bytes(b"\0" * (2 * 1024**2))
    cfg.set("mcp.allowed_input_dirs", [str(extra)])
    cfg.set("mcp.max_input_file_mb", 1)
    pol = Policy(cfg)
    with pytest.raises(SecurityError, match="exceeds"):
        pol.resolve_input(str(extra / "big.wav"))
    (extra / "ok.wav").write_bytes(b"\0" * 10)
    assert pol.resolve_input(str(extra / "ok.wav")).name == "ok.wav"
    with pytest.raises(SecurityError):
        pol.resolve_output(str(extra / "out.wav"))  # input-only directory
    with pytest.raises(InvalidRequestError):
        pol.resolve_input(str(extra / "missing.wav"))


def test_model_allowlist_and_duration_cap(cfg):
    cfg.set("mcp.allowed_models", ["musicgen-small"])
    cfg.set("mcp.max_duration_seconds", 60)
    pol = Policy(cfg)
    assert pol.check_model("musicgen-small").id == "musicgen-small"
    assert pol.check_model("auto") is None
    with pytest.raises(SecurityError):
        pol.check_model("ace-step-1.5")
    with pytest.raises(InvalidRequestError):
        pol.check_model("htdemucs", kind="music")
    with pytest.raises(SecurityError):
        pol.check_duration(61)


# -- real client over stdio -----------------------------------------------------------
def _attr(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(names)


def _payload(res: Any) -> dict[str, Any]:
    structured = _attr(res, "structured_content", "structuredContent")
    if structured is not None:
        return structured
    return json.loads("".join(getattr(c, "text", "") for c in res.content))


def _server_command() -> list[str]:
    exe = shutil.which("prompt2music-mcp", path=str(Path(sys.executable).parent))
    return [exe] if exe else [sys.executable, "-m", "prompt2music.mcp.server"]


async def _session(fn):
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    cmd = _server_command()
    params = StdioServerParameters(command=cmd[0], args=cmd[1:], env=dict(os.environ))
    with open(os.devnull, "w") as devnull:
        async with stdio_client(params, errlog=devnull) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await fn(session)


@needs_ffmpeg
def test_agent_workflow_over_stdio(cfg):
    async def flow(s):
        tools = await s.list_tools()
        names = {t.name for t in tools.tools}
        assert REQUIRED_TOOLS <= names
        schema = _attr(next(t for t in tools.tools if t.name == "music_generate"),
                       "input_schema", "inputSchema")
        assert schema["required"] == ["prompt"]
        assert set(schema["properties"]["format"]["anyOf"][0]["enum"]) >= {"wav", "flac"}

        models = _payload(await s.call_tool("music_models_list", {"installed_only": True}))
        assert any(m["id"] == "test-tone" for m in models["models"])
        job = _payload(await s.call_tool("music_generate", {
            "prompt": "agent test", "model": "test-tone", "duration": 1, "seed": 4,
            "output_path": "agent/test.wav"}))
        assert job["status"] == "queued" and job["job_id"]
        for _ in range(200):
            status = _payload(await s.call_tool("music_job_status", {"job_id": job["job_id"]}))
            if status["status"] in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(0.2)
        assert status["status"] == "completed", status
        result = _payload(await s.call_tool("music_job_result", {"job_id": job["job_id"]}))
        out = result["result"]["outputs"][0]
        assert out["exists"] and out["path"].endswith("agent/test.wav")
        assert out["uri"].startswith("file://")
        info = _payload(await s.call_tool("music_audio_info", {"path": out["path"]}))
        assert info["duration"] == pytest.approx(1.0, abs=0.05)
        resource = await s.read_resource(out["resource"])
        assert json.loads(resource.contents[0].text)["model"] == "test-tone"

        denied = {}
        for name, args in [
            ("music_audio_info", {"path": "/etc/passwd"}),
            ("music_generate", {"prompt": "x", "model": "test-tone",
                                "output_path": "../../escape.wav"}),
            ("music_model_download", {"model": "musicgen-small"}),
            ("music_model_remove", {"model": "musicgen-small", "confirm": True}),
            ("music_generate", {"prompt": "x", "model": "test-tone", "duration": 5000}),
        ]:
            res = await s.call_tool(name, args)
            assert _attr(res, "is_error", "isError"), name
            text = "".join(getattr(c, "text", "") for c in res.content)
            denied[name] = json.loads(text[text.index("{"):])["error"]["code"]
        assert set(denied.values()) == {"forbidden"}

        results = await asyncio.gather(*[s.call_tool("music_jobs_list", {"limit": 2})
                                         for _ in range(10)])
        assert not any(_attr(r, "is_error", "isError") for r in results)
        return True

    assert asyncio.run(_session(flow))
    audit = Path(os.environ["XDG_CACHE_HOME"]) / "ai-media" / "music" / "logs"
    lines = (audit / "mcp-audit.jsonl").read_text().splitlines()
    assert any('"outcome": "error"' in line and "music_audio_info" in line for line in lines)


def test_mcp_quota(cfg):
    cfg.set("mcp.max_queued_jobs", 1)
    cfg.save()
    from prompt2music.jobs import JobStore

    JobStore().create("generate", {}, source="mcp")  # occupies the quota

    async def flow(s):
        res = await s.call_tool("music_generate", {"prompt": "x", "model": "test-tone",
                                                   "duration": 1})
        text = "".join(getattr(c, "text", "") for c in res.content)
        return _attr(res, "is_error", "isError"), text

    is_error, text = asyncio.run(_session(flow))
    assert is_error and "quota" in text


def test_http_refuses_public_bind_without_token(cfg, monkeypatch):
    from prompt2music.errors import ConfigError
    from prompt2music.mcp.server import client_config, serve

    monkeypatch.delenv("PROMPT2MUSIC_MCP_TOKEN", raising=False)
    with pytest.raises(ConfigError, match="without authentication"):
        serve(cfg, transport="http", host="0.0.0.0", port=8799)
    assert "claude mcp add prompt2music -- prompt2music-mcp" in client_config("claude")
    assert json.loads(client_config("generic"))["mcpServers"]["prompt2music"]["command"] == \
        "prompt2music-mcp"


def test_bearer_middleware():
    from prompt2music.mcp.server import BearerAuth

    calls = []

    async def app(scope, receive, send):
        calls.append(scope["path"])

    sent = []

    async def send(msg):
        sent.append(msg)

    mw = BearerAuth(app, "s3cret")
    asyncio.run(mw({"type": "http", "path": "/mcp", "headers": []}, None, send))
    assert sent[0]["status"] == 401 and not calls
    asyncio.run(mw({"type": "http", "path": "/mcp",
                    "headers": [(b"authorization", b"Bearer s3cret")]}, None, send))
    assert calls == ["/mcp"]
