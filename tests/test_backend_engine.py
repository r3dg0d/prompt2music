"""Backend contract, GPU planning, worker protocol and engine lifecycle."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from prompt2music._shared.gpu import GpuInfo
from prompt2music.backends.audio_io import output_name, read_wav, write_wav
from prompt2music.backends.base import (
    Capabilities,
    GenerationRequest,
    Task,
    check_request_against,
)
from prompt2music.engine import Engine
from prompt2music.errors import (
    CancelledError,
    GenerationError,
    InvalidRequestError,
    ResourceLimitError,
    UnsupportedTaskError,
)
from prompt2music.gpu import GpuLock, parse_size_gb, plan_device
from prompt2music.registry import get_model


# -- request validation ---------------------------------------------------------
@pytest.mark.parametrize("kwargs,msg", [
    ({"duration": 0}, "duration"),
    ({"duration": float("nan")}, "duration"),
    ({"bpm": 5}, "bpm"),
    ({"batch_size": 0}, "batch_size"),
    ({"task": Task.CONTINUE}, "input audio"),
    ({"task": Task.SONG}, "lyrics"),
    ({"task": Task.INPAINT, "input_audio": "a.wav", "region_start": 5, "region_end": 2},
     "region"),
    ({"strength": 1.5}, "strength"),
])
def test_request_validation(kwargs, msg):
    with pytest.raises(InvalidRequestError, match=msg):
        GenerationRequest(prompt="x", model="m", **kwargs)


def test_request_roundtrip_and_negative_seed():
    req = GenerationRequest(prompt="x", model="m", seed=-1, input_audio="a.wav",
                            task="continue")
    assert req.seed is None and isinstance(req.input_audio, Path)
    again = GenerationRequest.from_dict(req.to_dict())
    assert again.task is Task.CONTINUE and again.input_audio == Path("a.wav")


def test_capability_checks_refuse_unsupported():
    mg = get_model("musicgen-small")
    with pytest.raises(UnsupportedTaskError):
        check_request_against(mg.id, mg.capabilities,
                              GenerationRequest(prompt="x", model="m", task=Task.INPAINT,
                                                input_audio="a", region_start=0, region_end=1))
    with pytest.raises(UnsupportedTaskError, match="vocals"):
        check_request_against(mg.id, mg.capabilities,
                              GenerationRequest(prompt="x", model="m", instrumental=False))
    with pytest.raises(InvalidRequestError, match="at most"):
        check_request_against(mg.id, mg.capabilities,
                              GenerationRequest(prompt="x", model="m", duration=999))
    sa = get_model("stable-audio-3-small-music")
    with pytest.raises(UnsupportedTaskError, match="lyrics"):
        check_request_against(sa.id, sa.capabilities,
                              GenerationRequest(prompt="x", model="m", lyrics="la"))


def test_capabilities_roundtrip():
    caps = get_model("ace-step-1.5").capabilities
    assert Capabilities.from_dict(caps.as_dict()) == caps


def test_wav_io_roundtrip(tmp_path):
    np = pytest.importorskip("numpy")
    data = (np.sin(np.linspace(0, 100, 4410)) * 0.5).astype("float32")[None].repeat(2, 0)
    ch, frames = write_wav(tmp_path / "a.wav", data, 44100)
    back, rate = read_wav(tmp_path / "a.wav")
    assert (ch, frames, rate) == (2, 4410, 44100)
    assert np.max(np.abs(back - data)) < 1e-3
    write_wav(tmp_path / "b.wav", [[0.0, 0.5, -0.5]], 8000)  # stdlib path
    assert read_wav(tmp_path / "b.wav")[0].shape == (1, 3)


def test_output_names():
    assert output_name(Path("/o/song.wav"), 0, 1) == Path("/o/song.wav")
    assert output_name(Path("/o/song.wav"), 1, 3) == Path("/o/song-2.wav")


# -- GPU planning ------------------------------------------------------------------
def gpu(total_gb=24, free_gb=23, name="NVIDIA GeForce RTX 4090"):
    return GpuInfo(0, name, int(total_gb * 1024), int((total_gb - free_gb) * 1024),
                   int(free_gb * 1024), 0, 40)


def test_parse_size():
    assert parse_size_gb("20G") == 20
    assert parse_size_gb("20GB") == 20
    assert parse_size_gb("20480M") == 20
    assert parse_size_gb(18.5) == 18.5
    with pytest.raises(InvalidRequestError):
        parse_size_gb("lots")


def test_plan_device_budget_and_headroom(cfg, monkeypatch):
    monkeypatch.setattr("prompt2music.gpu.compute_capability", lambda _i=0: 8.9)
    plan = plan_device(cfg, get_model("ace-step-1.5"), gpus=[gpu()])
    assert plan.device == "cuda:0" and plan.precision == "bf16"
    assert plan.vram_budget_gb == pytest.approx(22.0)  # 24 GB minus 2 GB headroom
    limited = plan_device(cfg, get_model("ace-step-1.5"), gpus=[gpu()], vram_limit_gb=20)
    assert limited.vram_budget_gb == 20
    with pytest.raises(ResourceLimitError, match="budget"):
        plan_device(cfg, get_model("yue2-3b"), gpus=[gpu()], vram_limit_gb=20)
    with pytest.raises(ResourceLimitError, match="free"):
        plan_device(cfg, get_model("ace-step-1.5-xl-turbo"), gpus=[gpu(free_gb=8)])


def test_plan_device_cpu(cfg):
    plan = plan_device(cfg, get_model("musicgen-small"), gpus=[])
    assert (plan.device, plan.precision) == ("cpu", "fp32")
    with pytest.raises(ResourceLimitError, match="needs a GPU"):
        plan_device(cfg, get_model("stable-audio-3-medium"), gpus=[])


def test_gpu_lock_is_exclusive(music_env):
    a, b = GpuLock(), GpuLock()
    assert a.acquire()
    assert not b.acquire(timeout=0)
    assert "pid=" in (b.holder() or "")
    a.release()
    assert b.acquire()
    b.release()


# -- engine + worker subprocess ------------------------------------------------------
@pytest.fixture()
def engine(cfg):
    eng = Engine(cfg)
    yield eng
    eng.unload()


def tone_request(**kw):
    base = {"prompt": "test", "model": "test-tone", "duration": 0.5}
    base.update(kw)
    return GenerationRequest(**base)


def test_engine_loads_reuses_and_generates(engine, tmp_path):
    spec = get_model("test-tone")
    info = engine.load(spec)
    assert info["reused"] is False and info["device"] == "cpu"
    pid = info["worker_pid"]
    assert engine.load(spec)["reused"] is True
    data = engine.generate(spec, tone_request(batch_size=2, seed=3), tmp_path)
    assert [o["seed"] for o in data["outputs"]] == [3, 4]
    assert all(Path(o["path"]).is_file() for o in data["outputs"])
    assert data["peak_rss_mb"] > 0
    assert engine.status()["worker_pid"] == pid


def test_cooperative_cancel_keeps_model_warm(engine, tmp_path):
    spec = get_model("test-tone")
    engine.load(spec)
    threading.Timer(0.4, engine.cancel).start()
    started = time.monotonic()
    with pytest.raises(CancelledError):
        engine.generate(spec, tone_request(duration=1, extra={"seconds_per_step": 0.5}),
                        tmp_path)
    assert time.monotonic() - started < 3
    assert engine.loaded is not None and engine.loaded.client.alive
    engine.generate(spec, tone_request(), tmp_path)  # still usable


def test_forced_stop_when_backend_ignores_cancel(engine, tmp_path):
    spec = get_model("test-tone")
    engine.load(spec)
    # A delay longer than the grace period simulates native code that never yields.
    threading.Timer(0.2, lambda: engine.cancel(grace=0.3)).start()
    with pytest.raises(CancelledError):
        engine.generate(spec, tone_request(duration=1, extra={"seconds_per_step": 5}),
                        tmp_path)
    assert engine.loaded is None  # the worker was terminated and the model unloaded


def test_unsupported_request_rejected_before_worker(engine, tmp_path):
    spec = get_model("test-tone")
    engine.load(spec)
    with pytest.raises(UnsupportedTaskError):
        engine.generate(spec, tone_request(instrumental=False), tmp_path)


def test_worker_crash_is_reported(engine, tmp_path):
    spec = get_model("test-tone")
    engine.load(spec)
    engine.loaded.client.kill()  # died while idle
    with pytest.raises(GenerationError, match="exited"):
        engine.generate(spec, tone_request(), tmp_path)
    assert engine.loaded is None
    engine.load(spec)  # recovers with a fresh worker
    client = engine.loaded.client
    threading.Timer(0.3, client.kill).start()  # dies mid-generation
    with pytest.raises(GenerationError):
        engine.generate(spec, tone_request(duration=1, extra={"seconds_per_step": 1}),
                        tmp_path)
    assert engine.loaded is None


def test_missing_backend_env_is_explicit(cfg, monkeypatch):
    from prompt2music import envs
    from prompt2music.errors import BackendNotReadyError

    monkeypatch.setenv("PROMPT2MUSIC_PYTHON_MUSICGEN", "/nonexistent/python")
    envs._probe_cache.clear()
    with pytest.raises(BackendNotReadyError, match="interpreter not found") as ei:
        envs.require_backend(cfg, "musicgen")
    assert "prompt2music backends install musicgen" in (ei.value.hint or "")


def test_install_commands_are_argv_lists(cfg, monkeypatch):
    from prompt2music import envs

    monkeypatch.setattr(envs.shutil, "which", lambda name: f"/usr/bin/{name}")
    cmds = envs.install_commands(cfg, "ace-step", ref="v1.5.0")
    assert cmds[0][:2] == ["/usr/bin/git", "clone"]
    assert ["/usr/bin/git", "-C", str(envs.src_dir(cfg, "ace-step")), "checkout",
            "v1.5.0"] in cmds
    assert any(c[1] == "sync" for c in cmds)
    pip = envs.install_commands(cfg, "musicgen", torch_index="https://example/cu128")
    assert any("--index-url" in c for c in pip)


def test_worker_env_is_isolated(cfg):
    from prompt2music import envs

    env = envs.worker_env(cfg)
    assert env["HF_HUB_OFFLINE"] == "1"
    shim = Path(env["PYTHONPATH"])
    assert (shim / "prompt2music").exists() and len(list(shim.iterdir())) == 1


def test_spec_options_reach_backend(engine, tmp_path):
    spec = replace(get_model("test-tone"), options={"seconds_per_step": 0.0})
    engine.load(spec)
    assert engine.generate(spec, tone_request(), tmp_path)["outputs"]
