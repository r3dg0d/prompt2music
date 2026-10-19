"""Job queue, daemon, generation pipeline, model selection and library."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from prompt2music import daemon, envs, service
from prompt2music.backends.base import GenerationRequest
from prompt2music.daemon import execute_job
from prompt2music.engine import Engine
from prompt2music.errors import (
    InvalidRequestError,
    JobNotFoundError,
    ModelNotInstalledError,
    ResourceLimitError,
    UnsupportedTaskError,
)
from prompt2music.jobs import JobStore
from prompt2music.library import Library
from prompt2music.manifest import FileEntry, Manifest
from prompt2music.selection import select_model
from tests.conftest import needs_ffmpeg


# -- job store ------------------------------------------------------------------------
def test_claim_is_atomic_across_connections(music_env):
    a, b = JobStore(), JobStore()
    ids = [a.create("generate", {"n": i}) for i in range(5)]
    claimed: list[str] = []
    lock = threading.Lock()

    def worker(store):
        while (job := store.claim_next(os.getpid())) is not None:
            with lock:
                claimed.append(job.id)

    threads = [threading.Thread(target=worker, args=(JobStore(),)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(claimed) == sorted(ids)  # every job exactly once
    assert {b.get(i).status for i in ids} == {"loading"}


def test_cancel_queued_vs_running(music_env):
    store = JobStore()
    queued = store.create("generate", {})
    assert store.request_cancel(queued).status == "cancelled"
    running = store.create("generate", {})
    store.update(running, status="running")
    job = store.request_cancel(running)
    assert job.status == "running" and job.cancel_requested


def test_orphaned_jobs_recovered(music_env):
    store = JobStore()
    job_id = store.create("generate", {})
    store.update(job_id, status="running", daemon_pid=2**22 + 12345)  # no such pid
    assert store.recover_orphans() == [job_id]
    assert store.get(job_id).status == "failed"


def test_queue_quota_and_prefix_lookup(music_env):
    store = JobStore()
    first = store.create("generate", {}, max_queued=2)
    store.create("generate", {}, max_queued=2)
    with pytest.raises(ResourceLimitError):
        store.create("generate", {}, max_queued=2)
    assert store.get(first[:6]).id == first
    with pytest.raises(JobNotFoundError):
        store.get("zzzz")


# -- service pipeline -------------------------------------------------------------------
def tone_params(cfg, **kw):
    raw = {"prompt": "pipeline test", "model": "test-tone", "duration": 0.5, "seed": 1}
    raw.update(kw)
    return service.normalise_params(cfg, raw)


def test_normalise_params_defaults_and_unknown(cfg):
    params = service.normalise_params(cfg, {"prompt": "x"})
    assert params["duration"] == 30 and params["model"] == "auto"
    assert params["instrumental"] is True and params["seed"] is None
    assert service.normalise_params(cfg, {"prompt": "x", "lyrics": "[Verse]\na"})[
        "instrumental"] is False
    with pytest.raises(InvalidRequestError, match="unknown parameters"):
        service.normalise_params(cfg, {"prompt": "x", "rm -rf": 1})


def install_fake(cfg, model_id: str) -> None:
    path = cfg.models_dir / model_id
    path.mkdir(parents=True)
    (path / "w.bin").write_bytes(b"x")
    Manifest(model_id=model_id, sources=[], files={"w.bin": FileEntry(1)},
             license="test").write(path)


def test_prepare_folds_bpm_key_for_text_only_models(cfg, monkeypatch):
    install_fake(cfg, "musicgen-small")
    monkeypatch.setattr(envs, "probe", lambda *_a, **_k: envs.ProbeResult("x", "py", "t", True))
    params = service.normalise_params(cfg, {"prompt": "techno", "model": "musicgen-small",
                                            "bpm": 145, "key": "Fm", "duration": 10})
    prepared = service.prepare(cfg, params)
    assert prepared.prompt_used == "techno, 145 BPM, in Fm"
    assert prepared.request.bpm is None and prepared.request.key is None
    raw = service.prepare(cfg, {**params, "raw_prompt": True})
    assert raw.prompt_used == "techno"


def test_prepare_keeps_native_bpm_for_ace_step(cfg):
    params = service.normalise_params(cfg, {"prompt": "techno", "model": "ace-step-1.5",
                                            "bpm": 145, "key": "F minor", "duration": 30})
    prepared = service.prepare(cfg, params)
    assert prepared.request.bpm == 145 and prepared.prompt_used == "techno"


def test_enhancement_is_opt_in(cfg):
    plain = service.prepare(cfg, tone_params(cfg, prompt="sad piano"))
    assert plain.prompt_used == "sad piano" and plain.enhancement is None
    enhanced = service.prepare(cfg, tone_params(cfg, prompt="sad piano", enhance=True))
    assert enhanced.prompt_used.startswith("sad piano,") and enhanced.enhancement


def test_explicit_model_never_replaced(cfg):
    sel = select_model(cfg, GenerationRequest(prompt="x", model="m"), explicit="yue2-3b")
    assert sel.spec.id == "yue2-3b" and not sel.automatic


def test_auto_selection_prefers_quality_and_explains_missing(cfg, monkeypatch):
    monkeypatch.setattr(envs, "probe", lambda *_a, **_k: envs.ProbeResult("x", "py", "t", True))
    req = GenerationRequest(prompt="x", model="auto", duration=20)
    with pytest.raises(ModelNotInstalledError) as ei:
        select_model(cfg, req, device="cpu")
    assert "prompt2music models download" in (ei.value.hint or "")
    install_fake(cfg, "musicgen-small")
    install_fake(cfg, "stable-audio-3-small-music")
    sel = select_model(cfg, req, device="cpu", quality="best")
    assert sel.spec.id == "stable-audio-3-small-music" and sel.automatic
    cfg.set("general.allow_noncommercial_models", False)
    assert "license" in select_model(cfg, req, device="cpu").rejected["musicgen-small"]
    vocal = GenerationRequest(prompt="x", model="auto", instrumental=False, duration=20)
    with pytest.raises(ModelNotInstalledError):
        select_model(cfg, vocal, device="cpu")  # no installed vocal model


def test_continue_modes(cfg):
    params = tone_params(cfg, task="continue", input_audio="/tmp/a.wav",
                         continue_mode="transition", continue_target="drum and bass")
    with pytest.raises(UnsupportedTaskError):  # the tone backend cannot continue audio
        service.prepare(cfg, params)
    with pytest.raises(InvalidRequestError, match="needs --to"):
        service.prepare(cfg, {**params, "continue_target": None, "model": "musicgen-small"})


@needs_ffmpeg
def test_run_generate_end_to_end(cfg):
    engine = Engine(cfg)
    try:
        out_dir = cfg.output_dir
        result = service.run_generate(cfg, engine, tone_params(
            cfg, batch_size=2, format="flac", normalize="peak", fade_out=0.1))
    finally:
        engine.unload()
    assert len(result["outputs"]) == 2
    for out in result["outputs"]:
        path = Path(out["path"])
        assert path.parent == out_dir and path.suffix == ".flac"
        assert out["validation"]["ok"], out["validation"]
        side = json.loads(Path(out["sidecar"]).read_text())
        assert side["model"] == "test-tone" and side["post_processing"][0]["op"] == "fade"
        assert side["parameters"]["prompt"] == "pipeline test"
    assert result["rtf"] is not None
    assert not (Path(cfg.output_dir) / "out.wav").exists()
    lib = Library()
    assert len(lib.search("pipeline")) == 2


@needs_ffmpeg
def test_outputs_never_clobber(cfg):
    engine = Engine(cfg)
    target = str(cfg.output_dir / "fixed.wav")
    try:
        a = service.run_generate(cfg, engine, tone_params(cfg, output=target))
        b = service.run_generate(cfg, engine, tone_params(cfg, output=target))
    finally:
        engine.unload()
    assert a["outputs"][0]["path"] != b["outputs"][0]["path"]
    assert b["outputs"][0]["path"].endswith("fixed-v2.wav")


@needs_ffmpeg
def test_execute_job_records_completion_and_cancellation(cfg):
    store = JobStore()
    engine = Engine(cfg)
    try:
        ok = store.create("generate", tone_params(cfg), status="loading")
        job = execute_job(cfg, engine, store, store.get(ok))
        assert job.status == "completed" and job.result["outputs"]
        assert "completed" in store.log_path(ok).read_text()
        slow = store.create("generate", tone_params(cfg, duration=2,
                                                     extra={"seconds_per_step": 0.5}),
                            status="loading")
        threading.Timer(1.0, lambda: JobStore().request_cancel(slow)).start()
        job = execute_job(cfg, engine, store, store.get(slow))
        assert job.status == "cancelled"
        bad = store.create("generate", tone_params(cfg, model="not-a-model"), status="loading")
        job = execute_job(cfg, engine, store, store.get(bad))
        assert job.status == "failed" and job.error["code"] == "unknown_model"
    finally:
        engine.unload()


@needs_ffmpeg
def test_library_export_reproduce_remove(cfg, tmp_path):
    engine = Engine(cfg)
    try:
        result = service.run_generate(cfg, engine, tone_params(cfg, seed=11))
    finally:
        engine.unload()
    lib = Library()
    tid = result["outputs"][0]["track_id"]
    exported = lib.export(tid, tmp_path / "exp", fmt="mp3")
    assert Path(exported["exported"]).suffix == ".mp3"
    assert Path(exported["exported"] + ".json").is_file()
    params = lib.reproduce_params(tid)
    assert params["seed"] == 11 and params["model"] == "test-tone"
    lib.remove(tid, delete_files=True)
    assert not Path(result["outputs"][0]["path"]).exists()


# -- daemon process ---------------------------------------------------------------------
@needs_ffmpeg
def test_background_job_survives_submitter_exit(cfg):
    """Submit from a short-lived process; the detached daemon finishes the job."""
    code = (
        "from prompt2music import daemon, service\n"
        "from prompt2music.config import load_music_config\n"
        "from prompt2music.jobs import JobStore\n"
        "cfg = load_music_config()\n"
        "p = service.normalise_params(cfg, {'prompt': 'bg', 'model': 'test-tone', "
        "'duration': 0.5})\n"
        "print(JobStore().create('generate', p))\n"
        "daemon.ensure_running(cfg)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=os.environ.copy(), timeout=60, check=True)
    job_id = proc.stdout.strip().splitlines()[-1]
    store = JobStore()
    job = daemon.follow(store, job_id, poll=0.2)
    assert job.status == "completed", job.error
    assert job.daemon_pid == daemon.status()["pid"]  # ran in the daemon, not the submitter
    # warm model reused by the next job
    second = store.create("generate", tone_params(cfg))
    assert daemon.follow(store, second, poll=0.2).result["load"]["reused"] is True
    assert daemon.stop()["stopped"]
    assert not daemon.status()["running"]


def test_daemon_single_instance(cfg):
    info = daemon.start(cfg, idle_exit=30)
    try:
        assert info["running"]
        again = daemon.start(cfg)
        assert again["started"] is False and again["pid"] == info["pid"]
    finally:
        daemon.stop()


def test_systemd_unit():
    unit = daemon.systemd_unit("/run/current-system/sw/bin/prompt2music")
    assert "ExecStart=/run/current-system/sw/bin/prompt2music daemon run --idle-exit 0" in unit



def test_waiting_on_queued_job_without_daemon_fails_fast(music_env):
    from prompt2music.errors import MusicError

    store = JobStore()
    job_id = store.create("generate", {})
    with pytest.raises(MusicError, match="daemon is not running"):
        daemon.follow(store, job_id, poll=0.05)
