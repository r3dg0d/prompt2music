"""Job daemon: ``prompt2music daemon run`` (foreground) or ``daemon start`` (detached).

One daemon per user. It claims queued jobs from SQLite, keeps the last model
loaded for consecutive jobs (unloading after ``jobs.idle_unload_seconds``),
watches for cancellation requests, and survives the CLI or MCP process that
queued the work. An auto-started daemon exits after
``jobs.daemon_idle_exit_seconds`` of inactivity; under systemd use ``run``
with ``--idle-exit 0``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from prompt2music import paths
from prompt2music.config import MusicConfig, load_music_config
from prompt2music.engine import Engine
from prompt2music.errors import CancelledError, MusicError
from prompt2music.jobs import JobRecord, JobStore
from prompt2music.registry import get_model


class JobLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def __call__(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] {message}\n")


def execute_job(cfg: MusicConfig, engine: Engine, store: JobStore, job: JobRecord, *,
                cancel_grace: float = 15.0, stopping: threading.Event | None = None,
                on_progress: Any = None) -> JobRecord:
    """Run one claimed job to a final state. Never raises for job failures."""
    from prompt2music import service

    log = JobLog(store.log_path(job.id))
    log(f"start {job.kind} model={job.model} source={job.source}")
    done = threading.Event()
    max_input = int(cfg["mcp"]["max_input_file_mb"]) * 1024**2 if job.source == "mcp" else None

    def monitor() -> None:
        watcher = JobStore()
        while not done.wait(0.5):
            if watcher.cancel_requested(job.id) or (stopping is not None and stopping.is_set()):
                log("cancellation requested; stopping inference")
                engine.cancel(cancel_grace)
                return

    last = [0.0]

    def progress(fraction: float | None, message: str) -> None:
        if on_progress is not None:
            on_progress(fraction, message)
        now = time.monotonic()
        if fraction is None or now - last[0] > 0.5 or fraction >= 1.0:
            last[0] = now
            store.progress(job.id, fraction, message)
            if fraction is None:
                log(message)

    def stage(name: str) -> None:
        if store.cancel_requested(job.id) or (stopping is not None and stopping.is_set()):
            raise CancelledError(f"cancelled before stage {name}")
        store.update(job.id, status=name)
        log(f"stage {name}")
        if on_progress is not None:
            on_progress(None, name)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        if store.cancel_requested(job.id):
            raise CancelledError("cancelled before start")
        params = job.params
        if job.kind == "generate":
            result = service.run_generate(cfg, engine, params, job_id=job.id, progress=progress,
                                          on_stage=stage, max_input_bytes=max_input)
        elif job.kind == "separate":
            stage("running")
            result = service.run_separate(cfg, engine, params, job_id=job.id,
                                          progress=progress, max_input_bytes=max_input)
        elif job.kind == "load":
            stage("loading")
            result = engine.load(get_model(params["model"]), device=params.get("device"),
                                 precision=params.get("precision"),
                                 vram_limit_gb=params.get("vram_limit_gb"), on_status=progress)
        elif job.kind == "unload":
            result = {"unloaded": engine.unload()}
        elif job.kind == "download":
            from prompt2music.downloader import HfHub, execute_plan, plan_download

            stage("running")
            spec = get_model(params["model"])
            hub = HfHub(cfg["models"]["hf_endpoint"] or None)
            plan = plan_download(spec, cfg.models_dir, hub=hub, token=cfg.hf_token(),
                                 offline=cfg.offline)
            limit = params.get("max_bytes")
            if limit and plan.total_bytes > limit:
                raise MusicError(f"{spec.id} is {plan.total_bytes / 1024**3:.1f} GB, above the "
                                 "allowed download size")
            manifest = execute_plan(plan, hub=hub, token=cfg.hf_token(),
                                    license_accepted=bool(params.get("accept_license")),
                                    max_workers=int(cfg["models"]["max_download_workers"]),
                                    quiet=True, progress=lambda m: progress(None, m))
            result = {"model": spec.id, "path": str(plan.destination),
                      "bytes": manifest.total_bytes, "revisions": manifest.revisions,
                      "skipped_code_files": plan.skipped_code_files}
        elif job.kind == "benchmark":
            from prompt2music.benchmark import run_benchmark

            stage("running")
            result = run_benchmark(cfg, engine, params, progress=progress)
        else:
            raise MusicError(f"unknown job kind {job.kind}")
        mem = (result or {}).get("memory") or {}
        resolved = (result or {}).get("model") if isinstance((result or {}).get("model"), str) \
            else job.model
        store.update(job.id, status="completed", result=result, progress=1.0, message="done",
                     model=resolved,
                     generation_seconds=(result or {}).get("generation_seconds"),
                     gpu_peak_mb=mem.get("cuda_peak_allocated_mb"),
                     worker_pid=(engine.loaded.client.pid if engine.loaded else None))
        log("completed")
    except CancelledError as exc:
        reason = "daemon stopping" if stopping is not None and stopping.is_set() else str(exc)
        store.update(job.id, status="cancelled", message=reason,
                     error={"code": "cancelled", "message": reason})
        log(f"cancelled: {reason}")
    except MusicError as exc:
        store.update(job.id, status="failed", error=exc.as_dict(), message=str(exc))
        log(f"failed: {exc}" + (f"\nhint: {exc.hint}" if exc.hint else ""))
    except Exception as exc:  # unexpected bug: record it, keep the daemon alive
        store.update(job.id, status="failed", message=str(exc),
                     error={"code": "internal_error", "message": f"{type(exc).__name__}: {exc}"})
        log("internal error:\n" + traceback.format_exc())
    finally:
        done.set()
        thread.join(timeout=2)
    return store.get(job.id)


# -- process management ---------------------------------------------------------
def _lock_file() -> Path:
    return paths.ensure(paths.daemon_dir()) / "daemon.lock"


def _pid_file() -> Path:
    return paths.ensure(paths.daemon_dir()) / "daemon.json"


def status() -> dict[str, Any]:
    info: dict[str, Any] = {"running": False, "pid": None}
    try:
        data = json.loads(_pid_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows: trust the pid file
        pid = data.get("pid")
        info.update(data, running=bool(pid))
        return info
    with open(_lock_file(), "a+", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            info.update(data, running=True)
            return info
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return info


class Daemon:
    def __init__(self, cfg: MusicConfig, *, idle_exit: float | None = None,
                 poll: float = 1.0) -> None:
        self.cfg = cfg
        self.idle_exit = (float(cfg["jobs"]["daemon_idle_exit_seconds"])
                          if idle_exit is None else idle_exit)
        self.idle_unload = float(cfg["jobs"]["idle_unload_seconds"])
        self.poll = poll
        self.stopping = threading.Event()
        self.engine = Engine(cfg)
        self.store = JobStore()
        self._lock_fh: Any = None

    def _acquire(self) -> bool:
        try:
            import fcntl
        except ImportError:  # pragma: no cover
            return True
        fh = open(_lock_file(), "a+", encoding="utf-8")  # noqa: SIM115 - held for lifetime
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.close()
            return False
        self._lock_fh = fh
        return True

    def _on_signal(self, signum: int, frame: Any) -> None:
        self.stopping.set()
        self.engine.cancel(grace=5.0)

    def run(self) -> int:
        if not self._acquire():
            print("prompt2music daemon is already running", file=sys.stderr)
            return 1
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)
        _pid_file().write_text(json.dumps({
            "pid": os.getpid(), "started_at": time.time(), "python": sys.executable,
            "idle_exit_seconds": self.idle_exit}), encoding="utf-8")
        recovered = self.store.recover_orphans()
        if recovered:
            print(f"marked {len(recovered)} orphaned job(s) failed", file=sys.stderr)
        last_activity = time.monotonic()
        try:
            while not self.stopping.is_set():
                job = self.store.claim_next(os.getpid())
                if job is not None:
                    execute_job(self.cfg, self.engine, self.store, job, stopping=self.stopping)
                    last_activity = time.monotonic()
                    continue
                if (self.engine.loaded is not None and self.idle_unload > 0
                        and self.engine.idle_seconds() >= self.idle_unload):
                    self.engine.unload()
                idle = time.monotonic() - last_activity
                if self.idle_exit > 0 and idle >= self.idle_exit and self.engine.loaded is None:
                    break
                self.stopping.wait(self.poll)
        finally:
            self.engine.unload()
            try:
                _pid_file().unlink()
            except OSError:
                pass
        return 0


def start(cfg: MusicConfig, *, idle_exit: float | None = None, wait: float = 15.0
          ) -> dict[str, Any]:
    current = status()
    if current["running"]:
        return {**current, "started": False}
    log = paths.ensure(paths.logs_dir()) / "daemon.log"
    argv = [sys.executable, "-m", "prompt2music.daemon"]
    if idle_exit is not None:
        argv += ["--idle-exit", str(idle_exit)]
    env = os.environ.copy()
    # Nix-wrapped (and some zipapp) launchers add dependencies via site.addsitedir
    # inside the script, so a bare `python -m` child would not see them.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p and os.path.isdir(p))
    with open(log, "ab") as fh:
        subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=fh, stderr=fh,
                         start_new_session=True, close_fds=True, env=env)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        current = status()
        if current["running"]:
            return {**current, "started": True, "log": str(log)}
        time.sleep(0.2)
    raise MusicError("daemon did not start", hint=f"see {log}")


def stop(timeout: float = 30.0) -> dict[str, Any]:
    current = status()
    pid = current.get("pid")
    if not current["running"] or not pid:
        return {"running": False, "stopped": False}
    os.kill(int(pid), signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not status()["running"]:
            return {"running": False, "stopped": True, "pid": pid}
        time.sleep(0.2)
    os.kill(int(pid), getattr(signal, "SIGKILL", signal.SIGTERM))
    return {"running": False, "stopped": True, "pid": pid, "killed": True}


def ensure_running(cfg: MusicConfig) -> dict[str, Any]:
    return start(cfg) if not status()["running"] else {**status(), "started": False}


def follow(store: JobStore, job_id: str, *, on_update: Any = None,
           poll: float = 0.5) -> JobRecord:
    """Block until the job is final, reporting progress changes."""
    seen: tuple[Any, ...] = ()
    while True:
        job = store.get(job_id)
        key = (job.status, job.progress, job.message)
        if key != seen and on_update:
            on_update(job)
            seen = key
        if job.final:
            return job
        if not status()["running"]:
            if job.status == "queued":
                raise MusicError(f"job {job.id} is queued but the job daemon is not running",
                                 hint="start it with `prompt2music daemon start`")
            if job.daemon_pid != os.getpid():
                store.recover_orphans()
        time.sleep(poll)


SYSTEMD_UNIT = """[Unit]
Description=prompt2music job daemon (ai-media-cli)
After=default.target

[Service]
Type=simple
ExecStart={exe} daemon run --idle-exit 0
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"""


def systemd_unit(exe: str) -> str:
    return SYSTEMD_UNIT.format(exe=exe)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prompt2music-daemon")
    parser.add_argument("--idle-exit", type=float, default=None)
    args = parser.parse_args(argv)
    return Daemon(load_music_config(), idle_exit=args.idle_exit).run()


if __name__ == "__main__":
    raise SystemExit(main())
