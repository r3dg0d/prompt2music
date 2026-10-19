"""Host side of the backend worker protocol.

:class:`Engine` keeps at most one model loaded (in one worker process) and
reuses it across consecutive requests. Loading a different model shuts the
old worker down first, which returns all of its GPU memory to the system.

Cancellation is cooperative first (backends check a token between steps)
and forced second: if the worker has not stopped within ``grace`` seconds it
is terminated. A forced stop also unloads the model, because the only safe
way to interrupt native inference code is to end the process.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from prompt2music import envs, paths
from prompt2music.backends.base import GenerationRequest, check_request_against
from prompt2music.config import MusicConfig
from prompt2music.errors import (
    BackendNotReadyError,
    CancelledError,
    GenerationError,
    MusicError,
    ResourceLimitError,
    error_from_dict,
)
from prompt2music.gpu import DevicePlan, GpuLock, plan_device
from prompt2music.manager import ModelManager
from prompt2music.registry import ModelSpec

ProgressFn = Callable[[float | None, str], None]


class WorkerClient:
    """One backend worker subprocess speaking JSON lines."""

    def __init__(self, python: Path, env: dict[str, str], log_path: Path) -> None:
        self.python = python
        self.env = env
        self.log_path = log_path
        self.proc: subprocess.Popen[bytes] | None = None
        self._log: BinaryIO | None = None
        self._messages: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._next_id = 0
        self._lock = threading.Lock()
        self._kill_timer: threading.Timer | None = None
        self.forced_stop = False

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def start(self, timeout: float = 120.0) -> dict[str, Any]:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab")  # noqa: SIM115 - lives with the worker
        self.proc = subprocess.Popen(
            [str(self.python), "-u", "-m", "prompt2music.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            env=self.env,
            start_new_session=True,  # own process group: killable without hitting us
        )
        threading.Thread(target=self._read, daemon=True).start()
        msg = self._wait(lambda m: m.get("type") == "ready", timeout, None)
        return msg

    def _read(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for raw in self.proc.stdout:
            try:
                self._messages.put(json.loads(raw.decode("utf-8")))
            except ValueError:
                continue
        self._messages.put(None)

    def log_tail(self, lines: int = 25) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def _wait(self, match: Callable[[dict[str, Any]], bool], timeout: float | None,
              on_progress: ProgressFn | None, msg_id: int | None = None) -> dict[str, Any]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining == 0.0:
                self.kill()
                raise GenerationError(f"worker timed out after {timeout:.0f}s; it was stopped")
            try:
                msg = self._messages.get(timeout=min(remaining or 1.0, 1.0))
            except queue.Empty:
                continue
            if msg is None:
                code = self.proc.wait() if self.proc else None
                if self.forced_stop:
                    raise CancelledError("generation cancelled (worker stopped)")
                raise GenerationError(
                    f"backend worker exited unexpectedly (code {code})",
                    hint=f"see {self.log_path}\n{self.log_tail()}",
                )
            if msg_id is not None and msg.get("id") == msg_id and msg.get("type") == "progress":
                if on_progress:
                    on_progress(msg.get("fraction"), str(msg.get("message", "")))
                continue
            if match(msg):
                return msg

    def call(self, op: str, payload: dict[str, Any] | None = None, *,
             timeout: float | None = None, on_progress: ProgressFn | None = None) -> dict[str, Any]:
        if not self.alive:
            raise GenerationError("backend worker is not running")
        with self._lock:
            self._next_id += 1
            msg_id = self._next_id
        self._send({"id": msg_id, "op": op, **(payload or {})})
        msg = self._wait(lambda m: m.get("id") == msg_id and m.get("type") in
                         ("result", "error"), timeout, on_progress, msg_id)
        if msg["type"] == "error":
            err = msg.get("error", {})
            exc = error_from_dict(err)
            if err.get("traceback"):
                exc.hint = (exc.hint + "\n" if exc.hint else "") + f"worker log: {self.log_path}"
            raise exc
        return dict(msg.get("data") or {})

    def _send(self, payload: dict[str, Any]) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        try:
            self.proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise GenerationError("backend worker is not accepting requests") from exc

    def cancel(self, grace: float = 15.0) -> None:
        """Ask nicely, then terminate the process group if it does not stop."""
        if not self.alive:
            return
        try:
            self._send({"op": "cancel"})
        except GenerationError:
            pass
        if self._kill_timer is None and grace >= 0:
            self._kill_timer = threading.Timer(grace, self._force_stop)
            self._kill_timer.daemon = True
            self._kill_timer.start()

    def cancel_settled(self) -> None:
        if self._kill_timer is not None:
            self._kill_timer.cancel()
            self._kill_timer = None

    def _force_stop(self) -> None:
        self.forced_stop = True
        self.kill()

    def kill(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, AttributeError):
            self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, AttributeError):
                self.proc.kill()
            self.proc.wait(timeout=10)

    def close(self) -> None:
        self.cancel_settled()
        if self.alive:
            try:
                self.call("shutdown", timeout=30)
            except MusicError:
                pass
            try:
                assert self.proc is not None
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.kill()
        if self._log is not None:
            self._log.close()
            self._log = None


@dataclass
class LoadedModel:
    spec: ModelSpec
    client: WorkerClient
    plan: DevicePlan
    load_info: dict[str, Any]
    lock: GpuLock | None
    loaded_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)


class Engine:
    """Loads one model at a time and runs requests against it."""

    def __init__(self, cfg: MusicConfig, *, manager: ModelManager | None = None,
                 gpu_lock_timeout: float = 600.0) -> None:
        self.cfg = cfg
        self.manager = manager or ModelManager(cfg)
        self.loaded: LoadedModel | None = None
        self._loading: WorkerClient | None = None  # worker being started/loaded
        self.gpu_lock_timeout = gpu_lock_timeout
        self._busy = threading.Lock()

    # -- lifecycle ------------------------------------------------------------
    def load(self, spec: ModelSpec, *, device: str | None = None, precision: str | None = None,
             vram_limit_gb: float | None = None, on_status: ProgressFn | None = None,
             log_path: Path | None = None) -> dict[str, Any]:
        plan = plan_device(self.cfg, spec, device=device, precision=precision,
                           vram_limit_gb=vram_limit_gb)
        current = self.loaded
        if (current is not None and current.spec.id == spec.id and current.client.alive
                and current.plan.device == plan.device
                and current.plan.precision == plan.precision):
            current.last_used = time.monotonic()
            return {**current.load_info, "reused": True}
        self.unload()
        model_dir = self.manager.require(spec.id)
        python = envs.require_backend(self.cfg, spec.backend)
        lock: GpuLock | None = None
        if plan.device.startswith("cuda"):
            lock = GpuLock()
            if not lock.acquire(timeout=0):
                if on_status:
                    on_status(None, f"waiting for GPU lock held by {lock.holder() or '?'}")
                if not lock.acquire(timeout=self.gpu_lock_timeout):
                    raise ResourceLimitError(
                        "another prompt2music process is using the GPU",
                        hint="wait for it, or stop the job daemon: prompt2music daemon stop",
                    )
        extra_env: dict[str, str] = {}
        if model_dir is not None and any(s.layout == "hf-cache" for s in spec.sources):
            extra_env["HF_HUB_CACHE"] = str(model_dir / "hf-cache")
        log = log_path or (paths.logs_dir() / f"worker-{spec.backend}.log")
        client = WorkerClient(python, envs.worker_env(self.cfg, extra_env), log)
        self._loading = client
        try:
            if on_status:
                on_status(None, f"starting {spec.backend} worker")
            client.start()
            options = {
                **spec.options,
                "cpu_offload": bool(self.cfg["hardware"]["cpu_offload"]),
                "torch_compile": bool(self.cfg["hardware"]["torch_compile"]),
                "allow_tf32": bool(self.cfg["hardware"]["allow_tf32"]),
            }
            if on_status:
                on_status(None, f"loading {spec.id} on {plan.device} ({plan.precision})")
            info = client.call("load", {
                "backend": spec.backend,
                "model_id": spec.id,
                "model_dir": str(model_dir) if model_dir else None,
                "options": options,
                "capabilities": spec.capabilities.as_dict(),
                "device": plan.device,
                "precision": plan.precision,
                "vram_limit_gb": plan.vram_budget_gb,
            }, timeout=float(self.cfg["jobs"]["timeout_seconds"]))
        except BaseException as exc:
            client.kill()
            client.close()
            if lock:
                lock.release()
            if client.forced_stop and not isinstance(exc, CancelledError):
                raise CancelledError("model load cancelled") from exc
            raise
        finally:
            self._loading = None
        info.update({"device_plan": plan.as_dict(), "python": str(python),
                     "worker_pid": client.pid, "reused": False})
        self.loaded = LoadedModel(spec, client, plan, info, lock)
        return info

    def unload(self) -> bool:
        current, self.loaded = self.loaded, None
        if current is None:
            return False
        try:
            current.client.close()
        finally:
            if current.lock:
                current.lock.release()
        return True

    close = unload

    def idle_seconds(self) -> float:
        return 0.0 if self.loaded is None else time.monotonic() - self.loaded.last_used

    # -- work -----------------------------------------------------------------
    def _client_for(self, spec: ModelSpec) -> LoadedModel:
        loaded = self.loaded
        if loaded is None or loaded.spec.id != spec.id:
            raise BackendNotReadyError(f"{spec.id} is not loaded")
        if not loaded.client.alive:
            tail = loaded.client.log_tail()
            self.unload()  # the next load() starts a fresh worker
            raise GenerationError("backend worker exited unexpectedly",
                                  hint=f"see {loaded.client.log_path}\n{tail}")
        return loaded

    def generate(self, spec: ModelSpec, request: GenerationRequest, output_dir: Path, *,
                 on_progress: ProgressFn | None = None) -> dict[str, Any]:
        check_request_against(spec.id, spec.capabilities, request)
        loaded = self._client_for(spec)
        with self._busy:
            try:
                data = loaded.client.call(
                    "generate", {"request": request.to_dict(), "output_dir": str(output_dir)},
                    timeout=float(self.cfg["jobs"]["timeout_seconds"]),
                    on_progress=on_progress)
            except CancelledError:
                self._after_cancel(loaded)
                raise
            except GenerationError:
                if not loaded.client.alive:
                    self.unload()
                raise
            finally:
                loaded.client.cancel_settled()
                loaded.last_used = time.monotonic()
        return data

    def separate(self, spec: ModelSpec, input_wav: Path, output_dir: Path,
                 stems: list[str] | None, *, on_progress: ProgressFn | None = None
                 ) -> dict[str, Any]:
        loaded = self._client_for(spec)
        with self._busy:
            try:
                return loaded.client.call(
                    "separate", {"input": str(input_wav), "output_dir": str(output_dir),
                                 "stems": stems},
                    timeout=float(self.cfg["jobs"]["timeout_seconds"]),
                    on_progress=on_progress)
            except CancelledError:
                self._after_cancel(loaded)
                raise
            finally:
                loaded.client.cancel_settled()
                loaded.last_used = time.monotonic()

    def _after_cancel(self, loaded: LoadedModel) -> None:
        if not loaded.client.alive or loaded.client.forced_stop:
            self.unload()

    def cancel(self, grace: float = 15.0) -> None:
        loading = self._loading
        if loading is not None:
            # A load cannot be interrupted cooperatively; stop that worker now.
            loading.forced_stop = True
            loading.kill()
        elif self.loaded is not None:
            self.loaded.client.cancel(grace)

    def status(self) -> dict[str, Any]:
        if self.loaded is None:
            return {"loaded": None}
        lm = self.loaded
        return {
            "loaded": lm.spec.id,
            "device": lm.plan.device,
            "precision": lm.plan.precision,
            "worker_pid": lm.client.pid,
            "alive": lm.client.alive,
            "idle_seconds": round(self.idle_seconds(), 1),
            "load_info": lm.load_info,
        }
