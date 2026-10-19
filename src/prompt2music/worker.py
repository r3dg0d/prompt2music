"""Backend worker process: ``python -m prompt2music.worker``.

Runs inside a backend's own virtualenv and keeps one model loaded between
requests. Protocol: one JSON object per line.

    host → worker  {"id": 1, "op": "load", "backend": ..., "model_id": ..., ...}
                   {"id": 2, "op": "generate", "request": {...}, "output_dir": ...}
                   {"id": 3, "op": "separate", ...}
                   {"op": "cancel"}            # handled immediately, no reply
                   {"id": 4, "op": "unload"} / {"id": 5, "op": "shutdown"}
    worker → host  {"id": 1, "type": "result", "data": {...}}
                   {"id": 1, "type": "error", "error": {"code": ..., "message": ...}}
                   {"id": 2, "type": "progress", "fraction": 0.4, "message": "..."}

The original stdout file descriptor is reserved for the protocol; fd 1 is
pointed at stderr so library prints cannot corrupt it. Only stdlib and
prompt2music.backends are imported here.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, TextIO

from prompt2music.backends import load_backend_class
from prompt2music.backends.base import (
    CancelToken,
    Capabilities,
    GenerationRequest,
    MusicBackend,
)
from prompt2music.errors import CancelledError, MusicError, OutOfMemoryError


def _peak_rss_mb() -> float | None:
    """Peak resident memory of this process. VmHWM is per address space (reset
    by exec); ru_maxrss can carry the parent's peak across fork/exec on Linux."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 1024 if sys.platform != "darwin" else peak / 2**20, 1)


class Worker:
    def __init__(self, out: TextIO) -> None:
        self.out = out
        self.write_lock = threading.Lock()
        self.backend: MusicBackend | None = None
        self.cancel = CancelToken()  # token of the op currently running
        self.tokens: dict[Any, CancelToken] = {}
        self.tokens_lock = threading.Lock()
        self.ops: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def send(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, default=str)
        with self.write_lock:
            self.out.write(line + "\n")
            self.out.flush()

    # -- stdin reader: cancel is handled out of band ------------------------
    def read_loop(self, stream: TextIO) -> None:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.send({"type": "log", "message": f"ignored malformed line: {line[:80]}"})
                continue
            with self.tokens_lock:
                if msg.get("op") == "cancel":
                    target = msg.get("target")
                    for key, token in self.tokens.items():
                        if target is None or key == target:
                            token.cancel()
                    continue
                self.tokens[msg.get("id")] = CancelToken()
            self.ops.put(msg)
        with self.tokens_lock:  # host went away: stop work and exit
            for token in self.tokens.values():
                token.cancel()
        self.ops.put(None)

    # -- ops ------------------------------------------------------------------
    def op_ping(self, msg: dict[str, Any]) -> dict[str, Any]:
        return {"pid": os.getpid(), "python": sys.executable,
                "loaded": self.backend.model_id if self.backend else None}

    def op_load(self, msg: dict[str, Any]) -> dict[str, Any]:
        if self.backend is not None:
            self.backend.unload()
            self.backend = None
        cls = load_backend_class(msg["backend"])
        model_dir = Path(msg["model_dir"]) if msg.get("model_dir") else None
        caps = Capabilities.from_dict(msg["capabilities"])
        backend = cls(msg["model_id"], model_dir, msg.get("options", {}), caps)
        device = msg.get("device", "cpu")
        self._apply_vram_limit(device, msg.get("vram_limit_gb"))
        started = time.perf_counter()
        backend.load(device, msg.get("precision", "fp32"))
        self.backend = backend
        return {"model_id": backend.model_id, "device": device,
                "load_seconds": round(backend.load_seconds or time.perf_counter() - started, 3),
                "memory": backend.memory_stats(), "peak_rss_mb": _peak_rss_mb()}

    @staticmethod
    def _apply_vram_limit(device: str, limit_gb: float | None) -> None:
        if not limit_gb or not device.startswith("cuda"):
            return
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            return
        if not torch.cuda.is_available():
            return
        idx = int(device.split(":", 1)[1]) if ":" in device else 0
        total = torch.cuda.get_device_properties(idx).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, limit_gb / total), idx)

    def _require_backend(self) -> MusicBackend:
        if self.backend is None:
            raise MusicError("no model loaded in worker")
        return self.backend

    def _progress(self, msg_id: Any) -> Any:
        def emit(fraction: float | None, message: str) -> None:
            if self.cancel.cancelled:
                raise CancelledError("cancelled")
            self.send({"id": msg_id, "type": "progress", "fraction": fraction,
                       "message": message})

        return emit

    def _reset_peak(self) -> None:
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

    def op_generate(self, msg: dict[str, Any]) -> dict[str, Any]:
        backend = self._require_backend()
        request = GenerationRequest.from_dict(msg["request"])
        self._reset_peak()
        result = backend.generate(request, Path(msg["output_dir"]), self._progress(msg.get("id")),
                                  self.cancel)
        data = result.as_dict()
        data["memory"] = backend.memory_stats()
        data["peak_rss_mb"] = _peak_rss_mb()
        return data

    def op_separate(self, msg: dict[str, Any]) -> dict[str, Any]:
        backend = self._require_backend()
        separate = getattr(backend, "separate", None)
        if separate is None:
            raise MusicError(f"{backend.model_id} cannot separate stems")
        self._reset_peak()
        data = separate(Path(msg["input"]), Path(msg["output_dir"]), msg.get("stems"),
                        self._progress(msg.get("id")), self.cancel)
        data["memory"] = backend.memory_stats()
        return data

    def op_unload(self, msg: dict[str, Any]) -> dict[str, Any]:
        if self.backend is not None:
            self.backend.unload()
            self.backend = None
        return {"unloaded": True}

    def handle(self, msg: dict[str, Any]) -> bool:
        op = msg.get("op")
        msg_id = msg.get("id")
        if op == "shutdown":
            self.send({"id": msg_id, "type": "result", "data": {"bye": True}})
            return False
        handler = getattr(self, f"op_{op}", None)
        with self.tokens_lock:
            self.cancel = self.tokens.get(msg_id) or CancelToken()
        try:
            if handler is None:
                raise MusicError(f"unknown op {op!r}")
            data = handler(msg)
            self.send({"id": msg_id, "type": "result", "data": data})
        except MusicError as exc:
            self.send({"id": msg_id, "type": "error", "error": exc.as_dict()})
        except Exception as exc:  # backend libraries raise anything; report, keep serving
            err = _classify(exc)
            err["traceback"] = traceback.format_exc()[-4000:]
            self.send({"id": msg_id, "type": "error", "error": err})
        finally:
            with self.tokens_lock:
                self.tokens.pop(msg_id, None)
        return True


def _classify(exc: BaseException) -> dict[str, Any]:
    text = f"{type(exc).__name__}: {exc}"
    if "out of memory" in text.lower() or type(exc).__name__ == "OutOfMemoryError":
        return OutOfMemoryError(text, hint="reduce duration/batch size or use a smaller "
                                "model; check other GPU processes").as_dict()
    return {"code": "generation_failed", "message": text}


def main() -> int:
    protocol_fd = os.dup(1)
    os.dup2(2, 1)  # anything printed by libraries goes to stderr (the job log)
    sys.stdout = sys.stderr
    out = os.fdopen(protocol_fd, "w", buffering=1, encoding="utf-8")
    worker = Worker(out)
    reader = threading.Thread(target=worker.read_loop, args=(sys.stdin,), daemon=True)
    reader.start()
    worker.send({"type": "ready", "pid": os.getpid(), "python": sys.version.split()[0]})
    while True:
        msg = worker.ops.get()
        if msg is None or not worker.handle(msg):
            break
    if worker.backend is not None:
        worker.backend.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
