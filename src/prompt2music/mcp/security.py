"""Guards for MCP tool inputs, which come from autonomous agents and are untrusted.

* Output paths must resolve (after following symlinks) inside the configured
  output directory or ``mcp.allowed_output_dirs``; existing symlinks are
  never written through.
* Input paths must resolve inside the output roots or ``mcp.allowed_input_dirs``
  and stay under ``mcp.max_input_file_mb``.
* Models must be catalogued and, if ``mcp.allowed_models`` is set, allowlisted.
* Every call is appended to a JSON-lines audit log.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from prompt2music import paths
from prompt2music.config import MusicConfig
from prompt2music.errors import InvalidRequestError, SecurityError
from prompt2music.registry import ModelSpec, get_model

_AUDIT_LOCK = threading.Lock()


def _resolve_root(raw: str | Path) -> Path:
    return paths.expand(raw).resolve()


def _inside(path: Path, roots: list[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


class Policy:
    def __init__(self, cfg: MusicConfig) -> None:
        self.cfg = cfg
        mcp = cfg["mcp"]
        self.output_roots = [_resolve_root(cfg.output_dir)] + [
            _resolve_root(p) for p in mcp["allowed_output_dirs"]]
        self.input_roots = self.output_roots + [_resolve_root(p)
                                                for p in mcp["allowed_input_dirs"]]
        self.max_input_bytes = int(mcp["max_input_file_mb"]) * 1024**2
        self.max_duration = float(mcp["max_duration_seconds"])
        self.allowed_models = {m.lower() for m in mcp["allowed_models"]}

    @staticmethod
    def _clean(raw: str) -> str:
        if not isinstance(raw, str) or not raw.strip():
            raise InvalidRequestError("path must be a non-empty string")
        if "\x00" in raw:
            raise SecurityError("path contains a NUL byte")
        if raw.startswith(("http://", "https://", "file://", "ftp://")):
            raise SecurityError("URLs are not accepted; pass a local path")
        return raw.strip()

    def resolve_output(self, raw: str | None, *, is_dir: bool = False) -> Path | None:
        """Absolute output path inside an allowed root (None → caller default)."""
        if raw is None or raw == "":
            return None
        text = self._clean(raw)
        candidate = Path(os.path.expanduser(text))
        if not candidate.is_absolute():
            candidate = self.output_roots[0] / candidate
        if candidate.is_symlink():
            raise SecurityError(f"refusing to write through symlink {candidate}")
        resolved = candidate.resolve()
        if not _inside(resolved, self.output_roots):
            raise SecurityError(
                f"output {resolved} is outside the allowed output directories",
                hint="allowed: " + ", ".join(str(r) for r in self.output_roots)
                + " (configure mcp.allowed_output_dirs)",
            )
        if not is_dir and resolved.exists() and resolved.is_dir():
            raise InvalidRequestError(f"{resolved} is a directory")
        return resolved

    def resolve_input(self, raw: str) -> Path:
        text = self._clean(raw)
        candidate = Path(os.path.expanduser(text))
        if not candidate.is_absolute():
            candidate = self.output_roots[0] / candidate
        resolved = candidate.resolve()
        if not _inside(resolved, self.input_roots):
            raise SecurityError(
                f"input {resolved} is outside the allowed input directories",
                hint="allowed: " + ", ".join(str(r) for r in self.input_roots)
                + " (configure mcp.allowed_input_dirs)",
            )
        if not resolved.is_file():
            raise InvalidRequestError(f"input file not found: {resolved}")
        if resolved.stat().st_size > self.max_input_bytes:
            raise SecurityError(f"input exceeds {self.max_input_bytes // 1024**2} MB limit")
        return resolved

    def check_model(self, model: str | None, *, kind: str | None = None) -> ModelSpec | None:
        if model is None or model == "auto":
            return None
        spec = get_model(model)
        if kind and spec.kind != kind:
            raise InvalidRequestError(f"{spec.id} is a {spec.kind} model")
        if self.allowed_models and spec.id not in self.allowed_models:
            raise SecurityError(f"model {spec.id} is not in mcp.allowed_models")
        return spec

    def check_duration(self, seconds: float | None) -> None:
        if seconds is not None and seconds > self.max_duration:
            raise SecurityError(f"duration {seconds:g}s exceeds mcp.max_duration_seconds "
                                f"({self.max_duration:g}s)")


def _redact(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 300:
        return value[:300] + f"… ({len(value)} chars)"
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value[:20]]
    return value


def audit(cfg: MusicConfig, tool: str, arguments: dict[str, Any], outcome: str,
          detail: Any = None) -> None:
    if not cfg["logging"]["audit_log"]:
        return
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pid": os.getpid(), "tool": tool,
              "arguments": _redact(arguments), "outcome": outcome}
    if detail is not None:
        record["detail"] = _redact(detail)
    path = paths.ensure(paths.logs_dir()) / "mcp-audit.jsonl"
    with _AUDIT_LOCK, path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
