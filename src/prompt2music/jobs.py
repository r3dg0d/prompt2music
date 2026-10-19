"""Persistent job queue (SQLite) used by the daemon, CLI and MCP server.

States: queued → loading → running → processing → completed | failed | cancelled.
"""

from __future__ import annotations

import builtins
import os
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt2music import db, paths
from prompt2music.errors import JobNotFoundError, ResourceLimitError

STATES = ("queued", "loading", "running", "processing", "completed", "failed", "cancelled")
ACTIVE = ("queued", "loading", "running", "processing")
FINAL = ("completed", "failed", "cancelled")
KINDS = ("generate", "separate", "load", "unload", "benchmark", "download")


@dataclass
class JobRecord:
    id: str
    kind: str
    status: str
    source: str
    model: str | None
    prompt: str | None
    params: dict[str, Any]
    created_at: str
    started_at: str | None
    finished_at: str | None
    progress: float | None
    message: str | None
    result: dict[str, Any] | None
    error: dict[str, Any] | None
    cancel_requested: bool
    daemon_pid: int | None
    worker_pid: int | None
    gpu_peak_mb: float | None
    generation_seconds: float | None
    log_path: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> JobRecord:
        data = dict(row)
        data["params"] = db.loads(data["params"]) or {}
        data["result"] = db.loads(data["result"])
        data["error"] = db.loads(data["error"])
        data["cancel_requested"] = bool(data["cancel_requested"])
        return cls(**data)

    @property
    def final(self) -> bool:
        return self.status in FINAL

    def as_dict(self, *, brief: bool = False) -> dict[str, Any]:
        data = dict(self.__dict__)
        if brief:
            for key in ("params", "result", "log_path"):
                data.pop(key, None)
            outputs = (self.result or {}).get("outputs") or []
            data["outputs"] = [o.get("path") for o in outputs]
        return data


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class JobStore:
    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self.conn = conn or db.connect()

    def create(self, kind: str, params: dict[str, Any], *, model: str | None = None,
               prompt: str | None = None, source: str = "cli", status: str = "queued",
               max_queued: int | None = None) -> str:
        if kind not in KINDS:
            raise ValueError(f"unknown job kind {kind}")
        if max_queued is not None and self.count_active(source=None) >= max_queued:
            raise ResourceLimitError(f"job queue is full ({max_queued} active jobs)",
                                     hint="wait for jobs to finish or cancel some")
        job_id = uuid.uuid4().hex[:12]
        log_path = str(paths.logs_dir() / "jobs" / f"{job_id}.log")
        self.conn.execute(
            "INSERT INTO jobs (id, kind, status, source, model, prompt, params, created_at, "
            "log_path) VALUES (?,?,?,?,?,?,?,?,?)",
            (job_id, kind, status, source, model, prompt, db.dumps(params), db.now(), log_path),
        )
        return job_id

    def get(self, job_id: str) -> JobRecord:
        row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            row = self.conn.execute("SELECT * FROM jobs WHERE id LIKE ? || '%'",
                                    (job_id,)).fetchall()
            if len(row) != 1:
                raise JobNotFoundError(f"job not found: {job_id}")
            row = row[0]
        return JobRecord.from_row(row)

    def list(self, *, status: str | None = None, limit: int = 50,
             source: str | None = None) -> builtins.list[JobRecord]:
        sql = "SELECT * FROM jobs"
        where: builtins.list[str] = []
        args: builtins.list[Any] = []
        if status == "active":
            where.append(f"status IN ({','.join('?' * len(ACTIVE))})")
            args.extend(ACTIVE)
        elif status:
            where.append("status = ?")
            args.append(status)
        if source:
            where.append("source = ?")
            args.append(source)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args.append(limit)
        return [JobRecord.from_row(r) for r in self.conn.execute(sql, args).fetchall()]

    def count_active(self, *, source: str | None = None) -> int:
        sql = f"SELECT COUNT(*) FROM jobs WHERE status IN ({','.join('?' * len(ACTIVE))})"
        args: list[Any] = list(ACTIVE)
        if source:
            sql += " AND source = ?"
            args.append(source)
        return int(self.conn.execute(sql, args).fetchone()[0])

    def claim_next(self, daemon_pid: int) -> JobRecord | None:
        """Atomically move the oldest queued job to ``loading``."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                "SELECT id FROM jobs WHERE status = 'queued' AND cancel_requested = 0 "
                "ORDER BY created_at, rowid LIMIT 1").fetchone()
            if row is None:
                self.conn.execute("COMMIT")
                return None
            self.conn.execute(
                "UPDATE jobs SET status='loading', started_at=?, daemon_pid=?, progress=0, "
                "message='claimed by daemon' WHERE id=?", (db.now(), daemon_pid, row["id"]))
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return self.get(row["id"])

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        for key in ("params", "result", "error"):
            if key in fields and not isinstance(fields[key], (str, type(None))):
                fields[key] = db.dumps(fields[key])
        if fields.get("status") in FINAL:
            fields.setdefault("finished_at", db.now())
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))

    def progress(self, job_id: str, fraction: float | None, message: str) -> None:
        if fraction is None:
            self.conn.execute("UPDATE jobs SET message=? WHERE id=?", (message, job_id))
        else:
            self.conn.execute("UPDATE jobs SET progress=?, message=? WHERE id=?",
                              (round(float(fraction), 4), message, job_id))

    def cancel_requested(self, job_id: str) -> bool:
        row = self.conn.execute("SELECT cancel_requested FROM jobs WHERE id=?",
                                (job_id,)).fetchone()
        return bool(row and row[0])

    def request_cancel(self, job_id: str) -> JobRecord:
        job = self.get(job_id)
        if job.final:
            return job
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                "UPDATE jobs SET status='cancelled', cancel_requested=1, finished_at=?, "
                "message='cancelled before start' WHERE id=? AND status='queued'",
                (db.now(), job.id))
            self.conn.execute(
                "UPDATE jobs SET cancel_requested=1, message='cancellation requested' "
                "WHERE id=? AND status IN ('loading','running','processing')", (job.id,))
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return self.get(job.id)

    def recover_orphans(self) -> builtins.list[str]:
        """Fail jobs left active by a daemon that no longer exists."""
        recovered = []
        for job in self.list(status="active", limit=1000):
            if job.status == "queued":
                continue
            if not _pid_alive(job.daemon_pid):
                status = "cancelled" if job.cancel_requested else "failed"
                self.update(job.id, status=status,
                            error={"code": "daemon_exited",
                                   "message": "the process running this job exited"})
                recovered.append(job.id)
        return recovered

    def log_path(self, job_id: str) -> Path:
        job = self.get(job_id)
        return Path(job.log_path) if job.log_path else paths.logs_dir() / f"{job.id}.log"

    def prune(self, *, keep: int = 500) -> int:
        cur = self.conn.execute(
            f"DELETE FROM jobs WHERE status IN ({','.join('?' * len(FINAL))}) AND id NOT IN "
            "(SELECT id FROM jobs ORDER BY created_at DESC LIMIT ?)", (*FINAL, keep))
        return cur.rowcount
