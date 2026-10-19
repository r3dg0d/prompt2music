"""SQLite storage shared by the CLI, job daemon and MCP server (WAL mode)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prompt2music import paths

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'cli',
    model TEXT,
    prompt TEXT,
    params TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    progress REAL,
    message TEXT,
    result TEXT,
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    daemon_pid INTEGER,
    worker_pid INTEGER,
    gpu_peak_mb REAL,
    generation_seconds REAL,
    log_path TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created_at);
CREATE TABLE IF NOT EXISTS tracks (
    id TEXT PRIMARY KEY,
    job_id TEXT,
    path TEXT NOT NULL,
    prompt TEXT,
    model TEXT,
    model_revision TEXT,
    task TEXT,
    params TEXT,
    duration REAL,
    sample_rate INTEGER,
    channels INTEGER,
    seed INTEGER,
    format TEXT,
    created_at TEXT NOT NULL,
    sidecar TEXT
);
CREATE INDEX IF NOT EXISTS tracks_created ON tracks(created_at);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: Path | None = None) -> sqlite3.Connection:
    db_path = path or paths.db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None,
                           check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < SCHEMA_VERSION:
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    return conn


def dumps(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def loads(value: str | None) -> Any:
    return json.loads(value) if value else None
