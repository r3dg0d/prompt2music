"""Local music library: SQLite index plus a portable JSON sidecar per track.

The sidecar (``song.wav.json``) carries everything needed to reproduce a
generation with the same model revision and seed. Reproduction is best
effort: bit-identical output is not guaranteed across GPUs, drivers or
library versions.
"""

from __future__ import annotations

import builtins
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from prompt2music import __version__, db
from prompt2music._shared.metadata import read_sidecar, write_sidecar
from prompt2music.errors import InvalidRequestError, MusicError


class Library:
    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self.conn = conn or db.connect()

    def add(self, *, path: Path, prompt: str, model: str, model_revision: str | None,
            task: str, params: dict[str, Any], duration: float, sample_rate: int,
            channels: int, seed: int | None, job_id: str | None = None,
            generation: dict[str, Any] | None = None,
            post_processing: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        track_id = uuid.uuid4().hex[:10]
        created = db.now()
        sidecar = {
            "schema": "prompt2music.track/1",
            "track_id": track_id,
            "job_id": job_id,
            "created_at": created,
            "prompt": prompt,
            "model": model,
            "model_revision": model_revision,
            "task": task,
            "seed": seed,
            "duration": round(duration, 3),
            "sample_rate": sample_rate,
            "channels": channels,
            "format": path.suffix.lstrip("."),
            "parameters": params,
            "generation": generation or {},
            "post_processing": post_processing or [],
            "app": {"name": "prompt2music", "package": "ai-media-cli", "version": __version__},
            "reproducibility": "same model revision + seed + parameters; not guaranteed "
                               "bit-identical across hardware or library versions",
        }
        side = write_sidecar(path, sidecar)
        self.conn.execute(
            "INSERT INTO tracks (id, job_id, path, prompt, model, model_revision, task, params, "
            "duration, sample_rate, channels, seed, format, created_at, sidecar) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (track_id, job_id, str(path), prompt, model, model_revision, task, db.dumps(params),
             duration, sample_rate, channels, seed, path.suffix.lstrip("."), created, str(side)),
        )
        return {"track_id": track_id, "sidecar": str(side), **{"path": str(path)}}

    def _row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["params"] = db.loads(data["params"]) or {}
        data["exists"] = Path(data["path"]).is_file()
        return data

    def list(self, *, limit: int = 50) -> builtins.list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM tracks ORDER BY created_at DESC, rowid DESC "
                                 "LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in rows]

    def search(self, query: str, *, limit: int = 50) -> builtins.list[dict[str, Any]]:
        like = f"%{query}%"
        rows = self.conn.execute(
            "SELECT * FROM tracks WHERE prompt LIKE ? OR model LIKE ? OR id LIKE ? OR "
            "path LIKE ? ORDER BY created_at DESC LIMIT ?", (like, like, like, like, limit),
        ).fetchall()
        return [self._row(r) for r in rows]

    def get(self, track_id: str) -> dict[str, Any]:
        rows = self.conn.execute("SELECT * FROM tracks WHERE id LIKE ? || '%'",
                                 (track_id,)).fetchall()
        if len(rows) != 1:
            raise MusicError(f"track not found: {track_id}")
        data = self._row(rows[0])
        data["sidecar_data"] = read_sidecar(data["path"]) if data["exists"] else None
        return data

    def export(self, track_id: str, dest: Path, *, fmt: str | None = None,
               overwrite: bool = False) -> dict[str, Any]:
        track = self.get(track_id)
        src = Path(track["path"])
        if not src.is_file():
            raise InvalidRequestError(f"audio file is missing: {src}")
        if dest.is_dir() or not dest.suffix:
            dest.mkdir(parents=True, exist_ok=True)
            ext = f".{fmt}" if fmt else src.suffix
            dest = dest / f"{src.stem}{ext}"
        if dest.exists() and not overwrite:
            raise InvalidRequestError(f"{dest} exists (use --force)")
        if dest.suffix.lower() == src.suffix.lower():
            shutil.copy2(src, dest)
        else:
            from prompt2music import audio

            audio.convert(src, dest, overwrite=overwrite,
                          metadata={"comment": f"prompt2music {track['id']}: {track['prompt']}"})
        side = Path(str(src) + ".json")
        if side.is_file():
            shutil.copy2(side, Path(str(dest) + ".json"))
        return {"track_id": track["id"], "exported": str(dest)}

    def reproduce_params(self, track_id: str) -> dict[str, Any]:
        track = self.get(track_id)
        params = dict(track["params"])
        params.update({"model": track["model"], "seed": track["seed"]})
        return params

    def remove(self, track_id: str, *, delete_files: bool = False) -> dict[str, Any]:
        track = self.get(track_id)
        self.conn.execute("DELETE FROM tracks WHERE id = ?", (track["id"],))
        if delete_files:
            Path(track["path"]).unlink(missing_ok=True)
            Path(str(track["path"]) + ".json").unlink(missing_ok=True)
        return {"removed": track["id"], "files_deleted": delete_files}
