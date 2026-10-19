"""Installed-model manifests and integrity verification.

A model directory counts as installed only when ``manifest.json`` exists in
it. The downloader writes the manifest last, inside a staging directory, and
then renames the whole directory into place, so an interrupted download never
looks installed.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
_SKIP_DIRS = {".cache", "blobs", "refs", ".no_exist", "locks"}
_CHUNK = 8 * 1024 * 1024


@dataclass
class FileEntry:
    size: int
    sha256: str | None = None
    git_sha1: str | None = None  # Hub blob id for non-LFS files


@dataclass
class Manifest:
    model_id: str
    sources: list[dict[str, Any]]
    files: dict[str, FileEntry]
    license: str
    installed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    license_accepted_at: str | None = None
    trusted_repo_code: bool = False
    app_version: str = ""
    version: int = MANIFEST_VERSION

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files.values())

    @property
    def revisions(self) -> dict[str, str]:
        return {s["repo_id"]: s.get("revision_resolved", "") for s in self.sources}

    def to_json(self) -> str:
        data = asdict(self)
        return json.dumps(data, indent=2, sort_keys=True) + "\n"

    def write(self, model_dir: Path) -> Path:
        path = model_dir / MANIFEST_NAME
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        os.replace(tmp, path)
        return path

    @classmethod
    def read(cls, model_dir: Path) -> Manifest | None:
        path = model_dir / MANIFEST_NAME
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            files = {k: FileEntry(**v) for k, v in data.pop("files", {}).items()}
            known = set(cls.__dataclass_fields__)
            return cls(files=files, **{k: v for k, v in data.items() if k in known})
        except (OSError, ValueError, TypeError):
            return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """Git object id of a file, which is what the Hub reports for non-LFS files."""
    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def iter_model_files(model_dir: Path) -> Iterator[Path]:
    """Payload files, skipping Hub bookkeeping and our own manifest."""
    for root, dirs, files in os.walk(model_dir):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".staging"))
        for name in sorted(files):
            if name in (MANIFEST_NAME, MANIFEST_NAME + ".tmp") or name.endswith(".incomplete"):
                continue
            yield Path(root) / name


@dataclass
class VerifyReport:
    model_id: str
    ok: bool
    checked: int
    problems: list[str]
    full: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_model_dir(model_dir: Path, manifest: Manifest, *, full: bool = True) -> VerifyReport:
    """Check presence and size (always) and content hashes (``full``)."""
    problems: list[str] = []
    checked = 0
    for rel, entry in sorted(manifest.files.items()):
        path = model_dir / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
            continue
        size = path.stat().st_size
        if size != entry.size:
            problems.append(f"size mismatch: {rel} ({size} != {entry.size})")
            continue
        if full:
            if entry.sha256 and sha256_file(path) != entry.sha256:
                problems.append(f"sha256 mismatch: {rel}")
            elif not entry.sha256 and entry.git_sha1 and git_blob_sha1(path) != entry.git_sha1:
                problems.append(f"git blob hash mismatch: {rel}")
        checked += 1
    return VerifyReport(manifest.model_id, not problems, checked, problems, full)


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            p = Path(root) / name
            try:
                if not p.is_symlink():
                    total += p.stat().st_size
            except OSError:
                continue
    return total
