"""Output path helpers and safe write utilities."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def slugify(text: str, *, max_len: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip().lower())
    s = re.sub(r"-+", "-", s).strip("-")
    return (s or "out")[:max_len]


def default_output_path(
    stem: str,
    *,
    directory: str | Path | None = None,
    ext: str = "png",
    seed: int | None = None,
) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    parts = [slugify(stem), ts]
    if seed is not None:
        parts.append(f"s{seed}")
    name = "_".join(parts) + f".{ext.lstrip('.')}"
    base = ensure_dir(directory or Path.cwd() / "outputs")
    return base / name
