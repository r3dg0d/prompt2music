"""Sidecar / PNG metadata helpers (no secrets)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_sidecar(path: str | Path, meta: dict[str, Any]) -> Path:
    path = Path(path)
    side = path.with_suffix(path.suffix + ".json")
    side.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return side


def read_sidecar(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    side = path.with_suffix(path.suffix + ".json")
    if not side.is_file():
        return None
    return json.loads(side.read_text(encoding="utf-8"))


def embed_png_text(path: str | Path, meta: dict[str, Any]) -> None:
    """Best-effort tEXt chunks via Pillow; no-op if Pillow missing or not PNG."""
    path = Path(path)
    if path.suffix.lower() != ".png":
        return
    try:
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
    except ImportError:
        return
    info = PngInfo()
    for k, v in meta.items():
        info.add_text(str(k)[:79], json.dumps(v) if not isinstance(v, str) else v)
    with Image.open(path) as im:
        im.save(path, pnginfo=info)
