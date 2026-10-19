"""Backend registry: maps backend names to classes, imported lazily.

Importing a backend module never imports its heavy dependencies; those load
inside ``MusicBackend.load`` in the worker process.
"""

from __future__ import annotations

import importlib

from prompt2music.backends.base import MusicBackend
from prompt2music.errors import BackendNotReadyError

BACKEND_CLASSES: dict[str, str] = {
    "tone": "prompt2music.backends.tone:ToneBackend",
    "musicgen": "prompt2music.backends.musicgen:MusicGenBackend",
    "ace-step": "prompt2music.backends.ace_step:AceStepBackend",
    "stable-audio": "prompt2music.backends.stable_audio:StableAudioBackend",
    "heartmula": "prompt2music.backends.heartmula:HeartMuLaBackend",
    "yue": "prompt2music.backends.yue:YuEBackend",
    "demucs": "prompt2music.backends.demucs_stems:DemucsBackend",
}


def load_backend_class(name: str) -> type[MusicBackend]:
    target = BACKEND_CLASSES.get(name)
    if target is None:
        raise BackendNotReadyError(f"unknown backend {name!r}")
    module_name, _, cls_name = target.partition(":")
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    assert issubclass(cls, MusicBackend)
    return cls
