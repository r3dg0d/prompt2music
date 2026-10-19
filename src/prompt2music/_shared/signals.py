"""Ctrl+C clean shutdown helpers."""

from __future__ import annotations

import signal
import sys
import threading
from collections.abc import Callable
from functools import wraps

_lock = threading.Lock()
_shutdown = False
_callbacks: list[Callable[[], None]] = []


def is_shutdown_requested() -> bool:
    return _shutdown


def on_shutdown(cb: Callable[[], None]) -> None:
    with _lock:
        _callbacks.append(cb)


def request_shutdown(signum: int | None = None, frame: object = None) -> None:
    global _shutdown
    with _lock:
        if _shutdown:
            return
        _shutdown = True
        cbs = list(_callbacks)
    for cb in cbs:
        try:
            cb()
        except Exception:
            pass
    # Soft exit message to stderr
    try:
        sys.stderr.write("\nInterrupted — shutting down cleanly.\n")
        sys.stderr.flush()
    except Exception:
        pass


def _handle_signal(signum: int, frame: object) -> None:
    # Keep locks and callbacks outside the signal handler. Unwind through the
    # workflow cleanup first; the CLI wrapper performs shutdown and exits 130.
    raise KeyboardInterrupt


def interruptible_cli(main: Callable[..., None]) -> Callable[..., None]:
    @wraps(main)
    def wrapped(*args: object, **kwargs: object) -> None:
        try:
            main(*args, **kwargs)
        except KeyboardInterrupt:
            try:
                request_shutdown()
            finally:
                raise SystemExit(130) from None

    return wrapped


def install_sigint_handler() -> None:
    """Install SIGINT handler for clean Ctrl+C."""
    signal.signal(signal.SIGINT, _handle_signal)
    # Also handle SIGTERM when available
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _handle_signal)
        except (ValueError, OSError):
            pass


def reset_for_tests() -> None:
    global _shutdown
    with _lock:
        _shutdown = False
        _callbacks.clear()
