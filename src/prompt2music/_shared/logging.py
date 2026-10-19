"""Structured-ish logging setup for quiet/verbose/debug."""

from __future__ import annotations

import logging
import sys


def setup_logging(
    *, quiet: bool = False, verbose: bool = False, debug: bool = False
) -> logging.Logger:
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    elif quiet:
        level = logging.ERROR
    else:
        level = logging.WARNING

    logger = logging.getLogger("prompt2music")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
