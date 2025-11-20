"""Simple logging helpers used across the project."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

_LOGGER_CONFIGURED = False
_LOG_PATH: Optional[Path] = None


def _resolve_level(level_name: str | None) -> int:
    """Return a logging level integer handling invalid names gracefully."""
    if not level_name:
        return logging.INFO

    level_name = level_name.upper()
    return getattr(logging, level_name, logging.INFO)


def configure_logging(
    *,
    level: str | None = None,
    log_dir: Optional[Path] = None,
) -> None:
    """Configure the root logger once for the entire application."""
    global _LOGGER_CONFIGURED, _LOG_PATH
    if _LOGGER_CONFIGURED:
        return

    resolved_level = _resolve_level(level or os.getenv("LOG_LEVEL"))
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        _LOG_PATH = log_dir / "calexa.log"
        file_handler = logging.FileHandler(_LOG_PATH, encoding="utf-8")
        handlers.append(file_handler)

    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

    _LOGGER_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger ensuring global configuration happens once."""
    if not _LOGGER_CONFIGURED:
        configure_logging()
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger", "_LOG_PATH"]
