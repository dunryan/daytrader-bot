"""Centralized logging configuration.

Provides a console handler plus a rotating file handler under ``logs/``.
Call :func:`setup_logging` once at process startup; use :func:`get_logger`
(a thin ``logging.getLogger`` wrapper) everywhere else.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_CONFIGURED = False

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_dir: Path | str = "logs",
    log_file: str = "daytrader.log",
) -> None:
    """Configure root logging with console + rotating file handlers.

    Idempotent: safe to call multiple times (only configures once).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Quiet noisy third-party loggers.
    for noisy in ("urllib3", "matplotlib", "apscheduler.executors"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (configures logging with defaults if needed)."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
