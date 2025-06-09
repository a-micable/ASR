"""Centralized structured logging configuration."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter for production environments."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)  # type: ignore[arg-type]
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    structured: bool = True,
) -> logging.Logger:
    """
    Configure application-wide logging.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to write logs to disk.
        structured: Use JSON structured formatting when True.

    Returns:
        Root logger instance.
    """
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if structured:
        formatter: logging.Formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
