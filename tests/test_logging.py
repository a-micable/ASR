"""Tests for logging configuration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from logging_config import StructuredFormatter, get_logger, setup_logging


class TestLoggingConfig:
    def test_setup_logging_console(self) -> None:
        logger = setup_logging(level="DEBUG", structured=False)
        assert logger.level == logging.DEBUG

    def test_setup_logging_with_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        setup_logging(level="INFO", log_file=log_file, structured=True)
        test_logger = get_logger("test.module")
        test_logger.info("test message")
        assert log_file.exists()

    def test_structured_formatter_with_exception(self) -> None:
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="error occurred",
            args=(),
            exc_info=None,
        )
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            record.exc_info = sys.exc_info()

        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "ERROR"
        assert "exception" in data

    def test_get_logger_returns_named_logger(self) -> None:
        logger = get_logger("whisper.test")
        assert logger.name == "whisper.test"
