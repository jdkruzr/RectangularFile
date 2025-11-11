"""Structured logging utility for better log parsing in concurrent operations."""

import logging
from typing import Optional


class StructuredLogger:
    """
    Wrapper around standard logger to add structured prefixes for easier parsing.

    Prefixes:
    - [STARTUP] - Application initialization
    - [FILE-SCAN] - File watcher operations
    - [OCR-QUEUE] - OCR queue management
    - [PDF-PROC] - PDF processing
    - [AI-MODEL] - AI model operations (loading, inference)
    - [DB] - Database operations
    - [CALDAV] - CalDAV integration
    - [CV-DETECT] - Computer vision annotation detection
    """

    def __init__(self, logger: logging.Logger, prefix: str):
        """
        Initialize structured logger.

        Args:
            logger: Standard Python logger instance
            prefix: Prefix to add to all log messages (e.g., "OCR-QUEUE")
        """
        self.logger = logger
        self.prefix = f"[{prefix}]"

    def _format_message(self, msg: str, doc_id: Optional[int] = None) -> str:
        """Format message with prefix and optional document ID."""
        if doc_id is not None:
            return f"{self.prefix} [Doc {doc_id}] {msg}"
        return f"{self.prefix} {msg}"

    def debug(self, msg: str, doc_id: Optional[int] = None):
        """Log debug message."""
        self.logger.debug(self._format_message(msg, doc_id))

    def info(self, msg: str, doc_id: Optional[int] = None):
        """Log info message."""
        self.logger.info(self._format_message(msg, doc_id))

    def warning(self, msg: str, doc_id: Optional[int] = None):
        """Log warning message."""
        self.logger.warning(self._format_message(msg, doc_id))

    def error(self, msg: str, doc_id: Optional[int] = None):
        """Log error message."""
        self.logger.error(self._format_message(msg, doc_id))

    def critical(self, msg: str, doc_id: Optional[int] = None):
        """Log critical message."""
        self.logger.critical(self._format_message(msg, doc_id))

    # Convenience methods for operations with document context
    def start_operation(self, operation: str, doc_id: int, filename: str = ""):
        """Log start of an operation on a document."""
        file_part = f" ({filename})" if filename else ""
        self.info(f"▶ Started {operation}{file_part}", doc_id)

    def complete_operation(self, operation: str, doc_id: int, details: str = ""):
        """Log completion of an operation on a document."""
        detail_part = f" - {details}" if details else ""
        self.info(f"✓ Completed {operation}{detail_part}", doc_id)

    def fail_operation(self, operation: str, doc_id: int, error: str = ""):
        """Log failure of an operation on a document."""
        error_part = f": {error}" if error else ""
        self.error(f"✗ Failed {operation}{error_part}", doc_id)
