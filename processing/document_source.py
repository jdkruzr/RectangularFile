"""
Abstract base class for document sources in RectangularFile.

Each document source (Boox PDFs, Saber notes, etc.) implements this interface
to provide a consistent way to detect, process, and ingest documents.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Result of processing a document from a source."""

    source_type: str  # 'boox_pdf', 'saber_note', 'html', etc.
    original_path: str  # Original file path
    title: str  # Document title
    page_images: List[Path]  # Paths to rendered page images for OCR
    metadata: Dict[str, Any]  # Source-specific metadata

    # Optional fields
    encrypted_filename: Optional[str] = None  # For encrypted sources like Saber
    num_pages: int = 0


class DocumentSource(ABC):
    """
    Abstract base class for document sources.

    Each source watches a directory for specific file types and processes
    them into a common format for OCR and database storage.
    """

    def __init__(self, source_name: str, watch_directory: Path, enabled: bool = True):
        """
        Initialize document source.

        Args:
            source_name: Identifier for this source (e.g., 'boox', 'saber')
            watch_directory: Directory to monitor for files
            enabled: Whether this source is enabled
        """
        self.source_name = source_name
        self.watch_directory = Path(watch_directory)
        self.enabled = enabled
        self.logger = logging.getLogger(f"{__name__}.{source_name}")

        if self.enabled:
            self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate source-specific configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def get_file_extensions(self) -> List[str]:
        """
        Get list of file extensions this source handles.

        Returns:
            List of extensions (e.g., ['.pdf', '.sbe'])
        """
        pass

    @abstractmethod
    def can_process_file(self, file_path: Path) -> bool:
        """
        Check if this source can process a given file.

        Args:
            file_path: Path to file to check

        Returns:
            True if this source can handle the file
        """
        pass

    @abstractmethod
    def process_file(self, file_path: Path) -> Optional[ProcessedDocument]:
        """
        Process a file from this source.

        Args:
            file_path: Path to file to process

        Returns:
            ProcessedDocument if successful, None if processing failed

        Raises:
            Exception: On processing errors
        """
        pass

    def get_watch_directory(self) -> Path:
        """Get the directory this source watches."""
        return self.watch_directory

    def is_enabled(self) -> bool:
        """Check if this source is enabled."""
        return self.enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable this source."""
        was_enabled = self.enabled
        self.enabled = enabled

        if enabled and not was_enabled:
            self.logger.info(f"Document source '{self.source_name}' enabled")
            self._validate_config()
        elif not enabled and was_enabled:
            self.logger.info(f"Document source '{self.source_name}' disabled")
