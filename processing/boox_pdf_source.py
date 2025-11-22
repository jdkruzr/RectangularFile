"""
Boox PDF Document Source

Handles PDF files exported from Boox e-ink tablets.
"""

from pathlib import Path
from typing import List, Optional
import logging

from processing.document_source import DocumentSource, ProcessedDocument
from processing.pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)


class BooxPDFSource(DocumentSource):
    """Document source for Boox PDF files."""

    def __init__(self, watch_directory: Path, enabled: bool = True):
        """
        Initialize Boox PDF source.

        Args:
            watch_directory: Directory containing exported PDFs
            enabled: Whether this source is enabled
        """
        super().__init__("boox_pdf", watch_directory, enabled)
        self.pdf_processor = PDFProcessor() if enabled else None

    def _validate_config(self) -> None:
        """Validate Boox PDF source configuration."""
        if not self.watch_directory.exists():
            logger.warning(f"Boox folder does not exist: {self.watch_directory}")
            self.watch_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created Boox folder: {self.watch_directory}")

    def get_file_extensions(self) -> List[str]:
        """Get file extensions for PDF files."""
        return ['.pdf']

    def can_process_file(self, file_path: Path) -> bool:
        """Check if file is a PDF."""
        return file_path.suffix.lower() == '.pdf' and file_path.exists()

    def process_file(self, file_path: Path) -> Optional[ProcessedDocument]:
        """
        Process a Boox PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            ProcessedDocument if successful, None if processing failed
        """
        if not self.enabled:
            logger.warning("BooxPDFSource is disabled, cannot process file")
            return None

        if not self.can_process_file(file_path):
            logger.warning(f"Cannot process file: {file_path}")
            return None

        try:
            # For PDFs, the file itself serves as the "page image"
            # The existing OCR pipeline will convert PDF pages to images
            # We don't need to pre-render them here

            # Extract basic metadata
            metadata = {
                'file_size': file_path.stat().st_size,
                'modification_time': file_path.stat().st_mtime,
            }

            # Check if PDF has text content
            if self.pdf_processor:
                has_text = self.pdf_processor.check_text_presence(file_path)
                metadata['has_embedded_text'] = has_text

            return ProcessedDocument(
                source_type='boox_pdf',
                original_path=str(file_path),
                title=file_path.stem,  # Use filename without extension as title
                page_images=[file_path],  # PDF processor will handle page extraction
                metadata=metadata,
                num_pages=0,  # Will be determined during PDF processing
            )

        except Exception as e:
            logger.error(f"Failed to process PDF {file_path}: {e}", exc_info=True)
            return None
