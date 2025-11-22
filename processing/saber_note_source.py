"""
Saber Note Document Source

Handles encrypted Saber notes from WebDAV sync.
"""

from pathlib import Path
from typing import List, Optional
import logging
import tempfile
import struct

from processing.document_source import DocumentSource, ProcessedDocument
from processing.saber_decryptor import SaberDecryptor
from processing.saber_processor import SaberProcessor
from processing.saber_renderer import SaberRenderer

logger = logging.getLogger(__name__)


class SaberNoteSource(DocumentSource):
    """Document source for Saber encrypted notes."""

    def __init__(
        self,
        watch_directory: Path,
        encryption_password: str,
        enabled: bool = True,
        temp_dir: Optional[Path] = None
    ):
        """
        Initialize Saber note source.

        Args:
            watch_directory: Directory containing Saber sync folder
            encryption_password: Saber encryption password
            enabled: Whether this source is enabled
            temp_dir: Directory for temporary rendered images (defaults to system temp)
        """
        self.encryption_password = encryption_password
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "saber_rendered"

        super().__init__("saber_note", watch_directory, enabled)

        # Initialize components if enabled
        self.decryptor = None
        self.renderer = None

        if self.enabled:
            self.decryptor = SaberDecryptor(encryption_password, watch_directory)
            self.renderer = SaberRenderer()
            self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _validate_config(self) -> None:
        """Validate Saber source configuration."""
        if not self.watch_directory.exists():
            raise ValueError(f"Saber folder does not exist: {self.watch_directory}")

        # Check for config.sbc
        config_path = self.watch_directory / "Saber" / "config.sbc"
        if not config_path.exists():
            raise ValueError(
                f"Saber config.sbc not found at {config_path}. "
                "Ensure Saber has synced at least once before enabling."
            )

        if not self.encryption_password:
            raise ValueError("Saber encryption password is required but not set")

    def get_file_extensions(self) -> List[str]:
        """Get file extensions for Saber encrypted notes."""
        return ['.sbe']

    def can_process_file(self, file_path: Path) -> bool:
        """Check if file is a Saber encrypted note."""
        return (
            file_path.suffix.lower() == '.sbe' and
            file_path.exists() and
            'Saber' in file_path.parts  # Ensure it's in a Saber folder
        )

    def process_file(self, file_path: Path) -> Optional[ProcessedDocument]:
        """
        Process a Saber encrypted note.

        Args:
            file_path: Path to .sbe file

        Returns:
            ProcessedDocument if successful, None if processing failed
        """
        if not self.enabled:
            logger.warning("SaberNoteSource is disabled, cannot process file")
            return None

        if not self.can_process_file(file_path):
            logger.warning(f"Cannot process file: {file_path}")
            return None

        try:
            # Decrypt the filename to get original note name
            encrypted_filename = file_path.stem  # Remove .sbe extension
            decrypted_path = self.decryptor.decrypt_filename(encrypted_filename)
            title = Path(decrypted_path).stem  # Remove .sbn2 extension

            logger.info(f"Processing Saber note: {encrypted_filename} -> {decrypted_path}")

            # Decrypt the file content
            decrypted_data = self.decryptor.decrypt_file(file_path)

            # Parse BSON structure
            note = SaberProcessor.parse_note(decrypted_data)

            # Create output directory for this note's rendered pages
            note_output_dir = self.temp_dir / encrypted_filename
            note_output_dir.mkdir(parents=True, exist_ok=True)

            # Render all pages to images
            page_images = self.renderer.render_note(note, note_output_dir, prefix="page")

            # Extract metadata
            metadata = note.get_metadata()
            metadata.update({
                'encrypted_filename': encrypted_filename,
                'decrypted_path': decrypted_path,
                'file_size': file_path.stat().st_size,
                'modification_time': file_path.stat().st_mtime,
            })

            logger.info(f"Successfully processed Saber note: {title} "
                       f"({note.get_metadata()['num_pages']} pages, "
                       f"{note.get_metadata()['total_strokes']} strokes)")

            return ProcessedDocument(
                source_type='saber_note',
                original_path=str(file_path),
                title=title,
                page_images=page_images,
                metadata=metadata,
                encrypted_filename=encrypted_filename,
                num_pages=len(note.pages),
            )

        except Exception as e:
            logger.error(f"Failed to process Saber note {file_path}: {e}", exc_info=True)
            return None

    def cleanup_temp_files(self, encrypted_filename: str) -> None:
        """
        Clean up temporary rendered files for a note.

        Args:
            encrypted_filename: Encrypted filename (used as directory name)
        """
        note_dir = self.temp_dir / encrypted_filename
        if note_dir.exists():
            import shutil
            shutil.rmtree(note_dir)
            logger.debug(f"Cleaned up temp files for {encrypted_filename}")
