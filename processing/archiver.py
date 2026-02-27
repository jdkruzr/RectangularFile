"""
Document Archiver

Moves processed documents from the watch/export directory to an archive location.
This enables detection of re-uploaded documents (if a file appears in the export
directory again, it must be new content).
"""
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ArchiveError(Exception):
    """Base exception for archive operations."""
    pass


class DocumentArchiver:
    """
    Handles archiving of processed documents.

    After a document is successfully processed, it gets moved from the
    watch directory to the archive directory. This keeps the watch directory
    clean and allows detection of re-uploaded documents.
    """

    def __init__(
        self,
        archive_root: Path,
        preserve_structure: bool = True,
        enabled: bool = True,
    ):
        """
        Initialize the archiver.

        Args:
            archive_root: Root directory for archived documents
            preserve_structure: If True, maintain the relative path structure
                               from the source directory in the archive
            enabled: If False, archiving is skipped (documents stay in place)
        """
        self.archive_root = Path(archive_root)
        self.preserve_structure = preserve_structure
        self.enabled = enabled

        if self.enabled:
            self.archive_root.mkdir(parents=True, exist_ok=True)
            logger.info(f"DocumentArchiver initialized with archive root: {self.archive_root}")
        else:
            logger.info("DocumentArchiver disabled - documents will remain in place")

    def compute_archive_path(
        self,
        source_path: Path,
        base_watch_dir: Path,
    ) -> Path:
        """
        Determine where a document should be archived.

        Args:
            source_path: Current path of the document
            base_watch_dir: The root watch directory this document came from

        Returns:
            The target path in the archive
        """
        source_path = Path(source_path)
        base_watch_dir = Path(base_watch_dir)

        if self.preserve_structure:
            # Maintain relative path structure
            try:
                relative_path = source_path.relative_to(base_watch_dir)
            except ValueError:
                # source_path is not under base_watch_dir, use just the filename
                relative_path = source_path.name

            return self.archive_root / relative_path
        else:
            # Flat structure - just use the filename
            return self.archive_root / source_path.name

    def _resolve_collision(self, target_path: Path) -> Path:
        """
        Handle the case where the target archive path already exists.

        Appends a timestamp or incrementing number to make the path unique.

        Args:
            target_path: The desired target path that already exists

        Returns:
            A unique path that doesn't exist
        """
        if not target_path.exists():
            return target_path

        stem = target_path.stem
        suffix = target_path.suffix
        parent = target_path.parent

        # Try with timestamp first
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = parent / f"{stem}_{timestamp}{suffix}"
        if not candidate.exists():
            return candidate

        # Fall back to incrementing number
        counter = 1
        while True:
            candidate = parent / f"{stem}_{timestamp}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1
            if counter > 1000:
                raise ArchiveError(f"Could not find unique archive path for {target_path}")

    def archive_document(
        self,
        source_path: Path,
        base_watch_dir: Path,
    ) -> Optional[Path]:
        """
        Move a document to the archive.

        Args:
            source_path: Current path of the document
            base_watch_dir: The root watch directory this document came from

        Returns:
            The new path in the archive, or None if archiving is disabled

        Raises:
            ArchiveError: If the archive operation fails
        """
        if not self.enabled:
            logger.debug(f"Archiving disabled, skipping: {source_path}")
            return None

        source_path = Path(source_path)

        if not source_path.exists():
            logger.warning(f"Source file does not exist, cannot archive: {source_path}")
            return None

        target_path = self.compute_archive_path(source_path, base_watch_dir)
        target_path = self._resolve_collision(target_path)

        try:
            # Ensure target directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Move the file
            shutil.move(str(source_path), str(target_path))

            logger.info(f"Archived document: {source_path} -> {target_path}")
            return target_path

        except PermissionError as e:
            raise ArchiveError(f"Permission denied archiving {source_path}: {e}")
        except OSError as e:
            raise ArchiveError(f"OS error archiving {source_path}: {e}")
        except Exception as e:
            raise ArchiveError(f"Unexpected error archiving {source_path}: {e}")

    def get_archive_stats(self) -> dict:
        """
        Get statistics about the archive.

        Returns:
            Dictionary with archive statistics
        """
        if not self.enabled or not self.archive_root.exists():
            return {
                "enabled": self.enabled,
                "total_files": 0,
                "total_size_bytes": 0,
            }

        total_files = 0
        total_size = 0

        for path in self.archive_root.rglob("*"):
            if path.is_file():
                total_files += 1
                total_size += path.stat().st_size

        return {
            "enabled": self.enabled,
            "archive_root": str(self.archive_root),
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }
