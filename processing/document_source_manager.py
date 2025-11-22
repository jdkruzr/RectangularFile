"""
Document Source Manager

Coordinates multiple document sources and integrates with FileWatcher.
"""

from pathlib import Path
from typing import List, Dict, Optional, Callable
import logging

from processing.document_source import DocumentSource, ProcessedDocument
from processing.file_watcher import FileWatcher

logger = logging.getLogger(__name__)


class DocumentSourceManager:
    """Manages multiple document sources and coordinates file watching."""

    def __init__(self, polling_interval: float = 30.0):
        """
        Initialize document source manager.

        Args:
            polling_interval: Polling interval for file watchers (seconds)
        """
        self.sources: List[DocumentSource] = []
        self.file_watchers: Dict[str, FileWatcher] = {}
        self.polling_interval = polling_interval
        self.process_callback: Optional[Callable[[ProcessedDocument], None]] = None

    def register_source(self, source: DocumentSource) -> None:
        """
        Register a document source.

        Args:
            source: DocumentSource to register
        """
        if source in self.sources:
            logger.warning(f"Source {source.source_name} already registered")
            return

        self.sources.append(source)
        logger.info(f"Registered document source: {source.source_name} "
                   f"(enabled={source.is_enabled()})")

        # Create file watcher for this source if enabled
        if source.is_enabled():
            self._create_watcher_for_source(source)

    def _create_watcher_for_source(self, source: DocumentSource) -> None:
        """Create a FileWatcher for a document source."""
        watch_dir = str(source.get_watch_directory())
        extensions = source.get_file_extensions()

        logger.info(f"Creating file watcher for {source.source_name}: "
                   f"directory={watch_dir}, extensions={extensions}")

        watcher = FileWatcher(
            directory_path=watch_dir,
            file_types=extensions,
            polling_interval=self.polling_interval,
            recursive=True
        )

        # Register callback to process files from this source
        def on_file_detected(rel_path: str):
            file_path = Path(watch_dir) / rel_path
            logger.info(f"New file detected by {source.source_name}: {rel_path}")

            # Process the file
            processed = source.process_file(file_path)

            if processed and self.process_callback:
                self.process_callback(processed)

        watcher.register_callback(on_file_detected)
        self.file_watchers[source.source_name] = watcher

    def set_process_callback(self, callback: Callable[[ProcessedDocument], None]) -> None:
        """
        Set callback to handle processed documents.

        Args:
            callback: Function that receives ProcessedDocument objects
        """
        self.process_callback = callback

    def start_all(self, trigger_initial_scan: bool = True) -> None:
        """
        Start all file watchers.

        Args:
            trigger_initial_scan: Whether to process existing files on startup
        """
        for source_name, watcher in self.file_watchers.items():
            logger.info(f"Starting file watcher for {source_name}")
            watcher.start(trigger_initial_scan=trigger_initial_scan)

    def stop_all(self) -> None:
        """Stop all file watchers."""
        for source_name, watcher in self.file_watchers.items():
            logger.info(f"Stopping file watcher for {source_name}")
            watcher.stop()

    def get_enabled_sources(self) -> List[DocumentSource]:
        """Get list of enabled document sources."""
        return [s for s in self.sources if s.is_enabled()]

    def get_source_by_name(self, name: str) -> Optional[DocumentSource]:
        """
        Get a document source by name.

        Args:
            name: Source name to find

        Returns:
            DocumentSource if found, None otherwise
        """
        for source in self.sources:
            if source.source_name == name:
                return source
        return None

    def process_file_manually(self, file_path: Path) -> Optional[ProcessedDocument]:
        """
        Manually process a file using the appropriate source.

        Args:
            file_path: Path to file to process

        Returns:
            ProcessedDocument if a source could process it, None otherwise
        """
        for source in self.sources:
            if source.is_enabled() and source.can_process_file(file_path):
                logger.info(f"Processing {file_path} with {source.source_name}")
                return source.process_file(file_path)

        logger.warning(f"No enabled source found to process {file_path}")
        return None
