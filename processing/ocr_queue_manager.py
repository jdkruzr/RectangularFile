import threading
import queue
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path
from utils.structured_logger import StructuredLogger

class OCRQueueManager:
    """Manages a queue of documents for OCR processing."""

    def __init__(self, db_manager, ocr_processor, archiver=None):
        """
        Initialize the OCR queue manager.

        Args:
            db_manager: Database manager instance
            ocr_processor: OCR processor (VisionAPIClient or similar)
            archiver: Optional DocumentArchiver for post-processing archival
        """
        self.queue = queue.Queue()
        self.processing_thread = None
        self.stop_requested = False
        self.currently_processing = None
        self.db_manager = db_manager
        self.ocr_processor = ocr_processor
        self.archiver = archiver
        base_logger = self._setup_logging()
        self.logger = StructuredLogger(base_logger, "OCR-QUEUE")
        self.is_running = False

    def _setup_logging(self):
        """Configure logging for the queue manager."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def add_to_queue(self, doc_id: int, filepath: Path, base_watch_dir: Path = None) -> bool:
        """
        Add a document to the processing queue.

        Args:
            doc_id: Document ID in the database
            filepath: Path to the document file
            base_watch_dir: Base directory the file came from (for archiving)
        """
        if not filepath.exists():
            self.logger.error(f"File not found: {filepath}", doc_id)
            return False

        self.logger.info(f"Added to queue ({filepath.name})", doc_id)
        self.queue.put({
            'doc_id': doc_id,
            'filepath': filepath,
            'base_watch_dir': base_watch_dir or filepath.parent,
            'queued_at': time.time()
        })

        # Start the processing thread if not already running
        if not self.is_running:
            self.start_processing()

        return True
    
    def start_processing(self):
        """Start the background processing thread."""
        if self.is_running:
            self.logger.info("Processing thread already running")
            return

        self.stop_requested = False
        self.processing_thread = threading.Thread(
            target=self._process_queue_worker,
            daemon=True
        )
        self.processing_thread.start()
        self.is_running = True
        self.logger.info("Started processing thread")
    
    def stop_processing(self):
        """Stop the background processing thread."""
        if not self.is_running:
            return

        self.logger.info("Stopping processing thread...")
        self.stop_requested = True

        if self.processing_thread:
            self.processing_thread.join(timeout=5)

        self.is_running = False
        self.logger.info("Processing thread stopped")
    
    def _process_queue_worker(self):
        """Worker thread that processes the queue items one at a time."""
        self.logger.info("Queue worker started")

        while not self.stop_requested:
            try:
                # Try to get an item from the queue with a timeout
                try:
                    item = self.queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                doc_id = item['doc_id']
                filepath = item['filepath']
                self.currently_processing = item

                # Check if document has been skipped before processing
                doc_info = self.db_manager.get_document_by_id(doc_id)
                if doc_info and doc_info.get('processing_status') == 'skipped':
                    self.logger.info(f"Document {doc_id} is skipped, removing from queue", doc_id)
                    self.queue.task_done()
                    self.currently_processing = None
                    continue

                self.logger.start_operation("OCR processing", doc_id, filepath.name)
                self.db_manager.update_processing_progress(
                    doc_id, 5.0, "Queued for OCR processing"
                )

                # Process the document with OCR
                try:
                    success = self.ocr_processor.process_document(
                        filepath, doc_id, self.db_manager
                    )
                except Exception as proc_error:
                    self.logger.error(f"Processing error: {proc_error}", doc_id)
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    success = False

                if success:
                    self.logger.complete_operation("OCR processing", doc_id)

                    # Archive the document if archiver is configured
                    if self.archiver and self.archiver.enabled:
                        try:
                            base_watch_dir = item.get('base_watch_dir', filepath.parent)
                            archived_path = self.archiver.archive_document(filepath, base_watch_dir)
                            if archived_path:
                                self.db_manager.update_archived_path(doc_id, str(archived_path))
                                self.logger.info(f"Archived to: {archived_path}", doc_id)
                        except Exception as archive_error:
                            # Log but don't fail the whole operation
                            self.logger.error(f"Archive failed: {archive_error}", doc_id)
                else:
                    self.logger.fail_operation("OCR processing", doc_id)

                self.queue.task_done()
                self.currently_processing = None

            except Exception as e:
                self.logger.error(f"Queue worker error: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                self.currently_processing = None

                # Sleep a bit to prevent rapid error loops
                time.sleep(1.0)
                continue

        self.logger.info("Queue worker stopped")    
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get the current status of the OCR processing queue."""
        queue_size = self.queue.qsize()
        
        status = {
            'is_running': self.is_running,
            'queue_size': queue_size,
            'currently_processing': None
        }
        
        if self.currently_processing:
            doc_id = self.currently_processing['doc_id']
            filename = self.currently_processing['filepath'].name
            queued_at = self.currently_processing['queued_at']
            processing_time = time.time() - queued_at
            
            status['currently_processing'] = {
                'doc_id': doc_id,
                'filename': filename,
                'queued_at': queued_at,
                'processing_time': processing_time
            }
            
        return status