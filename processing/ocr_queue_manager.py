import threading
import queue
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path

class OCRQueueManager:
    """Manages a queue of documents for OCR processing to avoid overloading the GPU."""
    
    def __init__(self, db_manager, ocr_processor):
        """Initialize the OCR queue manager."""
        self.queue = queue.Queue()
        self.processing_thread = None
        self.stop_requested = False
        self.currently_processing = None
        self.db_manager = db_manager
        self.ocr_processor = ocr_processor
        self.logger = self._setup_logging()
        self.is_running = False
    
    def _setup_logging(self):
        """Configure logging for the queue manager."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def add_to_queue(self, doc_id: int, filepath: Path) -> bool:
        """Add a document to the processing queue."""
        if not filepath.exists():
            self.logger.error(f"File not found: {filepath}")
            return False
        
        self.logger.info(f"Adding document {doc_id} ({filepath.name}) to OCR queue")
        self.queue.put({
            'doc_id': doc_id,
            'filepath': filepath,
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
        self.logger.info("OCR processing thread started")
    
    def stop_processing(self):
        """Stop the background processing thread."""
        if not self.is_running:
            return
            
        self.logger.info("Stopping OCR processing thread...")
        self.stop_requested = True
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
            
        self.is_running = False
        self.logger.info("OCR processing thread stopped")
    
    def _process_queue_worker(self):
        """Worker thread that processes the queue items one at a time."""
        self.logger.info("OCR queue worker started")
        
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
                
                self.logger.info(f"Started OCR processing for document {doc_id} ({filepath.name})")
                self.db_manager.update_processing_progress(
                    doc_id, 5.0, "Queued for OCR processing"
                )
                
                # Improved memory cleanup before processing each document
                if hasattr(self.ocr_processor, 'device') and self.ocr_processor.device == 'cuda':
                    import torch
                    import gc
                    
                    # More aggressive memory cleanup
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Wait for all CUDA operations to finish
                    gc.collect()  # Run Python garbage collector
                    
                    # Second round of cleanup
                    torch.cuda.empty_cache()
                    
                    # Log memory status
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / (1024**3)
                        reserved = torch.cuda.memory_reserved() / (1024**3)
                        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        free = total - reserved
                        self.logger.info(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {free:.2f} GB free")
                
                # Process the document with OCR
                try:
                    self.logger.info(f"Calling OCR processor for document {doc_id}...")
                    success = self.ocr_processor.process_document(
                        filepath, doc_id, self.db_manager
                    )
                    self.logger.info(f"OCR processor returned: {success}")
                except Exception as proc_error:
                    self.logger.error(f"Error processing document {doc_id}: {proc_error}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    success = False
                            
                if success:
                    self.logger.info(f"Successfully completed OCR for document {doc_id}")
                else:
                    self.logger.error(f"Failed to process document {doc_id} with OCR")
                
                # Final cleanup after processing
                if hasattr(self.ocr_processor, 'device') and self.ocr_processor.device == 'cuda':
                    import torch
                    import gc
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()
                
                self.queue.task_done()
                self.currently_processing = None
                
            except Exception as e:
                self.logger.error(f"Error in OCR queue worker: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                self.currently_processing = None
                
                # Sleep a bit to prevent rapid error loops
                time.sleep(1.0)
                continue
                
        self.logger.info("OCR queue worker stopped")    
    
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