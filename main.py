# main.py
import os
import signal
import atexit
import sys
from pathlib import Path

from app import create_app
from db.db_manager import DatabaseManager
from processing.file_watcher import FileWatcher
from processing.pdf_processor import PDFProcessor
from processing.qwen_processor import QwenVLProcessor
from processing.ocr_queue_manager import OCRQueueManager

# Configuration
UPLOAD_FOLDER = "/mnt/onyx"
DEFAULT_POLLING_INTERVAL = 30.0

# Create application components
db = DatabaseManager("/mnt/rectangularfile/pdf_index.db")
file_watcher = FileWatcher(UPLOAD_FOLDER, polling_interval=DEFAULT_POLLING_INTERVAL)
pdf_processor = PDFProcessor()
ocr_processor = QwenVLProcessor()
ocr_queue = OCRQueueManager(db, ocr_processor)

# Define callbacks for file watching
def process_new_file(relative_path):
    """Process a newly detected file."""
    filepath = Path(os.path.join(UPLOAD_FOLDER, relative_path))
    print(f"Processing new file: {filepath}")
    
    # Add to database
    doc_id = db.add_document(filepath)
    if not doc_id:
        print(f"Failed to add document to database: {filepath}")
        return
        
    # Process with text extraction
    pdf_processor.process_document(filepath, doc_id, db)
    
    # Queue for OCR processing
    ocr_queue.add_to_queue(doc_id, filepath)

def handle_removed_file(relative_path):
    """Handle a file that's been removed."""
    filepath = Path(os.path.join(UPLOAD_FOLDER, relative_path))
    print(f"Marking removed file: {filepath}")
    db.mark_document_removed(filepath)

# Register callbacks with the file watcher
file_watcher.register_callback(process_new_file)
file_watcher.register_removal_callback(handle_removed_file)

def cleanup():
    """Clean up resources before shutdown."""
    print("Shutting down application...")
    ocr_queue.stop_processing()
    file_watcher.stop()
    db.close()
    print("Application shutdown complete.")

def signal_handler(signum, _):
    """Handle shutdown signals gracefully."""
    print(f"\nSignal received: {signum}")
    cleanup()
    sys.exit(0)

# Register cleanup handlers
atexit.register(cleanup)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Create the Flask application
app = create_app(db, file_watcher, pdf_processor, ocr_processor, ocr_queue)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Start file watching and OCR queue processing
    file_watcher.start()
    ocr_queue.start_processing()
    
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)