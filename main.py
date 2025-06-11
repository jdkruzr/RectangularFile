# main.py
from app import create_app
from db.db_manager import DatabaseManager
from processing.file_watcher import FileWatcher
from processing.pdf_processor import PDFProcessor
from processing.qwen_processor import QwenVLProcessor
from processing.ocr_queue_manager import OCRQueueManager
import os
import signal
import atexit
import sys
from functools import partial

# Configuration
UPLOAD_FOLDER = "/mnt/onyx"
DEFAULT_POLLING_INTERVAL = 30.0

# Create application components
db = DatabaseManager("/mnt/rectangularfile/pdf_index.db")
file_watcher = FileWatcher(UPLOAD_FOLDER, polling_interval=DEFAULT_POLLING_INTERVAL)
pdf_processor = PDFProcessor()
ocr_processor = QwenVLProcessor()
ocr_queue = OCRQueueManager(db, ocr_processor)

# Create the Flask application
app = create_app(db, file_watcher, pdf_processor, ocr_queue)

def cleanup():
    """Clean up resources before shutdown."""
    print("Shutting down application...")
    ocr_queue.stop_processing()
    file_watcher.stop()
    db.close()
    print("Application shutdown complete.")

# Register cleanup handlers
atexit.register(cleanup)
signal.signal(signal.SIGINT, partial(lambda signum, _: (cleanup(), sys.exit(0)), signum=signal.SIGINT))
signal.signal(signal.SIGTERM, partial(lambda signum, _: (cleanup(), sys.exit(0)), signum=signal.SIGTERM))

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)