# main.py
import os
import signal
import atexit
import sys
from pathlib import Path

from config import config
from app import create_app
from db.db_manager import DatabaseManager
from processing.file_watcher import FileWatcher
from processing.pdf_processor import PDFProcessor
from processing.qwen_processor import QwenVLProcessor
from processing.ocr_queue_manager import OCRQueueManager
from processing.html_processor import HTMLProcessor

# Validate and print configuration
config.print_config()
validation_errors = config.validate()
if validation_errors:
    print("\n⚠️  Configuration warnings:")
    for error in validation_errors:
        print(f"  - {error}")
    print()

# Ensure required directories exist
config.ensure_directories()

# Create application components using centralized config
db = DatabaseManager(config.DATABASE_PATH)
file_watcher = FileWatcher(config.UPLOAD_FOLDER, polling_interval=config.FILE_WATCHER_POLLING_INTERVAL)
pdf_processor = PDFProcessor()
ocr_processor = QwenVLProcessor()
ocr_queue = OCRQueueManager(db, ocr_processor)
html_processor = HTMLProcessor()

# Define callbacks for file watching
def process_new_file(relative_path):
    """Process a newly detected file."""
    filepath = Path(os.path.join(config.UPLOAD_FOLDER, relative_path))
    print(f"Processing new file: {filepath}")
    
    # Check if this file is already in the database
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Try to find by relative path or filename
        filename = filepath.name
        cursor.execute("""
            SELECT id, processing_status FROM pdf_documents 
            WHERE relative_path = ? OR filename = ?
        """, (str(filepath.absolute()), filename))
        
        existing = cursor.fetchone()
    
    # If already in database and completed, skip processing
    if existing and existing['processing_status'] == 'completed':
        print(f"File already processed: {filepath}")
        return
    
    # If in database but not completed, reset and process
    if existing:
        doc_id = existing['id']
        print(f"Resetting existing document: {doc_id}")
        db.reset_document_status_by_id(doc_id)
    else:
        # Add to database if not found
        doc_id = db.add_document(filepath)
        if not doc_id:
            print(f"Failed to add document to database: {filepath}")
            return
    
    # Choose processor based on file extension
    file_extension = filepath.suffix.lower()
    
    # Process based on file type
    if file_extension in ['.html', '.htm']:
        print(f"Processing HTML file: {filepath}")
        # Use HTML processor
        html_processor.process_document(filepath, doc_id, db)
        print(f"HTML processing complete for {filepath}")
    else:
        # Default PDF processing
        print(f"Processing PDF file: {filepath}")
        pdf_processor.process_document(filepath, doc_id, db)
        
        # Only queue PDFs for OCR
        ocr_queue.add_to_queue(doc_id, filepath)
            
def handle_removed_file(relative_path):
    """Handle a file that's been removed."""
    filepath = Path(os.path.join(config.UPLOAD_FOLDER, relative_path))
    print(f"Marking removed file: {filepath}")
    db.mark_document_removed(filepath)

# File watcher callbacks are now registered in app/__init__.py via _init_file_watcher()
# to avoid duplicate processing

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
app = create_app(db, file_watcher, pdf_processor, ocr_processor, ocr_queue, html_processor)

# Start background services immediately (works for both gunicorn and direct execution)
# These services are thread-based and won't interfere with gunicorn workers
print("Starting file watcher...")
file_watcher.start()
print("File watcher started.")

print("Starting OCR queue processing...")
ocr_queue.start_processing()
print("OCR queue processing started.")

if __name__ == '__main__':
    # This block only runs when executing directly (not under gunicorn)
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
        use_reloader=False
    )