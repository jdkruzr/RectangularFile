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
from processing.document_source_manager import DocumentSourceManager
from processing.boox_pdf_source import BooxPDFSource
from processing.saber_note_source import SaberNoteSource

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
pdf_processor = PDFProcessor()
ocr_processor = QwenVLProcessor()
ocr_queue = OCRQueueManager(db, ocr_processor)
html_processor = HTMLProcessor()

# Initialize document source manager
doc_source_manager = DocumentSourceManager(polling_interval=config.FILE_WATCHER_POLLING_INTERVAL)

# Register Boox PDF source if enabled
if config.BOOX_ENABLED:
    print(f"Initializing Boox PDF source: {config.BOOX_FOLDER}")
    boox_source = BooxPDFSource(
        watch_directory=Path(config.BOOX_FOLDER),
        enabled=True
    )
    doc_source_manager.register_source(boox_source)
else:
    print("Boox PDF source is disabled")

# Register Saber note source if enabled
if config.SABER_ENABLED:
    if not config.SABER_PASSWORD:
        print("⚠️  WARNING: Saber is enabled but SABER_PASSWORD is not set. Disabling Saber source.")
    else:
        print(f"Initializing Saber note source: {config.SABER_FOLDER}")
        try:
            saber_source = SaberNoteSource(
                watch_directory=Path(config.SABER_FOLDER),
                encryption_password=config.SABER_PASSWORD,
                enabled=True
            )
            doc_source_manager.register_source(saber_source)
        except Exception as e:
            print(f"⚠️  WARNING: Failed to initialize Saber source: {e}")
            print("   Saber support will be disabled")
else:
    print("Saber note source is disabled")

# Keep legacy file_watcher for backward compatibility (used by app/__init__.py)
# This will be removed once app is updated to use doc_source_manager
file_watcher = FileWatcher(config.UPLOAD_FOLDER, polling_interval=config.FILE_WATCHER_POLLING_INTERVAL)

# Define callback for processed documents from document source manager
def process_document_from_source(processed_doc):
    """
    Handle a processed document from any document source.

    Args:
        processed_doc: ProcessedDocument object from a document source
    """
    print(f"Processing document from {processed_doc.source_type}: {processed_doc.title}")

    # Check if already in database
    with db.get_connection() as conn:
        cursor = conn.cursor()

        # For Saber notes, use encrypted filename as unique identifier
        # For PDFs, use the file path
        if processed_doc.source_type == 'saber_note':
            identifier = processed_doc.encrypted_filename
            cursor.execute("""
                SELECT id, processing_status FROM pdf_documents
                WHERE filename = ? OR relative_path LIKE ?
            """, (identifier, f"%{identifier}%"))
        else:
            cursor.execute("""
                SELECT id, processing_status FROM pdf_documents
                WHERE relative_path = ? OR filename = ?
            """, (processed_doc.original_path, Path(processed_doc.original_path).name))

        existing = cursor.fetchone()

    # Skip if already completed
    if existing and existing['processing_status'] == 'completed':
        print(f"Document already processed: {processed_doc.title}")
        return

    # Add or reset document in database
    if existing:
        doc_id = existing['id']
        print(f"Resetting existing document: {doc_id}")
        db.reset_document_status_by_id(doc_id)
    else:
        # Add to database
        # For Saber notes, we need to add with rendered page paths
        # For PDFs, use the original file
        primary_path = Path(processed_doc.original_path)
        doc_id = db.add_document(primary_path)

        if not doc_id:
            print(f"Failed to add document to database: {processed_doc.title}")
            return

    # Process based on source type
    if processed_doc.source_type == 'saber_note':
        # For Saber notes, queue each rendered page image for OCR
        print(f"Queueing {len(processed_doc.page_images)} Saber pages for OCR")
        for page_image in processed_doc.page_images:
            ocr_queue.add_to_queue(doc_id, page_image)

    elif processed_doc.source_type == 'boox_pdf':
        # For PDFs, use existing PDF processor
        filepath = Path(processed_doc.original_path)
        pdf_processor.process_document(filepath, doc_id, db)
        ocr_queue.add_to_queue(doc_id, filepath)

    else:
        print(f"Unknown source type: {processed_doc.source_type}")

# Set the callback on the document source manager
doc_source_manager.set_process_callback(process_document_from_source)

# Define callbacks for file watching (legacy - for backward compatibility)
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
    doc_source_manager.stop_all()
    file_watcher.stop()  # Legacy watcher
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
# Note: Still passing legacy file_watcher for backward compatibility
# TODO: Update create_app signature to accept doc_source_manager
app = create_app(db, file_watcher, pdf_processor, ocr_processor, ocr_queue, html_processor)

# Store doc_source_manager on app for access in routes if needed
app.doc_source_manager = doc_source_manager

# Start background services immediately (works for both gunicorn and direct execution)
# These services are thread-based and won't interfere with gunicorn workers

def _startup_initialization():
    """
    Perform startup initialization in background thread to avoid blocking gunicorn.
    This includes model loading, file watching, and initial document scanning.
    """
    print("[STARTUP] ▶ Starting OCR queue processing...")
    ocr_queue.start_processing()
    print("[STARTUP] ✓ OCR queue ready")

    # Pre-load the model before scanning files to avoid race conditions
    print("[STARTUP] ▶ Pre-loading AI model (this may take 30-60 seconds)...")
    if not ocr_processor._load_model():
        print("[STARTUP] ⚠ WARNING: Failed to pre-load model, will load on first use")
    else:
        print("[STARTUP] ✓ Model loaded successfully")

    print("[STARTUP] ▶ Starting document source watchers...")
    # Start the new modular document source manager
    doc_source_manager.start_all(trigger_initial_scan=True)
    # Also start legacy file watcher for backward compatibility
    file_watcher.start()
    print("[STARTUP] ✓ File watchers ready")

    # Perform initial scan to queue any unprocessed documents from database
    print("[STARTUP] ▶ Scanning for unprocessed documents in database...")
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, relative_path, processing_status
            FROM pdf_documents
            WHERE (processing_status = 'pending' OR processing_status IS NULL OR processing_status = 'failed')
            AND relative_path NOT LIKE '%.html'
            AND relative_path NOT LIKE '%.htm'
            ORDER BY file_modified_at DESC
        """)
        unprocessed = cursor.fetchall()

        if unprocessed:
            print(f"[STARTUP] Found {len(unprocessed)} unprocessed documents, queueing for processing...")
            for doc in unprocessed:
                doc_id = doc['id']
                doc_path = doc['relative_path']
                filepath = Path(doc_path) if os.path.isabs(doc_path) else Path(config.UPLOAD_FOLDER) / doc_path

                if filepath.exists():
                    print(f"[STARTUP]   Queueing doc {doc_id}: {filepath.name}")
                    # Process PDF immediately
                    pdf_processor.process_document(filepath, doc_id, db)
                    # Queue for OCR
                    ocr_queue.add_to_queue(doc_id, filepath)
                else:
                    print(f"[STARTUP]   Skipping doc {doc_id}: file not found")
            print(f"[STARTUP] ✓ Initial scan complete, {len(unprocessed)} documents queued")
        else:
            print("[STARTUP] ✓ No unprocessed documents found")

# Start initialization in background thread to avoid blocking gunicorn worker
import threading
print("[STARTUP] Starting background initialization...")
init_thread = threading.Thread(target=_startup_initialization, daemon=True, name="StartupInit")
init_thread.start()
print("[STARTUP] Web server is ready to accept connections")

if __name__ == '__main__':
    # This block only runs when executing directly (not under gunicorn)
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
        use_reloader=False
    )