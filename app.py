from flask import Flask, render_template, jsonify, request
import os
from typing import List
from file_watcher import FileWatcher
from db_manager import DatabaseManager
from pdf_processor import PDFProcessor
from pathlib import Path
import time
import signal
import atexit
import sys
from functools import partial
import multiprocessing
from handwriting_recognition import HandwritingRecognizer

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
DEFAULT_POLLING_INTERVAL = 30.0
file_watcher = FileWatcher(UPLOAD_FOLDER, polling_interval=DEFAULT_POLLING_INTERVAL)
db = DatabaseManager("pdf_index.db")
pdf_processor = PDFProcessor()
ocr_processor = HandwritingRecognizer()

new_files: List[str] = []
removed_files: List[str] = []

file_watcher_initialized = False
_cleanup_done = False

def calculate_processing_progress(doc: dict) -> float:
    """Calculate the processing progress percentage for a document."""
    status = doc.get('processing_status', '').lower()
    if status == 'completed':
        return 100.0
    elif status == 'failed':
        return 0.0
    elif status == 'pending':
        return 0.0
    elif status == 'processing':
        return doc.get('processing_progress', 50.0)
    else:
        return 0.0

def cleanup():
    """Clean up resources before shutdown."""
    global _cleanup_done
    if _cleanup_done:
        return

    print("DEBUG: Cleanup function called from:")
    import traceback
    traceback.print_stack()

    print("Shutting down file watcher...")
    if file_watcher_initialized:
        file_watcher.stop()
        time.sleep(0.2)

        if hasattr(multiprocessing, 'resource_tracker'):
            try:
                multiprocessing.resource_tracker._resource_tracker.clear()
                multiprocessing.resource_tracker._resource_tracker = None
            except Exception:
                pass

    print("File watcher stopped.")
    db.close()
    _cleanup_done = True

def signal_handler(signum, _):
    """Handle shutdown signals gracefully."""
    print(f"\nSignal received: {signum}")
    cleanup()

    if hasattr(multiprocessing, 'resource_tracker'):
        try:
            multiprocessing.resource_tracker._resource_tracker._kill_process()
        except Exception:
            pass

    sys.exit(0)

# Register cleanup handlers
atexit.register(cleanup)
signal.signal(signal.SIGINT, partial(signal_handler, signum=signal.SIGINT))
signal.signal(signal.SIGTERM, partial(signal_handler, signum=signal.SIGTERM))

@app.before_request
def initialize_file_watcher():
    """Initialize the file watcher before handling the first request."""
    global file_watcher_initialized
    if not file_watcher_initialized:
        print(f"Initializing file watcher at {time.strftime('%H:%M:%S')}")
        file_watcher.register_callback(on_new_file)
        file_watcher.register_removal_callback(on_file_removed)
        file_watcher.start()
        file_watcher_initialized = True
        print(f"File watcher started with polling interval of {file_watcher.polling_interval} seconds")

@app.route('/')
def index():
    """Render the main page."""
    print(f"Rendering template with files: {file_watcher.files}")
    print(f"Template folder is at: {app.template_folder}")
    return render_template('index.html',
                          files=file_watcher.files,
                          polling_interval=file_watcher.polling_interval)

@app.route('/settings/page')
def settings_page():
    """Render the settings page."""
    return render_template('settings.html')

@app.route('/reset/<int:doc_id>')
def reset_processing(doc_id):
    """Reset processing status for a document by ID and trigger reprocessing."""
    # Get the document details
    document = db.get_document_by_id(doc_id)
    if not document:
        return jsonify(success=False, message="Document not found"), 404

    # Get the filepath
    filepath = Path(os.path.join(UPLOAD_FOLDER, document['filename']))
    if not filepath.exists():
        return jsonify(success=False, message="File no longer exists"), 404

    # Reset status
    if db.reset_document_status_by_id(doc_id):
        # Try text extraction first
        extracted = pdf_processor.process_document(filepath, doc_id, db)

        # Always run OCR regardless of text extraction result
        ocr_result = ocr_processor.process_document(filepath, doc_id, db)

        if ocr_result:
            return jsonify(success=True, message=f"Reset and processed document {doc_id}")
        else:
            return jsonify(success=False, message="Reset succeeded but processing failed"), 500
    else:
        return jsonify(success=False, message="Failed to reset status"), 400

@app.route('/files')
def get_files():
    """Get status of all files."""
    status = request.args.get('status')
    sort_by = request.args.get('sort', 'last_indexed_at')
    order = request.args.get('order', 'desc')

    fs_files = file_watcher.files
    db_docs = db.get_active_documents(
        status=status,
        sort_by=sort_by,
        order=order
    )

    files_status = []
    for filename in fs_files:
        file_info = {
            'filename': filename,
            'in_filesystem': True,
            'in_database': False,
            'processing_status': None,
            'ocr_processed': False,
            'last_indexed': None,
            'file_size': None,
            'processing_progress': None,
            'id': None  # Add ID field
        }

        for doc in db_docs:
            if doc['filename'] == filename:
                file_info.update({
                    'in_database': True,
                    'processing_status': doc['processing_status'],
                    'ocr_processed': doc['ocr_processed'],
                    'last_indexed': doc['last_indexed_at'],
                    'file_size': doc.get('file_size_bytes'),
                    'processing_progress': calculate_processing_progress(doc),
                    'id': doc['id']  # Include the ID
                })
                break

        files_status.append(file_info)

    return jsonify(
        files=files_status,
        total_count=len(files_status),
        processing_count=sum(1 for f in files_status if f['processing_status'] == 'processing'),
        completed_count=sum(1 for f in files_status if f['processing_status'] == 'completed')
    )

@app.route('/new_files')
def get_new_files():
    """Get lists of new and removed files."""
    global new_files, removed_files
    
    new_discovered, removed_discovered = file_watcher.scan_now()
    
    files_to_return = new_files.copy()
    removed_to_return = removed_files.copy()
    
    for file in new_discovered:
        if file not in files_to_return:
            files_to_return.append(file)
    
    for file in removed_discovered:
        if file not in removed_to_return:
            removed_to_return.append(file)
    
    new_files = []
    removed_files = []
    
    return jsonify(new_files=files_to_return, removed_files=removed_to_return)

@app.route('/scan')
def scan_files():
    """Manually trigger a file scan."""
    new_discovered, removed_discovered = file_watcher.scan_now()
    
    for filename in new_discovered:
        filepath = Path(os.path.join(UPLOAD_FOLDER, filename))
        doc_id = db.add_document(filepath)
        if not doc_id:
            print(f"Failed to add document to database: {filename}")
    
    for filename in removed_discovered:
        filepath = Path(os.path.join(UPLOAD_FOLDER, filename))
        if not db.mark_document_removed(filepath):
            print(f"Failed to mark document as removed: {filename}")
    
    return jsonify(
        new_files=new_discovered, 
        removed_files=removed_discovered
    )

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Get or update application settings."""
    if request.method == 'POST':
        try:
            new_interval = float(request.form.get('polling_interval', DEFAULT_POLLING_INTERVAL))
            
            success = file_watcher.set_polling_interval(new_interval)
            
            if success:
                return jsonify(success=True, message=f"Polling interval updated to {new_interval} seconds")
            else:
                return jsonify(success=False, message="Invalid polling interval"), 400
                
        except ValueError:
            return jsonify(success=False, message="Invalid polling interval format"), 400
    else:
        # GET request - return current settings
        return jsonify(
            polling_interval=file_watcher.polling_interval,
            file_types=file_watcher.file_types
        )

@app.route('/settings/files', methods=['POST'])
def update_file_settings():
    """Update file type settings."""
    try:
        file_types = request.form.get('file_types', '')
        
        # Parse the file types
        types_list = [t.strip() for t in file_types.split(',') if t.strip()]
        
        # Update file watcher settings
        file_watcher.file_types = types_list
        
        return jsonify(
            success=True, 
            message=f"File types updated to: {', '.join(types_list) if types_list else 'all files'}"
        )
            
    except Exception as e:
        return jsonify(success=False, message=f"Error updating file types: {str(e)}"), 400

@app.route('/search')
def search_page():
    """Render the search page."""
    query = request.args.get('q', '')
    results = []
    
    if query:
        results = db.search_documents(query)
        
    return render_template('search.html', query=query, results=results)

@app.route('/view/<int:doc_id>')
def view_document(doc_id):
    """View a document with optional text highlighting."""
    highlight = request.args.get('highlight', '')
    
    # Get document details
    document = db.get_document_by_id(doc_id)
    if not document:
        return render_template('error.html', message="Document not found"), 404
        
    # Get document text content
    text_content = db.get_document_text(doc_id)
    
    return render_template(
        'view_document.html',
        document=document,
        text_content=text_content,
        highlight=highlight
    )

@app.route('/ocr/<int:doc_id>')
def process_ocr(doc_id):
    """Process a document with handwriting recognition."""
    # Get document details
    document = db.get_document_by_id(doc_id)
    if not document:
        return jsonify(success=False, message="Document not found"), 404

    # Get the filepath
    filepath = Path(os.path.join(UPLOAD_FOLDER, document['filename']))
    if not filepath.exists():
        return jsonify(success=False, message="File no longer exists"), 404

    # Start OCR processing
    if ocr_processor.process_document(filepath, doc_id, db):
        return jsonify(success=True, message=f"OCR processing started for document {doc_id}")
    else:
        return jsonify(success=False, message="Failed to start OCR processing"), 500
    
def on_new_file(filename: str):
    """Handle new file detection."""
    global new_files
    new_files.append(filename)
    print(f"New file detected: {filename} at {time.strftime('%H:%M:%S')}")

    filepath = Path(os.path.join(UPLOAD_FOLDER, filename))
    doc_id = db.add_document(filepath)
    if doc_id:
        print(f"Added document to database with ID: {doc_id}")

        # Try text extraction first
        extracted = pdf_processor.process_document(filepath, doc_id, db)

        # Always run OCR processing regardless of text extraction result
        print(f"Starting OCR processing for {filename}")
        ocr_result = ocr_processor.process_document(filepath, doc_id, db)

        if ocr_result:
            print(f"OCR processing started for {filename}")
        else:
            print(f"Failed to start OCR processing for {filename}")

def on_file_removed(filename: str):
    """Handle file removal."""
    global removed_files
    removed_files.append(filename)
    print(f"File removed: {filename} at {time.strftime('%H:%M:%S')}")

    filepath = Path(os.path.join(UPLOAD_FOLDER, filename))
    if db.mark_document_removed(filepath):
        print(f"Marked document as removed in database: {filename}")
    else:
        print(f"Failed to mark document as removed in database: {filename}")

@app.route('/training/correct/<int:doc_id>/<int:page_num>')
def training_correction_page(doc_id, page_num):
    """Show page for correcting OCR results for training."""
    document = db.get_document_by_id(doc_id)
    if not document:
        return render_template('error.html', message="Document not found"), 404

    # Get OCR text for the page
    page_texts = db.get_document_text(doc_id)
    if not page_texts or page_num not in page_texts:
        return render_template('error.html', message="Page text not found"), 404

    page_text = page_texts[page_num]
    
    return render_template(
        'training_correction.html',
        document=document,
        page_num=page_num,
        ocr_text=page_text.get('text', ''),
        confidence=page_text.get('confidence', 0)
    )

@app.route('/training/submit', methods=['POST'])
def submit_correction():
    """Submit a corrected text sample for training."""
    doc_id = request.form.get('doc_id', type=int)
    page_num = request.form.get('page_num', type=int)
    original_text = request.form.get('original_text', '')
    corrected_text = request.form.get('corrected_text', '')
    
    if not all([doc_id, page_num, original_text, corrected_text]):
        return jsonify(success=False, message="Missing required fields"), 400
    
    from handwriting_trainer import HandwritingTrainer
    trainer = HandwritingTrainer(db)
    
    success = trainer.collect_training_sample(
        doc_id=doc_id,
        page_num=page_num,
        original_text=original_text,
        corrected_text=corrected_text
    )
    
    if success:
        return jsonify(success=True, message="Thank you for your correction!")
    else:
        return jsonify(success=False, message="Failed to save correction"), 500

@app.route('/training/start', methods=['POST'])
def start_training():
    """Start a training job for handwriting recognition."""
    profile_name = request.form.get('profile_name', 'default')
    
    from handwriting_trainer import HandwritingTrainer
    trainer = HandwritingTrainer(db)
    
    job_id = trainer.start_training_job(profile_name)
    
    if job_id:
        # In a real app, you'd queue this for background processing
        # For simplicity, we'll just do it synchronously
        if trainer.generate_training_data(job_id):
            if trainer.execute_training(job_id):
                return jsonify(success=True, message=f"Training completed successfully! Job ID: {job_id}")
            else:
                return jsonify(success=False, message="Training execution failed"), 500
        else:
            return jsonify(success=False, message="Failed to generate training data"), 500
    else:
        return jsonify(success=False, message="Failed to start training job"), 500

@app.route('/training')
def training_management():
    """Manage handwriting recognition training."""
    # Get training stats and jobs
    success = request.args.get('success', None)
    message = request.args.get('message', None)
    
    if success is not None:
        success = success.lower() == 'true'
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Get stats
        cursor.execute("""
            SELECT COUNT(*) as total_samples 
            FROM handwriting_training_data
        """)
        total_samples = cursor.fetchone()['total_samples']
        
        cursor.execute("""
            SELECT COUNT(*) as total_jobs 
            FROM training_jobs 
            WHERE status = 'completed'
        """)
        total_jobs = cursor.fetchone()['total_jobs']
        
        cursor.execute("""
            SELECT AVG(accuracy_improvement) as avg_improvement 
            FROM training_jobs 
            WHERE status = 'completed' AND accuracy_improvement IS NOT NULL
        """)
        avg_improvement = cursor.fetchone()['avg_improvement'] or 0
        
        # Get recent jobs
        cursor.execute("""
            SELECT j.id, p.profile_name, j.status, j.started_at, j.completed_at,
                   j.sample_count, j.accuracy_improvement
            FROM training_jobs j
            JOIN handwriting_profiles p ON j.profile_id = p.id
            ORDER BY j.started_at DESC
            LIMIT 10
        """)
        jobs = [dict(row) for row in cursor.fetchall()]
        
        stats = {
            'total_samples': total_samples,
            'total_jobs': total_jobs,
            'avg_improvement': avg_improvement
        }
        
    return render_template(
        'training_management.html',
        stats=stats,
        jobs=jobs,
        success=success,
        message=message
    )

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    try:
        app.run(debug=True, use_reloader=False)  # Try disabling the reloader
    finally:
        cleanup()



