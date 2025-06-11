from flask import Flask, render_template, jsonify, request
import os
from typing import Dict, List
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
from qwen_processor import QwenVLProcessor
from ocr_queue_manager import OCRQueueManager
import glob
from flask import send_file
import json

app = Flask(__name__)

UPLOAD_FOLDER = "/mnt/onyx"
DEFAULT_POLLING_INTERVAL = 30.0
file_watcher = FileWatcher(UPLOAD_FOLDER, polling_interval=DEFAULT_POLLING_INTERVAL)
db = DatabaseManager("/mnt/rectangularfile/pdf_index.db")
pdf_processor = PDFProcessor()
ocr_processor = QwenVLProcessor()
ocr_queue = OCRQueueManager(db, ocr_processor)

new_files: List[str] = []
removed_files: List[str] = []

file_watcher_initialized = False
_cleanup_done = False

def extract_filename_metadata(filename: str) -> Dict[str, str]:
    """
    Extract useful metadata from filename patterns.
    For example: 20250514_Meeting_Name.pdf -> {'date': '2025-05-14', 'title': 'Meeting Name'}
    """
    metadata = {}
    
    # Extract date if filename starts with a date pattern (YYYYMMDD)
    date_match = re.match(r'^(\d{4})(\d{2})(\d{2})[ _-]?(.*)', filename)
    if date_match:
        year, month, day, remaining = date_match.groups()
        try:
            # Validate the date
            datetime.date(int(year), int(month), int(day))
            metadata['date'] = f"{year}-{month}-{day}"
            # Use the remaining part for title
            metadata['title'] = remaining.replace('_', ' ').replace('-', ' ')
        except ValueError:
            # If date is invalid, just use the whole filename
            metadata['title'] = filename.replace('.pdf', '')
    else:
        # No date pattern, just use the filename without extension
        metadata['title'] = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ')
    
    return metadata

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
    
    print("Shutting down OCR queue...")
    ocr_queue.stop_processing()
    
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

@app.template_filter('from_json')
def from_json(value):
    """Parse JSON string into Python object for templates."""
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}
    
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
    filepath = Path(document['relative_path'])
    if not filepath.exists():
        return jsonify(success=False, message="File no longer exists"), 404

    # Reset status
    if db.reset_document_status_by_id(doc_id):
        # Try text extraction first
        extracted = pdf_processor.process_document(filepath, doc_id, db)
        
        # Queue for OCR processing
        if ocr_queue.add_to_queue(doc_id, filepath):
            return jsonify(success=True, message=f"Reset and queued document {doc_id} for OCR processing")
        else:
            return jsonify(success=False, message="Reset succeeded but queuing failed"), 500
    else:
        return jsonify(success=False, message="Failed to reset status"), 400

@app.route('/ocr_queue')
def ocr_queue_status():
    """Get the status of the OCR processing queue."""
    status = ocr_queue.get_queue_status()
    return jsonify(status)

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
    """Render the search page with folder filtering."""
    query = request.args.get('q', '')
    folder_filter = request.args.get('folder', '')
    results = []
    
    # Get all unique folders for the filter dropdown
    folders = []
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT folder_path 
                FROM pdf_documents 
                WHERE processing_status != 'removed'
                ORDER BY folder_path
            """)
            folders = [row[0] for row in cursor.fetchall() if row[0]]
    except Exception as e:
        app.logger.error(f"Error fetching folders: {e}")
    
    # Add special options for common searches
    moffitt_folders = [f for f in folders if 'Moffitt' in f]
    
    if query:
        results = db.search_documents_with_folder_filter(query, folder_filter)
        
    return render_template(
        'search.html', 
        query=query, 
        results=results, 
        folders=folders,
        moffitt_folders=moffitt_folders,
        current_folder=folder_filter
    )

@app.route('/view/<int:doc_id>')
def view_document(doc_id):
    """Redirect to the unified document viewer."""
    highlight = request.args.get('highlight', '')
    if highlight:
        return redirect(url_for('document_viewer', doc_id=doc_id, highlight=highlight))
    return redirect(url_for('document_viewer', doc_id=doc_id))

@app.route('/document_image/<int:doc_id>/<int:page_num>')
def document_image(doc_id, page_num):
    """Serve a document page image."""
    document = db.get_document_by_id(doc_id)
    if not document:
        return jsonify(error="Document not found"), 404
    
    # Get the page text data which includes image path
    page_text = db.get_document_text(doc_id, page_num)
    
    if not page_text or not page_text.get('image_path'):
        return jsonify(error="Image path not found"), 404
        
    image_path = page_text['image_path']
    
    # Verify the file exists
    if not os.path.exists(image_path):
        return jsonify(error="Image file not found"), 404
        
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/files/<int:doc_id>')
def document_inspector(doc_id):
    """Redirect to the unified document viewer."""
    return redirect(url_for('document_viewer', doc_id=doc_id))

@app.route('/folders')
def folder_browser():
    """Browse documents by folder structure."""
    folder = request.args.get('path', '')
    
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all unique folders
            cursor.execute("""
                SELECT DISTINCT folder_path 
                FROM pdf_documents 
                WHERE processing_status != 'removed'
                ORDER BY folder_path
            """)
            
            all_folders = [row['folder_path'] for row in cursor.fetchall()]
            
            # Get files in the current folder
            if folder:
                # Exact folder match
                cursor.execute("""
                    SELECT id, filename, relative_path, folder_path, 
                           processing_status, last_indexed_at, ocr_processed
                    FROM pdf_documents
                    WHERE folder_path = ? AND processing_status != 'removed'
                    ORDER BY filename
                """, (folder,))
            else:
                # Root folder
                cursor.execute("""
                    SELECT id, filename, relative_path, folder_path, 
                           processing_status, last_indexed_at, ocr_processed
                    FROM pdf_documents
                    WHERE (folder_path = '' OR folder_path IS NULL) AND processing_status != 'removed'
                    ORDER BY filename
                """)
                
            files = [dict(row) for row in cursor.fetchall()]
            
            # Build folder tree
            folder_tree = {}
            for path in all_folders:
                if not path:  # Skip empty paths
                    continue
                    
                parts = path.split('/')
                current = folder_tree
                
                for i, part in enumerate(parts):
                    if part not in current:
                        current[part] = {}
                    current = current[part]
            
            return render_template(
                'folder_browser.html',
                current_folder=folder,
                files=files,
                folder_tree=folder_tree,
                all_folders=all_folders
            )
            
    except Exception as e:
        print(f"Error in folder browser: {e}")
        return render_template('error.html', message=f"Error browsing folders: {str(e)}"), 500
    
@app.route('/document/<int:doc_id>')
def document_viewer(doc_id):
    """Unified document viewer that combines inspection and viewing."""
    highlight = request.args.get('highlight', '')
    
    # Get document details
    document = db.get_document_by_id(doc_id)
    if not document:
        return render_template('error.html', message="Document not found"), 404
        
    # Get document text content
    text_content = db.get_document_text(doc_id) or {}
    
    # Get OCR and extracted content separately for comparison
    ocr_content = {}
    extracted_content = {}
    
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            # Get OCR content
            cursor.execute("""
                SELECT page_number, ocr_text, processed_at
                FROM pdf_text_content
                WHERE pdf_id = ? AND ocr_text IS NOT NULL
                ORDER BY page_number
            """, (doc_id,))
            
            for row in cursor.fetchall():
                ocr_content[row['page_number']] = {
                    'text': row['ocr_text'],
                    'processed_at': row['processed_at'],
                    'source': 'ocr'
                }
                
            # Get extracted content
            cursor.execute("""
                SELECT page_number, text_content, processed_at
                FROM pdf_text_content
                WHERE pdf_id = ? AND text_content IS NOT NULL
                ORDER BY page_number
            """, (doc_id,))
            
            for row in cursor.fetchall():
                extracted_content[row['page_number']] = {
                    'text': row['text_content'],
                    'processed_at': row['processed_at'],
                    'source': 'extracted'
                }
    except Exception as e:
        print(f"Error fetching content: {e}")
    
    # Pre-compute common pages
    common_pages = sorted(set(extracted_content.keys()) & set(ocr_content.keys()))
    has_comparison = len(common_pages) > 0
    
    return render_template(
        'document_viewer.html',
        document=document,
        text_content=text_content,
        extracted_content=extracted_content,
        ocr_content=ocr_content,
        common_pages=common_pages,
        has_comparison=has_comparison,
        highlight=highlight
    )

def on_new_file(rel_path: str):
    """Handle new file detection."""
    global new_files
    new_files.append(rel_path)
    print(f"New file detected: {rel_path} at {time.strftime('%H:%M:%S')}")

    filepath = Path(os.path.join(UPLOAD_FOLDER, rel_path))
    doc_id = db.add_document(filepath)
    if doc_id:
        print(f"Added document to database with ID: {doc_id}")

        # Try text extraction first
        extracted = pdf_processor.process_document(filepath, doc_id, db)
        
        # Queue the document for OCR processing instead of doing it immediately
        print(f"Queuing {rel_path} for OCR processing")
        if ocr_queue.add_to_queue(doc_id, filepath):
            print(f"Document {rel_path} queued for OCR processing")
        else:
            print(f"Failed to queue document {rel_path} for OCR processing")
            
def on_file_removed(rel_path: str):
    """Handle file removal."""
    global removed_files
    removed_files.append(rel_path)
    print(f"File removed: {rel_path} at {time.strftime('%H:%M:%S')}")

    filepath = Path(os.path.join(UPLOAD_FOLDER, rel_path))
    if db.mark_document_removed(filepath):
        print(f"Marked document as removed in database: {rel_path}")
    else:
        print(f"Failed to mark document as removed in database: {rel_path}")

@app.before_request
def initialize_file_watcher():
    """Initialize the file watcher before handling the first request."""
    global file_watcher_initialized
    if not file_watcher_initialized:
        print(f"Initializing file watcher at {time.strftime('%H:%M:%S')}")
        file_watcher.register_callback(on_new_file)
        file_watcher.register_removal_callback(on_file_removed)
        file_watcher.start()

        ocr_queue.start_processing()

        file_watcher_initialized = True
        print(f"File watcher started with polling interval of {file_watcher.polling_interval} seconds")

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    try:
        app.run(debug=True, use_reloader=False)  # Disable reloader to prevent double initialization
    finally:
        cleanup()
