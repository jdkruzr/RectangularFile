# app/__init__.py
from flask import Flask
import os

def create_app(db_manager, file_watcher, pdf_processor, ocr_processor, ocr_queue):
    """Application factory."""
    app = Flask(__name__)
    
    # Configure app
    app.config.update(
        UPLOAD_FOLDER="/mnt/onyx",
        MAX_CONTENT_LENGTH=50 * 1024 * 1024  # 50MB max upload
    )
    
    # Register components with app
    app.db = db_manager
    app.file_watcher = file_watcher
    app.pdf_processor = pdf_processor
    app.ocr_processor = ocr_processor
    app.ocr_queue = ocr_queue
    
    # Initialize file watcher
    _init_file_watcher(app)
    
    # Register routes
    from app.routes import register_routes
    register_routes(app)
    
    # Register template filters
    @app.template_filter('from_json')
    def from_json(value):
        """Parse JSON string into Python object for templates."""
        if not value:
            return {}
        try:
            import json
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    
    # Verify templates exist
    _verify_templates(app)
    
    return app

def _verify_templates(app):
    """Verify that required templates exist."""
    template_dir = app.template_folder
    required_templates = [
        'index.html',
        'search.html',
        'document_viewer.html',
        'folder_browser.html',
        'settings.html',
        'error.html'
    ]
    
    missing = []
    for template in required_templates:
        if not os.path.exists(os.path.join(template_dir, template)):
            missing.append(template)
    
    if missing:
        app.logger.warning(f"Missing templates: {', '.join(missing)}")
        app.logger.warning(f"Template dir: {template_dir}")
        app.logger.warning(f"Templates should be in: {os.path.abspath(template_dir)}")
    else:
        app.logger.info(f"All templates found in {template_dir}")

def _init_file_watcher(app):
    """Initialize the file watcher."""
    def on_new_file(rel_path):
        """Handle new file detection."""
        from pathlib import Path
        import os
        
        filepath = Path(os.path.join(app.config['UPLOAD_FOLDER'], rel_path))
        app.logger.info(f"New file detected: {rel_path}")
        
        doc_id = app.db.add_document(filepath)
        if doc_id:
            app.logger.info(f"Added document to database with ID: {doc_id}")
            
            # Try text extraction first
            app.pdf_processor.process_document(filepath, doc_id, app.db)
            
            # Queue for OCR processing
            if app.ocr_queue.add_to_queue(doc_id, filepath):
                app.logger.info(f"Document {rel_path} queued for OCR processing")
            else:
                app.logger.error(f"Failed to queue document {rel_path} for OCR processing")
    
    def on_file_removed(rel_path):
        """Handle file removal."""
        from pathlib import Path
        import os
        
        filepath = Path(os.path.join(app.config['UPLOAD_FOLDER'], rel_path))
        app.logger.info(f"File removed: {rel_path}")
        
        if app.db.mark_document_removed(filepath):
            app.logger.info(f"Marked document as removed in database: {rel_path}")
        else:
            app.logger.warning(f"Failed to mark document as removed in database: {rel_path}")
    
    # Register callbacks
    app.file_watcher.register_callback(on_new_file)
    app.file_watcher.register_removal_callback(on_file_removed)
    
    # Start file watcher
    app.file_watcher.start()
    
    # Start OCR queue
    app.ocr_queue.start_processing()