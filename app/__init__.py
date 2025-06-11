# app/__init__.py
from flask import Flask

def create_app(db_manager, file_watcher, pdf_processor, ocr_queue):
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
    
    return app

def _init_file_watcher(app):
    """Initialize the file watcher."""
    def on_new_file(rel_path):
        """Handle new file detection."""
        from pathlib import Path
        import os
        
        filepath = Path(os.path.join(app.config['UPLOAD_FOLDER'], rel_path))
        doc_id = app.db.add_document(filepath)
        if doc_id:
            # Try text extraction first
            app.pdf_processor.process_document(filepath, doc_id, app.db)
            
            # Queue for OCR processing
            app.ocr_queue.add_to_queue(doc_id, filepath)
    
    def on_file_removed(rel_path):
        """Handle file removal."""
        from pathlib import Path
        import os
        
        filepath = Path(os.path.join(app.config['UPLOAD_FOLDER'], rel_path))
        app.db.mark_document_removed(filepath)
    
    # Register callbacks
    app.file_watcher.register_callback(on_new_file)
    app.file_watcher.register_removal_callback(on_file_removed)
    
    # Start file watcher
    app.file_watcher.start()
    
    # Start OCR queue
    app.ocr_queue.start_processing()