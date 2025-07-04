"""Application initialization module for RectangularFile."""
import os
import json
from pathlib import Path
from flask import Flask
from flask_login import LoginManager

def create_app(
    db_manager, 
    file_watcher, 
    pdf_processor, 
    ocr_processor, 
    ocr_queue, 
    html_processor=None
):
    """
    Application factory.
    """
    # Get the absolute path to the templates directory
    template_dir = os.path.abspath('app/templates')
    static_dir = os.path.abspath('app/static')
    
    app = Flask(
        __name__, 
        template_folder=template_dir,
        static_folder=static_dir
    )
    
    # Configure app
    app.config.update(
        UPLOAD_FOLDER="/mnt/onyx",
        MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50MB max upload
        SECRET_KEY=os.environ.get('SECRET_KEY', 'your-default-secret-key-change-this'),
        SESSION_COOKIE_NAME='rectangularfile_session',
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax'
    )
    
    # Initialize Flask-Login AFTER app is created
    app.login_manager = LoginManager()
    app.login_manager.init_app(app)
    app.login_manager.login_view = 'login'
    app.login_manager.login_message = 'Please log in to access this page.'
    
    # Simple user class for single user
    from flask_login import UserMixin
    
    class User(UserMixin):
        def __init__(self, id):
            self.id = id
    
    @app.login_manager.user_loader
    def load_user(user_id):
        if user_id == "admin":
            return User(user_id)
        return None
    
    # Store the user class on app for routes to use
    app.User = User
    
    # Register components with app
    app.db = db_manager
    app.file_watcher = file_watcher
    app.pdf_processor = pdf_processor
    app.ocr_processor = ocr_processor
    app.ocr_queue = ocr_queue
    app.html_processor = html_processor
       
    # Initialize file watcher
    _init_file_watcher(app)
    
    # Import routes here to avoid circular imports
    from app.routes import register_routes
    
    # Register routes
    register_routes(app)
    
    # Register template filters
    @app.template_filter('from_json')
    def from_json(value):
        """Parse JSON string into Python object for templates."""
        if not value:
            return {}
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    
    # Verify templates exist
    _verify_templates(app)
    
    return app

def _verify_templates(app):
    """
    Verify that required templates exist.
    
    Args:
        app: Flask application instance
    """
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
    """
    Initialize the file watcher.
    
    Args:
        app: Flask application instance
    """
    # Use a non-protected attribute name
    if hasattr(app.file_watcher, 'initialized') and app.file_watcher.initialized:
        app.logger.info("File watcher already initialized")
        return
        
    app.file_watcher.initialized = True
    
    # Define callbacks
    def process_new_file(relative_path):
        """Process a newly detected file."""
        filepath = Path(os.path.join(app.config['UPLOAD_FOLDER'], relative_path))
        app.logger.info(f"Processing new file: {filepath}")
        
        # Check if this file is already in the database
        with app.db.get_connection() as conn:
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
            app.logger.info(f"File already processed: {filepath}")
            return
        
        # If in database but not completed, reset and process
        if existing:
            doc_id = existing['id']
            app.logger.info(f"Resetting existing document: {doc_id}")
            app.db.reset_document_status_by_id(doc_id)
        else:
            # Add to database if not found
            doc_id = app.db.add_document(filepath)
            if not doc_id:
                app.logger.error(f"Failed to add document to database: {filepath}")
                return
        
        # Choose processor based on file extension
        file_extension = filepath.suffix.lower()
        
        # Process based on file type
        if file_extension in ['.html', '.htm']:
            app.logger.info(f"Processing HTML file: {filepath}")
            # Use HTML processor if available
            if app.html_processor:
                app.html_processor.process_document(filepath, doc_id, app.db)
                app.logger.info(f"HTML processing complete for {filepath}")
            else:
                app.logger.warning(f"No HTML processor available for {filepath}")
                msg = "HTML file (no processor available)"
                app.db.update_processing_progress(doc_id, 100.0, msg)
        else:
            # Default PDF processing
            app.logger.info(f"Processing PDF file: {filepath}")
            app.pdf_processor.process_document(filepath, doc_id, app.db)
            
            # Only queue PDFs for OCR
            app.ocr_queue.add_to_queue(doc_id, filepath)

    def handle_removed_file(relative_path):
        """Handle a file that's been removed."""
        filepath = Path(os.path.join(app.config['UPLOAD_FOLDER'], relative_path))
        app.logger.info(f"Marking removed file: {filepath}")
        app.db.mark_document_removed(filepath)

    # Register callbacks
    app.file_watcher.register_callback(process_new_file)
    app.file_watcher.register_removal_callback(handle_removed_file)
    
    # Note: We don't start the file_watcher here to allow the main.py to
    # decide when to start it, typically after the app is initialized
    
    app.logger.info(f"File watcher initialized for: {app.config['UPLOAD_FOLDER']}")