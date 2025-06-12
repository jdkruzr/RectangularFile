# app/__init__.py
from flask import Flask
import os

def create_app(db_manager, file_watcher, pdf_processor, ocr_processor, ocr_queue):
    """Application factory."""
    # Get the absolute path to the templates directory
    template_dir = os.path.abspath('app/templates')
    static_dir = os.path.abspath('app/static')
    
    app = Flask(__name__, 
                template_folder=template_dir,
                static_folder=static_dir)
    
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
    # Rest of the function remains the same
    # ...