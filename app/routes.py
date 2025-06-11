# app/routes.py
from flask import render_template, jsonify, request, redirect, url_for, send_file
import os
from pathlib import Path
import time
import json

def register_routes(app):
    """Register application routes."""
    
    @app.route('/')
    def index():
        """Render the main page."""
        return render_template('index.html',
                              files=app.file_watcher.files,
                              polling_interval=app.file_watcher.polling_interval)
    
    @app.route('/settings/page')
    def settings_page():
        """Render the settings page."""
        return render_template('settings.html')
    
    @app.route('/reset/<int:doc_id>')
    def reset_processing(doc_id):
        """Reset processing status for a document by ID and trigger reprocessing."""
        # Get the document details
        document = app.db.get_document_by_id(doc_id)
        if not document:
            return jsonify(success=False, message="Document not found"), 404

        # Get the filepath
        filepath = Path(document['relative_path'])
        if not filepath.exists():
            return jsonify(success=False, message="File no longer exists"), 404

        # Reset status
        if app.db.reset_document_status_by_id(doc_id):
            # Try text extraction first
            app.pdf_processor.process_document(filepath, doc_id, app.db)
            
            # Queue for OCR processing
            if app.ocr_queue.add_to_queue(doc_id, filepath):
                return jsonify(success=True, message=f"Reset and queued document {doc_id} for OCR processing")
            else:
                return jsonify(success=False, message="Reset succeeded but queuing failed"), 500
        else:
            return jsonify(success=False, message="Failed to reset status"), 400
    
    # Continue defining the rest of the routes...
    # (files, search, document viewing, etc.)