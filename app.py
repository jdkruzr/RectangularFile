# app.py
"""
Backward compatibility for the new modular structure.
This file redirects to the new application entry point.
"""

# Import necessary components to create the app
from db.db_manager import DatabaseManager
from processing.file_watcher import FileWatcher
from processing.pdf_processor import PDFProcessor
from processing.qwen_processor import QwenVLProcessor
from processing.ocr_queue_manager import OCRQueueManager
from app import create_app
import os

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
app = create_app(db, file_watcher, pdf_processor, ocr_processor, ocr_queue)

# This allows gunicorn to use "app:app" as the entry point