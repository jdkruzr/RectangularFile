# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RectangularFile is a Flask-based document management system for handwritten notes from e-ink tablets. It uses local AI (Qwen2.5-VL-7B) for handwriting recognition and document processing.

## Key Commands

### Development & Testing
- `python main.py` - Run the application in development mode
- `python app.py` - Alternative entry point for compatibility
- `python model_test.py` - Test GPU and model functionality
- `python test_gpu.py` - Test GPU availability
- `python create_edits.py` - Utility for database edits
- `python db_migration_tool.py` - Database migration utility

### Dependencies
- `pip install -r requirements.txt` - Install all dependencies
- `python download_model.py` - Download the Qwen2.5-VL-7B model

### Production Deployment
- Uses systemd service file: `rectangular-file.service`
- Production path: `/mnt/onyx` for PDF storage
- Database path: `/mnt/rectangularfile/pdf_index.db`

## Architecture

### Core Components

1. **Flask Application** (`app/`)
   - Factory pattern with `create_app()` function
   - Single-user authentication via Flask-Login
   - Template-based UI with search, document viewer, and folder browser

2. **Database Layer** (`db/`)
   - SQLite database with schema versioning
   - `DatabaseManager` handles connections and operations
   - `SchemaManager` manages database migrations
   - Stores document metadata, OCR results, and annotations

3. **Document Processing** (`processing/`)
   - `FileWatcher` monitors filesystem changes
   - `PDFProcessor` handles PDF text extraction
   - `QwenVLProcessor` performs AI-powered handwriting recognition
   - `OCRQueueManager` manages async OCR processing
   - `HTMLProcessor` processes HTML files

4. **Utilities** (`utils/`)
   - `helpers.py` - Common utility functions
   - `wordcloud.py` - Word cloud generation

### Key Processing Flow

1. Files are detected by `FileWatcher` in monitored directories
2. `PDFProcessor` extracts basic text and metadata
3. `QwenVLProcessor` performs handwriting OCR on GPU
4. Results stored in database for search and browsing
5. Web interface provides search, viewing, and editing capabilities

### Configuration

All configuration is centralized in [config.py](config.py) with environment variable support:

**Required Environment Variables:**
- `SECRET_KEY` - Flask secret key (generate with `python -c 'import secrets; print(secrets.token_hex(32))'`)
- `APP_PASSWORD_HASH` - SHA256 hash of admin password

**Optional Environment Variables (with defaults):**
- `UPLOAD_FOLDER` - File monitoring directory (default: `/mnt/onyx`)
- `DATABASE_PATH` - SQLite database location (default: `/mnt/rectangularfile/pdf_index.db`)
- `MODEL_NAME` - HuggingFace model identifier (default: `Qwen/Qwen2.5-VL-7B-Instruct`)
- `MODEL_CACHE_DIR` - Transformers cache (default: `/mnt/rectangularfile/qwencache`)
- `DEBUG_IMAGES_DIR` - Debug output directory (default: `/mnt/rectangularfile/debug_images`)
- `POLLING_INTERVAL` - File watcher interval in seconds (default: `30.0`)
- `FLASK_HOST` - Server bind address (default: `0.0.0.0`)
- `FLASK_PORT` - Server port (default: `5000`)
- `FLASK_DEBUG` - Debug mode (default: `false`)

**Hardware Requirements:**
- GPU requirements: NVIDIA GPU with 16GB+ VRAM
- Model runs with INT8 quantization to fit in 16GB

### Database Schema

- `pdf_documents` - Document metadata and processing status
- `pdf_text_content` - Extracted and OCR text by page
- `document_annotations` - Detected annotations (green boxes, highlights)
- `edit_history` - Track user edits to transcriptions

### File Types Supported

- PDF files (primary focus)
- HTML files (basic support)
- Images (via PDF conversion)

## Important Notes

- This is a single-user application with simple authentication
- GPU processing is essential for handwriting recognition
- File watching uses polling by default (configurable interval)
- All processing happens locally - no cloud services
- Database uses SQLite with WAL mode for concurrent access