# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RectangularFile is a Flask-based document management system for handwritten notes from e-ink tablets. It uses any OpenAI-compatible vision API (vLLM, Ollama, OpenAI, Claude, etc.) for handwriting recognition and document processing.

## Key Commands

### Development & Testing
- `python main.py` - Run the application in development mode
- `python app.py` - Alternative entry point for compatibility
- `python create_edits.py` - Utility for database edits
- `python db_migration_tool.py` - Database migration utility

### Dependencies
- `pip install -r requirements.txt` - Install all dependencies

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
   - `VisionAPIClient` performs AI-powered handwriting recognition via OpenAI-compatible API
   - `OCRQueueManager` manages async OCR processing
   - `DocumentArchiver` moves processed files to archive directory
   - `HTMLProcessor` processes HTML files
   - Modular document sources: `BooxPDFSource`, `SaberNoteSource`

4. **Utilities** (`utils/`)
   - `helpers.py` - Common utility functions
   - `wordcloud.py` - Word cloud generation

### Key Processing Flow

1. Files are detected by `FileWatcher` in monitored directories
2. `PDFProcessor` extracts basic text and metadata
3. `VisionAPIClient` performs handwriting OCR via external inference API
4. Results stored in database for search and browsing
5. `DocumentArchiver` moves processed files to archive (enabling re-upload detection)
6. Web interface provides search, viewing, and editing capabilities

### Configuration

All configuration is centralized in [config.py](config.py) with environment variable support:

**Required Environment Variables:**
- `SECRET_KEY` - Flask secret key (generate with `python -c 'import secrets; print(secrets.token_hex(32))'`)
- `APP_PASSWORD_HASH` - SHA256 hash of admin password

**Inference API Configuration:**
- `INFERENCE_API_BASE` - OpenAI-compatible API endpoint (default: `http://localhost:8000/v1`)
- `INFERENCE_API_KEY` - API key for cloud providers (optional for local servers)
- `INFERENCE_MODEL` - Model identifier (default: `Qwen/Qwen2.5-VL-7B-Instruct`)
- `INFERENCE_MAX_TOKENS` - Max response tokens (default: `2048`)
- `INFERENCE_TIMEOUT` - Request timeout in seconds (default: `120`)

**File Storage:**
- `UPLOAD_FOLDER` - File monitoring directory (default: `/mnt/onyx`)
- `DATABASE_PATH` - SQLite database location (default: `/mnt/rectangularfile/pdf_index.db`)
- `DEBUG_IMAGES_DIR` - Debug output directory (default: `/mnt/rectangularfile/debug_images`)

**Archive Configuration:**
- `ARCHIVE_ENABLED` - Enable document archiving (default: `true`)
- `ARCHIVE_FOLDER` - Archive directory (default: `/mnt/rectangularfile/archive`)
- `ARCHIVE_PRESERVE_STRUCTURE` - Maintain folder structure in archive (default: `true`)

**Document Sources:**
- `BOOX_ENABLED` - Enable Boox PDF source (default: `true`)
- `BOOX_FOLDER` - Boox watch directory (defaults to `UPLOAD_FOLDER`)
- `SABER_ENABLED` - Enable Saber notes source (default: `false`)
- `SABER_FOLDER` - Saber sync folder (default: `/mnt/webdav/saber`)
- `SABER_PASSWORD` - Saber encryption password

**Server Configuration:**
- `POLLING_INTERVAL` - File watcher interval in seconds (default: `30.0`)
- `FLASK_HOST` - Server bind address (default: `0.0.0.0`)
- `FLASK_PORT` - Server port (default: `5000`)
- `FLASK_DEBUG` - Debug mode (default: `false`)

### Inference API Backends

RectangularFile works with any OpenAI-compatible vision API:

**Local Servers:**
- **vLLM**: `INFERENCE_API_BASE=http://localhost:8000/v1`
- **Ollama**: `INFERENCE_API_BASE=http://localhost:11434/v1`
- **llama.cpp**: `INFERENCE_API_BASE=http://localhost:8080/v1`

**Cloud APIs:**
- **OpenAI**: `INFERENCE_API_BASE=https://api.openai.com/v1` with `INFERENCE_API_KEY=sk-...`
- **Anthropic** (via proxy): Use an OpenAI-compatible proxy

### Annotation Detection

The system detects colored annotations in handwritten notes:
- **RED ink** → TODOs/action items (synced to CalDAV if configured)
- **GREEN ink** → Tags/categories

### Database Schema

- `pdf_documents` - Document metadata, processing status, archive paths
- `pdf_text_content` - Extracted and OCR text by page
- `document_annotations` - Detected annotations (red TODOs, green tags)
- `edit_history` - Track user edits to transcriptions
- `settings` - Application settings (CalDAV, inference API, etc.)

### File Types Supported

- PDF files (primary focus)
- Saber encrypted notes (.sbe files)
- HTML files (basic support)

## Important Notes

- This is a single-user application with simple authentication
- No local GPU required — inference is handled by external API
- File watching uses polling by default (configurable interval)
- Processed documents are moved to archive to enable re-upload detection
- Database uses SQLite with WAL mode for concurrent access
