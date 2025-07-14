# db_manager.py (modified)
import sqlite3
import logging
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
import threading
import os
import json
from db.schema_manager import SchemaManager  # Import the new schema manager

class DatabaseManager:
    def __init__(self, db_path: str = "pdf_index.db"):
        self.db_path = db_path
        self._local = threading.local()
        self.setup_logging()
        
        # Use the schema manager for initialization
        self.schema_manager = SchemaManager(db_path)
        self.initialize_database()
    
    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
            )
            self._local.conn.execute("PRAGMA foreign_keys = ON")
            self._local.conn.row_factory = sqlite3.Row
            
        return self._local.conn

    def initialize_database(self):
        """Initialize the database using the schema manager."""
        self.logger.info("Initializing database...")
        success = self.schema_manager.initialize_database()
        
        if success:
            self.logger.info("Database initialization completed successfully")
        else:
            self.logger.error("Failed to initialize database")
            raise RuntimeError("Database initialization failed")

    # (add_document, mark_document_removed, update_processing_progress, etc.)
    def add_document(self, filepath: Path) -> Optional[int]:
        """Add a new PDF document to the database or resurrect if marked as removed."""
        try:
            # Use absolute path for storage
            absolute_path = str(filepath.absolute())
            filename = filepath.name
            
            # Extract folder path relative to base directory
            folder_path = ""
            try:
                # Try multiple ways to get the base directory
                base_dir = None
                
                # Method 1: Try Flask current_app
                try:
                    from flask import current_app
                    if current_app:
                        base_dir = Path(current_app.config.get('UPLOAD_FOLDER', '/mnt/onyx'))
                except (ImportError, RuntimeError):
                    pass
                
                # Method 2: Use environment variable
                if not base_dir:
                    base_dir = Path(os.environ.get('UPLOAD_FOLDER', '/mnt/onyx'))
                
                # Method 3: Default to /mnt/onyx
                if not base_dir:
                    base_dir = Path('/mnt/onyx')
                
                # Now try to extract folder path
                if filepath.is_absolute() and filepath.is_relative_to(base_dir):
                    rel_path = filepath.relative_to(base_dir)
                    if len(rel_path.parts) > 1:
                        folder_path = str(rel_path.parent)
                    else:
                        folder_path = ""
                else:
                    # If not under base directory, try to extract meaningful folder structure
                    # Look for device patterns like Go103/Notebooks/Moffitt
                    path_str = str(filepath)
                    if '/mnt/onyx/' in path_str:
                        # Extract everything after /mnt/onyx/
                        idx = path_str.find('/mnt/onyx/') + 10
                        remaining = path_str[idx:]
                        parts = remaining.split('/')
                        if len(parts) > 1:
                            folder_path = '/'.join(parts[:-1])  # Everything except filename
                    
            except Exception as e:
                self.logger.warning(f"Could not determine folder path for {filepath}: {e}")
                folder_path = ""
            
            self.logger.info(f"Adding document: {filename}, folder_path: '{folder_path}', absolute_path: {absolute_path}")
            
            # Extract metadata from filename
            import re
            
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
                        datetime(int(year), int(month), int(day))
                        metadata['date'] = f"{year}-{month}-{day}"
                        # Use the remaining part for title
                        remaining = os.path.splitext(remaining)[0]  # Remove extension
                        metadata['title'] = remaining.replace('_', ' ').replace('-', ' ').strip()
                    except ValueError:
                        # If date is invalid, just use the whole filename
                        metadata['title'] = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').strip()
                else:
                    # No date pattern, just use the filename without extension
                    metadata['title'] = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').strip()
                
                return metadata
                
            # Extract and store filename metadata
            filename_metadata = extract_filename_metadata(filename)
            metadata_json = json.dumps(filename_metadata)
            
            file_stats = filepath.stat()
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # First check if this path exists but is marked as removed
                cursor.execute("""
                    SELECT id FROM pdf_documents 
                    WHERE relative_path = ? AND processing_status = 'removed'
                """, (absolute_path,))
                
                existing = cursor.fetchone()
                if existing:
                    # Resurrect the document
                    cursor.execute("""
                        UPDATE pdf_documents
                        SET processing_status = 'pending',
                            processing_progress = 0.0,
                            file_size_bytes = ?,
                            file_modified_at = ?,
                            last_indexed_at = ?,
                            folder_path = ?,
                            extracted_metadata = ?
                        WHERE id = ?
                    """, (
                        file_stats.st_size,
                        datetime.fromtimestamp(file_stats.st_mtime),
                        datetime.now(),
                        folder_path,
                        metadata_json,
                        existing['id']
                    ))
                    self.logger.info(f"Resurrected document {filepath.name} with ID {existing['id']}")
                    return existing['id']
                
                # If not found or not removed, insert new record
                cursor.execute("""
                    INSERT INTO pdf_documents (
                        filename,
                        relative_path,
                        folder_path,
                        file_size_bytes,
                        file_created_at,
                        file_modified_at,
                        first_indexed_at,
                        last_indexed_at,
                        processing_status,
                        extracted_metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    filename,
                    absolute_path,  # Store absolute path instead of relative path
                    folder_path,
                    file_stats.st_size,
                    datetime.fromtimestamp(file_stats.st_ctime),
                    datetime.fromtimestamp(file_stats.st_mtime),
                    datetime.now(),
                    datetime.now(),
                    'pending',
                    metadata_json
                ))
                doc_id = cursor.lastrowid
                self.logger.info(f"Added document {filepath.name} with ID {doc_id}")
                return doc_id
                    
        except sqlite3.Error as e:
            self.logger.error(f"Error adding document {filepath}: {e}")
            return None
                
    def store_document_annotations(self, doc_id: int, annotations: list) -> bool:
        """Store annotations for a document, preventing duplicates."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # First, ensure the table exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS document_annotations (
                        id INTEGER PRIMARY KEY,
                        doc_id INTEGER,
                        page_number INTEGER,
                        annotation_type TEXT,
                        text TEXT,
                        confidence FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (doc_id) REFERENCES pdf_documents(id)
                    )
                """)
                
                # Remove any existing annotations for this document to prevent duplicates
                cursor.execute("""
                    DELETE FROM document_annotations
                    WHERE doc_id = ?
                """, (doc_id,))
                
                # Now insert the new annotations
                for annotation in annotations:
                    cursor.execute("""
                        INSERT INTO document_annotations (
                            doc_id, 
                            page_number, 
                            annotation_type, 
                            text,
                            confidence
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        doc_id,
                        annotation['page_number'],
                        annotation['annotation_type'],
                        annotation['text'],
                        annotation.get('confidence', 0.0)
                    ))
                
                conn.commit()
                self.logger.info(f"Stored {len(annotations)} annotations for document {doc_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing annotations for document {doc_id}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def mark_document_removed(self, filepath: Path) -> bool:
        try:
            # Use absolute path
            absolute_path = str(filepath.absolute())

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE pdf_documents
                    SET processing_status = 'removed',
                        last_indexed_at = ?
                    WHERE relative_path = ?
                    RETURNING id
                """, (datetime.now(), absolute_path))

                result = cursor.fetchone()
                if result:
                    self.logger.info(f"Marked document as removed: {filepath}")
                    return True
                else:
                    self.logger.warning(f"Document not found for removal: {filepath}")
                    return False

        except sqlite3.Error as e:
            self.logger.error(f"Error marking document as removed {filepath}: {e}")
            return False

    def update_processing_progress(self, doc_id: int, progress: float, status_message: Optional[str] = None) -> bool:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                if status_message:
                    cursor.execute("""
                        UPDATE pdf_documents
                        SET processing_progress = ?,
                            processing_error = ?,
                            last_indexed_at = ?
                        WHERE id = ?
                    RETURNING id
                    """, (progress, status_message, datetime.now(), doc_id))
                else:
                    cursor.execute("""
                        UPDATE pdf_documents
                        SET processing_progress = ?,
                            last_indexed_at = ?
                        WHERE id = ?
                        RETURNING id
                    """, (progress, datetime.now(), doc_id))

                result = cursor.fetchone()
                if result:
                    self.logger.info(
                        f"Updated document {doc_id} progress to {progress:.1f}% "
                        f"{f'({status_message})' if status_message else ''}"
                    )
                    return True
                else:
                    self.logger.warning(f"Document not found: {doc_id}")
            return False

        except sqlite3.Error as e:
            self.logger.error(f"Error updating progress for document {doc_id}: {e}")
            return False

    def mark_ocr_started(self, doc_id: int) -> bool:
        """Mark a document as being processed for OCR."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE pdf_documents
                    SET processing_status = 'processing',
                        processing_progress = 0.0,
                        last_indexed_at = ?,
                        ocr_processed = FALSE
                    WHERE id = ?
                    RETURNING id
                """, (datetime.now(), doc_id))
                
                result = cursor.fetchone()
                if result:
                    self.logger.info(f"Marked document {doc_id} for OCR processing")
                    return True
                else:
                    self.logger.warning(f"Document not found: {doc_id}")
                    return False
                    
        except sqlite3.Error as e:
            self.logger.error(f"Error marking document {doc_id} for OCR: {e}")
            return False

    def store_ocr_results(self, doc_id: int, page_data: Dict[int, Dict[str, any]]) -> bool:
        """Store OCR results and metadata for a document."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get document metadata
                cursor.execute("""
                    SELECT filename, extracted_metadata 
                    FROM pdf_documents 
                    WHERE id = ?
                """, (doc_id,))
                
                doc_info = cursor.fetchone()
                if not doc_info:
                    self.logger.error(f"Document {doc_id} not found")
                    return False
                    
                filename = doc_info['filename']
                metadata_json = doc_info['extracted_metadata']
                
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except json.JSONDecodeError:
                    metadata = {}
                
                # Create a metadata header to append to the OCR text
                metadata_header = f"Document: {filename}\n"
                if 'date' in metadata:
                    metadata_header += f"Date: {metadata['date']}\n"
                if 'title' in metadata:
                    metadata_header += f"Title: {metadata['title']}\n"
                metadata_header += "\n---\n\n"
                
                # Calculate overall statistics
                total_words = sum(page['word_count'] for page in page_data.values())
                
                # Update the main document record
                cursor.execute("""
                    UPDATE pdf_documents
                    SET processing_status = 'completed',
                        ocr_processed = TRUE,
                        ocr_last_processed_at = ?,
                        last_indexed_at = ?,
                        processing_progress = 100.0
                    WHERE id = ?
                """, (datetime.now(), datetime.now(), doc_id))
                
                # Store individual page data - append metadata to first page
                for page_number, data in page_data.items():
                    text = data['text']
                    
                    # Add metadata header to first page only
                    if page_number == 1:
                        text = metadata_header + text
                    
                    cursor.execute("""
                        INSERT INTO pdf_text_content (
                            pdf_id,
                            page_number,
                            ocr_text,
                            processed_at,
                            image_path
                        ) VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (pdf_id, page_number) 
                        DO UPDATE SET
                            ocr_text = excluded.ocr_text,
                            processed_at = excluded.processed_at,
                            image_path = excluded.image_path
                    """, (
                        doc_id,
                        page_number,
                        text,
                        data['processed_at'],
                        data.get('image_path', None)
                    ))
                
                self.logger.info(
                    f"Stored OCR results for document {doc_id}: "
                    f"{total_words} words across {len(page_data)} pages"
                )
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Error storing OCR results for document {doc_id}: {e}")
            return False
        
    def search_documents_with_folder_filter(self, query: str, folder_filter: str = "", limit: int = 20) -> List[Dict]:
        """
        Search for documents with optional folder filtering.
        
        Args:
            query: The search query
            folder_filter: Optional folder path or device name to filter results
            limit: Maximum number of results to return
                
        Returns:
            List of matching documents with relevance info
        """
        try:
            # Split query into keywords
            keywords = [k.strip().lower() for k in query.split() if k.strip()]
            if not keywords:
                return []
                    
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Build search conditions
                like_clauses = []
                params = []
                
                for keyword in keywords:
                    # Search in text_content, ocr_text, and extracted_metadata
                    like_clauses.append("""(
                        LOWER(p.text_content) LIKE ? OR 
                        LOWER(p.ocr_text) LIKE ? OR 
                        LOWER(d.extracted_metadata) LIKE ?
                    )""")
                    params.append(f"%{keyword}%")
                    params.append(f"%{keyword}%")
                    params.append(f"%{keyword}%")
                
                # Add folder filter if provided
                folder_clause = ""
                if folder_filter:
                    # Check if this is a device filter (no slashes)
                    if '/' not in folder_filter:
                        folder_clause = "AND d.folder_path LIKE ?"
                        params.append(f"{folder_filter}/%")
                    else:
                        # Normal folder filter
                        folder_clause = "AND d.folder_path = ?"
                        params.append(folder_filter)
                
                # Query across all pages of all documents
                query = f"""
                    SELECT 
                        d.id as doc_id,
                        d.filename,
                        d.relative_path,
                        d.folder_path,
                        COUNT(DISTINCT p.page_number) as matching_pages,
                        d.pdf_title,
                        d.pdf_author,
                        d.word_count,
                        d.extracted_metadata,
                        GROUP_CONCAT(DISTINCT p.page_number) as page_numbers
                    FROM 
                        pdf_documents d
                    JOIN 
                        pdf_text_content p ON d.id = p.pdf_id
                    WHERE 
                        d.processing_status = 'completed'
                        AND ({" OR ".join(like_clauses)})
                        {folder_clause}
                    GROUP BY 
                        d.id
                    ORDER BY 
                        COUNT(*) DESC,
                        d.word_count DESC
                    LIMIT ?
                """
                
                cursor.execute(query, params + [limit])
                
                # Convert to list of dictionaries
                results = []
                for row in cursor.fetchall():
                    doc = dict(row)
                    
                    # Get text snippets containing query terms
                    snippets = self.get_search_snippets(doc['doc_id'], keywords)
                    doc['snippets'] = snippets
                    
                    results.append(doc)
                    
                return results
                    
        except sqlite3.Error as e:
            self.logger.error(f"Error searching documents: {e}")
            return []

    def get_active_documents(self, status: Optional[str] = None, sort_by: str = 'last_indexed_at', order: str = 'desc') -> List[Dict]:
        try:
            where_clauses = ["processing_status != 'removed'"]
            params = []

            if status:
                where_clauses.append("processing_status = ?")
                params.append(status)

            valid_sort_fields = {
                'last_indexed_at', 'filename', 'file_size_bytes',
                'processing_status', 'ocr_processed'
            }
            if sort_by not in valid_sort_fields:
                sort_by = 'last_indexed_at'

            order = 'DESC' if order.lower() != 'asc' else 'ASC'

            with self.get_connection() as conn:
                cursor = conn.cursor()
                query = f"""
                    SELECT id, filename, relative_path, processing_status,
                           last_indexed_at, ocr_processed, processing_progress,
                           file_size_bytes, folder_path, extracted_metadata
                    FROM pdf_documents
                    WHERE {' AND '.join(where_clauses)}
                    ORDER BY {sort_by} {order}
                """

                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            self.logger.error(f"Error fetching active documents: {e}")
            return []
    
    def get_setting(self, category: str, key: str, decrypt: bool = False) -> Optional[str]:
        """Get a setting value from the database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT value, encrypted FROM settings 
                    WHERE category = ? AND key = ?
                """, (category, key))
                
                result = cursor.fetchone()
                if result:
                    value, is_encrypted = result
                    if is_encrypted and decrypt:
                        # Import here to avoid circular imports
                        from processing.caldav_client import CalDAVTodoClient
                        client = CalDAVTodoClient()
                        return client.decrypt_password(value)
                    return value
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting setting {category}.{key}: {e}")
            return None
    
    def set_setting(self, category: str, key: str, value: str, encrypt: bool = False) -> bool:
        """Set a setting value in the database."""
        try:
            encrypted_value = value
            if encrypt and value:
                # Import here to avoid circular imports
                from processing.caldav_client import CalDAVTodoClient
                client = CalDAVTodoClient()
                encrypted_value = client.encrypt_password(value)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO settings (category, key, value, encrypted, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (category, key, encrypted_value, encrypt, datetime.now()))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error setting {category}.{key}: {e}")
            return False
    
    def get_caldav_settings(self) -> Dict[str, any]:
        """Get CalDAV settings."""
        settings = {
            'enabled': self.get_setting('caldav', 'enabled') == 'true',
            'url': self.get_setting('caldav', 'url'),
            'username': self.get_setting('caldav', 'username'),
            'password': self.get_setting('caldav', 'password', decrypt=True),
            'calendar': self.get_setting('caldav', 'calendar') or 'todos',
            'base_url': self.get_setting('caldav', 'base_url') or 'http://localhost:5000'
        }
        return settings

    def store_extracted_text(self, doc_id: int, page_data: Dict[int, Dict[str, any]]) -> bool:
        """Store extracted text and metadata for a document."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Calculate overall statistics
                total_words = sum(page['word_count'] for page in page_data.values())
                
                # Update main document record
                cursor.execute("""
                    UPDATE pdf_documents
                    SET processing_status = 'completed',
                        word_count = ?,
                        has_text_content = TRUE,
                        last_indexed_at = ?,
                        processing_progress = 100.0
                    WHERE id = ?
                """, (total_words, datetime.now(), doc_id))

                for page_number, data in page_data.items():
                    cursor.execute("""
                        INSERT INTO pdf_text_content (
                            pdf_id,
                            page_number,
                            text_content,
                            processed_at
                        ) VALUES (?, ?, ?, ?)
                        ON CONFLICT (pdf_id, page_number)
                        DO UPDATE SET
                            text_content = excluded.text_content,
                            processed_at = excluded.processed_at
                    """, (
                        doc_id,
                        page_number,
                        data['text'],
                        data['processed_at']
                    ))

                self.logger.info(f"Stored extracted text for document {doc_id}: {total_words} words across {len(page_data)} pages")
                return True

        except sqlite3.Error as e:
            self.logger.error(f"Error storing extracted text for document {doc_id}: {e}")
            return False

    def get_document_text(self, doc_id: int, page_number: Optional[int] = None) -> Optional[Dict]:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                if page_number is not None:
                    # Query for a specific page - include all needed columns
                    cursor.execute("""
                        SELECT text_content, ocr_text, processed_at, image_path
                        FROM pdf_text_content
                        WHERE pdf_id = ? AND page_number = ?
                    """, (doc_id, page_number))
                    row = cursor.fetchone()

                    if row:
                        # Choose the best text content available
                        text = self._choose_best_text_content(row)
                        
                        # Determine the source
                        source = 'ocr' if row['ocr_text'] and row['ocr_text'].strip() else 'extracted'
                        
                        # Safely access image_path
                        try:
                            image_path = row['image_path']
                        except (IndexError, KeyError):
                            image_path = None
                        
                        return {
                            'page_number': page_number,
                            'text': text,
                            'processed_at': row['processed_at'],
                            'image_path': image_path,
                            'source': source
                        }
                    return None

                else:
                    # Query for all pages - include all needed columns
                    cursor.execute("""
                        SELECT page_number, text_content, ocr_text, processed_at, image_path
                        FROM pdf_text_content
                        WHERE pdf_id = ?
                        ORDER BY page_number
                    """, (doc_id,))

                    results = {}
                    for row in cursor.fetchall():
                        # Choose the best text content available
                        text = self._choose_best_text_content(row)
                        
                        # Determine the source
                        source = 'ocr' if row['ocr_text'] and row['ocr_text'].strip() else 'extracted'
                        
                        # Safely access image_path
                        try:
                            image_path = row['image_path']
                        except (IndexError, KeyError):
                            image_path = None
                        
                        results[row['page_number']] = {
                            'text': text,
                            'processed_at': row['processed_at'],
                            'image_path': image_path,
                            'source': source
                        }
                    return results

        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving text for document {doc_id}: {e}")
            return None

    def _choose_best_text_content(self, row) -> str:
        """
        Choose the best text content from a database row.
        Prioritizes OCR text, but falls back to extracted text if OCR isn't available.
        
        Args:
            row: A database row containing text_content and ocr_text
            
        Returns:
            The best available text content
        """
        # Safely access the text columns
        try:
            ocr_text = row['ocr_text']
        except (IndexError, KeyError):
            ocr_text = None
            
        try:
            extracted_text = row['text_content']
        except (IndexError, KeyError):
            extracted_text = None
        
        # Prioritize OCR text if it exists and isn't just whitespace
        if ocr_text and ocr_text.strip():
            return ocr_text
        
        # Fall back to extracted text
        if extracted_text and extracted_text.strip():
            return extracted_text
        
        # If neither is available, return an empty string
        return ""

    def mark_text_extraction_started(self, doc_id: int) -> bool:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE pdf_documents
                    SET processing_status = 'processing',
                        processing_progress = 0.0,
                        last_indexed_at = ?
                    WHERE id = ?
                    RETURNING id
                """, (datetime.now(), doc_id))

                result = cursor.fetchone()
                if result:
                    self.logger.info(f"Marked document {doc_id} as processing")
                    return True
                else:
                    self.logger.warning(f"Document not found: {doc_id}")
                    return False

        except sqlite3.Error as e:
            self.logger.error(f"Error marking document {doc_id} as processing: {e}")
            return False

    def reset_document_status(self, filepath: Path) -> bool:
        """Reset a document's processing status to allow reprocessing."""
        try:
            relative_path = str(filepath.relative_to(Path.cwd()))
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE pdf_documents 
                    SET processing_status = 'pending',
                        processing_progress = 0.0,
                        ocr_processed = FALSE
                    WHERE relative_path = ?
                    RETURNING id
                """, (relative_path,))
                
                result = cursor.fetchone()
                if result:
                    self.logger.info(f"Reset processing status for: {filepath}")
                    return True
                else:
                    self.logger.warning(f"Document not found: {filepath}")
                    return False
                    
        except sqlite3.Error as e:
            self.logger.error(f"Error resetting document status: {e}")
            return False
        
    def get_document_by_id(self, doc_id: int) -> Optional[Dict]:
        """
        Get comprehensive document details by ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Dictionary with document details or None if not found
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        id, filename, relative_path, processing_status,
                        file_size_bytes, file_created_at, file_modified_at,
                        first_indexed_at, last_indexed_at, ocr_processed,
                        ocr_last_processed_at, processing_progress,
                        pdf_title, pdf_author, pdf_created_at, pdf_modified_at,
                        pdf_page_count, pdf_version, has_text_content,
                        has_images, word_count, language_detected,
                        folder_path, extracted_metadata
                    FROM pdf_documents
                    WHERE id = ?
                """, (doc_id,))
                
                result = cursor.fetchone()
                if result:
                    return dict(result)
                else:
                    self.logger.warning(f"Document ID not found: {doc_id}")
                    return None
                    
        except sqlite3.Error as e:
            self.logger.error(f"Error getting document by ID {doc_id}: {e}")
            return None

    def mark_ready_for_ocr(self, doc_id: int) -> bool:
        """Mark a document as ready for OCR processing."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE pdf_documents
                    SET processing_status = 'pending',
                        processing_progress = 0.0
                    WHERE id = ?
                    RETURNING id
                """, (doc_id,))
                
                result = cursor.fetchone()
                if result:
                    self.logger.info(f"Marked document {doc_id} as ready for OCR")
                    return True
                else:
                    self.logger.warning(f"Document not found: {doc_id}")
                    return False
                    
        except sqlite3.Error as e:
            self.logger.error(f"Error marking document {doc_id} for OCR: {e}")
            return False

    def search_documents(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Search for documents containing the query text in either extracted text or OCR text.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching documents with relevance info
        """
        try:
            # Split query into keywords
            keywords = [k.strip().lower() for k in query.split() if k.strip()]
            if not keywords:
                return []
                
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Build a query that ranks documents by number of keyword matches
                like_clauses = []
                params = []
                
                for keyword in keywords:
                    # Search in text_content, ocr_text, and extracted_metadata
                    like_clauses.append("""(
                        LOWER(p.text_content) LIKE ? OR 
                        LOWER(p.ocr_text) LIKE ? OR 
                        LOWER(d.extracted_metadata) LIKE ?
                    )""")

                    params.append(f"%{keyword}%")
                    params.append(f"%{keyword}%")
                    params.append(f"%{keyword}%")
                
                # Query across all pages of all documents
                cursor.execute(f"""
                    SELECT 
                        d.id as doc_id,
                        d.filename,
                        d.relative_path,
                        COUNT(DISTINCT p.page_number) as matching_pages,
                        d.pdf_title,
                        d.pdf_author,
                        d.word_count,
                        d.folder_path,
                        d.extracted_metadata,
                        GROUP_CONCAT(DISTINCT p.page_number) as page_numbers
                    FROM 
                        pdf_documents d
                    JOIN 
                        pdf_text_content p ON d.id = p.pdf_id
                    WHERE 
                        d.processing_status = 'completed'
                        AND ({" OR ".join(like_clauses)})
                    GROUP BY 
                        d.id
                    ORDER BY 
                        COUNT(*) DESC,
                        d.word_count DESC
                    LIMIT ?
                """, params + [limit])
                
                # Convert to list of dictionaries
                results = []
                for row in cursor.fetchall():
                    doc = dict(row)
                    
                    # Get text snippets containing query terms
                    snippets = self.get_search_snippets(doc['doc_id'], keywords)
                    doc['snippets'] = snippets
                    
                    results.append(doc)
                    
                return results
                    
        except sqlite3.Error as e:
            self.logger.error(f"Error searching documents: {e}")
            return []

    def get_search_snippets(self, doc_id: int, keywords: List[str], 
                            max_snippets: int = 3, 
                            context_chars: int = 100) -> List[Dict]:
        """
        Get text snippets containing search keywords with surrounding context.
        
        Args:
            doc_id: Document ID
            keywords: List of search keywords
            max_snippets: Maximum number of snippets to return
            context_chars: Number of characters of context around matches
            
        Returns:
            List of snippet dictionaries with page numbers and text
        """
        try:
            snippets = []
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get text content from all pages
                cursor.execute("""
                    SELECT page_number, text_content, ocr_text
                    FROM pdf_text_content
                    WHERE pdf_id = ?
                    ORDER BY page_number
                """, (doc_id,))
                
                pages = cursor.fetchall()
                
                # Find snippets in each page
                for page in pages:
                    page_number = page['page_number']
                    
                    # Process extracted text
                    if page['text_content']:
                        text = page['text_content']
                        text_lower = text.lower()
                        self._find_snippets_in_text(text, text_lower, keywords, page_number, snippets, max_snippets, context_chars, "extracted")
                    
                    # Process OCR text
                    if page['ocr_text']:
                        text = page['ocr_text']
                        text_lower = text.lower()
                        self._find_snippets_in_text(text, text_lower, keywords, page_number, snippets, max_snippets, context_chars, "ocr")
                    
                    # If we have enough snippets, stop searching
                    if len(snippets) >= max_snippets:
                        break
                        
                return snippets
                    
        except sqlite3.Error as e:
            self.logger.error(f"Error getting search snippets: {e}")
            return []
        
    def _find_snippets_in_text(self, text, text_lower, keywords, page_number, snippets, max_snippets, context_chars, source_type):
        """Helper method to find snippets in a text string."""
        for keyword in keywords:
            start_pos = 0
            while len(snippets) < max_snippets:
                pos = text_lower.find(keyword.lower(), start_pos)
                if pos == -1:
                    break
                    
                # Get snippet with context
                snippet_start = max(0, pos - context_chars)
                snippet_end = min(len(text), pos + len(keyword) + context_chars)
                
                # Find word boundaries
                while snippet_start > 0 and text[snippet_start].isalnum():
                    snippet_start -= 1
                    
                while snippet_end < len(text) and text[snippet_end].isalnum():
                    snippet_end += 1
                    
                snippet_text = text[snippet_start:snippet_end]
                
                # Add ellipsis if not at boundaries
                if snippet_start > 0:
                    snippet_text = "..." + snippet_text
                if snippet_end < len(text):
                    snippet_text = snippet_text + "..."
                    
                # Highlight the keyword (for HTML display)
                highlight_start = pos - snippet_start
                highlight_end = highlight_start + len(keyword)
                highlighted_text = (
                    snippet_text[:highlight_start] +
                    "<mark>" +
                    snippet_text[highlight_start:highlight_end] +
                    "</mark>" +
                    snippet_text[highlight_end:]
                )
                
                snippets.append({
                    'page': page_number,
                    'text': highlighted_text,
                    'source': source_type  # Indicate if this is from extracted text or OCR
                })
                
                # Move to next occurrence
                start_pos = pos + len(keyword)
    
    def reset_document_status_by_id(self, doc_id: int) -> bool:
        """
        Reset a document's processing status by ID.
        
        Args:
            doc_id: Document ID to reset
            
        Returns:
            bool: True if successful, False if document not found or error
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE pdf_documents 
                    SET processing_status = 'pending',
                        processing_progress = 0.0,
                        ocr_processed = FALSE
                    WHERE id = ?
                    RETURNING id
                """, (doc_id,))
                
                result = cursor.fetchone()
                if result:
                    self.logger.info(f"Reset processing status for document ID: {doc_id}")
                    return True
                else:
                    self.logger.warning(f"Document ID not found: {doc_id}")
                    return False
                    
        except sqlite3.Error as e:
            self.logger.error(f"Error resetting document status for ID {doc_id}: {e}")
            return False

    def close(self):
        """Close any open database connections."""
        if hasattr(self._local, 'conn'):
            try:
                self._local.conn.close()
                self.logger.info("Database connection closed")
            except sqlite3.Error as e:
                self.logger.error(f"Error closing database connection: {e}")