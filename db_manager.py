import sqlite3
import logging
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
import threading
import os

class DatabaseManager:
    def __init__(self, db_path: str = "pdf_index.db"):
        self.db_path = db_path
        self._local = threading.local()
        self.setup_logging()
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
        self.logger.info("Checking database status...")
        
        database_exists = os.path.exists(self.db_path)
        if database_exists:
            self.logger.info(f"Found existing database at {self.db_path}")
        else:
            self.logger.info(f"No database found at {self.db_path}, will create new database")
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='pdf_documents'
                """)
                table_exists = cursor.fetchone() is not None

                if table_exists:
                    cursor.execute("PRAGMA table_info(pdf_documents)")
                    columns = [column[1] for column in cursor.fetchall()]
                    migrations_needed = []
                    if 'processing_progress' not in columns:
                        migrations_needed.append('Add processing_progress column')
                    if migrations_needed:
                        self.logger.info(f"Database needs migrations: {', '.join(migrations_needed)}")

                        if 'processing_progress' not in columns:
                            self.logger.info("Adding processing_progress column...")
                            conn.execute("""
                                ALTER TABLE pdf_documents
                                ADD COLUMN processing_progress FLOAT DEFAULT 0.0
                            """)
                            self.logger.info("Added processing_progress column")
                    else:
                        self.logger.info("Database schema is up to date")
                else:
                    self.logger.info("Database exists but needs tables created")
        except sqlite3.Error as e:
            self.logger.info(f"Error checking database status: {e}")
            self.logger.info("Will attempt to create schema from scratch")

        schema = """
        CREATE TABLE IF NOT EXISTS pdf_documents (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            file_size_bytes INTEGER,
            file_created_at TIMESTAMP,
            file_modified_at TIMESTAMP,
            file_hash TEXT,
            
            first_indexed_at TIMESTAMP,
            last_indexed_at TIMESTAMP,
            ocr_processed BOOLEAN DEFAULT FALSE,
            ocr_last_processed_at TIMESTAMP,
            processing_status TEXT DEFAULT 'pending',
            processing_error TEXT,
            processing_progress FLOAT DEFAULT 0.0,
            
            pdf_title TEXT,
            pdf_author TEXT,
            pdf_created_at TIMESTAMP,
            pdf_modified_at TIMESTAMP,
            pdf_page_count INTEGER,
            pdf_version TEXT,
            
            has_text_content BOOLEAN DEFAULT FALSE,
            has_images BOOLEAN DEFAULT FALSE,
            confidence_score FLOAT,
            word_count INTEGER DEFAULT 0,
            language_detected TEXT,
            search_terms TEXT,
            last_accessed_at TIMESTAMP,
            access_count INTEGER DEFAULT 0,
            
            UNIQUE(relative_path)
        );

        CREATE TABLE IF NOT EXISTS pdf_text_content (
            pdf_id INTEGER,
            page_number INTEGER,
            text_content TEXT,
            confidence_score FLOAT,
            processed_at TIMESTAMP,
            PRIMARY KEY (pdf_id, page_number),
            FOREIGN KEY (pdf_id) REFERENCES pdf_documents(id)
        );

        CREATE TABLE IF NOT EXISTS processing_jobs (
            id INTEGER PRIMARY KEY,
            pdf_id INTEGER,
            job_type TEXT,
            status TEXT DEFAULT 'pending',
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            error_message TEXT,
            FOREIGN KEY (pdf_id) REFERENCES pdf_documents(id)
        );

        CREATE TABLE IF NOT EXISTS pdf_pages (
            pdf_id INTEGER,
            page_number INTEGER,
            width_px INTEGER,
            height_px INTEGER,
            has_images BOOLEAN DEFAULT FALSE,
            has_text BOOLEAN DEFAULT FALSE,
            rotation_angle INTEGER DEFAULT 0,
            PRIMARY KEY (pdf_id, page_number),
            FOREIGN KEY (pdf_id) REFERENCES pdf_documents(id)
        );

        CREATE TABLE IF NOT EXISTS document_links (
            id INTEGER PRIMARY KEY,
            source_doc_id INTEGER,
            target_doc_id INTEGER,
            link_type TEXT,
            link_notes TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY (source_doc_id) REFERENCES pdf_documents(id),
            FOREIGN KEY (target_doc_id) REFERENCES pdf_documents(id)
        );

        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            parent_topic_id INTEGER NULL,
            created_at TIMESTAMP,
            FOREIGN KEY (parent_topic_id) REFERENCES topics(id)
        );

        CREATE TABLE IF NOT EXISTS document_topics (
            doc_id INTEGER,
            topic_id INTEGER,
            confidence_score FLOAT,
            assigned_at TIMESTAMP,
            PRIMARY KEY (doc_id, topic_id),
            FOREIGN KEY (doc_id) REFERENCES pdf_documents(id),
            FOREIGN KEY (topic_id) REFERENCES topics(id)
        );

        CREATE TABLE IF NOT EXISTS handwriting_profiles (
            id INTEGER PRIMARY KEY,
            profile_name TEXT UNIQUE,
            created_at TIMESTAMP,
            last_updated_at TIMESTAMP,
            training_sample_count INTEGER DEFAULT 0,
            average_confidence_score FLOAT
        );

        CREATE TABLE IF NOT EXISTS handwriting_training_data (
            id INTEGER PRIMARY KEY,
            profile_id INTEGER,
            original_text TEXT,
            corrected_text TEXT,
            context_before TEXT,
            context_after TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY (profile_id) REFERENCES handwriting_profiles(id)
        );

        CREATE INDEX IF NOT EXISTS idx_pdf_docs_status ON pdf_documents(processing_status);
        CREATE INDEX IF NOT EXISTS idx_pdf_docs_path ON pdf_documents(relative_path);
        CREATE INDEX IF NOT EXISTS idx_text_content_pdf ON pdf_text_content(pdf_id);
        CREATE INDEX IF NOT EXISTS idx_doc_topics_doc ON document_topics(doc_id);
        CREATE INDEX IF NOT EXISTS idx_doc_topics_topic ON document_topics(topic_id);
        """
        
        try:
            with self.get_connection() as conn:
                conn.executescript(schema)
                if not database_exists:
                    self.logger.info("New database created successfully")
                else:
                    self.logger.info("Database schema verification complete")
        except sqlite3.Error as e:
            self.logger.error(f"Error managing database schema: {e}")
            raise

    def add_document(self, filepath: Path) -> Optional[int]:
        try:
            relative_path = str(filepath.relative_to(Path.cwd()))
            file_stats = filepath.stat()
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO pdf_documents (
                        filename,
                        relative_path,
                        file_size_bytes,
                        file_created_at,
                        file_modified_at,
                        first_indexed_at,
                        last_indexed_at,
                        processing_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    filepath.name,
                    relative_path,
                    file_stats.st_size,
                    datetime.fromtimestamp(file_stats.st_ctime),
                    datetime.fromtimestamp(file_stats.st_mtime),
                    datetime.now(),
                    datetime.now(),
                    'pending'
                ))
                doc_id = cursor.lastrowid
                self.logger.info(f"Added document {filepath.name} with ID {doc_id}")
                return doc_id
                
        except sqlite3.Error as e:
            self.logger.error(f"Error adding document {filepath}: {e}")
            return None

    def mark_document_removed(self, filepath: Path) -> bool:
        try:
            relative_path = str(filepath.relative_to(Path.cwd()))

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE pdf_documents
                    SET processing_status = 'removed',
                        last_indexed_at = ?
                    WHERE relative_path = ?
                    RETURNING id
                """, (datetime.now(), relative_path))

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
                           file_size_bytes
                    FROM pdf_documents
                    WHERE {' AND '.join(where_clauses)}
                    ORDER BY {sort_by} {order}
                """

                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            self.logger.error(f"Error fetching active documents: {e}")
            return []

    def store_extracted_text(self, doc_id: int, page_data: Dict[int, Dict[str, any]]) -> bool:
        """Store extracted text and metadata for a document."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Calculate overall statistics
                total_words = sum(page['word_count'] for page in page_data.values())
                
                # Debug confidence scores
                page_confidences = [page['confidence'] for page in page_data.values()]
                self.logger.info(f"Page confidences for doc {doc_id}: {page_confidences}")

                avg_confidence = sum(page_confidences) / len(page_data) if page_data else 0
                self.logger.info(f"Average confidence for doc {doc_id}: {avg_confidence}")
                
                # Update main document record
                cursor.execute("""
                    UPDATE pdf_documents
                    SET processing_status = 'completed',
                        word_count = ?,
                        confidence_score = ?,
                        has_text_content = TRUE,
                        last_indexed_at = ?,
                        processing_progress = 100.0
                    WHERE id = ?
                """, (total_words, avg_confidence, datetime.now(), doc_id))

                for page_number, data in page_data.items():
                    cursor.execute("""
                        INSERT INTO pdf_text_content (
                            pdf_id,
                            page_number,
                            text_content,
                            confidence_score,
                            processed_at
                        ) VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (pdf_id, page_number)
                        DO UPDATE SET
                            text_content = excluded.text_content,
                            confidence_score = excluded.confidence_score,
                            processed_at = excluded.processed_at
                    """, (
                        doc_id,
                        page_number,
                        data['text'],
                        data['confidence'],
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
                    cursor.execute("""
                        SELECT text_content, confidence_score, processed_at
                        FROM pdf_text_content
                        WHERE pdf_id = ? AND page_number = ?
                    """, (doc_id, page_number))
                    row = cursor.fetchone()

                    if row:
                        return {
                            'page_number': page_number,
                            'text': row['text_content'],
                            'confidence': row['confidence_score'],
                            'processed_at': row['processed_at']
                        }
                    return None

                else:
                    cursor.execute("""
                        SELECT page_number, text_content, confidence_score, processed_at
                        FROM pdf_text_content
                        WHERE pdf_id = ?
                        ORDER BY page_number
                    """, (doc_id,))

                    results = {}
                    for row in cursor.fetchall():
                        results[row['page_number']] = {
                            'text': row['text_content'],
                            'confidence': row['confidence_score'],
                            'processed_at': row['processed_at']
                        }
                    return results

        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving text for document {doc_id}: {e}")
            return None

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
                        has_images, confidence_score, word_count, language_detected
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

    def search_documents(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Search for documents containing the query text.
        
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
                
                # First get documents with exact matches
                results = []
                
                # Build a query that ranks documents by number of keyword matches
                placeholders = ", ".join(["?"] * len(keywords))
                like_clauses = []
                params = []
                
                for keyword in keywords:
                    like_clauses.append("LOWER(text_content) LIKE ?")
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
                        d.confidence_score,
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
                    SELECT page_number, text_content
                    FROM pdf_text_content
                    WHERE pdf_id = ?
                    ORDER BY page_number
                """, (doc_id,))
                
                pages = cursor.fetchall()
                
                # Find snippets in each page
                for page in pages:
                    page_number = page['page_number']
                    text = page['text_content']
                    text_lower = text.lower()
                    
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
                                'text': highlighted_text
                            })
                            
                            # Move to next occurrence
                            start_pos = pos + len(keyword)
                    
                return snippets
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting search snippets: {e}")
            return []
    
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
            if hasattr(self._local, 'conn'):
                self._local.conn.close()
                delattr(self._local, 'conn')
