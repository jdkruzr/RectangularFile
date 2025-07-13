# schema_manager.py
import os
import sqlite3
import logging
from typing import List, Dict, Tuple
import json

class SchemaManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def get_base_schema(self) -> Dict[str, str]:
        """Returns a dictionary of table names to their creation SQL."""
        return {
            "pdf_documents": """
                CREATE TABLE IF NOT EXISTS pdf_documents (
                    id INTEGER PRIMARY KEY,
                    filename TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    folder_path TEXT DEFAULT '',
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
                    word_count INTEGER DEFAULT 0,
                    language_detected TEXT,
                    search_terms TEXT,
                    last_accessed_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    extracted_metadata TEXT,
                    
                    UNIQUE(relative_path)
                )
            """,
            "pdf_text_content": """
                CREATE TABLE IF NOT EXISTS pdf_text_content (
                    pdf_id INTEGER,
                    page_number INTEGER,
                    text_content TEXT,
                    processed_at TIMESTAMP,
                    ocr_text TEXT,
                    image_path TEXT,
                    PRIMARY KEY (pdf_id, page_number),
                    FOREIGN KEY (pdf_id) REFERENCES pdf_documents(id)
                )
            """,
            "processing_jobs": """
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    id INTEGER PRIMARY KEY,
                    pdf_id INTEGER,
                    job_type TEXT,
                    status TEXT DEFAULT 'pending',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    FOREIGN KEY (pdf_id) REFERENCES pdf_documents(id)
                )
            """,
            "pdf_pages": """
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
                )
            """,
            "document_links": """
                CREATE TABLE IF NOT EXISTS document_links (
                    id INTEGER PRIMARY KEY,
                    source_doc_id INTEGER,
                    target_doc_id INTEGER,
                    link_type TEXT,
                    link_notes TEXT,
                    created_at TIMESTAMP,
                    FOREIGN KEY (source_doc_id) REFERENCES pdf_documents(id),
                    FOREIGN KEY (target_doc_id) REFERENCES pdf_documents(id)
                )
            """,
            "topics": """
                CREATE TABLE IF NOT EXISTS topics (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE,
                    parent_topic_id INTEGER NULL,
                    created_at TIMESTAMP,
                    FOREIGN KEY (parent_topic_id) REFERENCES topics(id)
                )
            """,
            "document_topics": """
                CREATE TABLE IF NOT EXISTS document_topics (
                    doc_id INTEGER,
                    topic_id INTEGER,
                    assigned_at TIMESTAMP,
                    PRIMARY KEY (doc_id, topic_id),
                    FOREIGN KEY (doc_id) REFERENCES pdf_documents(id),
                    FOREIGN KEY (topic_id) REFERENCES topics(id)
                )
            """,
            "edit_history": """
                CREATE TABLE IF NOT EXISTS edit_history (
                    id INTEGER PRIMARY KEY,
                    doc_id INTEGER,
                    page_number INTEGER,
                    field_type TEXT, -- 'transcription' or 'annotation'
                    annotation_id INTEGER NULL, -- if editing an annotation
                    original_text TEXT,
                    edited_text TEXT,
                    edited_by TEXT DEFAULT 'user',
                    edited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES pdf_documents(id),
                    FOREIGN KEY (annotation_id) REFERENCES document_annotations(id)
                )
            """,
            "settings": """
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    encrypted BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(category, key)
                )
            """            
        }
    
    def get_indexes(self) -> Dict[str, str]:
        """Returns a dictionary of index names to their creation SQL."""
        return {
            "idx_pdf_docs_status": "CREATE INDEX IF NOT EXISTS idx_pdf_docs_status ON pdf_documents(processing_status)",
            "idx_pdf_docs_path": "CREATE INDEX IF NOT EXISTS idx_pdf_docs_path ON pdf_documents(relative_path)",
            "idx_pdf_docs_folder": "CREATE INDEX IF NOT EXISTS idx_pdf_docs_folder ON pdf_documents(folder_path)",
            "idx_text_content_pdf": "CREATE INDEX IF NOT EXISTS idx_text_content_pdf ON pdf_text_content(pdf_id)",
            "idx_doc_topics_doc": "CREATE INDEX IF NOT EXISTS idx_doc_topics_doc ON document_topics(doc_id)",
            "idx_doc_topics_topic": "CREATE INDEX IF NOT EXISTS idx_doc_topics_topic ON document_topics(topic_id)"
        }
    
    def get_migrations(self) -> List[Dict]:
        """Returns a list of migration definitions in order."""
        return [
            {
                "version": 1,
                "description": "Add processing_progress column",
                "sql": "ALTER TABLE pdf_documents ADD COLUMN processing_progress FLOAT DEFAULT 0.0"
            },
            {
                "version": 2,
                "description": "Add folder_path column",
                "sql": "ALTER TABLE pdf_documents ADD COLUMN folder_path TEXT DEFAULT ''"
            },
            {
                "version": 3,
                "description": "Add extracted_metadata column",
                "sql": "ALTER TABLE pdf_documents ADD COLUMN extracted_metadata TEXT"
            },
            {
                "version": 4,
                "description": "Add settings table for CalDAV integration",
                "sql": """CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    encrypted BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(category, key)
                )"""
            },
            # Future migrations would be added here
        ]
    
    def get_current_db_version(self, conn: sqlite3.Connection) -> int:
        """Get the current database version."""
        try:
            cursor = conn.cursor()
            # First check if the version table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='db_version'
            """)
            if not cursor.fetchone():
                # Create version table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE db_version (
                        version INTEGER PRIMARY KEY,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                return 0
                
            # Get the current version
            cursor.execute("SELECT MAX(version) FROM db_version")
            result = cursor.fetchone()[0]
            return result or 0
        except Exception as e:
            self.logger.error(f"Error getting DB version: {e}")
            return 0
    
    def check_and_apply_migrations(self) -> bool:
        """Check for and apply any needed migrations."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            current_version = self.get_current_db_version(conn)
            self.logger.info(f"Current database version: {current_version}")
            
            migrations = self.get_migrations()
            pending_migrations = [m for m in migrations if m["version"] > current_version]
            
            if not pending_migrations:
                self.logger.info("Database schema is up to date.")
                conn.close()
                return True
                
            self.logger.info(f"Found {len(pending_migrations)} pending migrations")
            
            for migration in pending_migrations:
                version = migration["version"]
                description = migration["description"]
                sql = migration["sql"]
                
                self.logger.info(f"Applying migration {version}: {description}")
                conn.execute(sql)
                
                # Record the migration
                conn.execute(
                    "INSERT INTO db_version (version) VALUES (?)",
                    (version,)
                )
                
                self.logger.info(f"Migration {version} applied successfully")
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying migrations: {e}")
            return False
    
    def initialize_database(self) -> bool:
        """Initialize the database schema or apply migrations as needed."""
        database_exists = os.path.exists(self.db_path)
        
        if database_exists:
            self.logger.info(f"Found existing database at {self.db_path}")
            return self.check_and_apply_migrations()
        
        self.logger.info(f"Creating new database at {self.db_path}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create all tables
            for table_name, create_sql in self.get_base_schema().items():
                self.logger.info(f"Creating table: {table_name}")
                conn.execute(create_sql)
            
            # Create all indexes
            for index_name, create_sql in self.get_indexes().items():
                self.logger.info(f"Creating index: {index_name}")
                conn.execute(create_sql)
            
            # Create and initialize version table
            conn.execute("""
                CREATE TABLE db_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Set to the latest migration version
            latest_version = max(m["version"] for m in self.get_migrations()) if self.get_migrations() else 0
            conn.execute("INSERT INTO db_version (version) VALUES (?)", (latest_version,))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Database initialized successfully at version {latest_version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            return False