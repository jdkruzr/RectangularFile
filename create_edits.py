#!/usr/bin/env python3
import sqlite3
import sys

def migrate():
    db_path = "/mnt/rectangularfile/pdf_index.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create edit_history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edit_history (
                id INTEGER PRIMARY KEY,
                doc_id INTEGER,
                page_number INTEGER,
                field_type TEXT,
                annotation_id INTEGER NULL,
                original_text TEXT,
                edited_text TEXT,
                edited_by TEXT DEFAULT 'user',
                edited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES pdf_documents(id),
                FOREIGN KEY (annotation_id) REFERENCES document_annotations(id)
            )
        """)
        
        conn.commit()
        print("Successfully created edit_history table")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()