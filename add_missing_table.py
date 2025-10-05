#!/usr/bin/env python3
"""Add the missing document_annotations table to the database."""
import sqlite3
from config import config

def add_annotations_table():
    """Add the document_annotations table if it doesn't exist."""
    db_path = config.DATABASE_PATH

    print(f"Database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='document_annotations'
    """)

    if cursor.fetchone():
        print("✓ document_annotations table already exists")
    else:
        print("Creating document_annotations table...")
        cursor.execute("""
            CREATE TABLE document_annotations (
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
        conn.commit()
        print("✓ document_annotations table created successfully")

    conn.close()

if __name__ == "__main__":
    add_annotations_table()
