#!/usr/bin/env python3
import sqlite3
import os

db_path = "/mnt/rectangularfile/pdf_index.db"
base_path = "/mnt/onyx/"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Find all documents with empty folder_path
cursor.execute("""
    SELECT id, relative_path, folder_path 
    FROM pdf_documents 
    WHERE (folder_path = '' OR folder_path IS NULL) 
    AND relative_path IS NOT NULL
""")

documents = cursor.fetchall()
print(f"Found {len(documents)} documents with empty folder_path")

# Update each one
for doc_id, relative_path, current_folder in documents:
    # Extract folder path from relative path
    if relative_path.startswith(base_path):
        # Remove base path
        path_without_base = relative_path[len(base_path):]
        # Remove filename to get just the folder
        folder_path = os.path.dirname(path_without_base)
        
        print(f"Doc {doc_id}: '{relative_path}' -> folder: '{folder_path}'")
        
        # Update the database
        cursor.execute("""
            UPDATE pdf_documents 
            SET folder_path = ? 
            WHERE id = ?
        """, (folder_path, doc_id))

conn.commit()
print(f"Updated {len(documents)} documents")
conn.close()
