#!/usr/bin/env python3
"""
Clean up all stuck Saber documents
"""

from db.db_manager import DatabaseManager
from config import config

db = DatabaseManager(config.DATABASE_PATH)
conn = db.get_connection()
cursor = conn.cursor()

# Find all .sbe and page_*.jpg documents
cursor.execute("""
    SELECT id, filename, relative_path, processing_status
    FROM pdf_documents
    WHERE filename LIKE '%.sbe'
       OR filename LIKE 'page_%.jpg'
       OR relative_path LIKE '%/tmp/saber%'
    ORDER BY id
""")

docs = cursor.fetchall()

if not docs:
    print("No Saber documents found to clean up")
else:
    print(f"Found {len(docs)} Saber-related documents to delete:\n")
    for doc in docs:
        print(f"  ID {doc['id']}: {doc['filename']} ({doc['processing_status']})")

    confirm = input(f"\nDelete all {len(docs)} documents? (y/n): ").strip().lower()

    if confirm == 'y':
        for doc in docs:
            success = db.delete_document(doc['id'], delete_file=False)
            if success:
                print(f"  ✓ Deleted ID {doc['id']}")
            else:
                print(f"  ✗ Failed to delete ID {doc['id']}")
        print(f"\nDone! Deleted {len(docs)} documents.")
    else:
        print("Cancelled.")
