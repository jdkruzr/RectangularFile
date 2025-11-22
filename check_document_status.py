#!/usr/bin/env python3
"""
Quick script to check document status in database
"""

from db.db_manager import DatabaseManager
from config import config

db = DatabaseManager(config.DATABASE_PATH)

# Get all documents sorted by ID
with db.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, filename, processing_status, file_size_bytes, processing_error
        FROM pdf_documents
        ORDER BY id DESC
        LIMIT 10
    """)
    docs = cursor.fetchall()

print("=" * 80)
print("Recent documents in database:")
print("=" * 80)

for doc in docs:
    size_mb = doc['file_size_bytes'] / (1024 * 1024) if doc['file_size_bytes'] else 0
    status = doc['processing_status'] or 'unknown'
    error = doc['processing_error'] or ''

    print(f"\nID: {doc['id']}")
    print(f"  File: {doc['filename']}")
    print(f"  Status: {status}")
    print(f"  Size: {size_mb:.2f} MB")
    if error:
        print(f"  Reason: {error}")

print("\n" + "=" * 80)
