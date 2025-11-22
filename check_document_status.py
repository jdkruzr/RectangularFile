#!/usr/bin/env python3
"""
Quick script to check document status in database

Usage:
    python check_document_status.py         # Show last 10 documents
    python check_document_status.py <id>    # Show specific document
"""

import sys
from db.db_manager import DatabaseManager
from config import config

db = DatabaseManager(config.DATABASE_PATH)

if len(sys.argv) > 1:
    # Check specific document
    try:
        doc_id = int(sys.argv[1])
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid document ID")
        sys.exit(1)

    doc_info = db.get_document_by_id(doc_id)

    if not doc_info:
        print(f"Document ID {doc_id} not found")
        sys.exit(1)

    print("=" * 80)
    print(f"Document #{doc_id}")
    print("=" * 80)

    size_mb = doc_info.get('file_size_bytes', 0) / (1024 * 1024)

    print(f"\nFilename: {doc_info.get('filename')}")
    print(f"Path: {doc_info.get('relative_path')}")
    print(f"Status: {doc_info.get('processing_status', 'unknown')}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"OCR Processed: {doc_info.get('ocr_processed', False)}")
    print(f"Progress: {doc_info.get('processing_progress', 0):.1f}%")

    if doc_info.get('processing_error'):
        print(f"Error/Reason: {doc_info['processing_error']}")

    if doc_info.get('last_indexed_at'):
        print(f"Last Indexed: {doc_info['last_indexed_at']}")

    print("\n" + "=" * 80)

else:
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
