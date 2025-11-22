#!/usr/bin/env python3
"""
Utility script to mark documents as skipped.

Usage:
    python skip_document.py <doc_id> ["reason"]
    python skip_document.py --search "filename pattern"
"""

import sys
from pathlib import Path
from db.db_manager import DatabaseManager
from config import config

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python skip_document.py <doc_id> [reason]")
        print("  python skip_document.py --search <pattern>")
        print("\nExamples:")
        print("  python skip_document.py 123 \"Too large\"")
        print("  python skip_document.py --search \"My Daily Organizer\"")
        sys.exit(1)

    db = DatabaseManager(config.DATABASE_PATH)

    # Search mode
    if sys.argv[1] == "--search":
        if len(sys.argv) < 3:
            print("Error: --search requires a search pattern")
            sys.exit(1)

        pattern = sys.argv[2]
        print(f"Searching for documents matching: {pattern}")

        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, filename, relative_path, processing_status, file_size_bytes
                FROM pdf_documents
                WHERE filename LIKE ? OR relative_path LIKE ?
                ORDER BY file_size_bytes DESC
            """, (f"%{pattern}%", f"%{pattern}%"))

            results = cursor.fetchall()

        if not results:
            print("No documents found matching that pattern.")
            sys.exit(0)

        print(f"\nFound {len(results)} document(s):\n")
        for doc in results:
            size_mb = doc['file_size_bytes'] / (1024 * 1024) if doc['file_size_bytes'] else 0
            status = doc['processing_status'] or 'unknown'
            print(f"  ID: {doc['id']}")
            print(f"  File: {doc['filename']}")
            print(f"  Size: {size_mb:.2f} MB")
            print(f"  Status: {status}")
            print(f"  Path: {doc['relative_path']}")
            print()

        print(f"To skip a document, run: python skip_document.py <ID> [reason]")
        sys.exit(0)

    # Skip mode
    try:
        doc_id = int(sys.argv[1])
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid document ID")
        sys.exit(1)

    reason = sys.argv[2] if len(sys.argv) > 2 else "User excluded"

    # Get document info before skipping
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, filename, file_size_bytes, processing_status
            FROM pdf_documents
            WHERE id = ?
        """, (doc_id,))
        doc = cursor.fetchone()

    if not doc:
        print(f"Error: Document ID {doc_id} not found")
        sys.exit(1)

    size_mb = doc['file_size_bytes'] / (1024 * 1024) if doc['file_size_bytes'] else 0
    print(f"Document: {doc['filename']}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"Current status: {doc['processing_status']}")
    print(f"\nMarking as skipped with reason: {reason}")

    # Confirm
    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Skip the document
    success = db.mark_document_skipped(doc_id, reason)

    if success:
        print(f"✓ Document {doc_id} marked as skipped")
        print(f"  It will not be processed automatically.")
        print(f"  To re-enable processing, use: python skip_document.py --unskip {doc_id}")
    else:
        print(f"✗ Failed to skip document {doc_id}")
        sys.exit(1)

if __name__ == "__main__":
    main()
