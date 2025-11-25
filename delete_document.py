#!/usr/bin/env python3
"""
Utility script to delete documents from the database.

Usage:
    python delete_document.py <doc_id> [--delete-file]

Options:
    --delete-file    Also delete the actual file from disk (use with caution!)
"""

import sys
from db.db_manager import DatabaseManager
from config import config

def main():
    if len(sys.argv) < 2:
        print("Usage: python delete_document.py <doc_id> [--delete-file]")
        print("\nOptions:")
        print("  --delete-file    Also delete the actual file from disk")
        sys.exit(1)

    try:
        doc_id = int(sys.argv[1])
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid document ID")
        sys.exit(1)

    # Check for --delete-file flag
    delete_file = '--delete-file' in sys.argv

    db = DatabaseManager(config.DATABASE_PATH)

    # Get document info before deleting
    doc_info = db.get_document_by_id(doc_id)

    if not doc_info:
        print(f"Error: Document ID {doc_id} not found")
        sys.exit(1)

    size_mb = doc_info.get('file_size_bytes', 0) / (1024 * 1024)
    print(f"Document #{doc_id}:")
    print(f"  File: {doc_info.get('filename')}")
    print(f"  Path: {doc_info.get('relative_path')}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Status: {doc_info.get('processing_status')}")
    print(f"\nThis will PERMANENTLY delete the document from the database.")

    if delete_file:
        print(f"⚠️  WARNING: This will ALSO delete the actual file from disk!")
    else:
        print(f"(The actual file on disk will NOT be deleted)")

    # Confirm
    confirm = input("\nProceed with deletion? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Delete the document
    success = db.delete_document(doc_id, delete_file=delete_file)

    if success:
        print(f"✓ Document {doc_id} deleted from database")
    else:
        print(f"✗ Failed to delete document {doc_id}")
        sys.exit(1)

if __name__ == "__main__":
    main()
