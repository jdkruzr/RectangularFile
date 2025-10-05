#!/usr/bin/env python3
"""Populate the database with all files found in the upload folder."""
import os
import sys
from pathlib import Path
from config import config
from db.db_manager import DatabaseManager

def populate_database():
    """Scan upload folder and add all files to database."""
    upload_folder = config.UPLOAD_FOLDER
    db_path = config.DATABASE_PATH

    print(f"Upload folder: {upload_folder}")
    print(f"Database: {db_path}")
    print("=" * 70)
    print()

    if not os.path.exists(upload_folder):
        print(f"❌ Upload folder does not exist: {upload_folder}")
        return

    # Initialize database manager
    db = DatabaseManager(db_path)

    # Scan for files
    pdf_files = []
    html_files = []

    print("Scanning filesystem...")
    for root, dirs, files in os.walk(upload_folder):
        for filename in files:
            if filename.startswith('.'):
                continue

            file_path = os.path.join(root, filename)

            if filename.lower().endswith('.pdf'):
                pdf_files.append(Path(file_path))
            elif filename.lower().endswith(('.html', '.htm')):
                html_files.append(Path(file_path))

    print(f"Found {len(pdf_files)} PDF files")
    print(f"Found {len(html_files)} HTML files")
    print()

    if not pdf_files and not html_files:
        print("No files found to add!")
        return

    # Add to database
    added = 0
    skipped = 0

    print("Adding files to database...")
    for filepath in pdf_files + html_files:
        try:
            doc_id = db.add_document(filepath)
            if doc_id:
                added += 1
                print(f"  ✓ Added: {filepath.name}")
            else:
                skipped += 1
                print(f"  - Skipped: {filepath.name} (already exists)")
        except Exception as e:
            print(f"  ✗ Error adding {filepath.name}: {e}")

    print()
    print("=" * 70)
    print(f"✓ Added {added} new documents")
    print(f"- Skipped {skipped} existing documents")
    print()
    print("Next steps:")
    print("  1. Restart the service to start processing: sudo systemctl restart rectangular-file")
    print("  2. Or manually trigger processing by visiting: http://rf.broken.works/")

if __name__ == "__main__":
    populate_database()
