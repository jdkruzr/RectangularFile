#!/usr/bin/env python3
"""Scan the filesystem and compare with database."""
import os
import sqlite3
from pathlib import Path
from config import config

def scan_filesystem():
    """Scan the upload folder and show what files exist."""
    upload_folder = config.UPLOAD_FOLDER
    db_path = config.DATABASE_PATH

    print(f"Upload folder: {upload_folder}")
    print(f"Database: {db_path}")
    print("=" * 70)
    print()

    # Scan filesystem
    pdf_files = []
    html_files = []
    other_files = []

    if not os.path.exists(upload_folder):
        print(f"❌ Upload folder does not exist: {upload_folder}")
        return

    for root, dirs, files in os.walk(upload_folder):
        for filename in files:
            if filename.startswith('.'):
                continue

            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, upload_folder)

            if filename.lower().endswith('.pdf'):
                pdf_files.append(rel_path)
            elif filename.lower().endswith(('.html', '.htm')):
                html_files.append(rel_path)
            else:
                other_files.append(rel_path)

    print(f"📄 Found {len(pdf_files)} PDF files")
    print(f"📝 Found {len(html_files)} HTML files")
    print(f"📦 Found {len(other_files)} other files")
    print()

    # Show first 10 PDFs
    if pdf_files:
        print("Sample PDF files:")
        for pdf in sorted(pdf_files)[:10]:
            print(f"  {pdf}")
        if len(pdf_files) > 10:
            print(f"  ... and {len(pdf_files) - 10} more")
        print()

    # Check database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as count FROM pdf_documents WHERE processing_status != 'removed'")
    db_count = cursor.fetchone()['count']

    print(f"📊 Database has {db_count} documents")
    print()

    if db_count == 0 and len(pdf_files) > 0:
        print("⚠️  Files exist on disk but database is empty!")
        print("    This means the file watcher hasn't added them yet.")
        print()
        print("💡 To add them manually, you can:")
        print(f"    1. Visit http://rf.broken.works/scan in your browser")
        print(f"    2. Or run: python populate_database.py")

    conn.close()

if __name__ == "__main__":
    scan_filesystem()
