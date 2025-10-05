#!/usr/bin/env python3
"""
Fix missing folder_path values in the database.

This script updates documents that have empty folder_path values by
re-extracting the folder path from their relative_path.
"""
import sqlite3
import os
import sys
from pathlib import Path
from config import config

def fix_folder_paths():
    """Fix folder paths for all documents in the database."""
    db_path = config.DATABASE_PATH
    base_path = config.UPLOAD_FOLDER

    print(f"Database: {db_path}")
    print(f"Base path: {base_path}")
    print()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Find all documents with empty or missing folder_path
    cursor.execute("""
        SELECT id, filename, relative_path, folder_path
        FROM pdf_documents
        WHERE (folder_path = '' OR folder_path IS NULL)
        AND processing_status != 'removed'
    """)

    documents = cursor.fetchall()
    print(f"Found {len(documents)} documents with missing folder paths\n")

    if not documents:
        print("No documents need updating!")
        conn.close()
        return

    # Update each document
    updated_count = 0
    for doc in documents:
        doc_id = doc['id']
        relative_path = doc['relative_path']

        try:
            # Parse the relative path
            filepath = Path(relative_path)

            # Extract folder path
            folder_path = ""

            # If it's an absolute path, make it relative to base_path
            if filepath.is_absolute():
                try:
                    rel_to_base = filepath.relative_to(base_path)
                    if len(rel_to_base.parts) > 1:
                        folder_path = str(rel_to_base.parent)
                except ValueError:
                    # Path is not relative to base_path, try string manipulation
                    path_str = str(filepath)
                    base_str = str(base_path) + '/'
                    if base_str in path_str:
                        idx = path_str.find(base_str) + len(base_str)
                        remaining = path_str[idx:]
                        parts = remaining.split('/')
                        if len(parts) > 1:
                            folder_path = '/'.join(parts[:-1])
            else:
                # It's already relative, extract parent
                if len(filepath.parts) > 1:
                    folder_path = str(filepath.parent)

            if folder_path and folder_path != '.':
                print(f"Doc {doc_id}: '{relative_path}' -> folder: '{folder_path}'")

                cursor.execute("""
                    UPDATE pdf_documents
                    SET folder_path = ?
                    WHERE id = ?
                """, (folder_path, doc_id))
                updated_count += 1
            else:
                print(f"Doc {doc_id}: '{relative_path}' -> (root folder)")

        except Exception as e:
            print(f"Error processing doc {doc_id}: {e}")
            continue

    conn.commit()
    print(f"\nUpdated {updated_count} documents with folder paths")

    # Show folder summary
    cursor.execute("""
        SELECT DISTINCT folder_path, COUNT(*) as count
        FROM pdf_documents
        WHERE processing_status != 'removed'
        GROUP BY folder_path
        ORDER BY folder_path
    """)

    print("\nFolder summary:")
    for row in cursor.fetchall():
        folder = row[0] or '(root)'
        count = row[1]
        print(f"  {folder}: {count} documents")

    conn.close()

def scan_database_stats():
    """Scan the database and show statistics about documents."""
    db_path = config.DATABASE_PATH

    print(f"Database: {db_path}")
    print("=" * 70)
    print()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Total documents
    cursor.execute("SELECT COUNT(*) as count FROM pdf_documents WHERE processing_status != 'removed'")
    total_docs = cursor.fetchone()['count']
    print(f"📄 Total documents: {total_docs}")

    # Documents by processing status
    cursor.execute("""
        SELECT processing_status, COUNT(*) as count
        FROM pdf_documents
        WHERE processing_status != 'removed'
        GROUP BY processing_status
        ORDER BY count DESC
    """)
    print("\n📊 Processing status:")
    for row in cursor.fetchall():
        status = row['processing_status'] or 'unknown'
        count = row['count']
        print(f"  {status:15} {count:5} documents")

    # OCR processed count
    cursor.execute("SELECT COUNT(*) as count FROM pdf_documents WHERE ocr_processed = 1")
    ocr_count = cursor.fetchone()['count']
    print(f"\n🤖 OCR processed: {ocr_count} / {total_docs} ({ocr_count*100/total_docs:.1f}%)" if total_docs > 0 else "\n🤖 OCR processed: 0")

    # Documents by folder
    cursor.execute("""
        SELECT folder_path, COUNT(*) as count
        FROM pdf_documents
        WHERE processing_status != 'removed'
        GROUP BY folder_path
        ORDER BY count DESC
    """)
    print("\n📁 Documents by folder:")
    folders = cursor.fetchall()
    for row in folders[:10]:  # Show top 10 folders
        folder = row['folder_path'] or '(root)'
        count = row['count']
        print(f"  {folder:40} {count:5} documents")

    if len(folders) > 10:
        print(f"  ... and {len(folders) - 10} more folders")

    # Documents with missing folder paths
    cursor.execute("""
        SELECT COUNT(*) as count
        FROM pdf_documents
        WHERE (folder_path = '' OR folder_path IS NULL)
        AND processing_status != 'removed'
    """)
    missing_folders = cursor.fetchone()['count']
    if missing_folders > 0:
        print(f"\n⚠️  {missing_folders} documents have missing folder paths")

    # Total pages processed
    cursor.execute("SELECT COUNT(*) as count FROM pdf_text_content")
    total_pages = cursor.fetchone()['count']
    print(f"\n📖 Total pages processed: {total_pages}")

    # Total word count
    cursor.execute("SELECT SUM(word_count) as total FROM pdf_documents WHERE processing_status != 'removed'")
    total_words = cursor.fetchone()['total'] or 0
    print(f"📝 Total words indexed: {total_words:,}")

    # Annotations
    cursor.execute("SELECT COUNT(*) as count FROM document_annotations")
    total_annotations = cursor.fetchone()['count']
    if total_annotations > 0:
        cursor.execute("""
            SELECT annotation_type, COUNT(*) as count
            FROM document_annotations
            GROUP BY annotation_type
            ORDER BY count DESC
        """)
        print(f"\n🎨 Annotations found: {total_annotations}")
        for row in cursor.fetchall():
            ann_type = row['annotation_type']
            count = row['count']
            print(f"  {ann_type:20} {count:5} annotations")
    else:
        print("\n🎨 No annotations found yet")

    print("\n" + "=" * 70)

    conn.close()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'scan':
        scan_database_stats()
    else:
        fix_folder_paths()
