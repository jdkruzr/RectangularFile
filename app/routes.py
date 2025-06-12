# app/routes.py
from flask import render_template, jsonify, request, redirect, url_for, send_file
import os
from pathlib import Path
import time
import json
import re
from datetime import datetime
from utils.helpers import calculate_processing_progress

def register_routes(app):
    """Register application routes."""
    
    @app.route('/')
    def index():
        """Render the main page."""
        return render_template('index.html',
                              files=app.file_watcher.files,
                              polling_interval=app.file_watcher.polling_interval)
    
    @app.route('/settings/page')
    def settings_page():
        """Render the settings page."""
        return render_template('settings.html')
    
    @app.route('/reset/<int:doc_id>')
    def reset_processing(doc_id):
        """Reset processing status for a document by ID and trigger reprocessing."""
        # Get the document details
        document = app.db.get_document_by_id(doc_id)
        if not document:
            return jsonify(success=False, message="Document not found"), 404

        # Get the filepath
        filepath = Path(document['relative_path'])
        if not filepath.exists():
            return jsonify(success=False, message="File no longer exists"), 404

        # Reset status
        if app.db.reset_document_status_by_id(doc_id):
            # Try text extraction first
            app.pdf_processor.process_document(filepath, doc_id, app.db)
            
            # Queue for OCR processing
            if app.ocr_queue.add_to_queue(doc_id, filepath):
                return jsonify(success=True, message=f"Reset and queued document {doc_id} for OCR processing")
            else:
                return jsonify(success=False, message="Reset succeeded but queuing failed"), 500
        else:
            return jsonify(success=False, message="Failed to reset status"), 400

    @app.route('/reset_file')
    def reset_file():
        """Reset processing status for a document by its path."""
        file_path = request.args.get('path')
        if not file_path:
            return jsonify(success=False, message="No file path provided"), 400
            
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], file_path)
        filepath = Path(full_path)
        
        if not filepath.exists():
            return jsonify(success=False, message="File does not exist"), 404
            
        # Try to find document in database by filename
        base_filename = os.path.basename(file_path)
        with app.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM pdf_documents WHERE filename = ?", (base_filename,))
            doc = cursor.fetchone()
            
        if not doc:
            # If not found, add it first
            doc_id = app.db.add_document(filepath)
            if not doc_id:
                return jsonify(success=False, message="Failed to add document to database"), 500
        else:
            doc_id = doc['id']
            
        # Reset status and process
        if app.db.reset_document_status_by_id(doc_id):
            # Try text extraction first
            app.pdf_processor.process_document(filepath, doc_id, app.db)
            
            # Queue for OCR processing
            if app.ocr_queue.add_to_queue(doc_id, filepath):
                return jsonify(success=True, message=f"Reset and queued document for OCR processing")
            else:
                return jsonify(success=False, message="Reset succeeded but queuing failed"), 500
        else:
            return jsonify(success=False, message="Failed to reset status"), 400
    
    @app.route('/ocr_queue')
    def ocr_queue_status():
        """Get the status of the OCR processing queue."""
        status = app.ocr_queue.get_queue_status()
        return jsonify(status)

    @app.route('/files')
    def get_files():
        """Get status of all files."""
        status = request.args.get('status')
        sort_by = request.args.get('sort', 'last_indexed_at')
        order = request.args.get('order', 'desc')

        fs_files = app.file_watcher.files
        db_docs = app.db.get_active_documents(
            status=status,
            sort_by=sort_by,
            order=order
        )

        # Debug info
        print(f"File watcher files: {fs_files[:5]}...")  # Print first 5 as sample
        print(f"Database documents: {[doc['filename'] for doc in db_docs[:5]]}...")  # Print first 5

        files_status = []
        for rel_path in fs_files:
            # Extract just the base filename for comparison
            base_filename = os.path.basename(rel_path)
            # Extract folder path for display
            folder_path = os.path.dirname(rel_path)
            
            file_info = {
                'filename': rel_path,  # Full relative path
                'base_filename': base_filename,  # Just the file name part
                'folder_path': folder_path,      # Just the folder part
                'in_filesystem': True,
                'in_database': False,
                'processing_status': None,
                'ocr_processed': False,
                'last_indexed': None,
                'file_size': None,
                'processing_progress': None,
                'id': None
            }

            # Try to find matching document in database
            for doc in db_docs:
                doc_filename = doc['filename']
                doc_folder = doc.get('folder_path', '')
                
                # Debug single comparison
                if base_filename == doc_filename:
                    print(f"Potential match: {base_filename} == {doc_filename}")
                    print(f"Folder comparison: '{folder_path}' vs '{doc_folder}'")
                    
                # Try different matching strategies
                if doc_filename == base_filename:
                    # If filenames match, this is likely our document
                    file_info.update({
                        'in_database': True,
                        'processing_status': doc['processing_status'],
                        'ocr_processed': doc['ocr_processed'],
                        'last_indexed': doc['last_indexed_at'],
                        'file_size': doc.get('file_size_bytes'),
                        'processing_progress': calculate_processing_progress(doc),
                        'id': doc['id']
                    })
                    print(f"Found match: {base_filename} = {doc_filename}, ID: {doc['id']}")
                    break

            files_status.append(file_info)

        return jsonify(
            files=files_status,
            total_count=len(files_status),
            processing_count=sum(1 for f in files_status if f['processing_status'] == 'processing'),
            completed_count=sum(1 for f in files_status if f['processing_status'] == 'completed')
        )

    @app.route('/new_files')
    def get_new_files():
        """Get lists of new and removed files."""
        # Since we don't have global variables anymore, we'll use the scan_now method
        # to detect changes and return them directly
        new_discovered, removed_discovered = app.file_watcher.scan_now()
        
        return jsonify(new_files=new_discovered, removed_files=removed_discovered)

    @app.route('/scan')
    def scan_files():
        """Manually trigger a file scan."""
        new_discovered, removed_discovered = app.file_watcher.scan_now()
        
        for filename in new_discovered:
            filepath = Path(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            doc_id = app.db.add_document(filepath)
            if not doc_id:
                print(f"Failed to add document to database: {filename}")
        
        for filename in removed_discovered:
            filepath = Path(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            if not app.db.mark_document_removed(filepath):
                print(f"Failed to mark document as removed: {filename}")
        
        return jsonify(
            new_files=new_discovered, 
            removed_files=removed_discovered
        )

    @app.route('/settings', methods=['GET', 'POST'])
    def settings():
        """Get or update application settings."""
        if request.method == 'POST':
            try:
                new_interval = float(request.form.get('polling_interval', app.file_watcher.polling_interval))
                
                success = app.file_watcher.set_polling_interval(new_interval)
                
                if success:
                    return jsonify(success=True, message=f"Polling interval updated to {new_interval} seconds")
                else:
                    return jsonify(success=False, message="Invalid polling interval"), 400
                    
            except ValueError:
                return jsonify(success=False, message="Invalid polling interval format"), 400
        else:
            # GET request - return current settings
            return jsonify(
                polling_interval=app.file_watcher.polling_interval,
                file_types=app.file_watcher.file_types
            )

    @app.route('/settings/files', methods=['POST'])
    def update_file_settings():
        """Update file type settings."""
        try:
            file_types = request.form.get('file_types', '')
            
            # Parse the file types
            types_list = [t.strip() for t in file_types.split(',') if t.strip()]
            
            # Update file watcher settings
            app.file_watcher.file_types = types_list
            
            return jsonify(
                success=True, 
                message=f"File types updated to: {', '.join(types_list) if types_list else 'all files'}"
            )
                
        except Exception as e:
            return jsonify(success=False, message=f"Error updating file types: {str(e)}"), 400

    @app.route('/search')
    def search_page():
        """Render the search page with folder filtering."""
        query = request.args.get('q', '')
        folder_filter = request.args.get('folder', '')
        results = []
        
        # Get all unique folders for the filter dropdown
        all_folders = []
        try:
            with app.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT folder_path 
                    FROM pdf_documents 
                    WHERE processing_status != 'removed'
                    ORDER BY folder_path
                """)
                all_folders = [row[0] for row in cursor.fetchall() if row[0]]
        except Exception as e:
            app.logger.error(f"Error fetching folders: {e}")
        
        # Group folders by common patterns
        folder_categories = {}
        device_folders = {}
        common_categories = set()
        
        for folder in all_folders:
            # Skip empty folders
            if not folder:
                continue
                
            # Split the folder path into components
            parts = folder.split('/')
            
            # Skip empty parts
            if not parts or not parts[0]:
                continue
                
            # First part is typically the device
            device = parts[0]
            if device not in device_folders:
                device_folders[device] = []
            device_folders[device].append(folder)
            
            # Add all path components as potential categories
            # (except the device name which is usually the first component)
            for i, part in enumerate(parts):
                if i > 0 and part:  # Skip device name and empty parts
                    common_categories.add(part)
        
        # For each potential category, check if it appears across multiple devices
        cross_device_categories = {}
        
        # Only process if we have multiple devices
        if len(device_folders) > 1:
            for category in common_categories:
                # Count how many different devices have this category
                devices_with_category = set()
                category_folders = []
                
                for folder in all_folders:
                    if f"/{category}/" in f"/{folder}/" or folder.startswith(f"{category}/"):
                        parts = folder.split('/')
                        if parts and parts[0]:
                            devices_with_category.add(parts[0])
                            category_folders.append(folder)
                
                # If this category appears in multiple devices, add it to cross-device categories
                if len(devices_with_category) > 1:
                    cross_device_categories[f"All {category} folders"] = category_folders
        
        # Sort by folder count (most common categories first)
        folder_categories = {k: v for k, v in sorted(
            cross_device_categories.items(), 
            key=lambda item: len(item[1]), 
            reverse=True
        )}
        
        if query:
            # If the folder filter is a category (not a specific folder)
            if folder_filter in [k.replace('All ', '').replace(' folders', '') for k in folder_categories.keys()]:
                # Get all folders for this category
                category = folder_filter
                matching_folders = []
                
                for folder in all_folders:
                    if (f"/{category}/" in f"/{folder}/" or 
                        folder.startswith(f"{category}/") or 
                        folder.endswith(f"/{category}")):
                        matching_folders.append(folder)
                
                # Build a combined result from all matching folders
                all_results = []
                for folder in matching_folders:
                    folder_results = app.db.search_documents_with_folder_filter(query, folder)
                    all_results.extend(folder_results)
                
                # Remove duplicates (by doc_id)
                seen_ids = set()
                results = []
                for res in all_results:
                    if res['doc_id'] not in seen_ids:
                        seen_ids.add(res['doc_id'])
                        results.append(res)
            else:
                # Regular folder search
                results = app.db.search_documents_with_folder_filter(query, folder_filter)
            
        return render_template(
            'search.html', 
            query=query, 
            results=results, 
            all_folders=all_folders,
            folder_categories=folder_categories,
            device_folders=device_folders,
            current_folder=folder_filter
        )

    @app.route('/view/<int:doc_id>')
    def view_document(doc_id):
        """Redirect to the unified document viewer."""
        highlight = request.args.get('highlight', '')
        if highlight:
            return redirect(url_for('document_viewer', doc_id=doc_id, highlight=highlight))
        return redirect(url_for('document_viewer', doc_id=doc_id))

    @app.route('/document_image/<int:doc_id>/<int:page_num>')
    def document_image(doc_id, page_num):
        """Serve a document page image."""
        document = app.db.get_document_by_id(doc_id)
        if not document:
            return jsonify(error="Document not found"), 404
        
        # Get the page text data which includes image path
        page_text = app.db.get_document_text(doc_id, page_num)
        
        if not page_text or not page_text.get('image_path'):
            return jsonify(error="Image path not found"), 404
            
        image_path = page_text['image_path']
        
        # Verify the file exists
        if not os.path.exists(image_path):
            return jsonify(error="Image file not found"), 404
            
        return send_file(image_path, mimetype='image/jpeg')

    @app.route('/files/<int:doc_id>')
    def document_inspector(doc_id):
        """Redirect to the unified document viewer."""
        return redirect(url_for('document_viewer', doc_id=doc_id))

    @app.route('/folders')
    def folder_browser():
        """Browse documents by folder structure."""
        folder = request.args.get('path', '')
        
        try:
            with app.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get all unique folders
                cursor.execute("""
                    SELECT DISTINCT folder_path 
                    FROM pdf_documents 
                    WHERE processing_status != 'removed'
                    ORDER BY folder_path
                """)
                
                all_folders = [row['folder_path'] for row in cursor.fetchall()]
                
                # Get files in the current folder
                if folder:
                    # Exact folder match
                    cursor.execute("""
                        SELECT id, filename, relative_path, folder_path, 
                               processing_status, last_indexed_at, ocr_processed
                        FROM pdf_documents
                        WHERE folder_path = ? AND processing_status != 'removed'
                        ORDER BY filename
                    """, (folder,))
                else:
                    # Root folder
                    cursor.execute("""
                        SELECT id, filename, relative_path, folder_path, 
                               processing_status, last_indexed_at, ocr_processed
                        FROM pdf_documents
                        WHERE (folder_path = '' OR folder_path IS NULL) AND processing_status != 'removed'
                        ORDER BY filename
                    """)
                    
                files = [dict(row) for row in cursor.fetchall()]
                
                # Build folder tree
                folder_tree = {}
                for path in all_folders:
                    if not path:  # Skip empty paths
                        continue
                        
                    parts = path.split('/')
                    current = folder_tree
                    
                    for i, part in enumerate(parts):
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                
                return render_template(
                    'folder_browser.html',
                    current_folder=folder,
                    files=files,
                    folder_tree=folder_tree,
                    all_folders=all_folders
                )
                
        except Exception as e:
            print(f"Error in folder browser: {e}")
            return render_template('error.html', message=f"Error browsing folders: {str(e)}"), 500

    @app.route('/test_db')
    def test_db():
        """Test database connection and schema."""
        results = {
            "database_path": app.db.db_path,
            "exists": os.path.exists(app.db.db_path),
            "connection_test": False,
            "tables": [],
            "document_count": 0,
            "sample_documents": []
        }
        
        try:
            with app.db.get_connection() as conn:
                results["connection_test"] = True
                
                # Check tables
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                results["tables"] = [row[0] for row in cursor.fetchall()]
                
                # Check document count
                if "pdf_documents" in results["tables"]:
                    cursor.execute("SELECT COUNT(*) FROM pdf_documents")
                    results["document_count"] = cursor.fetchone()[0]
                    
                    # Get sample documents
                    cursor.execute("SELECT id, filename, folder_path FROM pdf_documents LIMIT 5")
                    results["sample_documents"] = [dict(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            results["error"] = str(e)
            
        return jsonify(results)
    
    # Add these routes to app/routes.py

# Add these routes to app/routes.py inside the register_routes function

    @app.route('/wordcloud')
    def wordcloud_page():
        """Render the word cloud page with automatically detected cross-device categories."""
        # Get all folders for filtering
        all_folders = []
        try:
            with app.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT folder_path 
                    FROM pdf_documents 
                    WHERE processing_status = 'completed'
                    ORDER BY folder_path
                """)
                all_folders = [row[0] for row in cursor.fetchall() if row[0]]
        except Exception as e:
            app.logger.error(f"Error fetching folders: {e}")
        
        # Group folders by common patterns
        folder_categories = {}
        device_folders = {}
        common_categories = set()
        
        for folder in all_folders:
            # Skip empty folders
            if not folder:
                continue
                
            # Split the folder path into components
            parts = folder.split('/')
            
            # Skip empty parts
            if not parts or not parts[0]:
                continue
                
            # First part is typically the device
            device = parts[0]
            if device not in device_folders:
                device_folders[device] = []
            device_folders[device].append(folder)
            
            # Add all path components as potential categories
            # (except the device name which is usually the first component)
            for i, part in enumerate(parts):
                if i > 0 and part:  # Skip device name and empty parts
                    common_categories.add(part)
        
        # For each potential category, check if it appears across multiple devices
        cross_device_categories = {}
        
        # Only process if we have multiple devices
        if len(device_folders) > 1:
            for category in common_categories:
                # Count how many different devices have this category
                devices_with_category = set()
                category_folders = []
                
                for folder in all_folders:
                    if f"/{category}/" in f"/{folder}/" or folder.startswith(f"{category}/"):
                        parts = folder.split('/')
                        if parts and parts[0]:
                            devices_with_category.add(parts[0])
                            category_folders.append(folder)
                
                # If this category appears in multiple devices, add it to cross-device categories
                if len(devices_with_category) > 1:
                    cross_device_categories[f"All {category} folders"] = category_folders
        
        # Sort by folder count (most common categories first)
        folder_categories = {k: v for k, v in sorted(
            cross_device_categories.items(), 
            key=lambda item: len(item[1]), 
            reverse=True
        )}
        
        return render_template(
            'wordcloud.html', 
            all_folders=all_folders,
            folder_categories=folder_categories,
            device_folders=device_folders
        )

    @app.route('/generate_wordcloud')
    def generate_wordcloud_image():
        """Generate and return a word cloud image."""
        from utils.wordcloud import get_document_texts, process_text_for_wordcloud, generate_wordcloud
        
        doc_id = request.args.get('doc_id', type=int)
        folder = request.args.get('folder', '')
        device = request.args.get('device', '')
        category = request.args.get('category', '')
        
        app.logger.info(f"Generating word cloud for - Category: {category}, Device: {device}, Folder: {folder}")
        
        try:
            texts = []
            
            # If a specific category is selected, get texts from all matching folders
            if category:
                app.logger.info(f"Finding folders matching category: {category}")
                
                # First, get all folders
                with app.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT DISTINCT folder_path 
                        FROM pdf_documents 
                        WHERE processing_status = 'completed'
                    """)
                    all_folders = [row[0] for row in cursor.fetchall() if row[0]]
                    
                # Find folders that contain the category as a path component
                matching_folders = []
                for folder_path in all_folders:
                    if not folder_path:
                        continue
                        
                    # Add more ways to match the category
                    path_parts = folder_path.split('/')
                    
                    # Match if the category appears as any path component
                    if category in path_parts:
                        matching_folders.append(folder_path)
                        continue
                        
                    # Match if the category appears in any path component
                    for part in path_parts:
                        if category.lower() in part.lower():
                            matching_folders.append(folder_path)
                            break
                
                app.logger.info(f"Found {len(matching_folders)} matching folders for category '{category}': {matching_folders}")
                
                # Now get documents from each matching folder
                for folder_path in matching_folders:
                    # Get text directly from database for this folder
                    with app.db.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT tc.text_content, tc.ocr_text
                            FROM pdf_documents d
                            JOIN pdf_text_content tc ON d.id = tc.pdf_id
                            WHERE d.folder_path = ? AND d.processing_status = 'completed'
                        """, (folder_path,))
                        
                        folder_texts = []
                        for row in cursor.fetchall():
                            # Prefer OCR text if available
                            if row['ocr_text'] and row['ocr_text'].strip():
                                folder_texts.append(row['ocr_text'])
                            elif row['text_content'] and row['text_content'].strip():
                                folder_texts.append(row['text_content'])
                        
                        app.logger.info(f"Found {len(folder_texts)} texts in folder {folder_path}")
                        texts.extend(folder_texts)
            
            # If a specific device is selected
            elif device:
                app.logger.info(f"Finding folders for device: {device}")
                # Use direct database query for device
                with app.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT tc.text_content, tc.ocr_text
                        FROM pdf_documents d
                        JOIN pdf_text_content tc ON d.id = tc.pdf_id
                        WHERE d.folder_path LIKE ? AND d.processing_status = 'completed'
                    """, (f"{device}/%",))
                    
                    for row in cursor.fetchall():
                        # Prefer OCR text if available
                        if row['ocr_text'] and row['ocr_text'].strip():
                            texts.append(row['ocr_text'])
                        elif row['text_content'] and row['text_content'].strip():
                            texts.append(row['text_content'])
                    
                    app.logger.info(f"Found {len(texts)} texts for device '{device}'")
            
            # If a specific folder is selected
            elif folder:
                app.logger.info(f"Getting texts from specific folder: {folder}")
                with app.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT tc.text_content, tc.ocr_text
                        FROM pdf_documents d
                        JOIN pdf_text_content tc ON d.id = tc.pdf_id
                        WHERE d.folder_path = ? AND d.processing_status = 'completed'
                    """, (folder,))
                    
                    for row in cursor.fetchall():
                        # Prefer OCR text if available
                        if row['ocr_text'] and row['ocr_text'].strip():
                            texts.append(row['ocr_text'])
                        elif row['text_content'] and row['text_content'].strip():
                            texts.append(row['text_content'])
                    
                    app.logger.info(f"Found {len(texts)} texts in folder '{folder}'")
            
            # If no filters, get all texts
            else:
                app.logger.info("Getting texts from all documents")
                with app.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT tc.text_content, tc.ocr_text
                        FROM pdf_documents d
                        JOIN pdf_text_content tc ON d.id = tc.pdf_id
                        WHERE d.processing_status = 'completed'
                    """)
                    
                    for row in cursor.fetchall():
                        # Prefer OCR text if available
                        if row['ocr_text'] and row['ocr_text'].strip():
                            texts.append(row['ocr_text'])
                        elif row['text_content'] and row['text_content'].strip():
                            texts.append(row['text_content'])
                    
                    app.logger.info(f"Found {len(texts)} texts across all documents")
            
            app.logger.info(f"Found {len(texts)} total texts for word cloud generation")
            
            if not texts:
                app.logger.warning("No text found for the given filters")
                return jsonify(error="No text found for the given filters"), 404
            
            # Process text
            processed_text = process_text_for_wordcloud(texts)
            app.logger.info(f"Processed text length: {len(processed_text)} characters")
            
            # Generate word cloud
            app.logger.info("Generating word cloud image")
            _, img_bytes = generate_wordcloud(processed_text)
            app.logger.info("Word cloud image generated successfully")
            
            # Return image
            return send_file(img_bytes, mimetype='image/png')
        except Exception as e:
            app.logger.error(f"Error generating word cloud: {e}")
            import traceback
            app.logger.error(traceback.format_exc())
            return jsonify(error=str(e)), 500

    @app.route('/document/<int:doc_id>')
    def document_viewer(doc_id):
        """Unified document viewer that combines inspection and viewing."""
        highlight = request.args.get('highlight', '')
        
        # Get document details
        document = app.db.get_document_by_id(doc_id)
        if not document:
            return render_template('error.html', message="Document not found"), 404
            
        # Get document text content
        text_content = app.db.get_document_text(doc_id) or {}
        
        # Get OCR and extracted content separately for comparison
        ocr_content = {}
        extracted_content = {}
        
        try:
            with app.db.get_connection() as conn:
                cursor = conn.cursor()
                # Get OCR content
                cursor.execute("""
                    SELECT page_number, ocr_text, processed_at
                    FROM pdf_text_content
                    WHERE pdf_id = ? AND ocr_text IS NOT NULL
                    ORDER BY page_number
                """, (doc_id,))
                
                for row in cursor.fetchall():
                    ocr_content[row['page_number']] = {
                        'text': row['ocr_text'],
                        'processed_at': row['processed_at'],
                        'source': 'ocr'
                    }
                    
                # Get extracted content
                cursor.execute("""
                    SELECT page_number, text_content, processed_at
                    FROM pdf_text_content
                    WHERE pdf_id = ? AND text_content IS NOT NULL
                    ORDER BY page_number
                """, (doc_id,))
                
                for row in cursor.fetchall():
                    extracted_content[row['page_number']] = {
                        'text': row['text_content'],
                        'processed_at': row['processed_at'],
                        'source': 'extracted'
                    }
        except Exception as e:
            print(f"Error fetching content: {e}")
        
        # Pre-compute common pages
        common_pages = sorted(set(extracted_content.keys()) & set(ocr_content.keys()))
        has_comparison = len(common_pages) > 0
        
        return render_template(
            'document_viewer.html',
            document=document,
            text_content=text_content,
            extracted_content=extracted_content,
            ocr_content=ocr_content,
            common_pages=common_pages,
            has_comparison=has_comparison,
            highlight=highlight
        )

    # Return the app for chaining
    return app
