# app/routes.py
import os
from pathlib import Path
from flask import render_template, jsonify, request, redirect, url_for, send_file
from utils.helpers import calculate_processing_progress

def register_routes(app):
    """Register Flask application routes."""

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
        document = app.db.get_document_by_id(doc_id)
        if not document:
            return jsonify(success=False, message="Document not found"), 404

        filepath = Path(document['relative_path'])
        if not filepath.exists():
            return jsonify(success=False, message="File no longer exists"), 404

        if app.db.reset_document_status_by_id(doc_id):
            app.pdf_processor.process_document(filepath, doc_id, app.db)
            
            if app.ocr_queue.add_to_queue(doc_id, filepath):
                return jsonify(success=True, message=f"Reset and queued document {doc_id} for OCR processing")
            return jsonify(success=False, message="Reset succeeded but queuing failed"), 500
        return jsonify(success=False, message="Failed to reset status"), 400

    @app.route('/reset_file')
    def reset_file():
        """Reset processing status for a document by its path."""
        file_path = request.args.get('path')
        if not file_path:
            return jsonify(success=False, message="No file path provided"), 400
            
        app.logger.info(f"Attempting to reset file: {file_path}")

        file_variations = [
            file_path,
            os.path.join(app.config['UPLOAD_FOLDER'], file_path),
            str(Path(file_path).absolute()),
            os.path.basename(file_path)
        ]
        
        # Try to find document in DB
        doc_id = None
        with app.db.get_connection() as conn:
            cursor = conn.cursor()
            for path_variant in file_variations:
                cursor.execute("""
                    SELECT id FROM pdf_documents
                    WHERE relative_path = ? OR filename = ?
                """, (path_variant, os.path.basename(path_variant)))
                
                doc = cursor.fetchone()
                if doc:
                    doc_id = doc['id']
                    break
        
        # If not found, try to add it
        if not doc_id:
            for path_variant in file_variations:
                if os.path.exists(path_variant):
                    filepath = Path(path_variant)
                    doc_id = app.db.add_document(filepath)
                    if doc_id:
                        break
                    
            if not doc_id:
                return jsonify(success=False, message="File not found or could not be added"), 404
        
        # Reset status and process
        if app.db.reset_document_status_by_id(doc_id):
            document = app.db.get_document_by_id(doc_id)
            filepath = Path(document['relative_path'])
            
            # Check if file exists or find alternate path
            if not filepath.exists():
                potential_paths = [
                    Path(document['relative_path']),
                    Path(os.path.join(app.config['UPLOAD_FOLDER'], document['filename'])),
                    Path(os.path.join(
                        app.config['UPLOAD_FOLDER'], 
                        document.get('folder_path', ''), 
                        document['filename']
                    ))
                ]
                
                for potential_path in potential_paths:
                    if potential_path.exists():
                        filepath = potential_path
                        break
                        
            if not filepath.exists():
                return jsonify(
                    success=False, 
                    message="File exists in database but not on disk"
                ), 404
                
            # Process the document
            app.pdf_processor.process_document(filepath, doc_id, app.db)
            
            if app.ocr_queue.add_to_queue(doc_id, filepath):
                return jsonify(success=True, message="Reset and queued document for OCR processing")
            return jsonify(success=False, message="Reset succeeded but queuing failed"), 500
        return jsonify(success=False, message="Failed to reset status"), 400

    @app.route('/ocr_queue')
    def ocr_queue_status():
        """Get the status of the OCR processing queue."""
        return jsonify(app.ocr_queue.get_queue_status())

    @app.route('/files')
    def get_files():
        """Get status of all files."""
        status = request.args.get('status')
        sort_by = request.args.get('sort', 'last_indexed_at')
        order = request.args.get('order', 'desc')

        fs_files = app.file_watcher.files
        db_docs = app.db.get_active_documents(status=status, sort_by=sort_by, order=order)

        files_status = []
        for rel_path in fs_files:
            base_filename = os.path.basename(rel_path)
            folder_path = os.path.dirname(rel_path)
            
            file_info = {
                'filename': rel_path,
                'base_filename': base_filename,
                'folder_path': folder_path,
                'in_filesystem': True,
                'in_database': False,
                'processing_status': None,
                'ocr_processed': False,
                'last_indexed': None,
                'file_size': None,
                'processing_progress': None,
                'id': None
            }

            # Find matching document
            for doc in db_docs:
                if doc['filename'] == base_filename:
                    file_info.update({
                        'in_database': True,
                        'processing_status': doc['processing_status'],
                        'ocr_processed': doc['ocr_processed'],
                        'last_indexed': doc['last_indexed_at'],
                        'file_size': doc.get('file_size_bytes'),
                        'processing_progress': calculate_processing_progress(doc),
                        'id': doc['id']
                    })
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
        new_discovered, removed_discovered = app.file_watcher.scan_now()
        return jsonify(new_files=new_discovered, removed_files=removed_discovered)

    @app.route('/scan')
    def scan_files():
        """Manually trigger a file scan."""
        new_discovered, removed_discovered = app.file_watcher.scan_now()
        
        for filename in new_discovered:
            filepath = Path(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            app.db.add_document(filepath)
        
        for filename in removed_discovered:
            filepath = Path(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            app.db.mark_document_removed(filepath)
        
        return jsonify(new_files=new_discovered, removed_files=removed_discovered)

    @app.route('/settings', methods=['GET', 'POST'])
    def settings():
        """Get or update application settings."""
        if request.method == 'POST':
            try:
                new_interval = float(request.form.get(
                    'polling_interval', 
                    app.file_watcher.polling_interval
                ))
                success = app.file_watcher.set_polling_interval(new_interval)
                
                if success:
                    return jsonify(
                        success=True, 
                        message=f"Polling interval updated to {new_interval} seconds"
                    )
                return jsonify(success=False, message="Invalid polling interval"), 400
            except ValueError:
                return jsonify(success=False, message="Invalid polling interval format"), 400
        else:
            return jsonify(
                polling_interval=app.file_watcher.polling_interval,
                file_types=app.file_watcher.file_types
            )

    @app.route('/settings/files', methods=['POST'])
    def update_file_settings():
        """Update file type settings."""
        try:
            file_types = request.form.get('file_types', '')
            types_list = [t.strip() for t in file_types.split(',') if t.strip()]
            app.file_watcher.file_types = types_list
            
            msg = f"File types updated to: {', '.join(types_list) if types_list else 'all files'}"
            return jsonify(success=True, message=msg)
        except Exception as e:
            return jsonify(success=False, message=f"Error updating file types: {str(e)}"), 400

    @app.route('/search')
    def search_page():
        """Render the search page with folder filtering."""
        query = request.args.get('q', '')
        folder_filter = request.args.get('folder', '')
        results = []
        
        # Get all unique folders
        all_folders = []
        try:
            with app.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT DISTINCT folder_path FROM pdf_documents "
                    "WHERE processing_status != 'removed' ORDER BY folder_path"
                )
                all_folders = [row[0] for row in cursor.fetchall() if row[0]]
        except Exception as e:
            app.logger.error(f"Error fetching folders: {e}")
        
        # Group folders by common patterns
        folder_categories = {}
        device_folders = {}
        common_categories = set()
        
        for folder in all_folders:
            if not folder:
                continue
                
            parts = folder.split('/')
            if not parts or not parts[0]:
                continue
                
            # First part is typically the device
            device = parts[0]
            if device not in device_folders:
                device_folders[device] = []
            device_folders[device].append(folder)
            
            # Add path components as categories
            for i, part in enumerate(parts):
                if i > 0 and part:
                    common_categories.add(part)
        
        # Find cross-device categories
        cross_device_categories = {}
        if len(device_folders) > 1:
            for category in common_categories:
                devices_with_category = set()
                category_folders = []
                
                for folder in all_folders:
                    pattern = f"/{category}/"
                    if (pattern in f"/{folder}/" or 
                        folder.startswith(f"{category}/") or 
                        folder.endswith(f"/{category}")):
                        parts = folder.split('/')
                        if parts and parts[0]:
                            devices_with_category.add(parts[0])
                            category_folders.append(folder)
                
                if len(devices_with_category) > 1:
                    cross_device_categories[f"All {category} folders"] = category_folders
        
        # Sort by folder count
        folder_categories = dict(sorted(
            cross_device_categories.items(),
            key=lambda item: len(item[1]),
            reverse=True
        ))
        
        if query:
            # If the folder filter is a category
            cat_keys = [k.replace('All ', '').replace(' folders', '') 
                      for k in folder_categories.keys()]
            if folder_filter in cat_keys:
                category = folder_filter
                matching_folders = []
                
                for folder in all_folders:
                    pattern = f"/{category}/"
                    if (pattern in f"/{folder}/" or 
                        folder.startswith(f"{category}/") or 
                        folder.endswith(f"/{category}")):
                        matching_folders.append(folder)
                
                # Build combined results from matching folders
                all_results = []
                for folder in matching_folders:
                    folder_results = app.db.search_documents_with_folder_filter(query, folder)
                    all_results.extend(folder_results)
                
                # Remove duplicates
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
        
        page_text = app.db.get_document_text(doc_id, page_num)
        if not page_text or not page_text.get('image_path'):
            return jsonify(error="Image path not found"), 404
            
        image_path = page_text['image_path']
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
                
                # Get all folders
                cursor.execute(
                    "SELECT DISTINCT folder_path FROM pdf_documents "
                    "WHERE processing_status != 'removed' ORDER BY folder_path"
                )
                all_folders = [row['folder_path'] for row in cursor.fetchall()]
                
                # Get files in current folder
                if folder:
                    cursor.execute("""
                        SELECT id, filename, relative_path, folder_path, 
                               processing_status, last_indexed_at, ocr_processed
                        FROM pdf_documents
                        WHERE folder_path = ? AND processing_status != 'removed'
                        ORDER BY filename
                    """, (folder,))
                else:
                    cursor.execute("""
                        SELECT id, filename, relative_path, folder_path, 
                               processing_status, last_indexed_at, ocr_processed
                        FROM pdf_documents
                        WHERE (folder_path = '' OR folder_path IS NULL) 
                              AND processing_status != 'removed'
                        ORDER BY filename
                    """)
                    
                files = [dict(row) for row in cursor.fetchall()]
                
                # Build folder tree
                folder_tree = {}
                for path in all_folders:
                    if not path:
                        continue
                        
                    parts = path.split('/')
                    current = folder_tree
                    
                    for part in parts:
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
            return render_template('error.html', message=f"Error browsing folders: {str(e)}"), 500

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
            app.logger.error(f"Error fetching content: {e}")
        
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

    @app.route('/reprocess_all_pending')
    def reprocess_all_pending():
        """Reprocess all documents with pending status."""
        try:
            # Get all pending documents
            with app.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, relative_path, filename, folder_path
                    FROM pdf_documents
                    WHERE processing_status = 'pending'
                """)
                
                pending_docs = cursor.fetchall()
                
            if not pending_docs:
                return jsonify(success=True, message="No pending documents found"), 200
                
            # Process each document
            processed_count = 0
            failed_count = 0
            
            for doc in pending_docs:
                doc_id = doc['id']
                
                # Try different path variations - using dict access instead of .get()
                folder_path = doc['folder_path'] if 'folder_path' in doc else ''
                
                potential_paths = [
                    Path(doc['relative_path']),
                    Path(os.path.join(app.config['UPLOAD_FOLDER'], doc['filename'])),
                    Path(os.path.join(app.config['UPLOAD_FOLDER'], folder_path, doc['filename']))
                ]
                
                filepath = None
                for path in potential_paths:
                    if path.exists():
                        filepath = path
                        break
                        
                if not filepath:
                    app.logger.warning(f"Could not find file for document {doc_id}: {doc['filename']}")
                    failed_count += 1
                    continue
                    
                # Process the document
                app.pdf_processor.process_document(filepath, doc_id, app.db)
                
                # Queue for OCR
                if app.ocr_queue.add_to_queue(doc_id, filepath):
                    processed_count += 1
                else:
                    failed_count += 1
                    
            return jsonify(
                success=True,
                message=f"Reprocessed {processed_count} documents, {failed_count} failed",
                processed=processed_count,
                failed=failed_count
            )
                
        except Exception as e:
            app.logger.error(f"Error reprocessing pending documents: {e}")
            import traceback
            app.logger.error(traceback.format_exc())
            return jsonify(success=False, message=f"Error: {str(e)}"), 500
            
    @app.route('/fix_file/<path:filename>')
    def fix_file(filename):
        """Special endpoint to fix a specific file."""
        try:
            app.logger.info(f"Attempting to fix file: {filename}")
            
            # Get the full path
            filepath = Path(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            base_filename = os.path.basename(filename)
            
            # Try to find in database by filename
            doc_id = None
            with app.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM pdf_documents WHERE filename = ?", (base_filename,))
                result = cursor.fetchone()
                if result:
                    doc_id = result['id']
            
            if not doc_id:
                # If not found, add to database
                doc_id = app.db.add_document(filepath)
                if not doc_id:
                    return jsonify(success=False, message="Failed to add document to database"), 500
            
            # Reset status and force processing
            app.db.reset_document_status_by_id(doc_id)
            
            # Process with text extraction
            app.pdf_processor.process_document(filepath, doc_id, app.db)
            
            # Queue for OCR
            app.ocr_queue.add_to_queue(doc_id, filepath)
            
            return jsonify(
                success=True, 
                message=f"File {base_filename} has been reset and queued for processing"
            )
        
        except Exception as e:
            app.logger.error(f"Error fixing file: {e}")
            import traceback
            app.logger.error(traceback.format_exc())
            return jsonify(success=False, message=f"Error: {str(e)}"), 500

    @app.route('/fix_html_file/<path:filename>')
    def fix_html_file(filename):
        """Fix an HTML file that might be in a bad state."""
        try:
            # Get the full path
            filepath = Path(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            if not filepath.exists():
                return jsonify(success=False, message="File not found"), 404
                
            # Get document by filename
            base_filename = os.path.basename(filename)
            with app.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM pdf_documents WHERE filename = ?", (base_filename,))
                result = cursor.fetchone()
                
            if not result:
                # Add to database if not found
                doc_id = app.db.add_document(filepath)
                if not doc_id:
                    return jsonify(success=False, message="Failed to add document to database"), 500
            else:
                doc_id = result['id']
                
            # Reset status
            app.db.reset_document_status_by_id(doc_id)
            
            # Process with HTML processor
            if hasattr(app, 'html_processor') and app.html_processor:
                app.html_processor.process_document(filepath, doc_id, app.db)
                return jsonify(success=True, message=f"HTML file processed: {base_filename}")
            else:
                return jsonify(success=False, message="HTML processor not available"), 500
                
        except Exception as e:
            app.logger.error(f"Error fixing HTML file: {e}")
            import traceback
            app.logger.error(traceback.format_exc())
            return jsonify(success=False, message=f"Error: {str(e)}"), 500
        
    @app.route('/fix_html_file')
    def fix_html_file():
        """Fix an HTML file that might be in a bad state."""
        try:
            file_path = request.args.get('path')
            if not file_path:
                return jsonify(success=False, message="No file path provided"), 400
                
            # Get the full path
            filepath = Path(os.path.join(app.config['UPLOAD_FOLDER'], file_path))
            
            if not filepath.exists():
                return jsonify(success=False, message="File not found"), 404
                
            # Get document by filename
            base_filename = os.path.basename(file_path)
            with app.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM pdf_documents WHERE filename = ?", (base_filename,))
                result = cursor.fetchone()
                
            if not result:
                # Add to database if not found
                doc_id = app.db.add_document(filepath)
                if not doc_id:
                    return jsonify(success=False, message="Failed to add document to database"), 500
            else:
                doc_id = result['id']
                
            # Reset status
            app.db.reset_document_status_by_id(doc_id)
            
            # Process with HTML processor
            if hasattr(app, 'html_processor') and app.html_processor:
                app.html_processor.process_document(filepath, doc_id, app.db)
                return jsonify(success=True, message=f"HTML file processed: {base_filename}")
            else:
                return jsonify(success=False, message="HTML processor not available"), 500
                
        except Exception as e:
            app.logger.error(f"Error fixing HTML file: {e}")
            import traceback
            app.logger.error(traceback.format_exc())
            return jsonify(success=False, message=f"Error: {str(e)}"), 500

    @app.route('/fix_pdf_file')
    def fix_pdf_file():
        """Fix a PDF file that might be in a bad state."""
        try:
            file_path = request.args.get('path')
            if not file_path:
                return jsonify(success=False, message="No file path provided"), 400
                
            # Get the full path
            filepath = Path(os.path.join(app.config['UPLOAD_FOLDER'], file_path))
            
            if not filepath.exists():
                return jsonify(success=False, message="File not found"), 404
                
            # Get document by filename
            base_filename = os.path.basename(file_path)
            with app.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM pdf_documents WHERE filename = ?", (base_filename,))
                result = cursor.fetchone()
                
            if not result:
                # Add to database if not found
                doc_id = app.db.add_document(filepath)
                if not doc_id:
                    return jsonify(success=False, message="Failed to add document to database"), 500
            else:
                doc_id = result['id']
                
            # Reset status
            app.db.reset_document_status_by_id(doc_id)
            
            # Process with PDF processor and OCR
            app.pdf_processor.process_document(filepath, doc_id, app.db)
            app.ocr_queue.add_to_queue(doc_id, filepath)
                
            return jsonify(success=True, message=f"PDF file queued for processing: {base_filename}")
                
        except Exception as e:
            app.logger.error(f"Error fixing PDF file: {e}")
            import traceback
            app.logger.error(traceback.format_exc())
            return jsonify(success=False, message=f"Error: {str(e)}"), 500

    return app