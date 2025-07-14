# app/routes.py
import os
from datetime import datetime
from pathlib import Path
from flask import render_template, flash, jsonify, request, redirect, url_for, send_file
from utils.helpers import calculate_processing_progress
from flask_login import login_user, logout_user, login_required, current_user
import hashlib
import sys


def register_routes(app):
    """Register Flask application routes."""

    # Configure password from environment or use default
    PASSWORD_HASH = os.environ.get('APP_PASSWORD_HASH', 
        # Default password is "changeme" - SHA256 hash
        'e8c3a5a189478215a016d16f5abf3e643c3a6ba2d286dcb779aa891004fd1c41'
    )
    
    # print(f"LOADED PASSWORD HASH: {PASSWORD_HASH}", file=sys.stderr)
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """Login page."""
        if request.method == 'POST':
            password = request.form.get('password', '')
            
            # Direct debug output
            # print(f"LOGIN ATTEMPT: Password provided: {password}", file=sys.stderr)
            # print(f"LOGIN ATTEMPT: Expected hash: {PASSWORD_HASH}", file=sys.stderr)
            
            # Hash the provided password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            # print(f"LOGIN ATTEMPT: Computed hash: {password_hash}", file=sys.stderr)
            # print(f"LOGIN ATTEMPT: Hashes match: {password_hash == PASSWORD_HASH}", file=sys.stderr)
            
            # Temporary: just try to log in without password check
            #if password == "testlogin":
            #    user = app.User('admin')
            #    login_user(user)
            #    return redirect(url_for('index'))
            
            if password_hash == PASSWORD_HASH:
                user = app.User('admin')
                login_user(user)
                
                # Redirect to the page they were trying to access, or home
                next_page = request.args.get('next')
                return redirect(next_page or url_for('index'))
            else:
                flash('Invalid password', 'error')
        
        return render_template('login.html')
    
    @app.route('/logout')
    @login_required
    def logout():
        """Logout the user."""
        logout_user()
        return redirect(url_for('login'))

    @app.route('/')
    @login_required
    def index():
        """Render the main page."""
        return render_template('index.html',
                              files=app.file_watcher.files,
                              polling_interval=app.file_watcher.polling_interval)
    
    @app.route('/settings/page')
    @login_required
    def settings_page():
        """Render the settings page."""
        return render_template('settings.html')
    
    @app.route('/reset/<int:doc_id>')
    @login_required
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
    @login_required
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
    @login_required
    def ocr_queue_status():
        """Get the status of the OCR processing queue."""
        return jsonify(app.ocr_queue.get_queue_status())

    @app.route('/files')
    @login_required
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
    @login_required
    def get_new_files():
        """Get lists of new and removed files."""
        new_discovered, removed_discovered = app.file_watcher.scan_now()
        return jsonify(new_files=new_discovered, removed_files=removed_discovered)

    @app.route('/scan')
    @login_required    
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
    @login_required
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
    @login_required
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
    
    @app.route('/settings/caldav', methods=['GET', 'POST'])
    @login_required
    def caldav_settings():
        """Get or update CalDAV settings."""
        if request.method == 'POST':
            try:
                enabled = request.form.get('enabled') == 'true'
                url = request.form.get('url', '').strip()
                username = request.form.get('username', '').strip()
                password = request.form.get('password', '').strip()
                calendar = request.form.get('calendar', 'todos').strip()
                base_url = request.form.get('base_url', '').strip()
                
                # Save settings
                app.db.set_setting('caldav', 'enabled', str(enabled).lower())
                app.db.set_setting('caldav', 'url', url)
                app.db.set_setting('caldav', 'username', username)
                app.db.set_setting('caldav', 'calendar', calendar)
                app.db.set_setting('caldav', 'base_url', base_url)
                
                # Encrypt and save password if provided
                if password:
                    app.db.set_setting('caldav', 'password', password, encrypt=True)
                
                return jsonify(success=True, message="CalDAV settings saved successfully")
                
            except Exception as e:
                return jsonify(success=False, message=f"Error saving CalDAV settings: {str(e)}"), 400
        else:
            # GET request - return current settings (without password)
            try:
                settings = app.db.get_caldav_settings()
                # Don't return the password for security
                settings.pop('password', None)
                return jsonify(success=True, **settings)
            except Exception as e:
                return jsonify(success=False, message=f"Error loading CalDAV settings: {str(e)}"), 400
    
    @app.route('/settings/caldav/test', methods=['POST'])
    @login_required
    def test_caldav_connection():
        """Test CalDAV connection."""
        try:
            url = request.form.get('url', '').strip()
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '').strip()
            
            if not all([url, username, password]):
                return jsonify(success=False, message="Missing connection details"), 400
            
            from processing.caldav_client import CalDAVTodoClient
            client = CalDAVTodoClient()
            result = client.test_connection(url, username, password)
            
            return jsonify(**result)
            
        except Exception as e:
            return jsonify(success=False, message=f"Connection test failed: {str(e)}"), 400
    
    @app.route('/todos')
    @login_required
    def todos_page():
        """Render the todos page."""
        return render_template('todos.html')
    
    @app.route('/todos/sync', methods=['POST'])
    @login_required
    def sync_todos():
        """Sync todos with CalDAV server."""
        try:
            settings = app.db.get_caldav_settings()
            
            if not settings['enabled']:
                return jsonify(success=False, message="CalDAV integration is disabled")
            
            if not all([settings['url'], settings['username'], settings['password']]):
                return jsonify(success=False, message="CalDAV settings incomplete")
            
            from processing.caldav_client import CalDAVTodoClient
            client = CalDAVTodoClient()
            
            # Connect to CalDAV server
            if not client.connect(settings['url'], settings['username'], 
                                  settings['password'], settings['calendar']):
                return jsonify(success=False, message="Failed to connect to CalDAV server")
            
            # Get todos from server
            todos = client.get_todos(include_completed=True)
            
            return jsonify(success=True, todos=todos)
            
        except Exception as e:
            app.logger.error(f"Error syncing todos: {e}")
            return jsonify(success=False, message=f"Sync failed: {str(e)}"), 500
    
    @app.route('/todos/status', methods=['POST'])
    @login_required
    def update_todo_status():
        """Update todo completion status."""
        try:
            uid = request.form.get('uid')
            completed = request.form.get('completed') == 'true'
            
            if not uid:
                return jsonify(success=False, message="Missing todo UID"), 400
            
            settings = app.db.get_caldav_settings()
            
            if not settings['enabled']:
                return jsonify(success=False, message="CalDAV integration is disabled")
            
            from processing.caldav_client import CalDAVTodoClient
            client = CalDAVTodoClient()
            
            # Connect and update status
            if not client.connect(settings['url'], settings['username'], 
                                  settings['password'], settings['calendar']):
                return jsonify(success=False, message="Failed to connect to CalDAV server")
            
            if client.update_todo_status(uid, completed):
                return jsonify(success=True, message="Todo status updated")
            else:
                return jsonify(success=False, message="Failed to update todo status")
            
        except Exception as e:
            app.logger.error(f"Error updating todo status: {e}")
            return jsonify(success=False, message=f"Update failed: {str(e)}"), 500
    
    @app.route('/todos/delete', methods=['POST'])
    @login_required
    def delete_todo():
        """Delete a todo."""
        try:
            uid = request.form.get('uid')
            
            if not uid:
                return jsonify(success=False, message="Missing todo UID"), 400
            
            settings = app.db.get_caldav_settings()
            
            if not settings['enabled']:
                return jsonify(success=False, message="CalDAV integration is disabled")
            
            from processing.caldav_client import CalDAVTodoClient
            client = CalDAVTodoClient()
            
            # Connect and delete todo
            if not client.connect(settings['url'], settings['username'], 
                                  settings['password'], settings['calendar']):
                return jsonify(success=False, message="Failed to connect to CalDAV server")
            
            if client.delete_todo(uid):
                return jsonify(success=True, message="Todo deleted")
            else:
                return jsonify(success=False, message="Failed to delete todo")
            
        except Exception as e:
            app.logger.error(f"Error deleting todo: {e}")
            return jsonify(success=False, message=f"Delete failed: {str(e)}"), 500

    @app.route('/search')
    @login_required
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
    @login_required
    def view_document(doc_id):
        """Redirect to the unified document viewer."""
        highlight = request.args.get('highlight', '')
        if highlight:
            return redirect(url_for('document_viewer', doc_id=doc_id, highlight=highlight))
        return redirect(url_for('document_viewer', doc_id=doc_id))

    @app.route('/document_image/<int:doc_id>/<int:page_num>')
    @login_required
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
    @login_required
    def document_inspector(doc_id):
        """Redirect to the unified document viewer."""
        return redirect(url_for('document_viewer', doc_id=doc_id))

    @app.route('/folders')
    @login_required
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
    @login_required
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
    @login_required
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
                        
    @app.route('/fix_html_file')
    @login_required
    def fix_html_file():
        """Fix an HTML file that might be in a bad state."""
        try:
            file_path = request.args.get('path')
            app.logger.info(f"Fix HTML file request received for: {file_path}")
            
            if not file_path:
                return jsonify(success=False, message="No file path provided"), 400
            
            # Get the full path
            filepath = Path(os.path.join(app.config['UPLOAD_FOLDER'], file_path))
            app.logger.info(f"Full path: {filepath}")
            
            if not filepath.exists():
                app.logger.error(f"File not found at path: {filepath}")
                return jsonify(success=False, message=f"File not found at: {filepath}"), 404
            
            # Get document by filename
            base_filename = os.path.basename(file_path)
            app.logger.info(f"Base filename: {base_filename}")
            
            try:
                with app.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM pdf_documents WHERE filename = ?", (base_filename,))
                    result = cursor.fetchone()
                    
                    if result:
                        doc_id = result['id']
                        app.logger.info(f"Found existing document with ID: {doc_id}")
                    else:
                        doc_id = None
                        app.logger.info("Document not found in database")
            except Exception as db_error:
                app.logger.error(f"Database error while looking up document: {db_error}")
                return jsonify(success=False, message=f"Database error: {str(db_error)}"), 500
            
            if not doc_id:
                # Add to database if not found
                try:
                    doc_id = app.db.add_document(filepath)
                    app.logger.info(f"Added document to database with ID: {doc_id}")
                    if not doc_id:
                        app.logger.error("Failed to add document to database")
                        return jsonify(success=False, message="Failed to add document to database"), 500
                except Exception as add_error:
                    app.logger.error(f"Error adding document to database: {add_error}")
                    return jsonify(success=False, message=f"Error adding document: {str(add_error)}"), 500
            
            # Reset status
            try:
                if not app.db.reset_document_status_by_id(doc_id):
                    app.logger.error(f"Failed to reset document status for ID: {doc_id}")
                    return jsonify(success=False, message="Failed to reset document status"), 500
                app.logger.info(f"Reset document status for ID: {doc_id}")
            except Exception as reset_error:
                app.logger.error(f"Error resetting document status: {reset_error}")
                return jsonify(success=False, message=f"Error resetting status: {str(reset_error)}"), 500
            
            # Process with HTML processor
            if hasattr(app, 'html_processor') and app.html_processor:
                try:
                    app.html_processor.process_document(filepath, doc_id, app.db)
                    app.logger.info(f"Processed document with HTML processor: {doc_id}")
                    return jsonify(success=True, message=f"HTML file processed: {base_filename}")
                except Exception as html_error:
                    app.logger.error(f"Error processing HTML: {html_error}")
                    return jsonify(success=False, message=f"Error processing HTML: {str(html_error)}"), 500
            else:
                app.logger.error("HTML processor not available")
                return jsonify(success=False, message="HTML processor not available"), 500
            
        except Exception as e:
            app.logger.error(f"Unexpected error fixing HTML file: {e}")
            import traceback
            tb = traceback.format_exc()
            app.logger.error(f"Traceback: {tb}")
            return jsonify(success=False, message=f"Unexpected error: {str(e)}"), 500

    @app.route('/fix_pdf_file')
    @login_required
    def fix_pdf_file():
        """Fix a PDF file that might be in a bad state."""
        try:
            file_path = request.args.get('path')
            app.logger.info(f"Fix PDF file request received for: {file_path}")
            
            if not file_path:
                return jsonify(success=False, message="No file path provided"), 400
            
            # Get the full path
            filepath = Path(os.path.join(app.config['UPLOAD_FOLDER'], file_path))
            app.logger.info(f"Full path: {filepath}")
            
            if not filepath.exists():
                app.logger.error(f"File not found at path: {filepath}")
                return jsonify(success=False, message=f"File not found at: {filepath}"), 404
            
            # Get document by filename
            base_filename = os.path.basename(file_path)
            app.logger.info(f"Base filename: {base_filename}")
            
            try:
                with app.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM pdf_documents WHERE filename = ?", (base_filename,))
                    result = cursor.fetchone()
                    
                    if result:
                        doc_id = result['id']
                        app.logger.info(f"Found existing document with ID: {doc_id}")
                    else:
                        doc_id = None
                        app.logger.info("Document not found in database")
            except Exception as db_error:
                app.logger.error(f"Database error while looking up document: {db_error}")
                return jsonify(success=False, message=f"Database error: {str(db_error)}"), 500
            
            if not doc_id:
                # Add to database if not found
                try:
                    doc_id = app.db.add_document(filepath)
                    app.logger.info(f"Added document to database with ID: {doc_id}")
                    if not doc_id:
                        app.logger.error("Failed to add document to database")
                        return jsonify(success=False, message="Failed to add document to database"), 500
                except Exception as add_error:
                    app.logger.error(f"Error adding document to database: {add_error}")
                    return jsonify(success=False, message=f"Error adding document: {str(add_error)}"), 500
            
            # Reset status
            try:
                if not app.db.reset_document_status_by_id(doc_id):
                    app.logger.error(f"Failed to reset document status for ID: {doc_id}")
                    return jsonify(success=False, message="Failed to reset document status"), 500
                app.logger.info(f"Reset document status for ID: {doc_id}")
            except Exception as reset_error:
                app.logger.error(f"Error resetting document status: {reset_error}")
                return jsonify(success=False, message=f"Error resetting status: {str(reset_error)}"), 500
            
            # Process with PDF processor
            try:
                app.pdf_processor.process_document(filepath, doc_id, app.db)
                app.logger.info(f"Processed document with PDF processor: {doc_id}")
            except Exception as pdf_error:
                app.logger.error(f"Error processing PDF: {pdf_error}")
                return jsonify(success=False, message=f"Error processing PDF: {str(pdf_error)}"), 500
            
            # Queue for OCR
            try:
                if not app.ocr_queue.add_to_queue(doc_id, filepath):
                    app.logger.error(f"Failed to add document to OCR queue: {doc_id}")
                    return jsonify(success=False, message="Failed to add to OCR queue"), 500
                app.logger.info(f"Added document to OCR queue: {doc_id}")
            except Exception as ocr_error:
                app.logger.error(f"Error adding to OCR queue: {ocr_error}")
                return jsonify(success=False, message=f"Error adding to OCR queue: {str(ocr_error)}"), 500
            
            app.logger.info(f"Successfully processed PDF file: {base_filename}")
            return jsonify(success=True, message=f"PDF file queued for processing: {base_filename}")
            
        except Exception as e:
            app.logger.error(f"Unexpected error fixing PDF file: {e}")
            import traceback
            tb = traceback.format_exc()
            app.logger.error(f"Traceback: {tb}")
            return jsonify(success=False, message=f"Unexpected error: {str(e)}"), 500

    @app.route('/document/<int:doc_id>/annotations')
    @login_required
    def document_annotations(doc_id):
        """View annotations for a document."""
        document = app.db.get_document_by_id(doc_id)
        if not document:
            return render_template('error.html', message="Document not found"), 404
            
        # Get annotations
        annotations = []
        try:
            with app.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, page_number, annotation_type, text, created_at
                    FROM document_annotations
                    WHERE doc_id = ?
                    ORDER BY page_number, id
                """, (doc_id,))
                
                annotations = [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            app.logger.error(f"Error fetching annotations: {e}")
            
        return render_template(
            'document_annotations.html',
            document=document,
            annotations=annotations
        )

    @app.route('/wordcloud')
    @login_required
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
        folder_categories = dict(sorted(
            cross_device_categories.items(),
            key=lambda item: len(item[1]),
            reverse=True
        ))

        return render_template(
            'wordcloud.html',
            all_folders=all_folders,
            folder_categories=folder_categories,
            device_folders=device_folders
        )

    @app.route('/generate_wordcloud')
    @login_required
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
    
    @app.route('/api/update_transcription', methods=['POST'])
    @login_required
    def update_transcription():
        """Update the transcription for a specific page."""
        try:
            data = request.json
            doc_id = data.get('doc_id')
            page_number = data.get('page_number')
            new_text = data.get('text')
            field_type = data.get('field_type', 'ocr_text')  # or 'text_content'
            
            if not all([doc_id, page_number, new_text is not None]):
                return jsonify(success=False, message="Missing required fields"), 400
            
            # Get current text for history
            with app.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT {} FROM pdf_text_content 
                    WHERE pdf_id = ? AND page_number = ?
                """.format(field_type), (doc_id, page_number))
                
                current = cursor.fetchone()
                if not current:
                    return jsonify(success=False, message="Page not found"), 404
                
                original_text = current[0]
                
                # Update the text
                cursor.execute("""
                    UPDATE pdf_text_content 
                    SET {} = ?, processed_at = ?
                    WHERE pdf_id = ? AND page_number = ?
                """.format(field_type), (new_text, datetime.now(), doc_id, page_number))
                
                # Record in history
                cursor.execute("""
                    INSERT INTO edit_history 
                    (doc_id, page_number, field_type, original_text, edited_text)
                    VALUES (?, ?, ?, ?, ?)
                """, (doc_id, page_number, 'transcription', original_text, new_text))
                
                conn.commit()
                
            return jsonify(success=True, message="Transcription updated successfully")
            
        except Exception as e:
            app.logger.error(f"Error updating transcription: {e}")
            return jsonify(success=False, message=str(e)), 500

    @app.route('/api/update_annotation', methods=['POST'])
    @login_required
    def update_annotation():
        """Update an annotation."""
        try:
            data = request.json
            annotation_id = data.get('annotation_id')
            new_text = data.get('text')
            
            if not all([annotation_id, new_text is not None]):
                return jsonify(success=False, message="Missing required fields"), 400
            
            with app.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get current annotation
                cursor.execute("""
                    SELECT doc_id, page_number, text 
                    FROM document_annotations 
                    WHERE id = ?
                """, (annotation_id,))
                
                current = cursor.fetchone()
                if not current:
                    return jsonify(success=False, message="Annotation not found"), 404
                
                # Update annotation
                cursor.execute("""
                    UPDATE document_annotations 
                    SET text = ?
                    WHERE id = ?
                """, (new_text, annotation_id))
                
                # Record in history
                cursor.execute("""
                    INSERT INTO edit_history 
                    (doc_id, page_number, field_type, annotation_id, original_text, edited_text)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (current['doc_id'], current['page_number'], 'annotation', 
                    annotation_id, current['text'], new_text))
                
                conn.commit()
                
            return jsonify(success=True, message="Annotation updated successfully")
            
        except Exception as e:
            app.logger.error(f"Error updating annotation: {e}")
            return jsonify(success=False, message=str(e)), 500

    @app.route('/api/delete_annotation/<int:annotation_id>', methods=['DELETE'])
    @login_required
    def delete_annotation(annotation_id):
        """Delete an annotation."""
        try:
            with app.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if annotation exists
                cursor.execute("SELECT id FROM document_annotations WHERE id = ?", (annotation_id,))
                if not cursor.fetchone():
                    return jsonify(success=False, message="Annotation not found"), 404
                
                # Delete the annotation
                cursor.execute("DELETE FROM document_annotations WHERE id = ?", (annotation_id,))
                conn.commit()
                
                app.logger.info(f"Deleted annotation {annotation_id}")
                return jsonify(success=True, message="Annotation deleted successfully")
                
        except Exception as e:
            app.logger.error(f"Error deleting annotation {annotation_id}: {e}")
            return jsonify(success=False, message=str(e)), 500
        
    @app.route('/annotations')
    @login_required
    def annotations_overview():
        """View all annotations across documents with filtering."""
        # Get filter parameters
        annotation_type = request.args.get('type', '')  # 'green_box', 'yellow_highlight', or ''
        date_from = request.args.get('from', '')
        date_to = request.args.get('to', '')
        category_filter = request.args.get('category', '')
                                    
        # First, get all folders for category building (same as search/wordcloud)
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
        
        # Build folder categories (same logic as other views)
        folder_categories = {}
        device_folders = {}
        common_categories = set()
        
        for folder in all_folders:
            if not folder:
                continue
                
            parts = folder.split('/')
            if not parts or not parts[0]:
                continue
                
            device = parts[0]
            if device not in device_folders:
                device_folders[device] = []
            device_folders[device].append(folder)
            
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
        
        folder_categories = dict(sorted(
            cross_device_categories.items(),
            key=lambda item: len(item[1]),
            reverse=True
        ))
        
        # Build query
        query = """
            SELECT 
                a.id,
                a.doc_id,
                a.page_number,
                a.annotation_type,
                a.text,
                a.created_at,
                d.filename,
                d.folder_path,
                d.file_created_at
            FROM document_annotations a
            JOIN pdf_documents d ON a.doc_id = d.id
            WHERE d.processing_status != 'removed'
        """
        
        params = []
        
        # Filter by annotation type
        if annotation_type:
            query += " AND a.annotation_type = ?"
            params.append(annotation_type)
                
        # Filter by category (folder pattern)
        if category_filter:
            # Get all folders matching this category
            matching_folders = []
            print(f"ANNOTATION DEBUG: Looking for category: '{category_filter}'", file=sys.stderr)
            print(f"ANNOTATION DEBUG: Total folders available: {len(all_folders)}", file=sys.stderr)
            
            for folder in all_folders:
                # Split the folder into parts
                parts = folder.split('/')
                # Check if the category appears anywhere in the path
                if category_filter in parts:
                    matching_folders.append(folder)
                    print(f"ANNOTATION DEBUG: MATCH: '{folder}' contains '{category_filter}'", file=sys.stderr)
            
            print(f"ANNOTATION DEBUG: Total matching folders: {len(matching_folders)}", file=sys.stderr)
            if matching_folders:
                print(f"ANNOTATION DEBUG: Matching folders: {matching_folders}", file=sys.stderr)
            
            if matching_folders:
                placeholders = ','.join(['?' for _ in matching_folders])
                
                # DEBUG: Check what annotations exist in these folders
                print(f"ANNOTATION DEBUG: Checking annotations in Moffitt folders...", file=sys.stderr)
                debug_cursor = conn.cursor()
                debug_cursor.execute(f"""
                    SELECT d.folder_path, a.annotation_type, COUNT(*) as count
                    FROM document_annotations a
                    JOIN pdf_documents d ON a.doc_id = d.id
                    WHERE d.folder_path IN ({placeholders})
                    GROUP BY d.folder_path, a.annotation_type
                """, matching_folders)
                
                for row in debug_cursor.fetchall():
                    print(f"ANNOTATION DEBUG: {row['folder_path']} has {row['count']} {row['annotation_type']} annotations", file=sys.stderr)
                query += f" AND d.folder_path IN ({placeholders})"
                params.extend(matching_folders)
            else:
                # If no folders match, we'll get no results
                query += " AND 1=0"  # Force no results
                print("ANNOTATION DEBUG: No matching folders found - forcing empty result set", file=sys.stderr)
                    
                                 
        # Filter by date range (using file creation date)
        if date_from:
            query += " AND DATE(d.file_created_at) >= DATE(?)"
            params.append(date_from)
        
        if date_to:
            query += " AND DATE(d.file_created_at) <= DATE(?)"
            params.append(date_to)
        
        query += " ORDER BY d.file_created_at DESC, a.page_number"
        
        annotations = []
        annotation_counts = {'green_box': 0, 'yellow_highlight': 0}
        
        try:
            with app.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Right before executing the query
                app.logger.info(f"Final query: {query}")
                app.logger.info(f"Query params: {params}")

                cursor.execute(query, params)
                
                for row in cursor.fetchall():
                    annotations.append(dict(row))
                    
                # Get counts by type (respecting current filters)
                count_query = """
                    SELECT annotation_type, COUNT(*) as count
                    FROM document_annotations a
                    JOIN pdf_documents d ON a.doc_id = d.id
                    WHERE d.processing_status != 'removed'
                """
                count_params = []
                
                if category_filter:
                    count_query += f" AND d.folder_path IN ({placeholders})"
                    count_params.extend(matching_folders)
                    
                count_query += " GROUP BY annotation_type"
                
                cursor.execute(count_query, count_params)
                
                for row in cursor.fetchall():
                    if row['annotation_type'] in annotation_counts:
                        annotation_counts[row['annotation_type']] = row['count']
                        
        except Exception as e:
            app.logger.error(f"Error fetching annotations: {e}")
        
        return render_template(
            'annotations_overview.html',
            annotations=annotations,
            annotation_counts=annotation_counts,
            folder_categories=folder_categories,
            device_folders=device_folders,
            current_type=annotation_type,
            current_category=category_filter,
            date_from=date_from,
            date_to=date_to
        )
                
    return app