from flask import Flask, render_template, jsonify, request
import os
from typing import List
from file_watcher import FileWatcher
import time

app = Flask(__name__)

# Directory to monitor
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
# Initialize the file watcher with default polling interval
DEFAULT_POLLING_INTERVAL = 30.0  # 30 seconds default
file_watcher = FileWatcher(UPLOAD_FOLDER, polling_interval=DEFAULT_POLLING_INTERVAL)

# Collect newly added and removed files for realtime updates
new_files: List[str] = []
removed_files: List[str] = []

# Flag to track whether the file watcher has been initialized
file_watcher_initialized = False

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html',
                          files=file_watcher.files,
                          polling_interval=file_watcher.polling_interval)

@app.route('/files')
def get_files():
    """API endpoint to get the current list of files."""
    return jsonify(files=file_watcher.files)

@app.route('/new_files')
def get_new_files():
    """API endpoint for polling new and removed files (for real-time updates)."""
    global new_files, removed_files
    
    # Perform an immediate scan to mitigate longer polling intervals
    new_discovered, removed_discovered = file_watcher.scan_now()
    
    # Capture any files that might have been detected by the background watcher
    files_to_return = new_files.copy()
    removed_to_return = removed_files.copy()
    
    # Add files from immediate scan (avoiding duplicates)
    for file in new_discovered:
        if file not in files_to_return:
            files_to_return.append(file)
    
    for file in removed_discovered:
        if file not in removed_to_return:
            removed_to_return.append(file)
    
    # Clear the pending lists
    new_files = []
    removed_files = []
    
    return jsonify(new_files=files_to_return, removed_files=removed_to_return)

@app.route('/scan')
def scan_files():
    """Manual endpoint to trigger an immediate scan."""
    new_discovered, removed_discovered = file_watcher.scan_now()
    return jsonify(new_files=new_discovered, removed_files=removed_discovered)

@app.route('/settings', methods=['POST'])
def update_settings():
    """Update file watcher settings."""
    try:
        # Get the new polling interval from form data
        new_interval = float(request.form.get('polling_interval', DEFAULT_POLLING_INTERVAL))
        
        # Update the polling interval
        success = file_watcher.set_polling_interval(new_interval)
        
        if success:
            return jsonify(success=True, message=f"Polling interval updated to {new_interval} seconds")
        else:
            return jsonify(success=False, message="Invalid polling interval"), 400
            
    except ValueError:
        return jsonify(success=False, message="Invalid polling interval format"), 400

@app.route('/settings')
def get_settings():
    """Get current settings."""
    return jsonify(
        polling_interval=file_watcher.polling_interval
    )

def on_new_file(filename: str):
    """Callback function for when a new file is detected."""
    global new_files
    new_files.append(filename)
    print(f"New file detected: {filename} at {time.strftime('%H:%M:%S')}")

def on_file_removed(filename: str):
    """Callback function for when a file is removed."""
    global removed_files
    removed_files.append(filename)
    print(f"File removed: {filename} at {time.strftime('%H:%M:%S')}")

@app.before_request
def initialize_file_watcher():
    """Initialize the file watcher before handling the first request."""
    global file_watcher_initialized
    if not file_watcher_initialized:
        print(f"Initializing file watcher at {time.strftime('%H:%M:%S')}")
        file_watcher.register_callback(on_new_file)
        file_watcher.register_removal_callback(on_file_removed)
        file_watcher.start()
        file_watcher_initialized = True
        print(f"File watcher started with polling interval of {file_watcher.polling_interval} seconds")

if __name__ == '__main__':
    # Make sure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Start the Flask app
    app.run(debug=True)