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

# Collect newly added files for realtime updates
new_files: List[str] = []

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
    """API endpoint for polling new files (for real-time updates)."""
    global new_files
    
    # Perform an immediate scan to mitigate longer polling intervals
    file_watcher.scan_now()
    
    files_to_return = new_files.copy()
    new_files = []
    return jsonify(new_files=files_to_return)

@app.route('/scan')
def scan_files():
    """Manual endpoint to trigger an immediate scan."""
    new_discovered = file_watcher.scan_now()
    return jsonify(new_files=new_discovered)

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

@app.before_first_request
def start_file_watcher():
    """Start the file watcher before serving the first request."""
    file_watcher.register_callback(on_new_file)
    file_watcher.start()

if __name__ == '__main__':
    # Make sure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Start the Flask app
    app.run(debug=True)