import os
import time
import threading
from typing import List, Callable, Optional, Dict, Set, Tuple
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

class FileWatcher:
    def __init__(
        self, 
        directory_path: str, 
        file_types: Optional[List[str]] = None,
        polling_interval: float = 60.0,
        recursive: bool = True
    ):
        self.directory_path = os.path.abspath(directory_path)
        self.file_types = file_types or []
        self.files: List[str] = []
        self.file_mtimes: Dict[str, float] = {}
        self.callbacks: List[Callable] = []
        self.removal_callbacks: List[Callable] = []
        self._observer: Optional[PollingObserver] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stopped = False
        self.polling_interval = polling_interval
        self.recursive = recursive
        
        if not file_types:
            file_types = ['.pdf', '.html', '.htm']
        self.file_types = file_types
        
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)
            
        self._load_existing_files()
    
    def _is_valid_file_type(self, filename: str) -> bool:
        """Check if the file has a valid extension."""
        if filename.startswith('.'):
            return False

        if not self.file_types:
            return True
        
        return any(filename.lower().endswith(ext.lower()) for ext in self.file_types)
    
    def _load_existing_files(self) -> None:
        """Load existing files, optionally scanning subdirectories."""
        files_with_times = []
        self.file_mtimes = {}
        
        # Walk through directory tree if recursive, otherwise just list top directory
        if self.recursive:
            for root, _, filenames in os.walk(self.directory_path):
                for filename in filenames:
                    if self._is_valid_file_type(filename):
                        # Store path relative to the base directory
                        file_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(file_path, self.directory_path)
                        mtime = os.path.getmtime(file_path)
                        files_with_times.append((rel_path, mtime))
                        self.file_mtimes[rel_path] = mtime
        else:
            # Original non-recursive behavior
            for filename in os.listdir(self.directory_path):
                file_path = os.path.join(self.directory_path, filename)
                if os.path.isfile(file_path) and self._is_valid_file_type(filename):
                    mtime = os.path.getmtime(file_path)
                    files_with_times.append((filename, mtime))
                    self.file_mtimes[filename] = mtime
        
        # Sort by modification time (most recent first) and extract just the filenames
        files_with_times.sort(key=lambda x: x[1], reverse=True)
        self.files = [f[0] for f in files_with_times]
        
    def register_callback(self, callback: Callable) -> None:
        self.callbacks.append(callback)
    
    def register_removal_callback(self, callback: Callable) -> None:
        self.removal_callbacks.append(callback)
    
    def set_polling_interval(self, interval: float) -> bool:
        if interval < 1.0:
            return False
        self.polling_interval = interval
        
        if self._running:
            self.stop()
            self.start()
            
        return True
    
    def start(self) -> None:
        if self._running:
            return
        
        class Handler(FileSystemEventHandler):
            def __init__(self, watcher: 'FileWatcher'):
                self.watcher = watcher
                
            def on_created(self, event):
                if not event.is_directory:
                    file_path = event.src_path
                    rel_path = os.path.relpath(file_path, self.watcher.directory_path)
                    filename = os.path.basename(file_path)
                    
                    if (self.watcher._is_valid_file_type(filename) and 
                        rel_path not in self.watcher.files):
                        # Add the new file
                        mtime = os.path.getmtime(file_path)
                        self.watcher.file_mtimes[rel_path] = mtime
                        
                        # Re-sort the entire list to maintain order
                        files_with_times = [(f, self.watcher.file_mtimes.get(f, 0)) 
                                            for f in self.watcher.files]
                        files_with_times.append((rel_path, mtime))
                        files_with_times.sort(key=lambda x: x[1], reverse=True)
                        self.watcher.files = [f[0] for f in files_with_times]
                        
                        for callback in self.watcher.callbacks:
                            callback(rel_path)
                            
            def on_deleted(self, event):
                if not event.is_directory:
                    file_path = event.src_path
                    rel_path = os.path.relpath(file_path, self.watcher.directory_path)
                    
                    if rel_path in self.watcher.files:
                        self.watcher.files.remove(rel_path)
                        if rel_path in self.watcher.file_mtimes:
                            del self.watcher.file_mtimes[rel_path]
                        for callback in self.watcher.removal_callbacks:
                            callback(rel_path)
        
        self._running = True
        
        self._observer = PollingObserver(timeout=self.polling_interval)
        self._observer.schedule(
            Handler(self), 
            self.directory_path, 
            recursive=self.recursive
        )
        
        def run_observer():
            self._observer.start()
            try:
                while self._running:
                    time.sleep(1)
            finally:
                self._observer.stop()
                self._observer.join()
        
        self._thread = threading.Thread(target=run_observer, daemon=True)
        self._thread.start()
        
        print(f"File watcher started with polling interval of {self.polling_interval} seconds")
        
    def scan_now(self) -> Tuple[List[str], List[str]]:
        """Scan for new and removed files."""
        new_files = []
        removed_files = []
        
        current_files = set()
        
        # Use os.walk for recursive scanning
        if self.recursive:
            for root, _, filenames in os.walk(self.directory_path):
                for filename in filenames:
                    if self._is_valid_file_type(filename):
                        file_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(file_path, self.directory_path)
                        current_files.add(rel_path)
        else:
            # Original non-recursive behavior
            for filename in os.listdir(self.directory_path):
                file_path = os.path.join(self.directory_path, filename)
                if os.path.isfile(file_path) and self._is_valid_file_type(filename):
                    current_files.add(filename)
        
        # Find new files
        for rel_path in current_files:
            if rel_path not in self.files:
                file_path = os.path.join(self.directory_path, rel_path)
                file_mtime = os.path.getmtime(file_path)
                
                # Check if this is truly a new file or just an updated version of existing file
                # Only consider as new if it's not in file_mtimes or has been modified
                is_new = (rel_path not in self.file_mtimes or 
                        file_mtime > self.file_mtimes.get(rel_path, 0))
                
                if is_new:
                    self.files.append(rel_path)
                    self.file_mtimes[rel_path] = file_mtime
                    new_files.append(rel_path)
                    
                    # Only call callbacks for genuinely new files
                    for callback in self.callbacks:
                        callback(rel_path)
        
        # Find removed files
        files_set = set(self.files)
        for rel_path in files_set:
            if rel_path not in current_files:
                self.files.remove(rel_path)
                if rel_path in self.file_mtimes:
                    del self.file_mtimes[rel_path]
                removed_files.append(rel_path)
                
                for callback in self.removal_callbacks:
                    callback(rel_path)
                    
        # In scan_now, after the "Find new files" section, before returning:
        # Re-sort the files list to maintain order
        files_with_times = []
        for rel_path in self.files:
            file_path = os.path.join(self.directory_path, rel_path)
            if os.path.exists(file_path):
                mtime = os.path.getmtime(file_path)
                files_with_times.append((rel_path, mtime))

        # Sort by modification time (most recent first)
        files_with_times.sort(key=lambda x: x[1], reverse=True)
        self.files = [f[0] for f in files_with_times]

        return new_files, removed_files
    
    def stop(self) -> None:
        if self._stopped:
            return

        print("Stopping file watcher...")
        self._running = False

        if self._observer:
            print("Stopping observer...")
            self._observer.stop()
            self._observer.join(timeout=3)

        if self._thread and self._thread.is_alive():
            print("Stopping watcher thread...")
            self._thread.join(timeout=3)

        print("File watcher stopped.")
        self._stopped = True