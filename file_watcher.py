import os
import time
import threading
from typing import List, Callable, Optional, Dict, Set, Tuple
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

class FileWatcher:
    """Service that monitors a directory for new and removed files using polling for compatibility with CephFS."""
    
    def __init__(
        self, 
        directory_path: str, 
        file_types: Optional[List[str]] = None,
        polling_interval: float = 30.0
    ):
        """
        Initialize the file watcher service.
        
        Args:
            directory_path: Path to the directory to monitor
            file_types: Optional list of file extensions to monitor (e.g., ['.txt', '.pdf'])
            polling_interval: Time in seconds between polls (higher for networked filesystems)
        """
        self.directory_path = os.path.abspath(directory_path)
        self.file_types = file_types or []
        self.files: List[str] = []
        self.file_mtimes: Dict[str, float] = {}  # Track modification times
        self.callbacks: List[Callable] = []
        self.removal_callbacks: List[Callable] = []  # New: Callbacks for file removals
        self._observer: Optional[PollingObserver] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.polling_interval = polling_interval
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)
            
        # Initialize with existing files
        self._load_existing_files()
    
    def _load_existing_files(self) -> None:
        """Load existing files from the directory into the files list."""
        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)
            if os.path.isfile(file_path) and self._is_valid_file_type(filename):
                self.files.append(filename)
                # Store modification time for change detection
                self.file_mtimes[filename] = os.path.getmtime(file_path)
    
    def _is_valid_file_type(self, filename: str) -> bool:
        """Check if the file has an allowed extension."""
        if not self.file_types:
            return True
        return any(filename.endswith(ext) for ext in self.file_types)
    
    def register_callback(self, callback: Callable) -> None:
        """Register a callback function to be called when new files are detected."""
        self.callbacks.append(callback)
    
    def register_removal_callback(self, callback: Callable) -> None:
        """Register a callback function to be called when files are removed."""
        self.removal_callbacks.append(callback)
    
    def set_polling_interval(self, interval: float) -> bool:
        """
        Update the polling interval.
        
        Args:
            interval: New polling interval in seconds (minimum 1 second)
            
        Returns:
            bool: True if the interval was changed successfully
        """
        if interval < 1.0:
            return False  # Don't allow very short intervals
            
        # Store the new interval
        self.polling_interval = interval
        
        # If the watcher is running, restart it with the new interval
        if self._running:
            self.stop()
            self.start()
            
        return True
    
    def start(self) -> None:
        """Start the file watcher service in a background thread with explicit polling."""
        if self._running:
            return
        
        class Handler(FileSystemEventHandler):
            def __init__(self, watcher: 'FileWatcher'):
                self.watcher = watcher
                
            def on_created(self, event):
                if not event.is_directory:
                    filename = os.path.basename(event.src_path)
                    if (self.watcher._is_valid_file_type(filename) and 
                        filename not in self.watcher.files):
                        self.watcher.files.append(filename)
                        self.watcher.file_mtimes[filename] = os.path.getmtime(event.src_path)
                        for callback in self.watcher.callbacks:
                            callback(filename)
                            
            def on_deleted(self, event):
                if not event.is_directory:
                    filename = os.path.basename(event.src_path)
                    if filename in self.watcher.files:
                        self.watcher.files.remove(filename)
                        if filename in self.watcher.file_mtimes:
                            del self.watcher.file_mtimes[filename]
                        for callback in self.watcher.removal_callbacks:
                            callback(filename)
        
        self._running = True
        
        # Use PollingObserver with a longer interval for CephFS
        self._observer = PollingObserver(timeout=self.polling_interval)
        self._observer.schedule(Handler(self), self.directory_path, recursive=False)
        
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
        """
        Manually scan for new and removed files immediately.
        
        Returns:
            Tuple containing lists of (new_files, removed_files)
        """
        new_files = []
        removed_files = []
        
        # Get current files in directory
        current_files = set()
        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)
            if os.path.isfile(file_path) and self._is_valid_file_type(filename):
                current_files.add(filename)
        
        # Check for new files
        for filename in current_files:
            if filename not in self.files:
                file_path = os.path.join(self.directory_path, filename)
                self.files.append(filename)
                self.file_mtimes[filename] = os.path.getmtime(file_path)
                new_files.append(filename)
                
                # Call the callbacks
                for callback in self.callbacks:
                    callback(filename)
        
        # Check for removed files
        files_set = set(self.files)
        for filename in files_set:
            if filename not in current_files:
                self.files.remove(filename)
                if filename in self.file_mtimes:
                    del self.file_mtimes[filename]
                removed_files.append(filename)
                
                # Call the removal callbacks
                for callback in self.removal_callbacks:
                    callback(filename)
                    
        return new_files, removed_files
    
    def stop(self) -> None:
        """Stop the file watcher service."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        if self._observer:
            self._observer.stop()
            if self._observer.is_alive():
                self._observer.join(timeout=1)