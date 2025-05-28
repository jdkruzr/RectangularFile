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
        polling_interval: float = 60.0
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
        
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)
            
        self._load_existing_files()
    
    def _load_existing_files(self) -> None:
        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)
            if os.path.isfile(file_path) and self._is_valid_file_type(filename):
                self.files.append(filename)
                self.file_mtimes[filename] = os.path.getmtime(file_path)
    
    def _is_valid_file_type(self, filename: str) -> bool:
        if filename.startswith('.'):
            return False

        if not self.file_types:
            return True
    
        return any(filename.lower().endswith(ext.lower()) for ext in self.file_types)

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
        new_files = []
        removed_files = []
        
        current_files = set()
        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)
            if os.path.isfile(file_path) and self._is_valid_file_type(filename):
                current_files.add(filename)
        
        for filename in current_files:
            if filename not in self.files:
                file_path = os.path.join(self.directory_path, filename)
                self.files.append(filename)
                self.file_mtimes[filename] = os.path.getmtime(file_path)
                new_files.append(filename)
                
                for callback in self.callbacks:
                    callback(filename)
        files_set = set(self.files)
        for filename in files_set:
            if filename not in current_files:
                self.files.remove(filename)
                if filename in self.file_mtimes:
                    del self.file_mtimes[filename]
                removed_files.append(filename)
                
                for callback in self.removal_callbacks:
                    callback(filename)
                    
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

        if hasattr(self._observer, '_observers'):
            for observer in self._observer._observers:
                try:
                    observer.stop()
                    observer.join(timeout=1)
                except Exception:
                    pass

        print("File watcher stopped.")
        self._stopped = True

