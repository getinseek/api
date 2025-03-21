import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import mimetypes
from typing import Optional, Set
from PIL import Image
import math
from .utils import generate_embedding, get_collection, SKIP_DIRS, MIN_IMAGE_SIZE

class ImageHandler(FileSystemEventHandler):
    def __init__(self, watch_dir: str):
        self.watch_dir = watch_dir
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        
    def _is_valid_image(self, file_path: str) -> bool:
        """Check if the file is a valid image and not in a skip directory."""
        if any(skip_dir in file_path for skip_dir in SKIP_DIRS):
            return False

        # Check the mimetype of the file using the file command
        mimetype = mimetypes.guess_type(file_path)[0]
        return mimetype and mimetype.startswith('image/')
        
    def _get_image_dimensions(self, file_path: str):
        """Get image dimensions and calculate a size score."""
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                # Calculate a score that favors larger images but with diminishing returns
                size_score = math.log(max(width, MIN_IMAGE_SIZE) * max(height, MIN_IMAGE_SIZE))
                return width, height, size_score
        except Exception as e:
            print(f"Error getting image dimensions for {file_path}: {str(e)}")
            return None, None, 0
        
    def _remove_from_db(self, file_path: str):
        """Remove a file from the vector database."""
        try:
            # First find any entries that match this path
            results = get_collection().get(
                where={"file_path": file_path},
                include=["metadatas", "documents", "embeddings"]
            )
            if results and results['ids']:
                get_collection().delete(ids=results['ids'])
                print(f"Removed from vector DB: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path} from DB: {str(e)}")
        
    def _handle_file_change(self, file_path: str):
        """Handle a file change event by updating the vector database."""
        try:
            if not self._is_valid_image(file_path):
                return
                
            # First remove any existing entries
            self._remove_from_db(file_path)
                
            # Get image dimensions
            width, height, size_score = self._get_image_dimensions(file_path)
            if width is None or height is None:
                return
                
            # Skip very small images
            if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                print(f"Skipping small image {file_path} ({width}x{height})")
                return
                
            # Generate embedding and metadata
            embedding = generate_embedding(file_path)
            if embedding is None:
                return
                
            # Add to collection with size information
            file_name = os.path.basename(file_path)
            get_collection().upsert(
                ids=[file_path],
                embeddings=[embedding.tolist()],
                metadatas=[{
                    "file_path": os.path.abspath(file_path),
                    "file_name": os.path.basename(file_path),
                    "width": width,
                    "height": height,
                    "size_score": size_score
                }]
            )
            print(f"Added to vector DB: {file_path} ({width}x{height})")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    def on_created(self, event):
        if not event.is_directory:
            self._handle_file_change(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._handle_file_change(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            # Remove the old path from the database
            if self._is_valid_image(event.src_path):
                self._remove_from_db(event.src_path)
            
            # Add the new path if it's an image
            if self._is_valid_image(event.dest_path):
                self._handle_file_change(event.dest_path)

    def on_deleted(self, event):
        if not event.is_directory and self._is_valid_image(event.src_path):
            self._remove_from_db(event.src_path)

class FileWatcher:
    def __init__(self, watch_dir: str):
        self.watch_dir = os.path.abspath(watch_dir)
        self.event_handler = ImageHandler(self.watch_dir)
        self.observer = Observer()
        
    def watch(self):
        """Start watching for file changes."""
        try:
            print(f"Starting to watch directory: {self.watch_dir}")
            print("Monitoring for new or modified image files...")
            self.observer.schedule(self.event_handler, self.watch_dir, recursive=True)
            self.observer.start()
            
            # Keep the thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping file watcher...")
                self.stop()
                
        except Exception as e:
            print(f"Error in file watcher: {str(e)}")
            self.stop()
            
    def stop(self):
        """Stop the file watcher."""
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
