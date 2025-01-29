import os
from pathlib import Path

SKIP_DIRS = {
    'Library',
    '.Trash',
    'System',
    '.git',
    'node_modules',
    '.cache',
    '__pycache__',
    'Applications',
    'Pictures Library.photoslibrary'  # Skip Photos app library structure
}


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".heic"]

def find_images(directory):
    """Recursively find image files in a directory, skipping macOS system and app folders."""
    image_paths = []
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and system folders
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in SKIP_DIRS]
        
        for file in files:
            # Skip hidden files
            if not file.startswith('.'):
                if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                    print("Found image:", os.path.join(root, file))
                    image_paths.append(os.path.join(root, file))
    return image_paths

directory = "/Users/shubhampatil/"
image_paths = find_images(directory)
print(f"Found {len(image_paths)} images.")