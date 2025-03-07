import mimetypes
import os
import argparse
from pathlib import Path

import torch
import clip
from PIL import Image
import chromadb
import numpy as np
import magic

# Constants
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".heic"]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Initialize CLIP model
print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# Initialize ChromaDB with cosine similarity
print("Initializing ChromaDB...")
client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection(
    name="photos",
    metadata={"hnsw:space": "cosine"}  # Explicitly set cosine similarity
)

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

def handle_applications(file_path):
    file_type = magic.from_file(file_path, mime=True)
    if file_type.startswith('application/pdf'):
            print("Handling PDF document...")
            # Add PDF-specific processing here
            return True
            
    elif file_type.startswith('application/msword') or file_type.startswith('application/vnd.openxmlformats-officedocument.wordprocessingml'):
        print("Handling Word document...")
        # Add Word document processing here
        return True
        
    elif file_type.startswith('application/vnd.ms-excel') or file_type.startswith('application/vnd.openxmlformats-officedocument.spreadsheetml'):
        print("Handling Excel spreadsheet...")
        # Add Excel-specific processing here
        return True

def find_files(directory):
    """Recursively find image files in a directory, skipping macOS system and app folders."""
    image_paths = []
    audio_paths = []
    video_paths = []
    pdf_paths = []
    docx_paths = []
    txt_paths = []
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and system folders
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in SKIP_DIRS]
        
        for file in files:
            # Skip hidden files
            fileType = magic.Magic(mime=True)
            mime = fileType.from_file(os.path.join(root, file))
            if not file.startswith('.'):
                if mime.startswith("text/"):
                    txt_paths.append(os.path.join(root, file))
                if mime.startswith("image/"):
                    print("Found image:", os.path.join(root, file))
                    image_paths.append(os.path.join(root, file))
                if mime.startswith("video/"):
                    video_paths.append(os.path.join(root, file))
                if mime.startswith("audio/"):
                    audio_paths.append(os.path.join(root, file))
                if mime.startswith("application/"):
                    audio_paths.append(os.path.join(root, file))
           
           
           
                


    return image_paths

def is_plaintext(file_path):
    mime = mimetypes.guess_type(file_path)[0]
    return mime is not None and mime.startswith("text/")

def generate_embedding(image_path):
    """Generate a normalized embedding for an image."""
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = model.encode_image(image)
            embedding /= embedding.norm(dim=-1, keepdim=True)  # Normalize
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Skipping {image_path}: {str(e)}")
        return None

def index_files(directory):
    """Index images with error handling and overwrite existing entries."""
    image_paths = find_files(directory)
    print(f"Found {len(image_paths)} images. Processing...")

    embeddings, ids, metadatas = [], [], []
    for image_path in image_paths:
        print(f"Indexing {image_path}...")
        embedding = generate_embedding(image_path)
        if embedding is not None:
            embeddings.append(embedding.tolist())
            ids.append(image_path)
            metadatas.append({"path": image_path})
    
    # Upsert to replace existing entries
    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
    print(f"Indexed {len(embeddings)} images.")

def encode_text_query(query):
    """Generate a normalized text embedding."""
    text = clip.tokenize([query]).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_text(text)
        embedding /= embedding.norm(dim=-1, keepdim=True)  # Normalize
    return embedding.cpu().numpy().flatten()

def search_images(query, n_results=5):
    """Search with normalized embeddings."""
    query_embedding = encode_text_query(query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        include=["metadatas", "distances"]
    )
    return results

def main():
    parser = argparse.ArgumentParser(description="Photo search with CLIP")
    parser.add_argument("--index", type=str, help="Directory to index")
    parser.add_argument("--n_results", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    if args.index:
        index_files(args.index)

    while True:
        query = input("Enter search query (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        if query:
            results = search_images(query, args.n_results)
            for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
                print(f"Distance: {distance:.3f} | Path: {metadata['path']}")
        else:
            print("Please enter a valid query.")


if __name__ == "__main__":
    main()