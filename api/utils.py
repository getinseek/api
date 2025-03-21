import os
import argparse
import math

import torch
import clip
from PIL import Image
import chromadb

import mimetypes

# Constants
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MIN_IMAGE_SIZE = 50  # Images smaller than this will be penalized in ranking

# Initialize CLIP model
print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# Initialize ChromaDB with cosine similarity
print("Initializing ChromaDB...")
client = chromadb.PersistentClient(path="db")

def get_collection():
    """Get a fresh instance of the ChromaDB collection."""
    return client.get_or_create_collection(
        name="photos",
        metadata={"hnsw:space": "cosine"}  # Explicitly set cosine similarity
    )

# Initial collection instance
collection = get_collection()

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

def get_image_dimensions(image_path):
    """Get image dimensions and calculate a size score."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            # Calculate a score that favors larger images but with diminishing returns
            size_score = math.log(max(width, MIN_IMAGE_SIZE) * max(height, MIN_IMAGE_SIZE))
            return width, height, size_score
    except Exception as e:
        print(f"Error getting image dimensions for {image_path}: {str(e)}")
        return None, None, 0

def find_images(directory):
    """Recursively find image files in a directory, skipping macOS system and app folders."""
    image_paths = []
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and system folders
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in SKIP_DIRS]
        
        for file in files:
            # Skip hidden files
            if not file.startswith('.'):
                if mimetypes.guess_type(file)[0].startswith('image/'):
                    print("Found image:", os.path.join(root, file))
                    image_paths.append(os.path.join(root, file))
    return image_paths

def generate_embedding(image_path):
    """Generate a normalized embedding for an image."""
    try:
        image = Image.open(image_path)
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        # Get image dimensions
        width, height, size_score = get_image_dimensions(image_path)
        if width is None or height is None:
            return None
            
        # Skip very small images
        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            print(f"Skipping small image {image_path} ({width}x{height})")
            return None
            
        image_input = preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = model.encode_image(image_input)
            embedding /= embedding.norm(dim=-1, keepdim=True)  # Normalize
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def index_images(directory):
    """Index images with error handling and overwrite existing entries."""
    image_paths = find_images(directory)
    embeddings = []
    ids = []
    metadatas = []
    
    for image_path in image_paths:
        embedding = generate_embedding(image_path)
        if embedding is not None:
            # Get image dimensions
            width, height, size_score = get_image_dimensions(image_path)
            if width is None or height is None:
                continue
                
            embeddings.append(embedding.tolist())
            ids.append(image_path)
            metadatas.append({
                "file_path": os.path.abspath(image_path),
                "file_name": os.path.basename(image_path),
                "width": width,
                "height": height,
                "size_score": size_score
            })
    
    if embeddings:
        # Upsert to replace existing entries
        get_collection().upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        print(f"Indexed {len(embeddings)} images.")

def encode_text_query(query):
    """Generate a normalized text embedding."""
    text = clip.tokenize([query]).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_text(text)
        embedding /= embedding.norm(dim=-1, keepdim=True)  # Normalize
    return embedding.cpu().numpy().flatten()

def search_images(query, n_results=5):
    """Search with normalized embeddings and adjust ranking based on image size."""
    query_embedding = encode_text_query(query)
    
    try:
        # Get more results than needed to allow for filtering
        results = get_collection().query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results * 3,  # Get extra results for filtering
            include=["metadatas", "distances"]
        )
        
        # Check if we have valid results
        if not results or 'metadatas' not in results or not results['metadatas']:
            print("No results found")
            return {"files": []}
            
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        if not metadatas or not distances:
            print("No valid metadata or distances")
            return {"files": []}
            
        # Filter and rerank results
        valid_results = []
        for metadata, distance in zip(metadatas, distances):
            if metadata and 'file_path' in metadata:
                file_path = metadata['file_path']
                if os.path.exists(file_path):
                    valid_results.append({
                        "file_path": file_path,
                        "file_name": metadata.get('file_name', os.path.basename(file_path)),
                        "distance": float(distance)
                    })
                
        # Sort by distance and take top n_results
        valid_results.sort(key=lambda x: x['distance'])
        valid_results = valid_results[:n_results]
        
        return {"files": valid_results}
        
    except Exception as e:
        print(f"Error searching images: {str(e)}")
        return {"files": []}

def main():
    parser = argparse.ArgumentParser(description="Photo search with CLIP")
    parser.add_argument("--index", type=str, help="Directory to index")
    parser.add_argument("--n-results", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()
    
    if args.index:
        index_images(args.index)
        return
    
    while True:
        query = input("Enter search query (or 'q' to quit): ")
        if query.lower() == 'q':
            break
        if query:
            results = search_images(query, args.n_results)
            for file in results["files"]:
                print(f"Distance: {file['distance']:.3f} | Path: {file['file_path']}")
        else:
            print("Please enter a valid query.")

if __name__ == "__main__":
    main()