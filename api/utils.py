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
                "file_path": image_path,
                "file_name": os.path.basename(image_path),
                "width": width,
                "height": height,
                "size_score": size_score
            })
    
    if embeddings:
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
    """Search with normalized embeddings and adjust ranking based on image size."""
    query_embedding = encode_text_query(query)
    
    # First get the total count of images in the collection
    try:
        all_results = collection.get()
        if not all_results['ids']:
            return {"ids": [], "metadatas": [], "distances": []}
    except Exception as e:
        print(f"Error getting collection: {str(e)}")
        return {"ids": [], "metadatas": [], "distances": []}
    
    # Perform the search
    try:
        # Get more results than needed to allow for reranking
        n_search = min(n_results * 3, len(all_results['ids']))
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_search,
            include=["metadatas", "distances"]
        )
        
        # Filter and rerank results
        valid_results = []
        for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
            if not os.path.exists(metadata['file_path']):
                continue
                
            # Get the size score
            size_score = metadata.get('size_score', 0)
            
            # Combine CLIP similarity with size score
            # Convert distance to similarity (1 - distance) and weight it with size
            similarity = 1 - distance
            adjusted_score = similarity * (0.7 + 0.3 * (size_score / 20))  # Adjust weights as needed
            
            valid_results.append({
                'index': i,
                'metadata': metadata,
                'distance': distance,
                'adjusted_score': adjusted_score
            })
        
        # Sort by adjusted score and take top n_results
        valid_results.sort(key=lambda x: x['adjusted_score'], reverse=True)
        valid_results = valid_results[:n_results]
        
        # Format the results
        filtered_results = {
            "ids": [results['ids'][0][r['index']] for r in valid_results],
            "metadatas": [r['metadata'] for r in valid_results],
            "distances": [r['distance'] for r in valid_results]
        }
        
        return filtered_results
    except Exception as e:
        print(f"Error searching images: {str(e)}")
        return {"ids": [], "metadatas": [], "distances": []}

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
            for metadata, distance in zip(results["metadatas"], results["distances"]):
                print(f"Distance: {distance:.3f} | Path: {metadata['file_path']}")
        else:
            print("Please enter a valid query.")

if __name__ == "__main__":
    main()