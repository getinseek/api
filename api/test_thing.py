import os
import argparse
from pathlib import Path
import torch
import clip
from PIL import Image
import chromadb
import numpy as np
import fitz  # PyMuPDF
import io
import mimetypes
from pdf2image import convert_from_path
import hashlib
from datetime import datetime
import docx  # python-docx
import csv
import json
import xml.etree.ElementTree as ET
import zipfile

# Initialize CLIP
print("Loading CLIP model...")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# Initialize ChromaDB
print("Initializing ChromaDB...")
client = chromadb.PersistentClient(path="universal_vectors")
collection = client.get_or_create_collection(
    name="files",
    metadata={"hnsw:space": "cosine"}
)

# Initialize mimetypes
mimetypes.init()

def get_file_hash(file_path):
    """Generate a hash of the file's content and modification time."""
    mtime = os.path.getmtime(file_path)
    with open(file_path, 'rb') as f:
        content = f.read()
    return hashlib.md5(f"{content}{mtime}".encode()).hexdigest()

def get_mime_type(file_path):
    """Get mime type based on file extension."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or 'application/octet-stream'

def extract_text_from_docx(file_path):
    """Extract text from .docx file."""
    doc = docx.Document(file_path)
    return ' '.join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_txt(file_path):
    """Extract text from .txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except:
            return ""

def extract_text_from_pdf(file_path):
    """Extract text from PDF using PyMuPDF."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except:
        return ""

def extract_text_from_csv(file_path):
    """Extract text from CSV file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            return ' '.join([' '.join(row) for row in reader])
    except:
        return ""

def extract_text_from_json(file_path):
    """Extract text from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return json.dumps(data, ensure_ascii=False)
    except:
        return ""

def extract_text_from_xml(file_path):
    """Extract text from XML file."""
    try:
        tree = ET.parse(file_path)
        return ET.tostring(tree.getroot(), encoding='unicode', method='text')
    except:
        return ""

def extract_text(file_path, mime_type):

    """Extract text based on file type."""
    ext = os.path.splitext(file_path)[1].lower()
    
    # Handle different file types
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        try:
            return extract_text_from_docx(file_path)
        except:
            return ""
    elif ext == '.txt':
        return extract_text_from_txt(file_path)
    elif ext == '.csv':
        return extract_text_from_csv(file_path)
    elif ext == '.json':
        return extract_text_from_json(file_path)
    elif ext == '.xml':
        return extract_text_from_xml(file_path)
    elif ext in ['.py', '.js', '.html', '.css', '.md', '.rst', '.yml', '.yaml', '.ini', '.conf']:

        return extract_text_from_txt(file_path)
    elif mime_type and mime_type.startswith('text/'):
        return extract_text_from_txt(file_path)
    
    return ""

def generate_thumbnail(file_path, mime_type):
    """Generate a thumbnail for various file types."""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        
        # Handle images
        if mime_type and mime_type.startswith('image/'):
            return Image.open(file_path)
        
        # Handle PDFs
        elif ext == '.pdf':
            images = convert_from_path(file_path, first_page=1, last_page=1)
            return images[0] if images else None
        
        # For text-based files, create a simple placeholder image
        elif mime_type and (mime_type.startswith('text/') or 'document' in mime_type):
            img = Image.new('RGB', (400, 300), color='white')
            return img
            
        return None
    except Exception as e:
        print(f"Thumbnail generation failed for {file_path}: {str(e)}")
        return None

def generate_embedding(file_path):
    """Generate embeddings for any file type."""
    try:
        # Get mime type
        mime_type = get_mime_type(file_path)
        
        # Extract text content
        text_content = extract_text(file_path, mime_type)
        
        # Generate thumbnail
        thumbnail = generate_thumbnail(file_path, mime_type)
        
        embeddings = []
        
        # Generate text embedding if text content exists
        if text_content:
            text_tokens = clip.tokenize([text_content[:77]]).to(DEVICE)
            with torch.no_grad():
                text_embedding = model.encode_text(text_tokens)
                text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
                embeddings.append(text_embedding)
        
        # Generate image embedding if thumbnail exists
        if thumbnail:
            image_tensor = preprocess(thumbnail).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                image_embedding = model.encode_image(image_tensor)
                image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
                embeddings.append(image_embedding)
        
        # If we have no embeddings, create a basic embedding from the file path
        if not embeddings:
            path_tokens = clip.tokenize([os.path.basename(file_path)]).to(DEVICE)
            with torch.no_grad():
                path_embedding = model.encode_text(path_tokens)
                path_embedding /= path_embedding.norm(dim=-1, keepdim=True)
                embeddings.append(path_embedding)
        
        # Combine all embeddings (average)
        final_embedding = torch.mean(torch.stack(embeddings), dim=0)
        
        # Get file stats
        stats = os.stat(file_path)
        
        return {
            'embedding': final_embedding.cpu().numpy().flatten(),
            'metadata': {
                'path': file_path,
                'type': mime_type or 'application/octet-stream',
                'size': stats.st_size,
                'modified': datetime.fromtimestamp(stats.st_mtime).isoformat(),
                'text_preview': text_content[:1000] if text_content else "",
                'hash': get_file_hash(file_path)
            }
        }
        
    except Exception as e:
        print(f"Skipping {file_path}: {str(e)}")
        return None

def index_files(directory):
    """Index all files in directory recursively."""
    print(f"Scanning directory: {directory}")
    
    # Get existing files in index
    existing_files = {meta['path']: meta['hash'] 
                     for meta in collection.get()['metadatas']} if collection.count() > 0 else {}
    
    embeddings, ids, metadatas = [], [], []
    total_files = 0
    processed_files = 0
    
    # Walk through directory
    for root, _, files in os.walk(directory):
        for file in files:
            total_files += 1
            file_path = os.path.join(root, file)
            print(file_path)
            try:
                # Skip hidden files and directories
                if any(part.startswith('.') for part in file_path.split(os.sep)):
                    continue
                
                # Check if file has changed
                current_hash = get_file_hash(file_path)
                if file_path in existing_files and existing_files[file_path] == current_hash:
                    continue
                
                # Process file
                result = generate_embedding(file_path)


                if result:
                    embeddings.append(result['embedding'].tolist())
                    ids.append(file_path)


                    metadatas.append(result['metadata'])
                    processed_files += 1
                
                # Batch update every 100 files
                if len(embeddings) >= 100:
                    collection.upsert(
                        ids=ids,
                        embeddings=embeddings,
                        metadatas=metadatas
                    )
                    embeddings, ids, metadatas = [], [], []
                
                print(f"\rProcessed {processed_files}/{total_files} files", end='')
                
            except Exception as e:
                print(f"\nError processing {file_path}: {str(e)}")
    
    # Final batch update
    if embeddings:

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    print(f"\nIndexed {processed_files} new/modified files out of {total_files} total files.")

def search_files(query, n_results=5):
    """Search files using text query."""
    text = clip.tokenize([query]).to(DEVICE)
    with torch.no_grad():
        query_embedding = model.encode_text(text)
        query_embedding /= query_embedding.norm(dim=-1, keepdim=True)
    
    results = collection.query(
        query_embeddings=[query_embedding.cpu().numpy().flatten().tolist()],
        n_results=n_results,
        include=["metadatas", "distances"]
    )
    return results

def format_size(size):
    """Convert size in bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"

def main():
    parser = argparse.ArgumentParser(description="Universal file search with CLIP")
    parser.add_argument("--index", type=str, help="Directory to index")
    parser.add_argument("--n_results", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    if args.index:
        index_files(args.index)

    while True:
        query = input("\nEnter search query (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break
        if query:
            results = search_files(query, args.n_results)
            for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
                print("\n" + "=" * 80)
                print(f"File: {metadata['path']}")
                print(f"Type: {metadata['type']}")
                print(f"Size: {format_size(metadata['size'])}")
                print(f"Modified: {metadata['modified']}")
                print(f"Relevance Score: {1 - distance:.3f}")
                if metadata.get('text_preview'):
                    print(f"\nPreview: {metadata['text_preview'][:200]}...")
                print("=" * 80)
        else:
            print("Please enter a valid query.")

if __name__ == "__main__":
    main()