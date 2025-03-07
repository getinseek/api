import mimetypes
import os
import argparse
from pathlib import Path
import re

import torch
import clip
from PIL import Image
import chromadb
import numpy as np
import magic
import PyPDF2

# Constants
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Initialize CLIP model
print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# Initialize ChromaDB with cosine similarity
print("Initializing ChromaDB...")
client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection(
    name="photos",  # Keep your existing collection name
    metadata={"hnsw:space": "cosine"}
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
    'Pictures Library.photoslibrary'
}

def extract_text_from_pdf(file_path):
    """Extract text content from a PDF file."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
                
            return text
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {str(e)}")
        return None

def create_text_summaries(text, max_length=70):
    """Create short summaries that fit within CLIP's token limit.
    
    CLIP has a context length of 77 tokens, but we'll aim for ~70 to be safe.
    Each summary will roughly correspond to a paragraph or section of text.
    """
    if not text:
        return []
    
    # Split text into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    summaries = []
    current_chunk = []
    
    for para in paragraphs:
        # If paragraph is already short enough, use it directly
        if len(para.split()) <= max_length:
            summaries.append(para)
            continue
            
        # For longer paragraphs, break into sentences
        sentences = re.split(r'(?<=[.!?])\s+', para)
        
        for sentence in sentences:
            # If this sentence would make current_chunk too long, finalize chunk
            if current_chunk and len(' '.join(current_chunk + [sentence]).split()) > max_length:
                summaries.append(' '.join(current_chunk))
                current_chunk = [sentence]
            else:
                current_chunk.append(sentence)
    
    # Add any remaining text in the current chunk
    if current_chunk:
        summaries.append(' '.join(current_chunk))
    
    # For very short summaries, combine them to reduce fragmentation
    i = 0
    while i < len(summaries) - 1:
        combined = summaries[i] + " " + summaries[i+1]
        if len(combined.split()) <= max_length:
            summaries[i] = combined
            del summaries[i+1]
        else:
            i += 1
            
    return summaries

def encode_text_with_clip(text):
    """Generate a normalized text embedding using CLIP."""
    text = clip.tokenize([text]).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_text(text)
        embedding /= embedding.norm(dim=-1, keepdim=True)  # Normalize
    return embedding.cpu().numpy().flatten()

def process_pdf(file_path):
    """Process a PDF file by extracting text and creating embeddings for chunks."""
    print(f"Processing PDF: {file_path}")
    
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(file_path)
    if not pdf_text or len(pdf_text.strip()) < 50:
        print(f"  No meaningful text extracted from {file_path}")
        return False
        
    # Create summaries that fit within CLIP's context length
    summaries = create_text_summaries(pdf_text)
    print(f"  Created {len(summaries)} summaries from PDF")
    
    # Create embeddings for each summary
    embeddings, ids, metadatas, documents = [], [], [], []
    
    for i, summary in enumerate(summaries):
        chunk_id = f"{file_path}#chunk{i}"
        
        try:
            # Generate embedding for this summary
            embedding = encode_text_with_clip(summary)
            
            # Store embedding and metadata
            embeddings.append(embedding.tolist())
            ids.append(chunk_id)
            metadatas.append({
                "path": file_path,
                "chunk_id": i,
                "total_chunks": len(summaries),
                "content_type": "pdf",
                "summary": summary
            })
            documents.append(summary)
        except Exception as e:
            print(f"  Error processing summary {i}: {str(e)}")
    
    # Upsert all chunks at once
    if embeddings:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f"  Successfully indexed {len(embeddings)} summaries from {file_path}")
        return True
    
    return False

def handle_applications(file_path):
    """Handle different application file types."""
    try:
        file_type = magic.from_file(file_path, mime=True)
        
        if file_type.startswith('application/pdf'):
            return process_pdf(file_path)
        elif file_type.startswith('application/msword') or file_type.startswith('application/vnd.openxmlformats-officedocument.wordprocessingml'):
            print(f"Found Word document: {file_path} (processing not implemented yet)")
            return False
        elif file_type.startswith('application/vnd.ms-excel') or file_type.startswith('application/vnd.openxmlformats-officedocument.spreadsheetml'):
            print(f"Found Excel document: {file_path} (processing not implemented yet)")
            return False
        return False
    except Exception as e:
        print(f"Error in handle_applications for {file_path}: {str(e)}")
        return False

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

def create_and_upsert_embedding(path, content, type="image"):
    embeddings, ids, metadatas = [], [], []
    
    print(f"Indexing {path}...")
    embedding = None
    if type == "image":
        embedding = generate_embedding(path)
    else:
        embedding = encode_text_with_clip(content)

    if embedding is not None:
        embeddings.append(embedding.tolist())
        ids.append(path)
        metadatas.append({"path": path, "content_type": type})
        
        # For text content, also store the original text
        if type != "image":
            collection.upsert(
                ids=ids, 
                embeddings=embeddings, 
                metadatas=metadatas,
                documents=[content]
            )
        else:
            collection.upsert(
                ids=ids, 
                embeddings=embeddings, 
                metadatas=metadatas
            )

def find_files(directory):
    """Recursively find files in a directory, skipping system folders."""
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and system folders
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in SKIP_DIRS]
        
        for file in files:
            # Skip hidden files
            if not file.startswith('.'):
                file_path = os.path.join(root, file)
                try:
                    fileType = magic.Magic(mime=True)
                    mime = fileType.from_file(file_path)
                    
                    if mime.startswith("text/"):
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        # For text files, break into smaller chunks if needed
                        if len(content) > 300:  # If text is long
                            summaries = create_text_summaries(content)
                            for i, summary in enumerate(summaries):
                                chunk_id = f"{file_path}#chunk{i}"
                                embedding = encode_text_with_clip(summary)
                                if embedding is not None:
                                    collection.upsert(
                                        ids=[chunk_id],
                                        embeddings=[embedding.tolist()],
                                        metadatas=[{
                                            "path": file_path,
                                            "chunk_id": i,
                                            "total_chunks": len(summaries),
                                            "content_type": "text",
                                            "summary": summary
                                        }],
                                        documents=[summary]
                                    )
                            print(f"Indexed {len(summaries)} chunks from text file: {file_path}")
                        else:
                            # For short text, use as is
                            create_and_upsert_embedding(file_path, content, "text")
                    elif mime.startswith("image/"):
                        print(f"Found image: {file_path}")
                        create_and_upsert_embedding(file_path, None, "image")
                    elif mime.startswith("application/"):
                        handle_applications(file_path)
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")

def smart_search(query, n_results=5):
    # encode the query as usual
    query_embedding = encode_text_with_clip(query)
    # retrieve more candidates to have room for filtering/re-ranking
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=20,
        include=["metadatas", "distances", "documents"]
    )

    adjusted_results = []
    for metadata, distance, document in zip(
        results["metadatas"][0],
        results["distances"][0],
        results["documents"][0] if "documents" in results else [None]*len(results["metadatas"][0])
    ):
        
        # apply a penalty to text content to lower its ranking

        if metadata.get("content_type") != "image":
            adjusted_distance = distance + 0.5  # adjust this factor based on your data
        else:
            adjusted_distance = distance
        adjusted_results.append((adjusted_distance, metadata, document))
    
    # sort results by the adjusted distance (lower is better)
    adjusted_results.sort(key=lambda x: x[0])

    final_results = adjusted_results[:n_results]
    
    # build output structure similar to your existing code
    output = {"metadatas": [[]], "distances": [[]], "documents": [[]]}
    for adj, metadata, document in final_results:
        output["metadatas"][0].append(metadata)
        output["distances"][0].append(adj)
        output["documents"][0].append(document)
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Photo search with CLIP")
    parser.add_argument("--index", type=str, help="Directory to index")
    parser.add_argument("--n_results", type=int, default=20, help="Number of results")
    args = parser.parse_args()

    if args.index:
        find_files(args.index)

    while True:
        query = input("Enter search query (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        if query:
            results = smart_search(query, args.n_results)
            
            if not results["metadatas"][0]:
                print("No matching results found.")
                continue
                
            print("\nSearch Results:")
            print("---------------")
            
            for i, (metadata, distance, document) in enumerate(zip(
                results["metadatas"][0], 
                results["distances"][0],
                results.get("documents", [[None] * len(results["metadatas"][0])])[0]
            )):
                print(f"\n{i+1}. {os.path.basename(metadata['path'])} (Similarity: {(1-distance):.2f})")
                print(f"   Path: {metadata['path']}")
                
                if "chunk_id" in metadata:
                    chunk_id = metadata.get("chunk_id")
                    total_chunks = metadata.get("total_chunks")
                    print(f"   Section: {chunk_id+1} of {total_chunks}")
                    
                    # For PDF chunks, show which part of the document matched
                    if metadata.get("content_type") == "pdf":
                        print(f"   Content: {metadata.get('summary', '')}")
                
                if document and not "summary" in metadata:
                    print(f"   Preview: {document[:150]}..." if len(document) > 150 else f"   Preview: {document}")
        else:
            print("Please enter a valid query.")

if __name__ == "__main__":
    main()