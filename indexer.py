import os
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from PIL import Image, ExifTags
import pdfplumber
from chromadb.config import Settings
from chromadb import PersistentClient

import torch
import clip.clip as clip

# Set up the device and load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Set up ChromaDB client
client = PersistentClient(
    path="./db",  # Directory to persist embeddings
    settings=Settings(
        allow_reset=True,  # Optional, resets the DB if needed
    ),
)

# Use SentenceTransformer for local embeddings
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create or get the collection
collection = client.get_or_create_collection(
    name="files", embedding_function=embedding_function
)


def index_files(directory):
    """Indexes all files in the specified directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            # Skip already indexed files
            existing_ids = collection.get(ids=[file_path])
            if existing_ids and existing_ids["ids"]:
                print(f"Already indexed: {file_path}")
                continue

            metadata = extract_metadata(file_path)
            if not metadata.get('text') or not isinstance(metadata['text'], str):
                print(f"Skipping: {file_path} (Invalid text metadata)")
                continue

            print(
                f"Storing metadata for: {file_path} with text: {metadata['text']}")

            embedding = embedding_function(metadata['text'])
            collection.add(
                documents=[metadata['text']],
                metadatas=[metadata],
                ids=[file_path]
            )
            print(f"Indexed: {file_path}")


def extract_metadata(file_path):
    """Extracts metadata and processes files based on type."""
    metadata = {
        "file_name": os.path.basename(file_path),
        "file_path": file_path,
        "file_type": os.path.splitext(file_path)[1],
    }

    # Process specific file types
    if metadata["file_type"].lower() in [".jpg", ".png", ".jpeg"]:
        data = process_image(file_path)
        metadata.update(data if data else {"text": "Image processing failed"})
    elif metadata["file_type"].lower() == ".pdf":
        data = process_pdf(file_path)
        metadata.update(data if data.get("text") else {"text": "Empty PDF"})
    else:
        metadata["text"] = f"Unsupported file type: {metadata['file_type']}"

    # Ensure metadata values are simple types
    metadata = {k: v for k, v in metadata.items(
    ) if isinstance(v, (str, int, float, bool))}
    return metadata


def process_image(file_path):
    """Processes image files to extract features and metadata using CLIP."""
    try:
        with Image.open(file_path) as img:
            # Extract EXIF metadata
            exif_data = img._getexif()
            exif_metadata = {}
            if exif_data:
                for tag, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag, tag)
                    exif_metadata[tag_name] = value

            # Preprocess the image for CLIP
            image = preprocess(img).unsqueeze(0).to(device)

            # Generate descriptive keywords using CLIP
            with torch.no_grad():
                # Use CLIP's text-to-image matching with potential descriptive labels
                text_inputs = clip.tokenize(["A photo of a landmark", "A travel destination",
                                             "A famous place", "Sunset", "Cloudy skies"]).to(device)
                image_features = model.encode_image(image)
                text_features = model.encode_text(text_inputs)
                similarity = torch.matmul(image_features, text_features.T)

                # Get the most relevant description
                top_match_index = similarity.argmax().item()
                description = ["A photo of a landmark", "A travel destination",
                               "A famous place", "Sunset", "Cloudy skies"][top_match_index]

            return {
                "text": description,  # CLIP-generated description
                "width": img.size[0],
                "height": img.size[1],
                "mode": img.mode,
                "exif": exif_metadata
            }
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return {"text": "Image processing failed"}


def process_pdf(file_path):
    """Processes PDF files to extract text."""
    try:
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join([page.extract_text()
                             for page in pdf.pages if page.extract_text()])
        return {"text": text}
    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        return {"text": "PDF processing failed"}
