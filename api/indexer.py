import os
from chromadb.config import Settings
from chromadb import PersistentClient
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Set up ChromaDB client
client = PersistentClient(
    path="./db",  # Directory to persist embeddings
    settings=Settings(allow_reset=True)
)

# Create or get the collection
collection = client.get_or_create_collection(name="files")


def process_image(file_path):
    """Processes an image file and generates a detailed description using BLIP."""
    try:
        # Load and preprocess the image
        image = Image.open(file_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)

        # Generate caption
        with torch.no_grad():
            outputs = model.generate(**inputs)
            description = processor.decode(outputs[0], skip_special_tokens=True)

        return {
            "text": description,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
        }
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return {"text": "Image processing failed"}


def extract_metadata(file_path):
    """Extracts metadata and processes files based on type."""
    metadata = {
        "file_name": os.path.basename(file_path),
        "file_path": file_path,
        "file_type": os.path.splitext(file_path)[1],
    }

    # Process image files
    if metadata["file_type"].lower() in [".jpg", ".png", ".jpeg"]:
        data = process_image(file_path)
        metadata.update(data if data else {"text": "Image processing failed"})
    else:
        metadata["text"] = f"Unsupported file type: {metadata['file_type']}"

    return metadata


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

            # Extract metadata
            metadata = extract_metadata(file_path)
            if not metadata.get("text") or not isinstance(metadata["text"], str):
                print(f"Skipping: {file_path} (Invalid text metadata)")
                continue

            # Debug metadata
            print(f"Indexing file: {file_path}")
            print(f"Metadata: {metadata}")

            # Add to collection
            collection.add(
                documents=[metadata["text"]],
                metadatas=[metadata],
                ids=[file_path],
            )
            print(f"Indexed: {file_path}")


if __name__ == "__main__":
    # Prompt user for directory to index
    directory = input("Enter the path to the directory you want to index: ")
    index_files(directory)
