import os
from chromadb.config import Settings
from chromadb import PersistentClient
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import moondream as md


import torch
from api.download_moondream import download_model



download_model()

local_model_path=os.path.join(os.getcwd(),"moondream_model","moondream-0_5b-int8.mf")
model = md.vl(model=local_model_path)


# Load BLIP model and processor
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
# model = AutoModelForCausalLM.from_pretrained(
#     local_model_path,
#     trust_remote_code=True,
#     # Uncomment to run on GPU.
#     # device_map={"": device}
# ).to(device)

# tokenizer = AutoTokenizer.from_pretrained(local_model_path)

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
        image = Image.open(file_path)
        # enc_image = model.encode_image(image)
        caption = model.caption(image)["caption"]



        return {
            "text": caption,
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
                documents=[str(metadata)],
                metadatas=[metadata],
                ids=[file_path],
            )
            print(f"Indexed: {file_path}")


if __name__ == "__main__":
    # Prompt user for directory to index
    print(device)
    directory = input("Enter the path to the directory you want to index: ")
    index_files(directory)
