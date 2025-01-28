import mimetypes
import os
import pathlib
import PyPDF2
import chardet
from chromadb.config import Settings
from chromadb import PersistentClient
from PIL import Image
import docx
import filetype
from transformers import pipeline

from transformers import Blip2Processor, Blip2ForConditionalGeneration, BlipProcessor, BlipForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
import moondream as md
from transformers.image_utils import load_image


import ollama
import torch
from api.analyzer import ImageAnalyzer
from api.download_moondream import download_model



# download_model()

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# local_model_path=os.path.join(os.getcwd(),"moondream_model","moondream-0_5b-int8.mf")
# model = md.vl(model=local_model_path)


# # Load BLIP model and processor
# device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
# # DEVICE=device

# # Initialize processor and model
# processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
# model = AutoModelForVision2Seq.from_pretrained(
#     "HuggingFaceTB/SmolVLM-256M-Instruct",
#     torch_dtype=torch.bfloat16,
#     _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
# ).to(DEVICE)

# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-itm-vit-g")
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-itm-vit-g", 
#     torch_dtype=torch.float16
# )

# # Move to GPU if available
# device = "mps" if torch.mps.is_available() else "cpu"
# model.to(device)

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


        
       

        # captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
        # caption = captioner(image)

        # inputs = processor(image, "a photography of", return_tensors="pt")
        # outputs = model.generate(**inputs, max_new_tokens=100)
        # caption = processor.decode(outputs[0], skip_special_tokens=True)

        analyzer = ImageAnalyzer()
        caption = analyzer.get_summary(file_path)

        ## [{'generated_text': 'two birds are standing next to each other '}]


        # caption = caption[0]['generated_text']

        print(caption)

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
    """
    Extracts content from files, with special handling for images and text-based files.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        dict: Dictionary containing file content and metadata
    """
    file_path = pathlib.Path(file_path)
    
    mime_type = mimetypes.guess_type(str(file_path))[0]

    # Base metadata
    result = {
        "file_name": file_path.name,
        "file_path": str(file_path.absolute()),
        "file_type": file_path.suffix.lower(),
        "mime_type": mime_type,
        "content_type": None,
        "content": None,
        "error": None
    }

    print(mime_type)

    try:
        # Image handling
        if result["mime_type"].startswith('image/'):
            try:
                with Image.open(file_path) as img:
                    # Convert image to RGB if necessary
                    if img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGB')
                    data = process_image(file_path)
                    result["content"] = data if data else {"text": "Image processing failed"}

                    result.update({
                        "content_type": "image",
                        "content": {
                            "format": img.format,
                            "mode": img.mode,
                            "width": img.width,
                            "height": img.height,
                            "text": data["text"]
                        }
                    })
            except Exception as e:
                result["error"] = f"Image processing error: {str(e)}"

        # Text content handling
        else:
            # PDF files
            if result["file_type"] == ".pdf":
                text_content = []
                with open(file_path, "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        text_content.append(page.extract_text())
                result["content_type"] = "text"
                result["content"] = "\n".join(text_content)

            # Word documents
            elif result["file_type"] in [".docx", ".doc"]:
                doc = docx.Document(file_path)
                result["content_type"] = "text"
                result["content"] = "\n".join(paragraph.text for paragraph in doc.paragraphs)

            # Plain text files (including .txt, .csv, .json, .xml, etc.)
            else:
                # Detect file encoding
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    detected = chardet.detect(raw_data)
                    encoding = detected['encoding'] or 'utf-8'

                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        result["content_type"] = "text"
                        result["content"] = content
                except UnicodeDecodeError:
                    # If text reading fails, check if it might be binary data
                    result["error"] = "File appears to be binary or uses an unsupported encoding"

    except Exception as e:
        result["error"] = str(e)

    return result


def index_files(directory):
    # Process all files in directory
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            try:
                # Check if file is already indexed
                if True:
                    existing_ids = collection.get(ids=[file_path])
                    if existing_ids and existing_ids["ids"]:
                        print(f"Already indexed: {file_path}")

                        continue

                # Extract content and metadata
                metadata = extract_metadata(file_path)
                print(metadata)

                # Verify valid text content
                if not metadata['content']['text'] or not isinstance(metadata["content"]["text"], str):
                    print(f"Skipping: {file_path} (Invalid text metadata)")

                    continue

                # Debug output
                print(f"Indexing file: {file_path}")
                print(f"Metadata: {type(metadata)}")

                # Add to collection
                collection.add(
                    documents=[str(metadata)],
                    metadatas=[{"info":str(metadata)}],
                    ids=[file_path],
                )
                print(f"Indexed: {file_path}")


            except Exception as e:
                print(f"Failed to process {file_path}: {str(e)}")



if __name__ == "__main__":
    # Prompt user for directory to index
    # print(device)
    directory = input("Enter the path to the directory you want to index: ")
    index_files(directory)
