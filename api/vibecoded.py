import os
import numpy as np
import logging
import uuid
from embed_anything import EmbedData, EmbeddingModel
import embed_anything
import chromadb
from chromadb.config import Settings
from PIL import Image
import fitz  # PyMuPDF
import soundfile as sf
import cv2
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    'text': ['.txt', '.md', '.csv', '.json', '.html', '.xml', '.py', '.js', '.css'],
    'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'],
    'pdf': ['.pdf'],
    'video': ['.mp4', '.avi', '.mov', '.mkv', '.webm']
}

# Optional audio support
AUDIO_SUPPORT = False
try:
    # Try to import the specific audio libraries
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    AUDIO_SUPPORT = True
    SUPPORTED_EXTENSIONS['audio'] = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    logger.info("Audio support enabled")
except ImportError:
    logger.warning("Audio libraries not available. Audio files will be skipped.")

class MultimodalIndexer:
    def __init__(self, db_path, collection_name, target_dim=512):
        self.target_dim = target_dim
        self.collection_name = collection_name
        embedding_function = OpenCLIPEmbeddingFunction()
        data_loader = ImageLoader()


        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name, embedding_function=embedding_function, data_loader=data_loader)
            logger.info(f"Using existing collection: {collection_name}")
        except:
            logger.info(f"Creating new collection: {collection_name}")
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
                embedding_function=embedding_function,
                data_loader=data_loader
            )
        
        # Load embed_anything model
        logger.info("Loading embedding models...")
        self.model = EmbeddingModel.from_pretrained_hf(
            embed_anything.WhichModel.Clip,
            model_id="openai/clip-vit-base-patch16"
        )
    
    def get_file_type(self, file_path):
        """Determine file type based on extension"""
        ext = os.path.splitext(file_path)[1].lower()
        
        for file_type, extensions in SUPPORTED_EXTENSIONS.items():
            if ext in extensions:
                return file_type
                
        return None
    
    def extract_text_embedding(self, file_path):
        """Extract embedding from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            
            data = embed_anything.embed_query([text], embedder=self.model)
            embedding = np.array([d.embedding for d in data])
            return embedding
        except Exception as e:
            logger.error(f"Error extracting text embedding from {file_path}: {e}")
            return None
    
    def extract_image_embedding(self, file_path):
        """Extract embedding from image file"""
        try:
            image = Image.open(file_path)
            data = embed_anything.embed_anything([image], embedder=self.model)
            embedding = np.array([d.embedding for d in data])
            return embedding
        except Exception as e:
            logger.error(f"Error extracting image embedding from {file_path}: {e}")
            return None
    
    def extract_audio_embedding(self, file_path):
        """Extract embedding from audio file"""
        if not AUDIO_SUPPORT:
            logger.warning(f"Audio support not available. Skipping {file_path}")
            return None
            
        try:
            audio, rate = sf.read(file_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
                
            # Ensure float32 data type
            audio = audio.astype(np.float32)
            
            # Limit audio length to avoid memory issues
            max_length = 16000 * 30  # 30 seconds
            if len(audio) > max_length:
                audio = audio[:max_length]
                
            # Resample to 16kHz if needed
            if rate != 16000:
                indices = np.round(np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / rate))).astype(int)
                audio = audio[indices]
                
            data = embed_anything.embed_audio_file(audio, embedder=self.model)
            audio_embedding = np.array([d.embedding for d in data])
            return audio_embedding
            
        except Exception as e:
            logger.error(f"Error extracting audio embedding from {file_path}: {e}")
            return None
    
    def extract_pdf_embedding(self, file_path):
        """Extract embedding from PDF file"""
        try:
            doc = fitz.open(file_path)
            
            # Extract text
            pdf_text = ""
            for page in doc:
                pdf_text += page.get_text()
            
            data = embed_anything.embed_query([pdf_text], embedder=self.model)
            embedding = np.array([d.embedding for d in data])
            return embedding
        except Exception as e:
            logger.error(f"Error extracting PDF embedding from {file_path}: {e}")
            return None
    
    def extract_video_embedding(self, file_path):
        """Extract embedding from video file"""
        pass
    
    def get_embedding(self, file_path):
        """Get embedding based on file type"""
        file_type = self.get_file_type(file_path)
        
        if file_type is None:
            logger.warning(f"Unsupported file type: {file_path}")
            return None
        
        logger.info(f"Processing {file_type} file: {file_path}")
        
        if file_type == 'text':
            return self.extract_text_embedding(file_path)
        elif file_type == 'image':
            return self.extract_image_embedding(file_path)
        # elif file_type == 'audio' and AUDIO_SUPPORT:
        #     return self.extract_audio_embedding(file_path)
        elif file_type == 'pdf':
            return self.extract_pdf_embedding(file_path)
        # elif file_type == 'video':
        #     return self.extract_video_embedding(file_path)
        else:
            return None
    
    def normalize_embedding(self, embedding):
        """Normalize embedding to unit length and resize to target dimension"""
        if embedding is None:
            return None
        
        # Normalize the embedding to unit length
        normalized = normalize(embedding)
        
        # Resize embedding to target dimension if needed
        embedding_dim = normalized.shape[1]
        if embedding_dim != self.target_dim:
            logger.info(f"Resizing embedding from dimension {embedding_dim} to {self.target_dim}")
            
            if embedding_dim > self.target_dim:
                resized = normalized[0, :self.target_dim]
            else:
                resized = np.zeros((1, self.target_dim))
                resized[0, :embedding_dim] = normalized[0, :]
                
            # Normalize again after resizing
            resized = normalize(resized)
            return resized.flatten().tolist()
            
        return normalized.flatten().tolist()
    
    def index_file(self, file_path):
        """Index a single file"""
        rel_path = os.path.basename(file_path)
        file_type = self.get_file_type(file_path)
        
        if file_type is None:
            return False
            
        # Get embedding
        embedding = self.get_embedding(file_path)
        if embedding is None:
            return False
            
        # Normalize and resize embedding
        embedding = self.normalize_embedding(embedding)
        
        # Create unique ID
        file_id = str(uuid.uuid4())
        
        # Add dummy document text (ChromaDB requires documents)
        document = f"File: {rel_path}, Type: {file_type}"
        
        # Upload to ChromaDB
        try:
            self.collection.add(
                ids=[file_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[{"file_name": rel_path, "file_type": file_type}],
            )
            logger.info(f"Indexed {file_type} file: {rel_path}")
            return True
        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            return False
    
    def index_directory(self, directory_path):
        """Index all files in a directory"""
        if not os.path.isdir(directory_path):
            logger.error(f"{directory_path} is not a valid directory.")
            return
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                if not self.index_file(file_path):
                    logger.warning(f"Failed to index {file_path}")
    
    def query(self, query_text, top_k=5):
        """Query the index and return top_k results"""
        try:
            # Embed the query
            query_embedding = self.get_embedding(query_text)
            if query_embedding is None:
                logger.warning("Query embedding failed.")
                return []
            
            query_embedding = self.normalize_embedding(query_embedding)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            return results
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            return []

# Helper function to normalize
def normalize(embeddings):
    """Normalize embeddings to unit length"""
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norm

# Example usage
if __name__ == "__main__":
    indexer = MultimodalIndexer(db_path='./db-test', collection_name='media_index')
    
    # Index all files in a directory
    directory_path = '/Users/shubhampatil/Documents/programming/inseek/api/test'  # Replace with your directory path
    indexer.index_directory(directory_path)
    
    # Query the index
    query_text = "hawaii"
    results = indexer.query(query_text)
    for result in results:
        print(result['document'])
