from chromadb import PersistentClient
from chromadb.config import Settings

# Set up ChromaDB client
client = PersistentClient(
    path="./db",  # Same directory used in your indexer
    settings=Settings(allow_reset=True)
)


def clear_index():
    """Clears all data in the collection."""
    try:
        # Delete the entire collection
        collection_name = "files"
        client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' has been cleared.")
    except Exception as e:
        print(f"Error clearing the collection: {e}")


if __name__ == "__main__":
    clear_index()
