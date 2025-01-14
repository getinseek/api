from indexer import collection


def search_files(query):
    """Queries ChromaDB and returns matching file paths."""
    # Query the collection with the user's input
    results = collection.query(
        query_texts=[query],
        n_results=5  # Number of results to return
    )

    # Extract file paths from metadata
    return [
        match["metadata"]["file_path"]
        for match in results.get("matches", [])
    ]
