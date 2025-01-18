from api.indexer import collection


def search_files(query):
    """Searches the indexed files based on a query."""
    # Normalize the query
    normalized_query = query.lower()

    # Perform a search within the collection
    results = collection.query(
        query_texts=[normalized_query],
        n_results=5,  # Return top 5 matches
    )

    # Extract file paths of matching results

    # matching_files = [
    #     match["metadata"]["file_path"]
    #     for match in results.get("matches", [])
    # ]

    matching_files=[]

    for match in results.get("metadatas", [])[0]:
        matching_files.append(match["file_path"])

    
    
    

    # If no results from ChromaDB, perform a fallback search
    if not matching_files:
        all_metadata = collection.get()["metadatas"]
        for metadata in all_metadata:
            if normalized_query in metadata["text"].lower():
                matching_files.append(metadata["file_path"])
    
    return matching_files

