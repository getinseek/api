from typing import List, Dict, Optional
from api.indexer import collection
import logging
import ast
import re
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Data class to store search result information"""
    file_path: str
    metadata: Dict
    relevance_score: float

def search_files(
    query: str,
    max_results: int = 5,
    file_types: Optional[List[str]] = None,
) -> List[SearchResult]:
    """
    Searches indexed files based on a query string, taking into account the ChromaDB storage structure
    where documents are string representations of metadata dictionaries.

    Args:
        query (str): The search query string
        max_results (int): Maximum number of results to return (default: 5)
        file_types (Optional[List[str]]): List of file extensions to filter by (e.g., ['.jpg', '.png'])

    Returns:
        List[SearchResult]: List of SearchResult objects containing matched files and metadata

    Raises:
        ValueError: If query is empty or invalid parameters are provided
    """
    # Input validation
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")
    if max_results < 1:
        raise ValueError("max_results must be positive")

    normalized_query = query.lower().strip()
    results = []

    try:
        # Get all items from collection
        all_items = collection.get()
        
        # Process each item
        for idx, (doc, metadata, item_id) in enumerate(zip(
            all_items["documents"],
            all_items["metadatas"],
            all_items["ids"]
        )):
            # Skip if file type filter is active and file doesn't match
            if file_types and metadata["file_type"].lower() not in file_types:
                continue

            # Calculate relevance score based on both metadata and image description
            relevance_score = calculate_relevance(normalized_query, metadata)
            
            if relevance_score > 0:
                results.append(SearchResult(
                    file_path=item_id,
                    metadata=metadata,
                    relevance_score=relevance_score
                ))

        # Sort by relevance score and limit results
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:max_results]

    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        raise

def calculate_relevance(query: str, metadata: Dict) -> float:
    """
    Calculates relevance score based on multiple factors.
    
    Args:
        query (str): Normalized search query
        metadata (Dict): File metadata including image description
        
    Returns:
        float: Relevance score between 0 and 1
    """
    score = 0.0
    
    # Check image description/caption
    if "text" in metadata and isinstance(metadata["text"], str):
        text = metadata["text"].lower()
        # Exact match in description
        if query in text:
            score += 0.6
        # Partial word matches
        query_words = set(query.split())
        text_words = set(text.split())
        common_words = query_words.intersection(text_words)
        score += 0.2 * (len(common_words) / len(query_words))

    # Check filename
    if "file_name" in metadata:
        filename = metadata["file_name"].lower()
        if query in filename:
            score += 0.3
        # Partial filename match
        if any(word in filename for word in query.split()):
            score += 0.1

    # Normalize final score to be between 0 and 1
    return min(1.0, score)

def format_search_result(result: SearchResult) -> Dict:
    """
    Formats a search result for display.
    
    Args:
        result (SearchResult): Search result to format
        
    Returns:
        Dict: Formatted result with relevant information
    """
    return {
        "file_path": result.file_path,
        "file_name": result.metadata["file_name"],
        "file_type": result.metadata["file_type"],
        "description": result.metadata.get("text", "No description available"),
        "dimensions": f"{result.metadata.get('width', 'N/A')}x{result.metadata.get('height', 'N/A')}"
        if "width" in result.metadata and "height" in result.metadata
        else "N/A",
        "relevance_score": f"{result.relevance_score:.2f}"
    }