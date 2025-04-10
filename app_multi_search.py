"""
app_multi_search.py

Module for multi-document search capabilities in the Aspire Academy Document Assistant.
Provides functions to search across multiple documents with a sensible limit.
"""

import logging
from typing import List
from app_rag import DocumentCollection, hybrid_retriever

# Configure logging
logger = logging.getLogger(__name__)

# Maximum number of documents to search simultaneously
MAX_DOCUMENTS = 10

def multi_document_search(query: str, vector_store, doc_names: List[str], limit_per_doc: int = 20):
    """
    Search across multiple documents and combine the results.
    
    Args:
        query (str): The search query
        vector_store: The vector store to search in
        doc_names (List[str]): List of document names to search
        limit_per_doc (int): Maximum number of results per document
        
    Returns:
        DocumentCollection: Combined search results from all documents
    """
    if not query or not query.strip():
        logger.warning("Empty query provided to multi_document_search")
        return DocumentCollection()
        
    if not doc_names:
        logger.warning("No documents provided to multi_document_search")
        return DocumentCollection()
        
    # Enforce maximum document limit
    if len(doc_names) > MAX_DOCUMENTS:
        logger.warning(f"Too many documents selected ({len(doc_names)}). Limiting to {MAX_DOCUMENTS}.")
        doc_names = doc_names[:MAX_DOCUMENTS]
    
    # Initialize combined results collection
    combined_results = DocumentCollection()
    
    # Track which documents were searched and had results
    docs_with_results = 0
    docs_searched = 0
    combined_results.set_metric("documents_total", len(doc_names))
    
    # Storage for seen content to prevent duplicates across documents
    seen_contents = set()
    
    # Search each document and combine results
    for doc_name in doc_names:
        docs_searched += 1
        logger.info(f"Searching document {docs_searched}/{len(doc_names)}: {doc_name}")
        
        try:
            # Get results for this document
            doc_results = hybrid_retriever(query, vector_store, doc_name, limit=limit_per_doc)
            
            if doc_results and len(doc_results) > 0:
                docs_with_results += 1
                logger.info(f"Found {len(doc_results)} chunks in '{doc_name}'")
                
                # Add unique results to the combined collection
                for doc in doc_results:
                    if doc.page_content and doc.page_content not in seen_contents:
                        combined_results.append(doc)
                        seen_contents.add(doc.page_content)
                        
                        # Track document source in metadata
                        if not hasattr(doc.metadata, "from_document"):
                            doc.metadata["from_document"] = doc_name
                
                # Accumulate metrics
                combined_results.sql_count += getattr(doc_results, 'sql_count', 0)
                combined_results.vector_count += getattr(doc_results, 'vector_count', 0)
                combined_results.table_count += getattr(doc_results, 'table_count', 0)
                combined_results.fallback_count += getattr(doc_results, 'fallback_count', 0)
            else:
                logger.info(f"No results found in document '{doc_name}'")
                
        except Exception as e:
            logger.error(f"Error searching document '{doc_name}': {str(e)}")
    
    # Add additional metrics
    combined_results.set_metric("documents_with_results", docs_with_results)
    combined_results.set_metric("documents_searched", docs_searched)
    
    logger.info(f"Multi-document search complete: {len(combined_results)} total chunks from {docs_with_results}/{docs_searched} documents")
    
    # Return results (empty collection if nothing found)
    return combined_results 