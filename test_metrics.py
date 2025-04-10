"""
test_metrics.py

Script to test the search metrics in our RAG implementation.
"""

import logging
from langchain_core.documents import Document
from app_rag import hybrid_retriever, fetch_parent_context
from app_vector import CustomPGVector
from app_embeddings import embeddings
from app_database import get_connection_string

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_search_metrics():
    """
    Test the metrics tracking in our hybrid retriever.
    """
    print("\n" + "="*50)
    print("Testing Search Metrics")
    print("="*50)
    
    # Create mock documents for testing
    docs = [
        Document(
            page_content="Document about sports technology",
            metadata={"source": "test_doc.pdf", "page": 1}
        ),
        Document(
            page_content="Information about Brian Moore",
            metadata={"source": "test_doc.pdf", "page": 2}
        ),
        Document(
            page_content="Details about discus cage",
            metadata={"source": "test_doc.pdf", "page": 3}
        ),
    ]
    
    # Simulate a search
    print("\nSimulating hybrid retrieval...")
    
    # Mock SQL search results
    sql_results = docs.copy()
    sql_results.sql_count = len(sql_results)
    sql_results.vector_count = 0
    sql_results.fallback_count = 0
    
    print(f"SQL search found: {sql_results.sql_count} chunks")
    print(f"Vector search found: {sql_results.vector_count} chunks")
    print(f"Emergency fallback used: {sql_results.fallback_count} chunks")
    
    # Simulate adding parent documents
    enhanced_results = sql_results.copy()
    parent_doc = Document(
        page_content="Parent document with broader context",
        metadata={"source": "test_doc.pdf", "page": 1, "is_parent": True}
    )
    enhanced_results.append(parent_doc)
    
    # Set parent count
    enhanced_results.parent_count = 1
    
    # Preserve metrics
    enhanced_results.sql_count = sql_results.sql_count
    enhanced_results.vector_count = sql_results.vector_count
    enhanced_results.fallback_count = sql_results.fallback_count
    
    print(f"Added {enhanced_results.parent_count} parent documents")
    print(f"Total chunks: {len(enhanced_results)}")
    
    # Print debug info as it would appear in the UI
    print("\nDebug Information (UI Example):")
    print("="*50)
    print("Query: 'test query'")
    print("Selected document: test_doc.pdf")
    print("\nSearch Results:")
    print(f"Hybrid search found: {len(enhanced_results)} chunks total")
    print(f"- SQL search found: {enhanced_results.sql_count} chunks")
    print(f"- Vector search found: {enhanced_results.vector_count} chunks")
    if enhanced_results.fallback_count > 0:
        print(f"- Emergency fallback used: {enhanced_results.fallback_count} chunks")
    print(f"- Added {enhanced_results.parent_count} parent documents for context")
    
    print("\n" + "="*50)
    print("Test Complete!")

if __name__ == "__main__":
    test_search_metrics() 