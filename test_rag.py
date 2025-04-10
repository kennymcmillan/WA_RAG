"""
test_rag.py

Test script for our enhanced RAG implementation.
This script tests the parent-child document retriever and hybrid search.
"""

import os
import logging
from dotenv import load_dotenv
from app_rag import simple_sql_search, hybrid_retriever, fetch_parent_context
from app_vector import CustomPGVector
from app_embeddings import embeddings
from app_database import get_connection_string

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

def test_rag():
    """
    Test our RAG implementation with different queries and documents
    """
    print("=== Testing Enhanced RAG Implementation ===")
    
    # Get connection string for vector store
    CONNECTION_STRING = get_connection_string()
    if not CONNECTION_STRING:
        print("Error: Could not get database connection string")
        return False
    
    # Initialize vector store
    vector_store = CustomPGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name="documents",
        use_jsonb=True
    )
    
    # Test documents and queries
    test_cases = [
        {
            "doc_name": "Sanguine Sports Tech Fund.pdf",
            "query": "Who are the founders?",
            "description": "Searching for founder information in Sanguine Sports Tech Fund"
        },
        {
            "doc_name": "Sanguine Sports Tech Fund.pdf", 
            "query": "What is sports technology?",
            "description": "Searching for general information about sports technology"
        },
        {
            "doc_name": "Drake Star Global Sports Tech Report_2023.pdf",
            "query": "investment trends",
            "description": "Searching for investment trends in sports tech"
        }
    ]
    
    for test_case in test_cases:
        doc_name = test_case["doc_name"]
        query = test_case["query"]
        description = test_case["description"]
        
        print(f"\n\n=== Test: {description} ===")
        print(f"Document: {doc_name}")
        print(f"Query: '{query}'")
        
        print("\n--- Testing Simple SQL Search ---")
        try:
            results = simple_sql_search(query, doc_name, limit=5)
            print(f"SQL Search found {len(results)} results")
            for i, doc in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")
        except Exception as e:
            print(f"SQL Search error: {str(e)}")
        
        print("\n--- Testing Hybrid Retriever ---")
        try:
            results = hybrid_retriever(query, vector_store, doc_name, limit=5)
            print(f"Hybrid Retriever found {len(results)} results")
            for i, doc in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")
                
            # Test parent context fetching
            if results:
                print("\n--- Testing Parent Context Fetching ---")
                enhanced_results = fetch_parent_context(results, doc_name)
                print(f"After fetching parents: {len(enhanced_results)} results (added {len(enhanced_results) - len(results)} parents)")
        except Exception as e:
            print(f"Hybrid Retriever error: {str(e)}")
            
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_rag() 