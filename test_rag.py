"""
test_rag.py

Test script to verify the implementation of the Parent Document Retriever.
This script demonstrates the RAG pattern described in:
https://pub.towardsai.net/introduction-to-retrieval-augmented-generation-rag-using-langchain-and-lamaindex-bd0047628e2a
"""

import os
import logging
from dotenv import load_dotenv
from langchain_core.documents import Document
from app_rag import (
    implement_parent_document_retriever,
    simple_sql_search,
    hybrid_retriever,
    fetch_parent_context,
    rank_docs_by_relevance
)
from app_vector import CustomPGVector
from app_embeddings import embeddings
from app_database import get_connection_string

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

def test_parent_document_retriever():
    """
    Test the parent document retriever implementation
    """
    print("=== Testing Parent Document Retriever ===")
    
    # Create sample documents
    documents = [
        Document(
            page_content="This is a sample document about sports technology. Sports technology (sports tech) represents a rapidly evolving sector that integrates innovation and digital solutions into athletic training, performance analysis, and sports management.",
            metadata={"source": "test_doc.pdf", "page": 1}
        ),
        Document(
            page_content="Dr. Brian Moore is an expert in AI applications for sports analytics. He has pioneered several breakthrough technologies in motion capture and performance optimization.",
            metadata={"source": "test_doc.pdf", "page": 2}
        ),
        Document(
            page_content="Discus Cage: All discus throws shall be made from an enclosure or cage to ensure the safety of spectators, officials and athletes. The cage specified in this Rule is intended for use when the event takes place in the Field of Play with other events taking place at the same time.",
            metadata={"source": "test_doc.pdf", "page": 3}
        ),
    ]
    
    # Test implementing parent document retriever
    success = implement_parent_document_retriever(documents, "test_doc.pdf")
    print(f"Parent Document Retriever implementation: {'Success' if success else 'Failed'}")
    
    # If successful, test search
    if success:
        # Get connection string and vector store
        connection_string = get_connection_string()
        
        # Create vector store
        vector_store = CustomPGVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name="documents",
            use_jsonb=True
        )
        
        # Test hybrid retriever
        print("\n=== Testing Hybrid Retriever ===")
        test_queries = [
            "who is Brian Moore",
            "what is sports technology",
            "tell me about discus throw cage"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            # Get results from hybrid retriever
            results = hybrid_retriever(query, vector_store, "test_doc.pdf", limit=10)
            
            print(f"Found {len(results)} results")
            if results:
                print("\nTop 3 results:")
                for i, doc in enumerate(results[:3]):
                    print(f"\n{i+1}. {doc.page_content[:200]}...")
                    print(f"   Metadata: {doc.metadata}")
            else:
                print("No results found")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_parent_document_retriever() 