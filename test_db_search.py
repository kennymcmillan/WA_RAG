import os
import streamlit as st
from dotenv import load_dotenv
import logging
import json
from langchain_community.vectorstores.pgvector import PGVector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting test script...")

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# Import utility functions
from app_database import get_db_connection, get_connection_string
from app_embeddings import embeddings
from app_vector import iterative_document_search

def get_document_list():
    """Get list of documents from database"""
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Could not connect to database"
        
        cursor = conn.cursor()
        
        # Get distinct document names from metadata
        cursor.execute("""
            SELECT DISTINCT metadata->>'source' as doc_name, COUNT(*) as chunk_count
            FROM documents 
            WHERE metadata->>'source' IS NOT NULL
            GROUP BY metadata->>'source'
            ORDER BY doc_name;
        """)
        docs = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        if not docs:
            return False, "No documents found in database"
        
        return True, docs
    except Exception as e:
        logging.error(f"Error getting document list: {str(e)}")
        return False, str(e)

def test_search(query, document_name=None):
    """Test search functionality"""
    # Get database connection string
    CONNECTION_STRING = get_connection_string()
    if not CONNECTION_STRING:
        return False, "Could not get database connection string"
    
    try:
        # Initialize PGVector store
        logging.info("Initializing PGVector store for search...")
        vector_store = PGVector(
            connection_string=CONNECTION_STRING,
            embedding_function=embeddings,
            collection_name="documents", # Corresponds to table name
            use_jsonb=True
        )
        logging.info("PGVector store initialized.")
        
        # Search without filter - we'll filter programmatically instead
        logging.info(f"Searching for: '{query}'")
        raw_retrieved_docs = iterative_document_search(query, vector_store, max_iterations=2, initial_k=10)
        
        if not raw_retrieved_docs or len(raw_retrieved_docs) == 0:
            return False, "No results found"
        
        logging.info(f"Search returned {len(raw_retrieved_docs)} results")
        
        # Log metrics from DocumentCollection
        vector_count = getattr(raw_retrieved_docs, 'vector_count', 0)
        sql_count = getattr(raw_retrieved_docs, 'sql_count', 0)
        fallback_count = getattr(raw_retrieved_docs, 'fallback_count', 0)
        
        logging.info(f"Vector search found: {vector_count} chunks")
        logging.info(f"SQL search found: {sql_count} chunks")
        if fallback_count > 0:
            logging.info(f"Emergency fallback used: {fallback_count} chunks")
        
        # If a document name is provided, filter results
        if document_name:
            # First, try exact match on 'source' field
            filtered_docs = [
                doc for doc in raw_retrieved_docs 
                if doc.metadata.get('source') == document_name
            ]
            
            logging.info(f"After filtering for '{document_name}': found {len(filtered_docs)} results")
            
            # If no results, try case-insensitive matching
            if not filtered_docs:
                filtered_docs = [
                    doc for doc in raw_retrieved_docs 
                    if (doc.metadata.get('source', '').lower() == document_name.lower())
                ]
                logging.info(f"After case-insensitive matching: found {len(filtered_docs)} results")
            
            # If still no results, try string contains matching
            if not filtered_docs:
                filtered_docs = [
                    doc for doc in raw_retrieved_docs 
                    if document_name.lower() in str(doc.metadata).lower()
                ]
                logging.info(f"After contains matching: found {len(filtered_docs)} results")
            
            # Use filtered results - create a new DocumentCollection with the filtered docs
            from app_rag import DocumentCollection
            results = DocumentCollection(filtered_docs)
            
            # Preserve the metrics from the original collection
            results.vector_count = raw_retrieved_docs.vector_count
            results.sql_count = raw_retrieved_docs.sql_count
            results.fallback_count = raw_retrieved_docs.fallback_count
        else:
            # Use all results
            results = raw_retrieved_docs
        
        # Print first 3 results
        logging.info("First 3 results:")
        for i, doc in enumerate(list(results)[:3]):
            logging.info(f"Result {i+1}:")
            logging.info(f"Content: {doc.page_content[:100]}...")
            logging.info(f"Metadata: {doc.metadata}")
            logging.info("---")
        
        return True, results
    except Exception as e:
        logging.error(f"Error testing search: {str(e)}")
        return False, str(e)

def main():
    print("=== Database Search Test ===")
    
    # Get document list
    success, docs = get_document_list()
    if not success:
        print(f"Error: {docs}")
        return
    
    print(f"Found {len(docs)} documents in database:")
    for doc in docs:
        print(f"- {doc[0]}: {doc[1]} chunks")
    
    # Get user input
    print("\nEnter a query to test search functionality:")
    query = input("> ")
    
    # Ask if user wants to filter by document
    print("\nDo you want to filter by document? (y/n)")
    filter_choice = input("> ").lower()
    
    document_name = None
    if filter_choice == 'y':
        print("\nSelect a document by number:")
        for i, doc in enumerate(docs):
            print(f"{i+1}. {doc[0]}")
        
        doc_index = int(input("> ")) - 1
        if 0 <= doc_index < len(docs):
            document_name = docs[doc_index][0]
        else:
            print("Invalid selection. Searching all documents.")
    
    # Run search
    print("\nRunning search...")
    success, results = test_search(query, document_name)
    
    if not success:
        print(f"Error: {results}")
        return
    
    # Print results
    print(f"\nFound {len(results)} results")
    
    # Display DocumentCollection metrics
    vector_count = getattr(results, 'vector_count', 0)
    sql_count = getattr(results, 'sql_count', 0)
    fallback_count = getattr(results, 'fallback_count', 0)
    
    print(f"Vector search found: {vector_count} chunks")
    print(f"SQL search found: {sql_count} chunks")
    if fallback_count > 0:
        print(f"Emergency fallback used: {fallback_count} chunks")
    
    print("\nFirst 3 results:")
    result_list = list(results)
    for i, doc in enumerate(result_list[:3]):
        print(f"Result {i+1}:")
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print("---")
    
    print("\nTest complete.")

if __name__ == "__main__":
    main() 