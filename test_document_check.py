"""
Test script to check document contents in the database
"""

import os
import logging
from dotenv import load_dotenv
from app_search import check_document_exists, sql_keyword_search

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

def test_documents():
    print("=== Document Database Contents Check ===")
    
    # Check all documents in the database
    doc_check = check_document_exists()
    
    if "error" in doc_check:
        print(f"Error checking database: {doc_check['error']}")
        return
    
    print(f"\nTotal document chunks in database: {doc_check['total_docs']}")
    print(f"Available documents: {doc_check['doc_names']}")
    
    # Test specific document if available
    if "Sanguine Sports Tech Fund.pdf" in doc_check['doc_names']:
        print("\n=== Checking Sanguine Sports Tech Fund.pdf ===")
        doc_info = check_document_exists("Sanguine Sports Tech Fund.pdf")
        
        if doc_info['specific_doc']['exists']:
            print(f"Document exists with {doc_info['specific_doc']['chunk_count']} chunks")
            print(f"\nSample content: {doc_info['specific_doc']['sample_content']}")
            
            # Try a basic search for any content
            print("\n=== Testing search for 'fund' ===")
            results = sql_keyword_search("fund", doc_name="Sanguine Sports Tech Fund.pdf")
            if results:
                print(f"Found {len(results)} results for 'fund'")
                print(f"First result: {results[0].page_content[:150]}...")
            else:
                print("No results found for 'fund'")
            
            # Try searching for team members by using a more general term
            print("\n=== Testing search for 'team' ===")
            results = sql_keyword_search("team", doc_name="Sanguine Sports Tech Fund.pdf")
            if results:
                print(f"Found {len(results)} results for 'team'")
                for i, doc in enumerate(results[:3]):
                    print(f"\nResult {i+1}: {doc.page_content[:150]}...")
            else:
                print("No results found for 'team'")
                
            # Check any content about a founder
            print("\n=== Testing search for 'founder' ===")
            results = sql_keyword_search("founder", doc_name="Sanguine Sports Tech Fund.pdf")
            if results:
                print(f"Found {len(results)} results for 'founder'")
                for i, doc in enumerate(results[:3]):
                    print(f"\nResult {i+1}: {doc.page_content[:150]}...")
            else:
                print("No results found for 'founder'")
        else:
            print("Document doesn't exist in the database")
    
    # Check for any mention of Brian or Moore in any document
    print("\n=== Searching any document for 'Brian' ===")
    results = sql_keyword_search("Brian")
    if results:
        print(f"Found {len(results)} results for 'Brian' across all documents")
        for i, doc in enumerate(results[:3]):
            print(f"\nResult {i+1} (from {doc.metadata.get('source', 'Unknown')}):")
            print(f"{doc.page_content[:150]}...")
    else:
        print("No results found for 'Brian' in any document")
    
    print("\n=== Searching any document for 'Moore' ===")
    results = sql_keyword_search("Moore")
    if results:
        print(f"Found {len(results)} results for 'Moore' across all documents")
        for i, doc in enumerate(results[:3]):
            print(f"\nResult {i+1} (from {doc.metadata.get('source', 'Unknown')}):")
            print(f"{doc.page_content[:150]}...")
    else:
        print("No results found for 'Moore' in any document")

if __name__ == "__main__":
    test_documents() 