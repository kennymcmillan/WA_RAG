"""
Test script for SQL-based search functionality
"""

import os
import logging
from dotenv import load_dotenv
from app_search import direct_document_search

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

def main():
    print("=== SQL Search Test ===")
    
    # List available documents
    print("\nAvailable documents for search:")
    print("1. Sanguine Sports Tech Fund.pdf")
    print("2. Competition and Technical Rules – 2024 Edition (1).pdf")
    print("3. Drake Star Global Sports Tech Report_2023.pdf")
    
    doc_choice = input("\nSelect document number (or press Enter for all): ")
    
    doc_name = None
    if doc_choice == '1':
        doc_name = "Sanguine Sports Tech Fund.pdf"
    elif doc_choice == '2':
        doc_name = "Competition and Technical Rules – 2024 Edition (1).pdf"
    elif doc_choice == '3':
        doc_name = "Drake Star Global Sports Tech Report_2023.pdf"
    
    # Get search query
    query = input("\nEnter search query: ")
    
    # Choose search method
    print("\nSelect search method:")
    print("1. Full-text search (PostgreSQL to_tsvector)")
    print("2. Keyword search (SQL ILIKE)")
    
    method_choice = input("Enter choice (1-2): ")
    use_fulltext = method_choice != '2'  # Use fulltext unless option 2 is selected
    
    # Perform search
    print(f"\nPerforming {'full-text' if use_fulltext else 'keyword'} search for: '{query}'")
    if doc_name:
        print(f"Filtering by document: {doc_name}")
    
    results = direct_document_search(query, doc_name=doc_name, use_fulltext=use_fulltext)
    
    # Display results
    if results:
        print(f"\nFound {len(results)} results:")
        for i, doc in enumerate(results[:10]):  # Show top 10 results
            print(f"\nResult {i+1}:")
            print(f"Content: {doc.page_content[:150]}...")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Page: {doc.metadata.get('page', 'Unknown')}")
            print("-" * 50)
    else:
        print("\nNo results found.")

if __name__ == "__main__":
    main() 