import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores.pgvector import PGVector
from app_embeddings import embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

def get_connection_string():
    """Get database connection string from environment variables"""
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    
    if all([DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT]):
        connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        return connection_string
    else:
        return None

def main():
    print("=== Simple Database Search Test ===")
    
    # Get database connection string
    connection_string = get_connection_string()
    if not connection_string:
        print("Error: Could not get database connection string")
        return
    
    # Initialize PGVector
    try:
        print("Initializing PGVector store...")
        vector_store = PGVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name="documents",  # Table name
            use_jsonb=True
        )
        print("PGVector initialized successfully!")
    except Exception as e:
        print(f"Error initializing PGVector: {str(e)}")
        return
    
    # Get search query
    query = input("Enter a simple search query: ")
    
    # Search without filter
    try:
        print(f"\nSearching for: '{query}'")
        docs = vector_store.similarity_search(
            query,
            k=20  # Return top 20 results
        )
        
        # Check results
        if docs:
            print(f"\nFound {len(docs)} results:")
            for i, doc in enumerate(docs[:5]):  # Show top 5
                print(f"\nResult {i+1}:")
                print(f"Content: {doc.page_content[:150]}...")
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"Page: {doc.metadata.get('page', 'Unknown')}")
                print("-" * 40)
        else:
            print("\nNo results found")
    except Exception as e:
        print(f"Error during search: {str(e)}")

if __name__ == "__main__":
    main() 