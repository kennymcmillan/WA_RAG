"""
run_app.py

Script to run the Aspire Academy Document Assistant with the improved
Parent Document Retriever RAG implementation with robust error handling.
"""

import subprocess
import os
import sys
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app_run.log')
    ]
)

def run_app():
    """
    Run the Streamlit app with the new RAG implementation
    """
    logging.info("Starting Aspire Academy Document Assistant")
    logging.info("Using Parent Document Retriever RAG implementation with improved error handling")
    
    # Check if all required files exist
    required_files = [
        'app.py',
        'app_rag.py',
        'app_vector.py',
        'app_llm.py',
        'app_documents.py',
        'app_embeddings.py',
        'app_database.py'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            logging.error(f"Required file {file} not found")
            return
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        logging.warning(".env file not found. Make sure you have environment variables set")
    
    try:
        # Run the app with streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
        
        # Print instructions
        print("\n" + "="*50)
        print("Aspire Academy Document Assistant")
        print("="*50)
        print("\nImplemented with Parent Document Retriever RAG pattern")
        print("This pattern creates:")
        print("- Larger parent chunks (2000 chars) for context")
        print("- Smaller child chunks (400 chars) for precise retrieval")
        print("- Maintains relationships between parent and child chunks")
        print("\nThe app uses a hybrid retriever that combines:")
        print("- SQL search (reliable keyword matching)")
        print("- Vector search (semantic similarity)")
        print("- Document ranking by relevance")
        print("- Parent context fetching for comprehensive answers")
        print("\nImproved Error Handling:")
        print("- Robust SQL query error handling")
        print("- Emergency fallback retrieval mechanisms")
        print("- Graceful degradation when components fail")
        print("- Comprehensive error logging for troubleshooting")
        print("\nStarting the application...")
        print("="*50 + "\n")
        
        # Start the process
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        logging.info("Application stopped by user")
    except Exception as e:
        logging.error(f"Error running the application: {str(e)}")

if __name__ == "__main__":
    run_app() 