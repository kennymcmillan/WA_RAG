"""
app.py

Main application file for the Aspire Academy Document Assistant.
This Streamlit application provides a user interface for:
- Uploading and processing PDF documents
- Integrating with Dropbox to access PDF files
- Searching through documents using vector similarity
- Asking questions about document content using LLM
- Viewing and interacting with documents

Relies on several utility modules:
- app_database.py: Database operations
- app_documents.py: PDF processing
- app_embeddings.py: Vector embeddings
- app_llm.py: LLM interactions
- app_vector.py: Vector store operations
- app_dropbox.py: Dropbox integration
"""

import os
import streamlit as st
from dotenv import load_dotenv
import logging
import json
import io
import numpy as np
from langchain_community.vectorstores.pgvector import PGVector
import psycopg2

# --- Specify .env file path explicitly ---
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)
# --- End explicit .env loading ---

# Import utility functions
from app_database import get_db_connection, initialize_pgvector, save_document_to_db, get_connection_string, load_documents_from_database, inspect_database_contents
from app_documents import get_pdf_pages, get_text_chunks, process_pdf_file, aspire_academy_css
from app_embeddings import embeddings
from app_llm import get_answer
from app_vector import create_vector_store, iterative_document_search, CustomPGVector
from app_dropbox import (get_dropbox_client, list_dropbox_folders,
                        list_dropbox_pdf_files, download_dropbox_file, create_file_like_object,
                        upload_to_dropbox)
from app_search import direct_document_search, sql_keyword_search, check_document_exists, diagnose_vector_search_issues
from app_rag import hybrid_retriever, fetch_parent_context

# --- New import for warnings ---
import warnings

# --- Suppress the LangChain deprecation warning ---
warnings.filterwarnings("ignore", category=Warning)

st.set_page_config(
    page_title="Aspire Academy Document Assistant",
    page_icon="📚",
    layout="wide"
)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting Streamlit app...")
# --- End Logging Configuration ---

# Load environment variables
logging.info("Loaded environment variables.")

# Get database connection string
CONNECTION_STRING = get_connection_string()

# Aspire Academy colors
ASPIRE_MAROON = "#7A0019"
ASPIRE_GOLD = "#FFD700"
ASPIRE_GRAY = "#F0F0F0"

# --- Enhanced Search Strategy Config ---
# Prioritizing SQL search for reliability in certain scenarios
PRIORITIZE_SQL_SEARCH = True  # Set to True to try SQL search first
MIN_VECTOR_RESULTS = 5  # Minimum number of vector results before trying SQL search

def main():
    # Apply custom CSS
    st.markdown(aspire_academy_css(), unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = {}  # Store processed documents
    if "selected_docs" not in st.session_state:
        st.session_state.selected_docs = []  # Store selected documents for search
    if "save_to_dropbox" not in st.session_state:
        st.session_state.save_to_dropbox = False
    if "auto_process" not in st.session_state:
        st.session_state.auto_process = False
    if "pdf_viewer_doc_name" not in st.session_state:
        st.session_state.pdf_viewer_doc_name = None
    
    # Debug: Inspect database contents to understand what's actually stored
    db_inspection = inspect_database_contents()
    if db_inspection:
        logging.info("Database inspection completed successfully")
    else:
        logging.warning("Database inspection failed or returned no data")
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"<h2 style='color: {ASPIRE_MAROON};'>Aspire Academy</h2>", unsafe_allow_html=True)
        
        # SECTION 1: Document selection - most important
        st.markdown("### Select Documents to Search")
        
        # Load documents from database directly
        db_docs = load_documents_from_database()
        
        # Update session state with documents from database
        if db_docs:
            # Replace session state with database info
            st.session_state.processed_docs = db_docs
            st.success(f"Found {len(db_docs)} documents in database")
        else:
            st.warning("No documents found in database. Please process some documents first.")
        
        # Debug display - show what's available
        if st.session_state.processed_docs:
            available_docs = sorted(list(st.session_state.processed_docs.keys()))
            chunk_counts = [f"{doc} ({st.session_state.processed_docs[doc]['chunk_count']} chunks)" for doc in available_docs]
            st.write("Available documents:")
            for doc_info in chunk_counts:
                st.write(f"- {doc_info}")
            
            # Define callback to ensure selection is saved and set pdf_viewer_doc_name
            def update_selected_docs():
                selected_docs = list(st.session_state.doc_selector)
                logging.info(f"Updated selected docs: {selected_docs}")
                st.session_state.selected_docs = selected_docs
                if selected_docs:
                    # Set the first selected doc as the doc to view in PDF Viewer
                    st.session_state.pdf_viewer_doc_name = selected_docs[0]
                else:
                    st.session_state.pdf_viewer_doc_name = None # Clear if no docs selected

            # Select documents for search
            st.multiselect(
                "Choose documents to search",
                options=available_docs,
                default=st.session_state.selected_docs,
                key="doc_selector",
                on_change=update_selected_docs
            )

            # Show confirmation of selection
            if st.session_state.selected_docs:
                st.success(f"Selected: {', '.join(st.session_state.selected_docs)}")
            else:
                st.warning("Please select documents to search")
        else:
            st.info("No processed documents available. Please upload and process documents.")
        
        st.markdown("---")
        
        # SECTION 2: Upload files to Dropbox and process
        st.markdown("### Upload Documents")
        
        # Simple file uploader
        pdf_docs = st.file_uploader(
            "Upload PDF documents",
            type="pdf",
            accept_multiple_files=True
        )
        
        if pdf_docs:
            # Use root folder for Dropbox
            rag_folder_path = "/"  # Root folder
        
        # Process button
        if st.button("Upload to Dropbox & Process"):
            with st.spinner("Uploading and processing files..."):
                dbx = get_dropbox_client()
                if dbx:
                    success_count = 0
                    for pdf in pdf_docs:
                        try:
                            # Upload to Dropbox
                            pdf.seek(0)
                            file_data = pdf.read()
                            upload_result = upload_to_dropbox(file_data, pdf.name, rag_folder_path, dbx)
                            
                            if upload_result:
                                # Process the file
                                pdf.seek(0)
                                process_result = process_pdf_file(pdf, False)
                                if process_result:
                                    success_count += 1
                        except Exception as e:
                            logging.error(f"Error processing {pdf.name}: {str(e)}")
                            st.error(f"Error with {pdf.name}: {str(e)}")
                    
                    st.success(f"Uploaded and processed {success_count} of {len(pdf_docs)} files")
                else:
                    st.error("Could not connect to Dropbox. Please check your credentials.")
        
        st.markdown("---")
        
        # SECTION 3: Check and process Dropbox files
        st.markdown("### Dropbox Files")
        
        # Check Dropbox for unprocessed files
        dbx = get_dropbox_client()
        if dbx:
            try:
                # Use root folder for Dropbox 
                rag_folder_path = "/"  # Root folder
                
                # List PDF files - function handles paths internally
                files = list_dropbox_pdf_files(rag_folder_path, dbx)
                
                if files:
                    # Check for unprocessed files
                    unprocessed_files = [f for f in files if f not in st.session_state.processed_docs]
                    
                    # Show all files found for debugging
                    st.write(f"Found {len(files)} PDF files in Dropbox: {', '.join(files)}")
                    st.write(f"Currently processed: {list(st.session_state.processed_docs.keys())}")
                    
                    # Only show warning about unprocessed files
                    if unprocessed_files:
                        st.warning(f"{len(unprocessed_files)} files need processing")
                        
                        # Button to process all unprocessed files
                        if st.button("Process All Unprocessed Files"):
                            with st.spinner(f"Processing {len(unprocessed_files)} files..."):
                                success_count = 0
                                for file_name in unprocessed_files:
                                    try:
                                        # Construct path with leading slash for root folder
                                        file_path = f"/{file_name}"
                                        logging.info(f"Attempting to download file: {file_path}")
                                        
                                        file_data = download_dropbox_file(file_path, dbx)
                                        if file_data:
                                            logging.info(f"Successfully downloaded file: {file_name} ({len(file_data)} bytes)")
                                            pdf_file_like = create_file_like_object(file_data, file_name)
                                            
                                            # Process file
                                            logging.info(f"Processing file: {file_name}")
                                            pages_data = get_pdf_pages([pdf_file_like])
                                            if pages_data:
                                                documents = get_text_chunks(pages_data)
                                                if create_vector_store(documents, file_name):
                                                    st.session_state.processed_docs[file_name] = {
                                                        "chunk_count": len(documents)
                                                    }
                                                    success_count += 1
                                    except Exception as e:
                                        logging.error(f"Error processing {file_name}: {str(e)}")
                                
                                if success_count > 0:
                                    st.success(f"Successfully processed {success_count} file(s)")
                                    # Force UI to refresh after processing
                                    st.rerun()
                                else:
                                    st.error("Failed to process any files")
                    else:
                        st.success("All Dropbox files have been processed")
                else:
                    st.info("No PDF files found in Dropbox root folder")
            except Exception as e:
                st.error(f"Error accessing Dropbox: {str(e)}")
        else:
            st.error("Could not connect to Dropbox. Please check your credentials.")
        
        # Admin section - hidden by default
        with st.expander("Admin Controls", expanded=False):
            st.markdown(f"<h3 style='color: {ASPIRE_MAROON};'>Database Administration</h3>", unsafe_allow_html=True)
            st.warning("Warning: These actions can cause data loss and should be used with caution.")
            
            # Clear database button
            if st.button("Clear Document Database"):
                try:
                    conn = get_db_connection()
                    if not conn:
                        st.error("Failed to connect to database")
                        return
                    
                    # Get count before deletion
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM documents;")
                    count_before = cursor.fetchone()[0]
                    
                    # Clear the table
                    cursor.execute("DELETE FROM documents;")
                    conn.commit()
                    
                    # Close connection
                    cursor.close()
                    conn.close()
                    
                    # Update session state
                    st.session_state.processed_docs = {}
                    st.session_state.selected_docs = []
                    
                    st.success(f"Successfully deleted {count_before} document chunks from the database.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing database: {str(e)}")
            
            # Initialize pgvector button 
            st.markdown("### Vector Search Setup")
            st.info("Use this to initialize or repair pgvector extension and tables")
            if st.button("Initialize PGVector Tables"):
                try:
                    from app_database import initialize_pgvector, check_pgvector_extension
                    
                    # First check the current status
                    pgvector_status = check_pgvector_extension()
                    if pgvector_status:
                        st.success("PGVector extension and tables are already set up correctly!")
                    else:
                        # Try to initialize
                        result = initialize_pgvector()
                        if result:
                            st.success("Successfully initialized PGVector extension and tables!")
                            st.info("You should now reprocess your documents to create embeddings")
                        else:
                            st.error("Failed to initialize PGVector. Check the logs for details.")
                            st.info("Make sure the pgvector extension is installed on your PostgreSQL server")
                except Exception as e:
                    st.error(f"Error initializing PGVector: {str(e)}")
            
            # Complete Reset button (clear data + initialize tables)
            st.markdown("### Complete Reset")
            st.warning("⚠️ This will clear ALL document data and recreate tables")
            if st.button("Complete Reset"):
                try:
                    # 1. Clear documents table
                    conn = get_db_connection()
                    if conn:
                        try:
                            cursor = conn.cursor()
                            # Get count before deletion
                            cursor.execute("SELECT COUNT(*) FROM documents;")
                            docs_count = cursor.fetchone()[0]
                            
                            # Get count from langchain_pg_embedding if exists
                            embeddings_count = 0
                            try:
                                cursor.execute("""
                                    SELECT COUNT(*) FROM langchain_pg_embedding;
                                """)
                                embeddings_count = cursor.fetchone()[0]
                            except:
                                pass  # Table might not exist yet
                            
                            # Clear tables
                            cursor.execute("DELETE FROM documents;")
                            
                            # Try to delete from langchain_pg_embedding if it exists
                            try:
                                cursor.execute("DELETE FROM langchain_pg_embedding;")
                            except:
                                pass  # Table might not exist yet
                            
                            conn.commit()
                            cursor.close()
                            conn.close()
                            
                            st.info(f"Cleared {docs_count} documents and {embeddings_count} embeddings")
                        except Exception as e:
                            st.error(f"Error clearing tables: {str(e)}")
                            if conn and not conn.closed:
                                conn.close()
                    
                    # 2. Initialize tables
                    from app_database import initialize_pgvector
                    result = initialize_pgvector()
                    if result:
                        st.success("Successfully reset database and initialized tables!")
                        # Update session state
                        st.session_state.processed_docs = {}
                        st.session_state.selected_docs = []
                        st.info("You can now upload and process your documents")
                        st.rerun()
                    else:
                        st.error("Tables initialization failed. Check the logs for details.")
                except Exception as e:
                    st.error(f"Error performing complete reset: {str(e)}")
            
            # Reset chat button
            if st.button("Reset Chat"):
                st.session_state.messages = []
    
    # Main content
    tab1, tab2 = st.tabs(["Chat", "PDF Viewer"])
    
    with tab1:
        st.markdown(f"<h1 style='color: {ASPIRE_MAROON};'>Aspire Academy Document Assistant</h1>", unsafe_allow_html=True)
        st.markdown("Ask questions about your uploaded documents and get instant answers.")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display sources if available
                if message["role"] == "assistant" and "sources" in message:
                    st.markdown("<div class='source-citation'>", unsafe_allow_html=True)
                    st.markdown("**Sources:**", unsafe_allow_html=True)
                    for source in message["sources"]:
                        st.markdown(f"- {source}", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
        
        # Chat input
        if user_question := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.chat_message("user").markdown(user_question)
            st.session_state.messages.append({"role": "user", "content": user_question})
            logging.info(f"User question received: {user_question}")
            
            try:
                if not st.session_state.selected_docs:
                    logging.warning("User asked question but no documents selected.")
                    st.error("Please select at least one document to search!")
                elif not CONNECTION_STRING:
                     logging.error("User asked question but DB connection string is missing.")
                     st.error("Database connection is not configured. Please check your .env file.")
                elif not embeddings:
                     logging.error("User asked question but embeddings model failed to initialize.")
                     st.error("Embeddings model not available. Please check logs.")
                else:
                    logging.info(f"Searching selected documents: {st.session_state.selected_docs}")
                    # Initialize PGVector store - Using our custom implementation
                    logging.info("Initializing CustomPGVector store for search...")
                    
                    # Use our custom PGVector implementation instead of the standard one
                    vector_store = CustomPGVector(
                        connection_string=CONNECTION_STRING,
                        embedding_function=embeddings,
                        collection_name="documents", # Corresponds to table name
                        use_jsonb=True
                    )
                    logging.info("CustomPGVector store initialized with our metadata matching function.")
                    
                    # Debug information about the selected documents
                    logging.info(f"Selected documents for search: {st.session_state.selected_docs}")
                    
                    # Initialize results containers
                    raw_retrieved_docs = []
                    selected_doc = st.session_state.selected_docs[0] if st.session_state.selected_docs else None
                    
                    # Use our hybrid retriever that combines SQL and vector search
                    logging.info(f"Using hybrid retriever for '{user_question}' in {selected_doc}")
                    raw_retrieved_docs = hybrid_retriever(user_question, vector_store, selected_doc, limit=50)
                    logging.info(f"Hybrid retriever found {len(raw_retrieved_docs)} documents")
                    
                    # Enhance results with parent context
                    if raw_retrieved_docs:
                        raw_retrieved_docs = fetch_parent_context(raw_retrieved_docs, selected_doc)
                        logging.info(f"After fetching parent context: {len(raw_retrieved_docs)} documents")
                    
                    # Show debug info in UI if enabled
                    with st.expander("Debug Information", expanded=True):
                        st.write("### Query Analysis")
                        st.write(f"Query: '{user_question}'")
                        st.write(f"Selected document: {st.session_state.selected_docs[0] if st.session_state.selected_docs else 'None'}")
                        
                        # Add more detailed breakdown of results
                        st.write("### Search Results")
                        # Add these lines to track metrics from the hybrid retriever
                        sql_count = getattr(raw_retrieved_docs, 'sql_count', 0)
                        vector_count = getattr(raw_retrieved_docs, 'vector_count', 0)
                        fallback_count = getattr(raw_retrieved_docs, 'fallback_count', 0)
                        
                        st.write(f"Hybrid search found: {len(raw_retrieved_docs)} chunks total")
                        st.write(f"- SQL search found: {sql_count} chunks")
                        st.write(f"- Vector search found: {vector_count} chunks")
                        if fallback_count > 0:
                            st.write(f"- Emergency fallback used: {fallback_count} chunks")
                        
                        # If there are parent documents, show that info
                        parent_count = getattr(raw_retrieved_docs, 'parent_count', 0)
                        if parent_count > 0:
                            st.write(f"- Added {parent_count} parent documents for context")
                            
                        # Database Contents Check section
                        st.write("### Database Contents Check")
                        doc_check = check_document_exists(selected_doc)
                        if "error" not in doc_check:
                            st.write(f"Total document chunks in DB: {doc_check['total_docs']}")
                            st.write(f"Available documents: {', '.join(doc_check['doc_names'])}")
                            
                            if selected_doc and doc_check['specific_doc']:
                                if doc_check['specific_doc']['exists']:
                                    st.success(f"Document '{selected_doc}' exists with {doc_check['specific_doc']['chunk_count']} chunks")
                                    st.write("#### Sample Content:")
                                    st.write(doc_check['specific_doc']['sample_content'])
                                    st.write("#### Sample Metadata:")
                                    st.json(doc_check['specific_doc']['sample_metadata'])
                                else:
                                    st.error(f"Document '{selected_doc}' NOT FOUND in database!")
                        else:
                            st.error(f"Error checking database: {doc_check['error']}")
                            
                        # Vector Search Diagnostics section
                        st.write("### Vector Search Diagnostics")
                        try:
                            diagnostics = diagnose_vector_search_issues(selected_doc)
                            
                            # Show summary statistics
                            st.write(f"Total documents: {diagnostics['total_documents']}")
                            st.write(f"Documents with embeddings: {diagnostics['documents_with_embeddings']}")
                            
                            # Document-specific information
                            if selected_doc and 'doc_count' in diagnostics:
                                st.write(f"Selected document '{selected_doc}':")
                                st.write(f"- Document chunks: {diagnostics['doc_count']}")
                                st.write(f"- Chunks with embeddings: {diagnostics.get('doc_embedding_count', 0)}")
                            
                            # Show embedding samples
                            if diagnostics['embedding_samples']:
                                with st.expander("Embedding Samples"):
                                    for i, sample in enumerate(diagnostics['embedding_samples']):
                                        st.write(f"Sample {i+1}:")
                                        st.write(f"- Has embedding: {sample['has_embedding']}")
                                        st.write(f"- Embedding dimensions: {sample['embedding_dimensions']}")
                                        st.write(f"- Content: {sample['content_preview']}")
                                        st.write(f"- Metadata: {sample['metadata']}")
                                        st.write("---")
                            
                            # Show issues if any
                            if diagnostics['issues']:
                                st.error("Issues detected:")
                                for issue in diagnostics['issues']:
                                    st.write(f"- {issue}")
                            elif 'status' in diagnostics:
                                st.success(diagnostics['status'])
                        except Exception as e:
                            st.error(f"Error running vector diagnostics: {str(e)}")
                    
                    # Use the retrieved documents directly
                    if st.session_state.selected_docs and raw_retrieved_docs:
                        retrieved_docs = raw_retrieved_docs
                        logging.info(f"Using {len(retrieved_docs)} chunks from hybrid retriever")
                    else:
                        retrieved_docs = []
                        logging.warning("No documents retrieved")
                        
                    logging.info("--- After filtering ---")
                    logging.info(f"Final results: {len(retrieved_docs)} chunks.")
                
                    # Generate answer only if we have relevant chunks
                    if not retrieved_docs:
                        answer = "I don't have enough information in the selected documents to answer that question."
                        sources = []
                    else:
                        # Extract context and sources from retrieved LangChain Documents
                        logging.info("Extracting context and sources from retrieved chunks.")
                        context_parts = []
                        sources = set() # Use a set to avoid duplicate source listings
                        
                        # Sort documents by source and page for better organization
                        sorted_docs = sorted(retrieved_docs, 
                                            key=lambda x: (x.metadata.get('source', 'Unknown'), 
                                                          x.metadata.get('page', 0)))
                        
                        # Process documents with better formatting
                        for doc in sorted_docs:
                            source = doc.metadata.get('source', 'Unknown Source')
                            page = doc.metadata.get('page', 'Unknown Page')
                            
                            # Check if this is a parent or child document for debugging
                            doc_type = "Parent" if doc.metadata.get('is_parent', False) else "Child"
                            logging.info(f"Adding {doc_type} document from {source} (Page {page})")
                            
                            # Add formatted context with clear source attribution
                            context_parts.append(f"From {source} (Page {page}):\n{doc.page_content.strip()}")
                            sources.add(f"{source} (Page {page})")
                        
                        # Join with double newlines for clear separation between chunks
                        context = "\n\n".join(context_parts)
                        sources = sorted(list(sources)) # Convert back to sorted list for display
                        logging.info(f"Generated context for LLM with {len(context_parts)} chunks from {len(sources)} sources")

                        # Debug: Show context length and structure
                        st.info(f"Context for question has {len(context)} characters from {len(sources)} sources")
                        
                        # Add sample of context with visible chunk separators for debugging
                        with st.expander("Sample of context sent to LLM"):
                            # Show the first 3 chunks for debugging
                            st.markdown("```")
                            for i, chunk in enumerate(context_parts[:3]):
                                st.markdown(f"Chunk {i+1}:\n{chunk[:300]}...\n\n")
                            st.markdown("```")
                            st.info(f"Showing 3 of {len(context_parts)} total chunks")

                        # Add keyword detection
                        keywords = [k for k in user_question.lower().split() if len(k) > 3]
                        if keywords:
                            keyword_counts = {}
                            for chunk in context_parts:
                                chunk_lower = chunk.lower()
                                for keyword in keywords:
                                    if keyword in chunk_lower:
                                        if keyword in keyword_counts:
                                            keyword_counts[keyword] += 1
                                        else:
                                            keyword_counts[keyword] = 1
                                                
                            st.markdown("**Keyword Frequency in Context:**")
                            for keyword, count in keyword_counts.items():
                                st.text(f"'{keyword}': found in {count} of {len(context_parts)} chunks")

                        # Generate answer
                        logging.info("--- Before get_answer ---")
                        with st.spinner("Generating answer..."):
                            answer = get_answer(user_question, context)
                            logging.info("--- After get_answer ---")
                        logging.info("Answer received from OpenRouter.")

                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    
                    # Display sources (only if sources were found)
                    if sources:
                        st.markdown("<div class='source-citation'>", unsafe_allow_html=True)
                        st.markdown("**Sources:**", unsafe_allow_html=True)
                        for source in sources:
                            st.markdown(f"- {source}", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab2:
        st.markdown(f"<h1 style='color: {ASPIRE_MAROON};'>Document Viewer</h1>", unsafe_allow_html=True)
        
        # Display PDF viewer for selected document
        if st.session_state.pdf_viewer_doc_name:
            try:
                # Show which document is being displayed
                st.subheader(f"Viewing: {st.session_state.pdf_viewer_doc_name}")
                
                # Get Dropbox client
                dbx = get_dropbox_client()
                if dbx:
                    # Construct path with leading slash for root folder
                    file_path = f"/{st.session_state.pdf_viewer_doc_name}"
                    logging.info(f"Attempting to download file for viewer: {file_path}")
                    
                    # Download file data
                    file_data = download_dropbox_file(file_path, dbx)
                    if file_data:
                        # Display the PDF using streamlit-pdf-viewer
                        from streamlit_pdf_viewer import pdf_viewer
                        pdf_viewer(file_data, width=700)
                        st.success(f"Displaying PDF: {st.session_state.pdf_viewer_doc_name}")
                    else:
                        st.error(f"Could not download {st.session_state.pdf_viewer_doc_name} from Dropbox")
                else:
                    st.error("Dropbox client not available. Could not display PDF.")
            except Exception as e:
                st.error(f"Error displaying PDF: {str(e)}")
        else:
            st.info("Select a document from the sidebar to view it here.")

if __name__ == "__main__":
    main()