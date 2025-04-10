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

# Must be the first Streamlit command
st.set_page_config(
    page_title="Aspire Academy Document Assistant",
    page_icon="📚",
    layout="wide"
)

# --- Environment setup and variable loading ---
# Load environment variables from .env file or Streamlit secrets
logging.info("Loading environment variables.")
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")

# First try loading from .env file (local development)
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    logging.info("Loaded environment variables from .env file.")

# Check if we're in a Streamlit Cloud environment and use secrets if available
try:
    if 'database' in st.secrets:
        # Map database section to environment variables
        os.environ["DB_NAME"] = st.secrets.database.get("DB_NAME", os.environ.get("DB_NAME", ""))
        os.environ["DB_USER"] = st.secrets.database.get("DB_USER", os.environ.get("DB_USER", ""))
        os.environ["DB_PASSWORD"] = st.secrets.database.get("DB_PASSWORD", os.environ.get("DB_PASSWORD", ""))
        os.environ["DB_HOST"] = st.secrets.database.get("DB_HOST", os.environ.get("DB_HOST", ""))
        os.environ["DB_PORT"] = st.secrets.database.get("DB_PORT", os.environ.get("DB_PORT", ""))
        logging.info("Loaded database credentials from Streamlit secrets.")
        
    if 'dropbox' in st.secrets:
        # Map Dropbox section to environment variables
        os.environ["DROPBOX_APPKEY"] = st.secrets.dropbox.get("DROPBOX_APPKEY", os.environ.get("DROPBOX_APPKEY", ""))
        os.environ["DROPBOX_APPSECRET"] = st.secrets.dropbox.get("DROPBOX_APPSECRET", os.environ.get("DROPBOX_APPSECRET", ""))
        os.environ["DROPBOX_REFRESH_TOKEN"] = st.secrets.dropbox.get("DROPBOX_REFRESH_TOKEN", os.environ.get("DROPBOX_REFRESH_TOKEN", ""))
        os.environ["DROPBOX_TOKEN"] = st.secrets.dropbox.get("DROPBOX_TOKEN", os.environ.get("DROPBOX_TOKEN", ""))
        logging.info("Loaded Dropbox credentials from Streamlit secrets.")
        
    if 'openrouter' in st.secrets:
        # Map OpenRouter section to environment variables
        os.environ["OPENROUTER_API_KEY"] = st.secrets.openrouter.get("OPENROUTER_API_KEY", os.environ.get("OPENROUTER_API_KEY", ""))
        logging.info("Loaded OpenRouter API key from Streamlit secrets.")
        
    if 'general' in st.secrets:
        # Map any other variables from general section
        for key, value in st.secrets.general.items():
            os.environ[key] = value
        logging.info("Loaded general variables from Streamlit secrets.")
except Exception as e:
    logging.warning(f"Could not load from Streamlit secrets: {str(e)}")
    logging.info("Continuing with environment variables from .env file or system environment.")

logging.info("Environment variables loaded successfully.")

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
from app_rag import hybrid_retriever, fetch_parent_context, DocumentCollection
from app_multi_search import multi_document_search, MAX_DOCUMENTS

# --- New import for warnings ---
import warnings

# --- Suppress the LangChain deprecation warning ---
warnings.filterwarnings("ignore", category=Warning)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting Streamlit app...")
# --- End Logging Configuration ---

# Load environment variables
logging.info("Loaded environment variables.")

# Get database connection string
CONNECTION_STRING = get_connection_string()

# --- Color scheme based on the image ---
# Replace the maroon with a blue color scheme
APP_PRIMARY_COLOR = "#29477F"  # Dark blue
APP_SECONDARY_COLOR = "#486CA5"  # Medium blue
APP_ACCENT_COLOR = "#F2CA52"  # Gold/Yellow accent
APP_TEXT_COLOR = "#333333"  # Dark gray (almost black) for text
APP_BACKGROUND_COLOR = "#FFFFFF"  # White background
APP_LIGHT_GRAY = "#F5F5F7"  # Light gray for bars/sections

# --- Enhanced Search Strategy Config ---
# Prioritizing SQL search for reliability in certain scenarios
PRIORITIZE_SQL_SEARCH = True  # Set to True to try SQL search first
MIN_VECTOR_RESULTS = 5  # Minimum number of vector results before trying SQL search

# --- Load custom CSS ---
def load_css():
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Try to load custom CSS, if file doesn't exist yet, we'll create it later
try:
    load_css()
except FileNotFoundError:
    pass

def main():
    logging.info("Starting Aspire Academy Document Assistant")
    
    if os.getenv("DATABASE_URL"):
        logging.info("Database URL found in environment variables")
    else:
        logging.warning("Database URL not found in environment")
    
    # Apply custom CSS from style.css instead of the old method
    # st.markdown(aspire_academy_css(), unsafe_allow_html=True)
    
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
        # Aspire Academy branding and title - adjust column ratio for larger logo
        col1, col2 = st.columns([1.2, 3])
        
        with col1:
            # Use a much larger logo size and add some CSS styling
            st.markdown("""
            <style>
            [data-testid="stImage"] img {
                width: 160px !important;
                margin-top: 0;
                vertical-align: top;
            }
            </style>
            """, unsafe_allow_html=True)
            st.image("aspire_logo.png", use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <style>
            .title-container {{
                padding-top: 15px;
            }}
            </style>
            <div class="title-container">
            <h2 style='color: {APP_PRIMARY_COLOR}; margin-top: 0; margin-bottom: 0; padding-top: 0; padding-bottom: 0;'>
            Aspire Academy<br>
            <span style='font-size: 1rem;'>Sports Dept<br>Document Search</span>
            </h2>
            </div>
            """, unsafe_allow_html=True)
        
        # SECTION 1: Document selection - most important
        #st.markdown("### Select Documents to Search")
        
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
            
            # Documents to hide from the display list
            hidden_documents = ["Competition and Technical Rules – 2024 Edition (1).pdf"]
            
            # Filter displayed documents (but keep them in the session state)
            displayed_docs = [doc for doc in available_docs if doc not in hidden_documents]
            
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

            # Select documents for search - use the full available list including hidden docs
            st.multiselect(
                f"Choose documents to search (max {MAX_DOCUMENTS})",
                options=available_docs,
                default=st.session_state.selected_docs,
                key="doc_selector",
                on_change=update_selected_docs
            )

            # Show confirmation of selection
            if st.session_state.selected_docs:
                if len(st.session_state.selected_docs) > MAX_DOCUMENTS:
                    st.warning(f"You've selected {len(st.session_state.selected_docs)} documents. Only the first {MAX_DOCUMENTS} will be used for search.")
                    # The actual enforcement happens in the multi_document_search function
                else:
                    st.success(f"Selected: {', '.join(st.session_state.selected_docs)}")
            else:
                st.warning("Please select documents to search")
        else:
            st.info("No processed documents available. Please upload and process documents.")
        
        st.markdown("---")
        
        # SECTION 2: Check and process Dropbox files - MOVED UP
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
                    
                    # Only show warning about unprocessed files
                    if unprocessed_files:
                        # Use markdown with red text for the warning
                        st.markdown(f"<span style='color: red; font-weight: bold;'>{len(unprocessed_files)} files need processing:</span>", unsafe_allow_html=True)
                        # List the unprocessed files with red bullet points
                        for file in unprocessed_files:
                            st.markdown(f"<span style='color: red;'>• {file}</span>", unsafe_allow_html=True)
                        
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
            
        st.markdown("---")
        
        # SECTION 3: Upload files to Dropbox and process
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
        
        # Admin section - hidden by default
        with st.expander("Admin Controls", expanded=False):
            st.markdown(f"<h3 style='color: {APP_PRIMARY_COLOR};'>Database Administration</h3>", unsafe_allow_html=True)
            st.warning("Warning: These actions can cause data loss and should be used with caution.")
            
            # Add diagnostics toggle
            st.markdown("### Display Options")
            # Initialize the show_diagnostics state if it doesn't exist
            if 'show_diagnostics' not in st.session_state:
                st.session_state.show_diagnostics = False
            
            show_diag = st.toggle("Show Diagnostic Containers", value=st.session_state.show_diagnostics)
            if show_diag != st.session_state.show_diagnostics:
                st.session_state.show_diagnostics = show_diag
                st.success(f"Diagnostic information is now {'visible' if show_diag else 'hidden'}")
            
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
    # Use plain text with emoji for tab labels - no HTML
    tab1, tab2 = st.tabs(["💬 Chat", "📄 PDF Viewer"])
    
    with tab1:
        # Remove the heading that says "Aspire Academy Document Assistant"
        # st.markdown(f"<h1 style='color: {APP_PRIMARY_COLOR};'>Aspire Academy Document Assistant</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {APP_SECONDARY_COLOR};'>Ask questions about your uploaded documents and get instant answers.</p>", unsafe_allow_html=True)
    
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
                
                # Check if any documents are selected
                if not st.session_state.selected_docs:
                    st.warning("Please select one or more documents to search.")
                    st.stop()
                
                # Check if too many documents are selected
                if len(st.session_state.selected_docs) > MAX_DOCUMENTS:
                    st.warning(f"You've selected {len(st.session_state.selected_docs)} documents. For optimal performance, the search will be limited to the first {MAX_DOCUMENTS} documents.")
                
                # Use multi-document search to search across all selected documents
                logging.info(f"Searching across {len(st.session_state.selected_docs)} documents for: '{user_question}'")
                raw_retrieved_docs = multi_document_search(
                    user_question, 
                    vector_store, 
                    st.session_state.selected_docs,
                    limit_per_doc=20
                )
                
                # Log search results summary
                docs_with_results = raw_retrieved_docs.metrics.get("documents_with_results", 0)
                docs_searched = raw_retrieved_docs.metrics.get("documents_searched", 0)
                logging.info(f"Found {len(raw_retrieved_docs)} relevant chunks from {docs_with_results}/{docs_searched} documents")
                
                # Collect and display details about the retrieved docs
                if raw_retrieved_docs and len(raw_retrieved_docs) > 0:
                    logging.info(f"Found {len(raw_retrieved_docs)} docs, fetching parent contexts...")
                    # Add parent context to improve coherence
                    retrieved_docs = fetch_parent_context(raw_retrieved_docs, parent_limit=2)
                    logging.info(f"After parent context: {len(retrieved_docs)} docs")
                    
                    # Display diagnostic info about docs - Only if diagnostics are enabled
                    if st.session_state.show_diagnostics:
                        with st.expander("Debug Information", expanded=False):
                            st.write("### Query Analysis")
                            st.text(f"Query: {user_question}")
                            st.text(f"Documents searched: {len(st.session_state.selected_docs)}")
                            doc_names = ", ".join(st.session_state.selected_docs[:5])
                            if len(st.session_state.selected_docs) > 5:
                                doc_names += "..."
                            st.text(f"Documents: {doc_names}")
                            st.text(f"Raw retrieved docs: {len(raw_retrieved_docs)}")
                            st.text(f"Retrieved docs with parent context: {len(retrieved_docs)}")
                            
                            # Multi-document metrics
                            docs_with_results = raw_retrieved_docs.metrics.get("documents_with_results", 0)
                            docs_searched = raw_retrieved_docs.metrics.get("documents_searched", 0)
                            st.text(f"Documents with results: {docs_with_results}/{docs_searched}")
                            
                            # Additional metrics
                            st.text(f"SQL search found: {getattr(retrieved_docs, 'sql_count', 0)} chunks")
                            st.text(f"Vector search found: {getattr(retrieved_docs, 'vector_count', 0)} chunks")
                            st.text(f"Table content chunks: {getattr(retrieved_docs, 'table_count', 0)}")
                            st.text(f"Fallback search used: {getattr(retrieved_docs, 'fallback_count', 0)} chunks")
                            st.text(f"Parent documents added: {getattr(retrieved_docs, 'parent_count', 0)} chunks")
                        
                        # Only show diagnostics containers if enabled
                        with st.expander("Diagnostics", expanded=False):
                            st.write("### Multi-Document Search Results")
                            
                            # Prepare sources list by document
                            doc_sources = {}
                            for doc in retrieved_docs:
                                source_doc = doc.metadata.get('source', 'Unknown')
                                page = doc.metadata.get('page', 'Unknown')
                                
                                if source_doc not in doc_sources:
                                    doc_sources[source_doc] = []
                                    
                                source_info = f"Page {page}"
                                if source_info not in doc_sources[source_doc]:
                                    doc_sources[source_doc].append(source_info)
                                    
                            # Context metrics
                            total_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
                            char_count = len(total_content)
                            
                            # Add the context feedback
                            st.text(f"Context for question has {char_count} characters from {len(doc_sources)} documents")
                            
                            # Display sources by document
                            st.write("### Sources Used")
                            for doc_name, pages in doc_sources.items():
                                pages_str = ", ".join([str(p) for p in pages[:10]])
                                if len(pages) > 10:
                                    pages_str += "..."
                                st.text(f"📄 {doc_name}: {pages_str}")
                            
                            # Display database contents check
                            st.write("### Retrieved Content Sample")
                            # Group samples by document
                            samples_by_doc = {}
                            for doc in retrieved_docs[:10]:  # Take first 10 docs for samples
                                doc_name = doc.metadata.get('source', 'Unknown')
                                if doc_name not in samples_by_doc:
                                    samples_by_doc[doc_name] = []
                                if len(samples_by_doc[doc_name]) < 2:  # Max 2 samples per document
                                    samples_by_doc[doc_name].append(doc)
                            
                            # Display max 3 documents with 1-2 samples each
                            for doc_name, samples in list(samples_by_doc.items())[:3]:
                                st.write(f"**Document: {doc_name}**")
                                for i, doc in enumerate(samples):
                                    st.text(f"Page {doc.metadata.get('page', 'Unknown')}:")
                                    st.text(doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content)
                        
                        with st.expander("Vector Diagnostics", expanded=False):
                            st.write("### Vector Store Analysis")
                            st.text(f"Vector similarity search used: {getattr(retrieved_docs, 'vector_count', 0) > 0}")
                            
                            if hasattr(retrieved_docs, 'vector_count') and retrieved_docs.vector_count > 0:
                                st.text("Vector search is functioning correctly!")
                                st.text(f"Number of vector search results: {retrieved_docs.vector_count}")
                            else:
                                st.warning("Vector search returned no results. This may indicate:")
                                st.text("1. No embeddings in the database")
                                st.text("2. pgvector extension not properly configured")
                                st.text("3. Filter conditions excluding all results")
                                st.text("4. Issue with the langchain_pg_embedding table")
                    
                    # Use sources to create context
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    if len(context) > 0:
                        # Get answer from LLM using the context
                        answer, sources_text = get_answer(user_question, context, retrieved_docs)
                        logging.info("--- After get_answer ---")
                        
                        # Display the answer
                        st.markdown("### Answer")
                        # Process the final answer
                        with st.chat_message("assistant"):
                            if answer:
                                # Split at each citation marker and wrap in a message
                                parts = []
                                current_part = ""
                                sources = []
                                
                                for line in answer.split('\n'):
                                    if line.startswith('Source:'):
                                        if current_part.strip():
                                            parts.append(("text", current_part.strip()))
                                        # Collect sources separately instead of adding them directly
                                        sources.append(line)
                                        current_part = ""
                                    else:
                                        current_part += line + "\n"
                                
                                if current_part.strip():
                                    parts.append(("text", current_part.strip()))
                                
                                # Display all text parts
                                for part_type, part_content in parts:
                                    if part_type == "text":
                                        st.write(part_content)
                                
                                # Only show the top 5 most relevant sources
                                if sources:
                                    st.markdown(f"<p style='color: {APP_SECONDARY_COLOR}; font-weight: bold;'>Top Sources:</p>", unsafe_allow_html=True)
                                    # Display at most 5 sources
                                    for source in sources[:5]:
                                        st.markdown(f"> _{source}_")
                                    
                                    # Show a count of additional sources if there are more than 5
                                    if len(sources) > 5:
                                        st.markdown(f"<p style='color: {APP_SECONDARY_COLOR}; font-style: italic; font-size: 0.9em;'>Plus {len(sources) - 5} additional sources</p>", unsafe_allow_html=True)
                            else:
                                st.error("Failed to generate an answer. Please try again.")
                    else:
                        st.warning("No context found to generate an answer.")
                else:
                    logging.warning("No documents retrieved")
                    st.warning("No relevant documents found for your question. Please try rephrasing your question or select a different document.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    with tab2:
        # Display PDF viewer for selected document
        if st.session_state.selected_docs:
            try:
                # If only one document is selected, automatically show it
                if len(st.session_state.selected_docs) == 1:
                    st.session_state.pdf_viewer_doc_name = st.session_state.selected_docs[0]
                # If multiple documents are selected, show a dropdown to select which one to view
                else:
                    # Initialize pdf_viewer_doc_name to first selected doc if not set
                    if not st.session_state.pdf_viewer_doc_name or st.session_state.pdf_viewer_doc_name not in st.session_state.selected_docs:
                        st.session_state.pdf_viewer_doc_name = st.session_state.selected_docs[0]
                    
                    # Show dropdown to select which document to view
                    selected_doc = st.selectbox(
                        "Select document:",
                        options=st.session_state.selected_docs,
                        index=st.session_state.selected_docs.index(st.session_state.pdf_viewer_doc_name)
                    )
                    
                    # Update the document to view if changed
                    if selected_doc != st.session_state.pdf_viewer_doc_name:
                        st.session_state.pdf_viewer_doc_name = selected_doc
                
                # Get Dropbox client
                dbx = get_dropbox_client()
                if dbx:
                    # Construct path with leading slash for root folder
                    file_path = f"/{st.session_state.pdf_viewer_doc_name}"
                    logging.info(f"Attempting to download file for viewer: {file_path}")
                    
                    # Download file data
                    file_data = download_dropbox_file(file_path, dbx)
                    if file_data:
                        # Use the original streamlit-pdf-viewer but with a smaller size
                        from streamlit_pdf_viewer import pdf_viewer
                        
                        # Display PDF in a container with controlled size
                        with st.container():
                            # Display the PDF with a smaller width to ensure it fits on screen
                            pdf_viewer(file_data, width=600, height=500)
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