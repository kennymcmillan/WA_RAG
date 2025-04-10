"""
app_documents.py

This module handles PDF document processing.
Provides utilities for:
- Extracting text and metadata from PDF documents
- Splitting text into manageable chunks for embedding and retrieval
- Preserving document source and page information in metadata
- Optimized chunking for better retrieval performance
"""

import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import time
import json
import tempfile
from app_vector import create_vector_store
from app_dropbox import save_file_to_dropbox, is_dropbox_configured
from app_rag import implement_parent_document_retriever

# Add table extraction imports
import tabula
import pandas as pd

# Aspire Academy colors
ASPIRE_MAROON = "#7A0019"
ASPIRE_GOLD = "#FFD700"
ASPIRE_GRAY = "#F0F0F0"

def aspire_academy_css():
    aspire_css = f"""
    <style>
        .main .block-container {{
            padding-top: 1rem;
        }}
        .stApp {{
            background-color: white;
        }}
        .stSidebar {{
            background-color: {ASPIRE_GRAY};
        }}
        h1, h2, h3 {{
            color: {ASPIRE_MAROON};
        }}
        .stButton>button {{
            background-color: {ASPIRE_MAROON};
            color: white;
        }}
        .stButton>button:hover {{
            background-color: {ASPIRE_MAROON};
            color: {ASPIRE_GOLD};
        }}
        .source-citation {{
            font-size: 0.8em;
            color: gray;
            border-left: 3px solid {ASPIRE_MAROON};
            padding-left: 10px;
            margin-top: 10px;
        }}
    </style>
    """
    return aspire_css

# Get PDF text and metadata - Enhanced with table extraction
def get_pdf_pages(pdf_docs, extract_tables=True):
    pages_data = []
    for pdf in pdf_docs:
        doc_name = pdf.name
        try:
            # Create a temporary file to use with tabula if we need to extract tables
            tmp_file = None
            tmp_file_path = None
            
            if extract_tables:
                try:
                    # Create a temporary file for tabula to process
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                    tmp_file_path = tmp_file.name
                    
                    # Write the uploaded PDF to the temporary file
                    pdf.seek(0)
                    tmp_file.write(pdf.read())
                    tmp_file.close()
                    logging.info(f"Created temporary file for table extraction: {tmp_file_path}")
                except Exception as e:
                    logging.warning(f"Failed to create temporary file for table extraction: {str(e)}")
                    extract_tables = False
            
            # Reset file pointer for PyPDF2 to read
            pdf.seek(0)
            pdf_reader = PdfReader(pdf)
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                tables_data = []
                
                # Extract tables from the page if enabled
                if extract_tables and tmp_file_path:
                    try:
                        # Extract tables from the current page
                        page_tables = tabula.read_pdf(
                            tmp_file_path, 
                            pages=page_num+1,  # tabula uses 1-indexed pages
                            multiple_tables=True
                        )
                        
                        if page_tables:
                            logging.info(f"Found {len(page_tables)} tables on page {page_num+1} of {doc_name}")
                            
                            # Process each table
                            for i, table in enumerate(page_tables):
                                if not table.empty:
                                    # Convert table to various formats for flexible usage
                                    table_str = table.to_string(index=False)
                                    
                                    # Handle NaN values in the table data by converting to None (null in JSON)
                                    table_dict = table.replace({float('nan'): None}).to_dict(orient='records')
                                    
                                    # Clean up table HTML - replace NaN with empty strings
                                    table_html = table.fillna('').to_html(index=False)
                                    
                                    # Create a formatted string representation of the table
                                    table_text = f"\n[TABLE {i+1}]\n{table_str}\n[/TABLE {i+1}]\n"
                                    
                                    # Add table to the list of tables for this page
                                    tables_data.append({
                                        "table_id": i+1,
                                        "table_data": table_dict,
                                        "table_html": table_html,
                                        "table_text": table_text
                                    })
                                    
                                    # Append the table text to the page text for unified search
                                    page_text = f"{page_text}\n{table_text}"
                    except Exception as e:
                        logging.warning(f"Table extraction failed for page {page_num+1} of {doc_name}: {str(e)}")
                
                # Only add non-empty pages
                if page_text:  
                    # Clean up the text
                    page_text = re.sub(r'\s+', ' ', page_text)  # Normalize whitespace
                    page_text = page_text.strip()
                    
                    # Create metadata with table info if tables were found
                    metadata = {
                        "source": doc_name, 
                        "page": page_num + 1
                    }
                    
                    if tables_data:
                        metadata["has_tables"] = True
                        metadata["tables_count"] = len(tables_data)
                        metadata["tables"] = tables_data
                    else:
                        metadata["has_tables"] = False
                    
                    pages_data.append({
                        "text": page_text,
                        "metadata": metadata
                    })
            
            # Clean up temporary file if we created one
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                    logging.info(f"Removed temporary file: {tmp_file_path}")
                except Exception as e:
                    logging.warning(f"Failed to remove temporary file {tmp_file_path}: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error processing {doc_name}: {str(e)}")
            logging.error(f"Error processing {doc_name}: {str(e)}", exc_info=True)
    
    return pages_data

# Process PDFs in parallel for better performance
def process_pdfs_in_parallel(pdf_docs, max_workers=4):
    """
    Process multiple PDF documents in parallel to improve performance.
    
    Args:
        pdf_docs: List of PDF file objects
        max_workers: Maximum number of worker threads
        
    Returns:
        List of page data dictionaries
    """
    if not pdf_docs:
        return []
        
    # Process PDFs in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process each PDF in a separate thread
        results = list(executor.map(
            lambda pdf: get_pdf_pages([pdf]), 
            pdf_docs
        ))
    
    # Flatten results
    pages_data = []
    for result in results:
        pages_data.extend(result)
        
    return pages_data

# Split text into chunks using LangChain Documents - Optimized for retrieval
def get_text_chunks(pages_data, chunk_size=500, chunk_overlap=50):
    """
    Split text into optimized chunks for better retrieval performance.
    
    Args:
        pages_data: List of page data dictionaries
        chunk_size: Size of each chunk (smaller chunks are better for retrieval)
        chunk_overlap: Overlap between chunks to maintain context
        
    Returns:
        List of LangChain Document objects with page content and metadata
    """
    if not pages_data:
        return []
        
    # Check if we need to adjust chunk size based on content length
    total_text_length = sum(len(page["text"]) for page in pages_data)
    avg_text_length = total_text_length / len(pages_data)
    
    # Adjust chunk size based on average page length
    if avg_text_length < 1000:
        # For short texts, use smaller chunks
        adjusted_chunk_size = min(300, chunk_size)
        adjusted_overlap = min(30, chunk_overlap)
        logging.info(f"Using smaller chunks ({adjusted_chunk_size}/{adjusted_overlap}) for short content")
    elif avg_text_length > 5000:
        # For very long texts, use larger chunks
        adjusted_chunk_size = max(1000, chunk_size)
        adjusted_overlap = max(100, chunk_overlap)
        logging.info(f"Using larger chunks ({adjusted_chunk_size}/{adjusted_overlap}) for long content")
    else:
        # Use default values
        adjusted_chunk_size = chunk_size
        adjusted_overlap = chunk_overlap
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=adjusted_chunk_size,
        chunk_overlap=adjusted_overlap,
        length_function=len,
        # Use more nuanced separators for better semantic boundaries
        separators=[
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            "! ",    # Exclamations
            "? ",    # Questions
            "; ",    # Semi-colons
            ", ",    # Commas
            " ",     # Words
            ""       # Characters
        ]
    )
    
    # Prepare texts and metadatas for create_documents
    texts = [page["text"] for page in pages_data]
    metadatas = [page["metadata"] for page in pages_data]

    # create_documents handles splitting text while preserving metadata links
    documents = text_splitter.create_documents(texts, metadatas=metadatas)
    
    logging.info(f"Created {len(documents)} chunks from {len(pages_data)} pages")
    return documents

def process_pdf_file(file, save_to_dropbox=False):
    """Process a PDF file to extract text, create embeddings, and save to vector store.
    
    Args:
        file: The PDF file to process (file-like object)
        save_to_dropbox: Whether to save the file to Dropbox
        
    Returns:
        bool: True if processing was successful
    """
    try:
        logging.info(f"Processing PDF file: {file.name}")
        
        # Get PDF text by page
        pages_data = get_pdf_pages([file])
        
        if not pages_data:
            logging.error(f"No text extracted from {file.name}")
            return False
        
        logging.info(f"Extracted {len(pages_data)} pages from {file.name}")
        
        # Create document chunks with the default chunking strategy
        documents = get_text_chunks(pages_data)
        
        if not documents:
            logging.error(f"Failed to create text chunks from {file.name}")
            return False
            
        logging.info(f"Created {len(documents)} chunks from {file.name}")
        
        # Process documents with Parent Document Retriever pattern
        success = implement_parent_document_retriever(documents, file.name)
        
        if not success:
            logging.error(f"Failed to implement Parent Document Retriever for {file.name}")
            # Fall back to traditional vector store creation
            if not create_vector_store(documents, file.name):
                logging.error(f"Failed to create vector store for {file.name}")
                return False
        
        # Save to Dropbox if requested
        if save_to_dropbox and is_dropbox_configured():
            try:
                # Reset file pointer to beginning
                file.seek(0)
                
                # Save to Dropbox
                dropbox_path = f"/{file.name}"  # Root folder
                save_file_to_dropbox(file, dropbox_path)
                logging.info(f"Saved {file.name} to Dropbox")
            except Exception as e:
                logging.error(f"Error saving to Dropbox: {str(e)}")
                # Don't fail the whole process if just Dropbox save fails
        
        logging.info(f"Successfully processed {file.name}")
        return True
    except Exception as e:
        logging.error(f"Error processing file {file.name}: {str(e)}")
        return False
