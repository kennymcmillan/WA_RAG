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
from app_vector import create_vector_store
from app_dropbox import save_file_to_dropbox, is_dropbox_configured
from app_rag import implement_parent_document_retriever

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

# Get PDF text and metadata - Refactored
def get_pdf_pages(pdf_docs):
    pages_data = []
    for pdf in pdf_docs:
        doc_name = pdf.name
        try:
            pdf_reader = PdfReader(pdf)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:  # Only add non-empty pages
                    # Clean up the text
                    page_text = re.sub(r'\s+', ' ', page_text)  # Normalize whitespace
                    page_text = page_text.strip()
                    pages_data.append({
                        "text": page_text,
                        "metadata": {"source": doc_name, "page": page_num + 1}
                    })
        except Exception as e:
            st.error(f"Error processing {doc_name}: {str(e)}")
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
        
        # Use advanced RAG with parent-child document approach
        result = implement_parent_document_retriever(pages_data, file.name)
        
        # Save to Dropbox if requested
        if save_to_dropbox and result:
            try:
                # Reset file pointer to beginning
                file.seek(0)
                file_bytes = file.read()
                
                # Get Dropbox client
                dbx = get_dropbox_client()
                if dbx:
                    upload_result = upload_to_dropbox(file_bytes, file.name, "/", dbx)
                    if upload_result:
                        logging.info(f"Successfully uploaded {file.name} to Dropbox")
                    else:
                        logging.error(f"Failed to upload {file.name} to Dropbox")
            except Exception as e:
                logging.error(f"Error uploading to Dropbox: {str(e)}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error processing PDF file: {str(e)}")
        return False
