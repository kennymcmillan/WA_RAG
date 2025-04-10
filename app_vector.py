"""
app_vector.py

This module manages vector operations for document retrieval.
Provides:
- Vector store creation functionality
- Conversion of document content to vector embeddings
- Integration with the database for storing vectorized documents
- Metadata preparation for document retrieval
"""

import streamlit as st
import logging
from app_database import initialize_pgvector, save_document_to_db, get_connection_string
from app_embeddings import embeddings

# Import PGVector for custom extension
from langchain_community.vectorstores.pgvector import PGVector
from sqlalchemy import text, or_, and_
import json

# Define a custom PGVector class that uses our custom_metadata_match function
# This bypasses the jsonb_path_match function that's causing errors
class CustomPGVector(PGVector):
    """
    A custom PGVector implementation that uses our custom_metadata_match function
    instead of the jsonb_path_match function.
    """
    
    def _and(self, clauses):
        """
        Add the missing _and method required by the filter implementation.
        """
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return and_(*clauses)
            
    def _or(self, clauses):
        """
        Add the missing _or method required by the filter implementation.
        """
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return or_(*clauses)
    
    def _handle_metadata_filter(self, field, operator, value):
        """
        Override the metadata filtering to use our custom_metadata_match function.
        This replaces the jsonb_path_match function that's causing errors.
        """
        if operator == "$eq":
            # Fix ambiguous column reference by specifying the table
            return text("custom_metadata_match(langchain_pg_embedding.cmetadata, :field, :value)").bindparams(
                field=field, value=str(value)
            )
        # Handle other operators as needed
        elif operator == "$in":
            # For $in operator, check if value is in a list
            clauses = []
            for item in value:
                clauses.append(
                    text("custom_metadata_match(langchain_pg_embedding.cmetadata, :field, :value)").bindparams(
                        field=field, value=str(item)
                    )
                )
            return self._or(clauses)
        
        # Fall back to original implementation for other operators
        return super()._handle_metadata_filter(field, operator, value)
    
    def _create_filter_clause(self, filters):
        """
        Modified filter clause creation that handles our custom metadata filtering.
        """
        if not filters:
            return None
            
        # Special handling for our fixed format filter
        if "metadata" in filters and isinstance(filters["metadata"], dict):
            metadata_filters = filters["metadata"]
            clauses = []
            
            for field, operators in metadata_filters.items():
                if isinstance(operators, dict):
                    for operator, value in operators.items():
                        clauses.append(self._handle_metadata_filter(field, operator, value))
                else:
                    # Direct value match (implicit $eq)
                    clauses.append(self._handle_metadata_filter(field, "$eq", operators))
                    
            return self._and(clauses) if clauses else None
        
        # Handle old-style filters
        return super()._create_filter_clause(filters)

# Create vector store using LangChain Documents
def create_vector_store(documents, doc_name):
    """
    Creates a vector store using LangChain Documents.
    Saves documents both to documents table and to langchain_pg_embedding
    for proper vector search.
    
    Args:
        documents: List of Document objects
        doc_name: Name of the document
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize database if needed
        if not initialize_pgvector():
            logging.error("Failed to initialize PGVector database")
            return None

        logging.info(f"Creating vector store for {doc_name} with {len(documents)} chunks")
        
        # Save to our documents table for regular SQL operations
        # Extract content and metadata
        chunks = [doc.page_content for doc in documents]
        metadata_list = [doc.metadata for doc in documents]

        # Generate embeddings for chunks
        embeddings_list = embeddings.embed_documents(chunks)

        # Save to documents table
        if not save_document_to_db(doc_name, chunks, embeddings_list, metadata_list):
            logging.error(f"Failed to save documents to documents table")
            return None
            
        # Now also save to langchain_pg_embedding table using PGVector directly
        try:
            logging.info(f"Saving embeddings to langchain_pg_embedding table...")
            connection_string = get_connection_string()
            
            if not connection_string:
                logging.error("Could not get database connection string")
                return None
                
            # Use our custom PGVector implementation
            vector_store = CustomPGVector.from_documents(
                documents=documents,
                embedding=embeddings,
                connection_string=connection_string,
                collection_name="documents",  # Corresponds to table prefix
                use_jsonb=True
            )
            
            logging.info(f"Successfully created vector store and saved embeddings for {doc_name}")
            return True
        except Exception as e:
            logging.error(f"Error saving to vector store: {str(e)}")
            return None
    except Exception as e:
        logging.error(f"Error creating vector store for {doc_name}: {str(e)}")
        return None

def iterative_document_search(question, vectorstore, max_iterations=5, initial_k=50, filter_dict=None):
    """
    Performs an iterative search over documents using various strategies to find the most relevant chunks.
    
    Args:
        question (str): The user's question or query
        vectorstore: The vector store to search in
        max_iterations (int): Maximum number of search iterations
        initial_k (int): Initial number of results to retrieve (increased to 50)
        filter_dict (dict): Optional filter to limit search to specific documents
                           Format should be: {"metadata": {"source": {"$eq": "document_name.pdf"}}}
        
    Returns:
        DocumentCollection: Collection of relevant document chunks with metrics
    """
    # Import DocumentCollection here to avoid circular imports
    from app_rag import DocumentCollection
    
    # Initialize a DocumentCollection to store results
    all_relevant_chunks = DocumentCollection()
    current_k = initial_k
    
    # Log filter information for debugging
    if filter_dict:
        logging.info(f"Searching with filter: {filter_dict}")
        # Debug filter structure
        if isinstance(filter_dict, dict) and "metadata" in filter_dict:
            metadata_filter = filter_dict.get("metadata", {})
            if isinstance(metadata_filter, dict):
                for field, condition in metadata_filter.items():
                    logging.info(f"Filter field: {field}, condition: {condition}")
    else:
        logging.warning("No filter provided for search - will retrieve from all documents")
    
    # Debug vectorstore
    logging.info(f"Vector store type: {type(vectorstore).__name__}")
    
    # Try different search strategies
    search_strategies = [
        {"search_type": "mmr", "lambda_mult": 0.5},  # Balanced approach
        {"search_type": "similarity"},  # Pure similarity
        {"search_type": "mmr", "lambda_mult": 0.8}  # More diverse results
    ]
    
    logging.info(f"Starting iterative search with question: '{question}'")
    
    # DEBUGGING: Try a direct unfiltered search first to verify vectorstore works at all
    try:
        logging.info("Attempting direct similarity search without filters (debug)")
        direct_results = vectorstore.similarity_search(
            question,
            k=10
        )
        logging.info(f"Direct unfiltered search found {len(direct_results)} documents")
        if direct_results and len(direct_results) > 0:
            first_doc = direct_results[0]
            logging.info(f"First unfiltered result sample: {first_doc.page_content[:100]}...")
            logging.info(f"First unfiltered result metadata: {first_doc.metadata}")
    except Exception as e:
        logging.error(f"Direct unfiltered search failed: {str(e)}")
    
    for strategy_index, strategy in enumerate(search_strategies):
        strategy_name = strategy.get("search_type", "unknown")
        logging.info(f"Trying search strategy {strategy_index+1}/{len(search_strategies)}: {strategy_name}")
        
        current_k = initial_k
        for iteration in range(max_iterations):
            logging.info(f"Iteration {iteration+1}/{max_iterations} with k={current_k}")
            
            # Search with current parameters and filter
            search_kwargs = strategy.copy()
            
            # Correctly pass the filter if provided
            if filter_dict:
                try:
                    search_kwargs["filter"] = filter_dict
                    logging.info(f"Making PGVector search with: strategy={strategy_name}, k={current_k}, filter={filter_dict}")
                    
                    # Attempt search with filter
                    docs = vectorstore.similarity_search(
                        question,
                        k=current_k,
                        **search_kwargs
                    )
                    
                    logging.info(f"Search returned {len(docs)} documents")
                    
                    # Process results (same as below)
                    # Log sample of returned docs
                    if docs and len(docs) > 0:
                        first_doc = docs[0]
                        logging.info(f"First result sample: {first_doc.page_content[:100]}...")
                        logging.info(f"First result metadata: {first_doc.metadata}")
                    else:
                        logging.warning(f"Search returned empty results with filter {filter_dict}")
                    
                    # Add new chunks to our collection
                    new_chunks_added = 0
                    for doc in docs:
                        # Check if this is a unique document
                        is_unique = True
                        for existing_doc in all_relevant_chunks:
                            if doc.page_content == existing_doc.page_content:
                                is_unique = False
                                break
                        
                        if doc.page_content is not None and is_unique:
                            all_relevant_chunks.append(doc)
                            new_chunks_added += 1
                    
                    logging.info(f"Added {new_chunks_added} new unique chunks in this iteration")
                    logging.info(f"Total unique chunks found so far: {len(all_relevant_chunks)}")
                    
                    # Increase search scope for next iteration
                    current_k *= 2
                    
                    # If we've found enough relevant chunks, we can stop
                    if len(all_relevant_chunks) >= 30:  # Increased from 15 to get more content
                        logging.info(f"Found enough chunks ({len(all_relevant_chunks)}), stopping search")
                        break
                        
                except Exception as e:
                    logging.error(f"Error during search iteration with filter: {str(e)}")
                    # Try without filter instead of failing entirely
                    logging.info("Attempting search without filter as fallback...")
                    search_kwargs.pop("filter", None)
            
            # If we don't have a filter or the filter search failed, do unfiltered search
            if not filter_dict or "filter" not in search_kwargs:
                # Debug the exact query being made
                logging.info(f"Making PGVector search with: strategy={strategy_name}, k={current_k}, no filter")
                    
                try:
                    # Perform search without filter - we filter results programmatically later
                    docs = vectorstore.similarity_search(
                        question,
                        k=current_k,
                        **search_kwargs
                    )
                    
                    logging.info(f"Search returned {len(docs)} documents")
                    
                    # Log sample of returned docs
                    if docs and len(docs) > 0:
                        first_doc = docs[0]
                        logging.info(f"First result sample: {first_doc.page_content[:100]}...")
                        logging.info(f"First result metadata: {first_doc.metadata}")
                    else:
                        logging.warning("Search returned empty results even without filter")
                    
                    # Add new chunks to our collection
                    new_chunks_added = 0
                    for doc in docs:
                        # Check if this is a unique document
                        is_unique = True
                        for existing_doc in all_relevant_chunks:
                            if doc.page_content == existing_doc.page_content:
                                is_unique = False
                                break
                        
                        if doc.page_content is not None and is_unique:
                            all_relevant_chunks.append(doc)
                            new_chunks_added += 1
                    
                    logging.info(f"Added {new_chunks_added} new unique chunks in this iteration")
                    logging.info(f"Total unique chunks found so far: {len(all_relevant_chunks)}")
                    
                    # Increase search scope for next iteration
                    current_k *= 2
                    
                    # If we've found enough relevant chunks, we can stop
                    if len(all_relevant_chunks) >= 30:  # Increased from 15 to get more content
                        logging.info(f"Found enough chunks ({len(all_relevant_chunks)}), stopping search")
                        break
                        
                except Exception as e:
                    logging.error(f"Error during search iteration: {str(e)}")
                    # Try to continue with the next strategy instead of failing entirely
    
    logging.info(f"Iterative search completed. Found {len(all_relevant_chunks)} relevant chunks")
    
    # Update metrics - all are vector results
    all_relevant_chunks.vector_count = len(all_relevant_chunks) 
    return all_relevant_chunks 