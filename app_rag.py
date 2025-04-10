"""
app_rag.py

This module provides advanced RAG (Retrieval Augmented Generation) techniques
for improving document search and retrieval based on best practices.
"""

import os
import psycopg2
import logging
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app_embeddings import embeddings
from app_database import save_document_to_db, get_db_connection

def implement_parent_document_retriever(pages_data, doc_name):
    """
    Implements the Parent Document Retriever pattern for better context preservation.
    Creates parent (large) and child (small) chunks with relationships maintained.
    
    Args:
        pages_data: List of tuples containing (page_number, page_content)
        doc_name: Name of the document
        
    Returns:
        bool: Success status
    """
    logging.info(f"Implementing Parent Document Retriever for {doc_name} with {len(pages_data)} pages")
    
    # Create parent chunks (larger) and child chunks (smaller)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    # Process parent documents
    parent_docs = []
    for page_num, page_text in pages_data:
        parent_chunks = parent_splitter.split_text(page_text)
        for chunk_idx, chunk in enumerate(parent_chunks):
            parent_id = f"{doc_name}_p{page_num}_c{chunk_idx}"
            parent_docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": doc_name, 
                    "page": page_num, 
                    "is_parent": True,
                    "parent_id": parent_id
                }
            ))
    
    # Process child documents with links to parents
    all_docs = []
    
    # First add all parents
    for parent_doc in parent_docs:
        all_docs.append(parent_doc)
    
    # Then create and add children linked to parents
    for parent_doc in parent_docs:
        parent_id = parent_doc.metadata["parent_id"]
        
        # Create and add children
        child_chunks = child_splitter.split_text(parent_doc.page_content)
        for child_idx, chunk in enumerate(child_chunks):
            child_id = f"{parent_id}_child{child_idx}"
            all_docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": doc_name, 
                    "page": parent_doc.metadata["page"],
                    "is_child": True,
                    "parent_id": parent_id,
                    "child_id": child_id
                }
            ))
    
    logging.info(f"Created {len(parent_docs)} parent chunks and {len(all_docs) - len(parent_docs)} child chunks")
    
    try:
        # Generate embeddings for all docs
        doc_contents = [doc.page_content for doc in all_docs]
        doc_metadata = [doc.metadata for doc in all_docs]
        
        embeddings_list = embeddings.embed_documents(doc_contents)
        
        # Save to database with parent-child relationships maintained
        success = save_document_to_db(doc_name, doc_contents, embeddings_list, doc_metadata)
        
        if success:
            logging.info(f"Successfully saved {len(all_docs)} chunks with parent-child relationships")
            return True
        else:
            logging.error("Failed to save document chunks to database")
            return False
            
    except Exception as e:
        logging.error(f"Error in implementing parent document retriever: {str(e)}")
        return False

def enhanced_vector_search(query, vectorstore, selected_doc):
    """
    Enhanced vector search using multiple search strategies
    
    Args:
        query (str): The search query
        vectorstore: The vector store to search in
        selected_doc (str): Document name to filter by
        
    Returns:
        list: List of Document objects with search results
    """
    logging.info(f"Running enhanced vector search for '{query}' in {selected_doc}")
    
    # Try different search strategies as in the article
    search_strategies = [
        {"search_type": "mmr", "lambda_mult": 0.5},  # Balanced relevance/diversity 
        {"search_type": "similarity"},               # Pure similarity
        {"search_type": "mmr", "lambda_mult": 0.8}   # More diverse results
    ]
    
    all_results = []
    
    # Create proper filter with metadata
    filter_dict = {
        "metadata": {
            "source": {
                "$eq": selected_doc
            }
        }
    }
    
    # Try each strategy
    for strategy in search_strategies:
        try:
            search_kwargs = strategy.copy()
            search_kwargs["filter"] = filter_dict
            
            logging.info(f"Trying search strategy: {strategy}")
            
            docs = vectorstore.similarity_search(
                query,
                k=30,  # Increased from typical 10
                **search_kwargs
            )
            
            logging.info(f"Strategy {strategy.get('search_type')} returned {len(docs)} results")
            
            # Add unique results
            for doc in docs:
                if doc.page_content not in [d.page_content for d in all_results]:
                    all_results.append(doc)
                    
            # If we have enough results, stop trying more strategies
            if len(all_results) >= 30:
                logging.info(f"Found {len(all_results)} results, stopping search")
                break
                
        except Exception as e:
            logging.error(f"Error with search strategy {strategy}: {str(e)}")
    
    logging.info(f"Enhanced vector search found total of {len(all_results)} unique results")
    return all_results

def simple_sql_search(query, doc_name, limit=30):
    """
    A simplified SQL search as a fallback method
    
    Args:
        query (str): The search query
        doc_name (str): Document name to search within
        limit (int): Maximum number of results
        
    Returns:
        list: List of Document objects with search results
    """
    logging.info(f"Running simple SQL search for '{query}' in {doc_name}")
    
    conn = get_db_connection()
    if not conn:
        logging.error("Failed to connect to database")
        return []
    
    try:
        cursor = conn.cursor()
        
        # First verify document exists
        cursor.execute("SELECT COUNT(*) FROM documents WHERE doc_name = %s", (doc_name,))
        count = cursor.fetchone()[0]
        
        if count == 0:
            logging.warning(f"Document '{doc_name}' not found in database")
            cursor.close()
            conn.close()
            return []
        
        logging.info(f"Found {count} chunks for document '{doc_name}'")
        
        # Extract keywords from query for simple search
        keywords = query.lower().split()
        if not keywords:
            # If no keywords, just return some chunks from the document
            sql = """
                SELECT id, doc_name, page_number, content, metadata 
                FROM documents 
                WHERE doc_name = %s
                ORDER BY page_number
                LIMIT %s;
            """
            cursor.execute(sql, (doc_name, limit))
        else:
            # Simple ILIKE search for any keyword
            conditions = []
            for keyword in keywords:
                if len(keyword) > 3:  # Skip very short words
                    conditions.append(f"content ILIKE '%{keyword}%'")
            
            if conditions:
                condition_sql = " OR ".join(conditions)
                sql = f"""
                    SELECT id, doc_name, page_number, content, metadata 
                    FROM documents 
                    WHERE doc_name = %s AND ({condition_sql})
                    ORDER BY page_number
                    LIMIT %s;
                """
                cursor.execute(sql, (doc_name, limit))
            else:
                # Fallback to simple document query
                sql = """
                    SELECT id, doc_name, page_number, content, metadata 
                    FROM documents 
                    WHERE doc_name = %s
                    ORDER BY page_number
                    LIMIT %s;
                """
                cursor.execute(sql, (doc_name, limit))
        
        rows = cursor.fetchall()
        logging.info(f"SQL query returned {len(rows)} results")
        
        documents = []
        for row in rows:
            try:
                doc_id, doc_name, page_num, content, meta = row
                
                # Ensure metadata is a dict
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except:
                        meta = {"source": doc_name, "page": page_num}
                elif meta is None:
                    meta = {"source": doc_name, "page": page_num}
                
                doc = Document(
                    page_content=content,
                    metadata=meta
                )
                documents.append(doc)
            except Exception as e:
                logging.error(f"Error processing row: {str(e)}")
        
        cursor.close()
        conn.close()
        return documents
        
    except Exception as e:
        logging.error(f"SQL search error: {str(e)}")
        if conn and not conn.closed:
            conn.close()
        return []

def rank_docs_by_relevance(docs, query):
    """
    Rank documents by keyword relevance to the query
    
    Args:
        docs: List of Document objects
        query: User query string
        
    Returns:
        list: Sorted list of Document objects by relevance
    """
    # Extract keywords from query, lowercase
    keywords = [k.lower() for k in query.split() if len(k) > 3]
    
    # Calculate relevance scores for each document
    scored_docs = []
    for doc in docs:
        score = 0
        content = doc.page_content.lower()
        
        # Score based on keyword presence
        for keyword in keywords:
            if keyword in content:
                # Count occurrences with word boundaries for more precision
                # This helps prioritize exact matches over partial matches
                import re
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', content))
                score += count * 2  # Give higher weight to exact keyword matches
                
                # If keyword appears in first sentence, give bonus points (likely more relevant)
                first_sentence = content.split('.')[0] if '.' in content else content
                if keyword in first_sentence:
                    score += 5
        
        # Give bonus for having multiple keywords
        keyword_matches = sum(1 for keyword in keywords if keyword in content)
        if keyword_matches > 1:
            score += keyword_matches * 3
            
        # Score exact phrase matches even higher
        if query.lower() in content:
            score += 10
            
        # Prioritize documents with relevant headings/beginnings
        if any(keyword in content[:100].lower() for keyword in keywords):
            score += 8
            
        # Add document with its score
        scored_docs.append((doc, score))
    
    # Sort by score in descending order
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Return sorted documents
    return [doc for doc, score in scored_docs]

def hybrid_retriever(query, vector_store, doc_name, limit=30):
    """
    Combines vector search and SQL search for better results
    
    Args:
        query (str): The search query
        vector_store: The vector store to search in
        doc_name (str): Document name to filter by
        limit (int): Maximum number of results
        
    Returns:
        list: List of Document objects with search results
    """
    logging.info(f"Running hybrid retrieval for '{query}' in {doc_name}")
    
    # Start with SQL search as reliable fallback
    sql_results = simple_sql_search(query, doc_name, limit=limit*2)  # Get more for ranking
    logging.info(f"SQL search found {len(sql_results)} results")
    
    vector_results = []
    # Try vector search to supplement only if SQL search found too few results
    if len(sql_results) < 10:
        try:
            vector_results = enhanced_vector_search(query, vector_store, doc_name)
            logging.info(f"Vector search found {len(vector_results)} results")
        except Exception as e:
            logging.error(f"Vector search failed: {str(e)}")
            # We already have SQL results as fallback
    
    # If we have zero results from both, return empty list
    if len(sql_results) == 0 and len(vector_results) == 0:
        logging.warning("No results found from either SQL or vector search")
        return []
        
    # Combine unique results
    all_results = []
    seen_contents = set()
    
    # First add all SQL results
    for doc in sql_results:
        if doc.page_content not in seen_contents:
            all_results.append(doc)
            seen_contents.add(doc.page_content)
    
    # Then add vector results that aren't duplicates
    for doc in vector_results:
        if doc.page_content not in seen_contents:
            all_results.append(doc)
            seen_contents.add(doc.page_content)
    
    logging.info(f"Combined results before ranking: {len(all_results)} chunks")
    
    # If we have at least some results, rank them
    if all_results:
        # Rank documents by keyword relevance
        ranked_results = rank_docs_by_relevance(all_results, query)
        logging.info(f"Ranked {len(ranked_results)} documents by keyword relevance")
        
        # Log top 3 documents for debugging
        for i, doc in enumerate(ranked_results[:3]):
            logging.info(f"Top result {i+1}: {doc.page_content[:100]}...")
        
        # Limit to requested number
        return ranked_results[:limit]
    else:
        # No results after combining
        return []

def fetch_parent_context(child_docs, doc_name):
    """
    For child documents, fetch their parent documents to provide additional context
    
    Args:
        child_docs: List of Document objects that are children
        doc_name (str): Document name
        
    Returns:
        list: List of Document objects with parents included
    """
    logging.info(f"Fetching parent context for {len(child_docs)} child documents")
    
    # Extract parent IDs from child docs
    parent_ids = []
    for doc in child_docs:
        if "parent_id" in doc.metadata and doc.metadata["parent_id"] not in parent_ids:
            parent_ids.append(doc.metadata["parent_id"])
    
    if not parent_ids:
        logging.info("No parent IDs found in child documents")
        return child_docs
    
    logging.info(f"Found {len(parent_ids)} unique parent IDs")
    
    # Fetch parent documents
    conn = get_db_connection()
    if not conn:
        logging.error("Failed to connect to database")
        return child_docs
    
    try:
        cursor = conn.cursor()
        
        parent_docs = []
        for parent_id in parent_ids:
            # Find parents with the given parent_id in metadata
            sql = """
                SELECT id, doc_name, page_number, content, metadata 
                FROM documents 
                WHERE doc_name = %s AND 
                      metadata->>'parent_id' = %s AND
                      metadata->>'is_parent' = 'true'
                LIMIT 1;
            """
            cursor.execute(sql, (doc_name, parent_id))
            row = cursor.fetchone()
            
            if row:
                doc_id, doc_name, page_num, content, meta = row
                
                # Ensure metadata is a dict
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except:
                        meta = {"source": doc_name, "page": page_num}
                elif meta is None:
                    meta = {"source": doc_name, "page": page_num}
                
                doc = Document(
                    page_content=content,
                    metadata=meta
                )
                parent_docs.append(doc)
        
        cursor.close()
        conn.close()
        
        logging.info(f"Found {len(parent_docs)} parent documents")
        
        # Combine child and parent documents
        all_docs = child_docs.copy()
        
        # Add parent docs that aren't already included
        seen_contents = [doc.page_content for doc in all_docs]
        for doc in parent_docs:
            if doc.page_content not in seen_contents:
                all_docs.append(doc)
                
        logging.info(f"Combined result has {len(all_docs)} documents")
        return all_docs
        
    except Exception as e:
        logging.error(f"Error fetching parent context: {str(e)}")
        if conn and not conn.closed:
            conn.close()
        return child_docs 