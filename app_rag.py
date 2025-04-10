"""
app_rag.py

This module implements advanced RAG techniques based on the Parent Document Retriever pattern.
It creates parent documents (large chunks) and child documents (small chunks) to maintain 
context while enabling precise retrieval.

Based on: https://pub.towardsai.net/introduction-to-retrieval-augmented-generation-rag-using-langchain-and-lamaindex-bd0047628e2a
"""

import logging
import os
import streamlit as st
from typing import List, Dict, Any, Optional
import psycopg2
from sqlalchemy import text
from langchain_core.documents import Document
from app_embeddings import embeddings
from app_database import get_db_connection, get_connection_string
from app_vector import CustomPGVector
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for chunk sizes
PARENT_CHUNK_SIZE = 2000  # Larger chunks for context
CHILD_CHUNK_SIZE = 400    # Smaller chunks for precise retrieval
CHUNK_OVERLAP = 50        # Overlap to maintain context between chunks

# Add this class near the top of the file, after imports
class DocumentCollection:
    """
    A class to hold documents and track search metrics.
    """
    def __init__(self, documents=None):
        self.documents = documents or []
        self.sql_count = 0
        self.vector_count = 0
        self.fallback_count = 0
        self.parent_count = 0
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            result = DocumentCollection(self.documents[idx])
            result.sql_count = self.sql_count
            result.vector_count = self.vector_count
            result.fallback_count = self.fallback_count
            result.parent_count = self.parent_count
            return result
        return self.documents[idx]
    
    def __iter__(self):
        return iter(self.documents)
    
    def append(self, item):
        self.documents.append(item)
    
    def extend(self, items):
        self.documents.extend(items)
    
    def copy(self):
        result = DocumentCollection(self.documents.copy())
        result.sql_count = self.sql_count
        result.vector_count = self.vector_count
        result.fallback_count = self.fallback_count
        result.parent_count = self.parent_count
        return result

def implement_parent_document_retriever(documents: List[Document], doc_name: str) -> bool:
    """
    Implements the Parent Document Retriever pattern for the given documents.
    
    Args:
        documents: List of documents to process
        doc_name: Name of the document being processed
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Implementing Parent Document Retriever for {doc_name}")
        
        # Create parent and child splitters with appropriate chunk sizes
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Storage layer for parent documents
        store = InMemoryStore()
        
        # Create vector store for the child documents
        connection_string = get_connection_string()
        if not connection_string:
            logger.error("Failed to get database connection string")
            return False
            
        # Create vector store 
        vectorstore = CustomPGVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name="documents",
            use_jsonb=True
        )
        
        # Create the parent document retriever
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )
        
        # Add documents to the retriever
        logger.info(f"Adding {len(documents)} documents to the retriever")
        retriever.add_documents(documents)
        
        logger.info(f"Successfully implemented Parent Document Retriever for {doc_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error implementing Parent Document Retriever: {str(e)}")
        return False

def simple_sql_search(query: str, doc_name: str, limit: int = 30) -> List[Document]:
    """
    Performs a simple SQL search for documents containing keywords from the query.
    This is a fallback for when vector search fails.
    
    Args:
        query: User query
        doc_name: Document name to filter by
        limit: Maximum number of results
        
    Returns:
        List of matching documents
    """
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return []
        
    try:
        cursor = conn.cursor()
        
        # Extract keywords from query (words longer than 2 characters)
        # Using shorter words to increase chances of finding matches
        keywords = [word.lower() for word in query.split() if len(word) > 2]
        logger.info(f"SQL search keywords (length > 2): {keywords}")
        
        # Also try with longer keywords for more precision
        longer_keywords = [word.lower() for word in query.split() if len(word) > 3]
        logger.info(f"SQL search longer keywords (length > 3): {longer_keywords}")
        
        # Generate trigrams from query for more fuzzy matching
        def generate_trigrams(text):
            words = text.lower().split()
            trigrams = []
            for word in words:
                if len(word) > 5:  # Only for longer words
                    for i in range(len(word) - 2):
                        trigrams.append(word[i:i+3])
            return trigrams
            
        trigrams = generate_trigrams(query)
        logger.info(f"Generated trigrams for fuzzy matching: {trigrams}")
        
        if not keywords and not trigrams:
            # If no keywords found, return documents from the specified doc
            sql = """
                SELECT id, doc_name, page_number, content, metadata 
                FROM documents 
                WHERE doc_name = %s
                LIMIT %s;
            """
            cursor.execute(sql, (doc_name, limit))
        else:
            # Try to find exact matches first
            exact_conditions = []
            
            # For exact phrase match
            if len(query.split()) > 1:
                # Try the whole query as a phrase
                exact_conditions.append(f"LOWER(content) LIKE '%{query.lower()}%'")
            
            # Build query conditions for each keyword
            keyword_conditions = []
            for keyword in keywords:
                # More precise match with word boundaries where possible
                if len(keyword) > 4:
                    keyword_conditions.append(f"LOWER(content) ~ '\\m{keyword}\\M'")
                else:
                    keyword_conditions.append(f"LOWER(content) LIKE '%{keyword}%'")
            
            # Add trigram conditions for fuzzy matching
            trigram_conditions = []
            for trigram in trigrams:
                trigram_conditions.append(f"LOWER(content) LIKE '%{trigram}%'")
            
            # Build where clause with weighted prioritization
            where_parts = []
            
            # Exact matches get highest priority
            if exact_conditions:
                where_parts.append(f"({' OR '.join(exact_conditions)})")
            
            # Keyword matches get medium priority
            if keyword_conditions:
                where_parts.append(f"({' OR '.join(keyword_conditions)})")
                
            # Trigram matches get lowest priority
            if trigram_conditions:
                where_parts.append(f"({' OR '.join(trigram_conditions)})")
                
            # Combine with OR between the groups
            combined_where = " OR ".join(where_parts)
            
            # Execute query with document filter
            sql = f"""
                SELECT id, doc_name, page_number, content, metadata 
                FROM documents 
                WHERE doc_name = %s AND ({combined_where})
                LIMIT %s;
            """
            
            logger.info(f"SQL where clause: {combined_where}")
            cursor.execute(sql, (doc_name, limit))
            
        # Process results
        rows = cursor.fetchall()
        logger.info(f"SQL search found {len(rows)} results")
        
        documents = []
        for row in rows:
            try:
                # Add robust error handling for row unpacking
                if row is None or len(row) < 4:
                    logger.warning(f"Invalid row format: {row}")
                    continue
                    
                doc_id, row_doc_name, page_num, content, meta = row
                
                # Skip empty content
                if not content or len(content.strip()) == 0:
                    continue
                
                # Build metadata
                metadata = {}
                if meta and isinstance(meta, dict):
                    metadata = meta
                else:
                    metadata = {"source": row_doc_name, "page": page_num}
                
                # Create document
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
            except Exception as e:
                logger.error(f"Error processing row in SQL search: {str(e)}")
        
        logger.info(f"Returning {len(documents)} documents from SQL search")
        cursor.close()
        conn.close()
        return documents
        
    except Exception as e:
        logger.error(f"Error in SQL search: {str(e)}")
        if conn and not conn.closed:
            conn.close()
        return []

def enhanced_vector_search(query: str, vector_store, doc_name: str) -> List[Document]:
    """
    Performs an enhanced vector search using multiple strategies.
    
    Args:
        query: User query
        vector_store: Vector store to search
        doc_name: Document name to filter by
        
    Returns:
        List of relevant documents
    """
    try:
        logger.info(f"Enhanced vector search for '{query}' in {doc_name}")
        
        # Filter for the specific document
        filter_dict = {
            "metadata": {
                "source": {
                    "$eq": doc_name
                }
            }
        }
        
        # Try different search strategies
        all_results = []
        seen_content = set()
        
        # Strategy 1: MMR with balanced diversity (0.5)
        try:
            results = vector_store.max_marginal_relevance_search(
                query,
                k=20,
                fetch_k=50,
                lambda_mult=0.5,
                filter=filter_dict
            )
            
            # Add unique results
            for doc in results:
                if doc.page_content not in seen_content:
                    all_results.append(doc)
                    seen_content.add(doc.page_content)
                    
            logger.info(f"MMR (0.5) search found {len(results)} results")
        except Exception as e:
            logger.error(f"Error with MMR (0.5) search: {str(e)}")
        
        # Strategy 2: Pure similarity search
        try:
            results = vector_store.similarity_search(
                query,
                k=20,
                filter=filter_dict
            )
            
            # Add unique results
            for doc in results:
                if doc.page_content not in seen_content:
                    all_results.append(doc)
                    seen_content.add(doc.page_content)
                    
            logger.info(f"Similarity search found {len(results)} results")
        except Exception as e:
            logger.error(f"Error with similarity search: {str(e)}")
        
        # Strategy 3: MMR with higher diversity (0.8)
        try:
            results = vector_store.max_marginal_relevance_search(
                query,
                k=20,
                fetch_k=50,
                lambda_mult=0.8,
                filter=filter_dict
            )
            
            # Add unique results
            for doc in results:
                if doc.page_content not in seen_content:
                    all_results.append(doc)
                    seen_content.add(doc.page_content)
                    
            logger.info(f"MMR (0.8) search found {len(results)} results")
        except Exception as e:
            logger.error(f"Error with MMR (0.8) search: {str(e)}")
        
        logger.info(f"Enhanced vector search found {len(all_results)} total results")
        return all_results
        
    except Exception as e:
        logger.error(f"Enhanced vector search error: {str(e)}")
        return []

def hybrid_retriever(query: str, vector_store, doc_name: str, limit: int = 30) -> DocumentCollection:
    """
    Combines vector search and SQL search for better results
    
    Args:
        query (str): The search query
        vector_store: The vector store to search in
        doc_name (str): Document name to filter by
        limit (int): Maximum number of results
        
    Returns:
        DocumentCollection: Collection of Document objects with search results and metrics
    """
    logger.info(f"Running hybrid retrieval for '{query}' in {doc_name}")
    
    # Validate inputs
    if not query or not query.strip():
        logger.warning("Empty query provided to hybrid_retriever")
        return DocumentCollection()
        
    if not doc_name:
        logger.warning("No document name provided to hybrid_retriever")
        return DocumentCollection()
    
    # Try both search methods in parallel for better results
    sql_results = []
    vector_results = []
    fallback_used = False
    
    # Natural language SQL search for semantically relevant results
    try:
        logger.info("Running natural language SQL search...")
        sql_results = natural_language_sql_search(query, doc_name, limit=limit)
        logger.info(f"Natural language SQL search found {len(sql_results)} results")
        
        # If natural language SQL search fails, try simple SQL search as fallback
        if not sql_results:
            logger.info("Natural language SQL search found no results, trying simple SQL search...")
            sql_results = simple_sql_search(query, doc_name, limit=limit)
            logger.info(f"Simple SQL search found {len(sql_results)} results")
    except Exception as e:
        logger.error(f"SQL search failed: {str(e)}")
        # Try simple SQL search as fallback
        try:
            sql_results = simple_sql_search(query, doc_name, limit=limit)
            logger.info(f"Fallback simple SQL search found {len(sql_results)} results")
        except Exception as inner_e:
            logger.error(f"Fallback SQL search also failed: {str(inner_e)}")
    
    # Vector search for semantic similarity
    try:
        logger.info("Running vector search...")
        # Create filter for vector search
        filter_dict = {
            "metadata": {
                "source": {
                    "$eq": doc_name
                }
            }
        }
        
        # Check if vector_store is properly initialized
        if vector_store is None:
            logger.error("Vector store is None - cannot perform vector search")
        else:
            # Use enhanced vector search for better results
            vector_results = enhanced_vector_search(query, vector_store, doc_name)
            logger.info(f"Vector search found {len(vector_results)} results")
    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
    
    # Emergency fallback if both methods fail
    if len(sql_results) == 0 and len(vector_results) == 0:
        logger.warning("No results from either method, using emergency fallback...")
        try:
            # Emergency fallback - just get some documents from this doc
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                # Super simple query to get any content from this document
                fallback_sql = """
                    SELECT id, doc_name, page_number, content
                    FROM documents 
                    WHERE doc_name = %s
                    LIMIT 10;
                """
                cursor.execute(fallback_sql, (doc_name,))
                rows = cursor.fetchall()
                
                for row in rows:
                    if row and len(row) >= 4 and row[3]:
                        # Extract fields safely
                        row_doc_name = row[1] if len(row) > 1 else doc_name
                        page_num = row[2] if len(row) > 2 else 0
                        content = row[3]
                        
                        # Add document
                        sql_results.append(Document(
                            page_content=content,
                            metadata={"source": row_doc_name, "page": page_num}
                        ))
                
                cursor.close()
                conn.close()
                fallback_used = True
                logger.info(f"Emergency fallback found {len(sql_results)} results")
            else:
                logger.error("Failed to connect to database for emergency fallback")
        except Exception as e:
            logger.error(f"Emergency fallback failed: {str(e)}")
            
        # If still no results, return empty collection
        if len(sql_results) == 0:
            logger.warning("All retrieval methods failed, returning empty results")
            return DocumentCollection()
    
    # Combine unique results
    result_collection = DocumentCollection()
    seen_contents = set()
    
    # First add all SQL results
    for doc in sql_results:
        if doc.page_content and doc.page_content not in seen_contents:
            result_collection.append(doc)
            seen_contents.add(doc.page_content)
    
    # Then add vector results that aren't duplicates
    for doc in vector_results:
        if doc.page_content and doc.page_content not in seen_contents:
            result_collection.append(doc)
            seen_contents.add(doc.page_content)
    
    # Track metrics
    result_collection.sql_count = len(sql_results)
    result_collection.vector_count = len(vector_results)
    result_collection.fallback_count = len(sql_results) if fallback_used else 0
    
    logger.info(f"Combined results before ranking: {len(result_collection)} chunks")
    logger.info(f"Metrics - SQL: {result_collection.sql_count}, Vector: {result_collection.vector_count}, Fallback: {result_collection.fallback_count}")
    
    # If we have at least some results, rank them
    if len(result_collection) > 0:
        try:
            # Rank documents by keyword relevance
            ranked_docs = rank_docs_by_relevance(result_collection, query)
            logger.info(f"Ranked {len(ranked_docs)} documents by keyword relevance")
            
            # Create a new collection with the ranked docs but preserve metrics
            ranked_collection = DocumentCollection(ranked_docs[:limit])
            ranked_collection.sql_count = result_collection.sql_count
            ranked_collection.vector_count = result_collection.vector_count
            ranked_collection.fallback_count = result_collection.fallback_count
            
            # Log top 3 documents for debugging
            for i, doc in enumerate(ranked_collection[:3] if len(ranked_collection) >= 3 else ranked_collection):
                logger.info(f"Top result {i+1}: {doc.page_content[:100]}...")
            
            # Return the limited results
            return ranked_collection
        except Exception as e:
            logger.error(f"Error during document ranking: {str(e)}")
            # Fall back to unranked results but keep metrics
            limited_collection = result_collection[:limit]
            return limited_collection
    else:
        # No results after combining
        return DocumentCollection()

def rank_docs_by_relevance(docs, query):
    """
    Rank documents by keyword relevance to the query
    
    Args:
        docs: List of Document objects
        query: User query string
        
    Returns:
        list: Sorted list of Document objects by relevance
    """
    # If no docs or query, return as is
    if not docs or not query:
        return docs
        
    # Extract keywords from query, lowercase
    keywords = [k.lower() for k in query.split() if len(k) > 3]
    
    # If no meaningful keywords, return docs as is
    if not keywords:
        return docs
    
    # Calculate relevance scores for each document
    scored_docs = []
    for doc in docs:
        try:
            score = 0
            # Skip if content is empty
            if not doc.page_content:
                continue
                
            content = doc.page_content.lower()
            
            # Score based on keyword presence
            for keyword in keywords:
                if keyword in content:
                    # Count occurrences with word boundaries for more precision
                    # This helps prioritize exact matches over partial matches
                    try:
                        import re
                        count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', content))
                        score += count * 2  # Give higher weight to exact keyword matches
                    except:
                        # Fallback if regex fails
                        score += content.count(keyword) * 2
                    
                    # If keyword appears in first sentence, give bonus points (likely more relevant)
                    try:
                        first_sentence = content.split('.')[0] if '.' in content else content
                        if keyword in first_sentence:
                            score += 5
                    except:
                        # Skip this bonus if we can't split sentences
                        pass
            
            # Give bonus for having multiple keywords
            try:
                keyword_matches = sum(1 for keyword in keywords if keyword in content)
                if keyword_matches > 1:
                    score += keyword_matches * 3
            except:
                # Skip this bonus if counting fails
                pass
                
            # Score exact phrase matches even higher
            if query.lower() in content:
                score += 10
                
            # Prioritize documents with relevant headings/beginnings
            try:
                if any(keyword in content[:100].lower() for keyword in keywords):
                    score += 8
            except:
                # Skip this bonus if slicing fails
                pass
                
            # Add document with its score
            scored_docs.append((doc, score))
        except Exception as e:
            logger.error(f"Error scoring document: {str(e)}")
            # Add with zero score so it at least appears in results
            scored_docs.append((doc, 0))
    
    try:
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
    except Exception as e:
        logger.error(f"Error sorting documents by score: {str(e)}")
        # Return unsorted if sorting fails
        return [doc for doc, _ in scored_docs]
    
    # Return sorted documents
    return [doc for doc, score in scored_docs]

def fetch_parent_context(child_docs, parent_limit=3):
    """
    Fetch parent documents for each chunk to provide more context to the LLM
    
    Args:
        child_docs (DocumentCollection): The document chunks that were retrieved
        parent_limit (int): Max number of parent docs to retrieve
    
    Returns:
        DocumentCollection: Combined child and parent documents
    """
    if not child_docs or len(child_docs) == 0:
        logger.warning("No child documents provided to fetch_parent_context")
        return DocumentCollection()
    
    # Create result collection starting with the child docs
    result_docs = DocumentCollection(list(child_docs))
    
    # Get DB connection
    conn = None
    try:
        conn = get_db_connection()
    except Exception as e:
        logger.error(f"Error connecting to database in fetch_parent_context: {str(e)}")
        return result_docs  # Return child docs if can't connect
    
    if not conn:
        logger.error("Failed to get DB connection in fetch_parent_context")
        return result_docs  # Return child docs if no connection
    
    # Process each child document
    seen_parents = set()
    try:
        cursor = conn.cursor()
        
        for doc in child_docs:
            doc_name = doc.metadata.get("source")
            page_num = doc.metadata.get("page")
            
            if not doc_name or not page_num:
                logger.warning(f"Missing metadata in document: {doc.metadata}")
                continue
                
            # Find parent pages based on proximity
            # Prefer the parent in the immediate vicinity (+/- 1-2 pages)
            parent_nums = []
            curr_page = int(page_num)
            
            # Create a series of ranges, focusing on proximity
            # First try exact -1, +1 pages
            ranges = [
                (curr_page - 1, curr_page - 1),  # Previous page
                (curr_page + 1, curr_page + 1),  # Next page
            ]
            
            # Then try pages further away if needed
            if parent_limit > 2:
                ranges.extend([
                    (curr_page - 3, curr_page - 2),  # Previous 2-3 pages
                    (curr_page + 2, curr_page + 3),  # Next 2-3 pages
                ])
            
            # Execute queries for each range until we have enough parents
            for page_min, page_max in ranges:
                if page_min <= 0:
                    continue  # Skip invalid page numbers
                    
                parent_query = """
                    SELECT id, doc_name, page_number, content
                    FROM documents
                    WHERE doc_name = %s
                    AND page_number BETWEEN %s AND %s
                    AND page_number != %s
                    ORDER BY ABS(page_number - %s)
                    LIMIT %s;
                """
                
                cursor.execute(parent_query, (doc_name, page_min, page_max, 
                                             curr_page, curr_page, parent_limit))
                parents = cursor.fetchall()
                
                # Process found parents
                for parent in parents:
                    if len(parent) < 4:
                        continue
                        
                    parent_id = parent[0]
                    parent_doc = parent[1]
                    parent_page = parent[2]
                    parent_content = parent[3]
                    
                    # Create a unique key for this parent
                    parent_key = f"{parent_doc}_{parent_page}"
                    
                    # Skip if we've already seen this parent
                    if parent_key in seen_parents:
                        continue
                        
                    # Add this parent to results
                    parent_doc_obj = Document(
                        page_content=parent_content,
                        metadata={
                            "source": parent_doc, 
                            "page": parent_page,
                            "is_parent": True,  # Mark as parent
                            "parent_of_page": curr_page
                        }
                    )
                    
                    result_docs.append(parent_doc_obj)
                    seen_parents.add(parent_key)
                    
                # Stop querying if we have enough parents
                if len(seen_parents) >= parent_limit:
                    break
    
    except Exception as e:
        logger.error(f"Error fetching parent context: {str(e)}")
    finally:
        if conn:
            conn.close()
    
    # Count how many parent docs were added
    added_parents = len(result_docs) - len(child_docs)
    logger.info(f"Added {added_parents} parent documents for context")
    
    # Preserve metrics from child_docs and add parent_count
    result_docs.sql_count = getattr(child_docs, 'sql_count', 0)
    result_docs.vector_count = getattr(child_docs, 'vector_count', 0)
    result_docs.fallback_count = getattr(child_docs, 'fallback_count', 0)
    result_docs.parent_count = added_parents
    
    return result_docs 

def natural_language_sql_search(query: str, doc_name: str, limit: int = 30) -> List[Document]:
    """
    Performs a more intelligent SQL search using LangChain's SQLDatabaseChain
    to convert natural language queries to SQL.
    
    Args:
        query: User query in natural language
        doc_name: Document name to filter by
        limit: Maximum number of results
        
    Returns:
        List of matching documents
    """
    try:
        # Import LangChain modules for SQL
        from langchain.utilities import SQLDatabase
        from langchain.chains import SQLDatabaseChain
        from langchain.llms import OpenRouter
        import os

        # Get DB connection string
        conn_string = get_connection_string()
        if not conn_string:
            logger.error("Failed to get database connection string")
            return []
        
        # Connect to the database
        db = SQLDatabase.from_uri(conn_string)
        
        # Get LLM using OpenRouter instead of OpenAI
        # Check if OPENROUTER_API_KEY exists in environment variables
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            logger.warning("No OPENROUTER_API_KEY found, using simple SQL search instead")
            return simple_sql_search(query, doc_name, limit)
            
        # Use Claude 3 Haiku (fastest/cheapest) for SQL generation tasks
        llm = OpenRouter(
            api_key=openrouter_api_key,
            model="anthropic/claude-3-haiku", 
            temperature=0
        )
        
        # Create a prompt that emphasizes searching for relevant content
        # and filtering by document name
        nl_query = f"""
        Find content in the 'documents' table that is relevant to: '{query}'.
        Only look for documents where doc_name = '{doc_name}'.
        The table has columns: id, doc_name, page_number, content, metadata.
        Return at most {limit} rows.
        Focus on finding the most relevant content based on semantic meaning, 
        not just exact keyword matches.
        """
        
        # Create the chain
        db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
        
        # Run the chain to generate and execute SQL
        logger.info(f"Generating SQL for query: '{query}' in document '{doc_name}'")
        result = db_chain.run(nl_query)
        logger.info(f"SQLDatabaseChain result: {result}")
        
        # If the chain doesn't return the actual documents, we need to parse the result
        # and execute the query ourselves
        
        # Extract SQL query if present in the result
        import re
        sql_match = re.search(r"```sql\s*(.*?)\s*```", result, re.DOTALL)
        
        if not sql_match:
            # Try another pattern without the markdown code block
            sql_match = re.search(r"SELECT\s+.*?\s+FROM\s+.*?(?:;|$)", result, re.DOTALL | re.IGNORECASE)
        
        if sql_match:
            sql_query = sql_match.group(0).strip()
            logger.info(f"Extracted SQL query: {sql_query}")
            
            # Execute the query
            conn = get_db_connection()
            if not conn:
                logger.error("Failed to connect to database")
                return []
                
            cursor = conn.cursor()
            try:
                cursor.execute(sql_query)
                rows = cursor.fetchall()
                logger.info(f"SQL query returned {len(rows)} results")
                
                # Process the results
                documents = []
                for row in rows:
                    try:
                        # The columns will depend on what was selected in the generated SQL
                        # We'll need to adapt this based on the actual results
                        
                        # If we have a full row from the documents table
                        if len(row) >= 4:
                            doc_id = row[0]
                            row_doc_name = row[1] if len(row) > 1 else doc_name
                            page_num = row[2] if len(row) > 2 else 0
                            content = row[3] if len(row) > 3 else ""
                            
                            # Skip empty content
                            if not content:
                                continue
                                
                            # Create document
                            documents.append(Document(
                                page_content=content,
                                metadata={"source": row_doc_name, "page": page_num}
                            ))
                    except Exception as e:
                        logger.error(f"Error processing row: {str(e)}")
                
                cursor.close()
                conn.close()
                return documents
                
            except Exception as e:
                logger.error(f"Error executing generated SQL: {str(e)}")
                cursor.close()
                conn.close()
        
        # If we couldn't extract or execute SQL, fall back to simple SQL search
        logger.warning("Falling back to simple SQL search")
        return simple_sql_search(query, doc_name, limit)
        
    except ImportError as e:
        logger.error(f"Missing required packages for natural language SQL search: {str(e)}")
        # Fall back to simple SQL search
        return simple_sql_search(query, doc_name, limit)
    except Exception as e:
        logger.error(f"Error in natural language SQL search: {str(e)}")
        # Fall back to simple SQL search
        return simple_sql_search(query, doc_name, limit) 