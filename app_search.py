"""
app_search.py

This module provides direct SQL-based search functionality as a fallback
when the PGVector search fails due to compatibility issues.
"""

import os
import psycopg2
import logging
import json
from dotenv import load_dotenv
from langchain_core.documents import Document

def get_db_connection():
    """Establish a connection to the database"""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        logging.info("Database connection successful for SQL search")
        return conn
    except Exception as e:
        logging.error(f"Error connecting to database: {str(e)}")
        return None

def inspect_table_structure():
    """
    Check the structure of the documents table
    """
    conn = get_db_connection()
    if not conn:
        logging.error("Failed to connect to database")
        return {"error": "Database connection failed"}
    
    try:
        cursor = conn.cursor()
        
        # Get table schema
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'documents'
            ORDER BY ordinal_position;
        """)
        
        columns = cursor.fetchall()
        logging.info("Table Structure:")
        for col in columns:
            logging.info(f"Column: {col[0]}, Type: {col[1]}")
        
        # Get sample row to verify structure
        cursor.execute("SELECT * FROM documents LIMIT 1")
        
        # Get column names
        col_names = [desc[0] for desc in cursor.description]
        sample = cursor.fetchone()
        
        # Close connection
        cursor.close()
        conn.close()
        
        if sample:
            result = {"columns": col_names, "sample": {}}
            for i, col_name in enumerate(col_names):
                # Truncate content to avoid huge output
                if col_name == 'content' and sample[i]:
                    result["sample"][col_name] = sample[i][:100] + "..." if len(sample[i]) > 100 else sample[i]
                else:
                    result["sample"][col_name] = sample[i]
            return result
        else:
            return {"columns": col_names, "sample": None}
    
    except Exception as e:
        logging.error(f"Error inspecting table: {str(e)}")
        if conn and not conn.closed:
            conn.close()
        return {"error": str(e)}

def check_document_exists(doc_name=None):
    """
    Check if a specific document exists in the database
    
    Args:
        doc_name (str): Optional document name to check
        
    Returns:
        dict: Information about database contents
    """
    conn = get_db_connection()
    if not conn:
        logging.error("Failed to connect to database")
        return {"error": "Database connection failed"}
    
    try:
        cursor = conn.cursor()
        
        # Count total documents
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]
        
        # Get list of unique document names
        cursor.execute("SELECT DISTINCT doc_name FROM documents")
        doc_names = [row[0] for row in cursor.fetchall()]
        
        # If specific document name provided, check if it exists
        doc_info = {}
        if doc_name:
            cursor.execute("SELECT COUNT(*) FROM documents WHERE doc_name = %s OR metadata->>'source' = %s", 
                         (doc_name, doc_name))
            doc_count = cursor.fetchone()[0]
            doc_info["exists"] = doc_count > 0
            doc_info["chunk_count"] = doc_count
            
            # Sample first chunk if document exists
            if doc_count > 0:
                cursor.execute("""
                    SELECT content, metadata FROM documents 
                    WHERE doc_name = %s OR metadata->>'source' = %s
                    LIMIT 1
                """, (doc_name, doc_name))
                sample = cursor.fetchone()
                if sample:
                    doc_info["sample_content"] = sample[0][:200] + "..." if sample[0] else "None"
                    doc_info["sample_metadata"] = sample[1]
        
        cursor.close()
        conn.close()
        
        return {
            "total_docs": total_docs,
            "doc_names": doc_names,
            "specific_doc": doc_info if doc_name else None
        }
    
    except Exception as e:
        logging.error(f"Error checking document: {str(e)}")
        if conn and not conn.closed:
            conn.close()
        return {"error": str(e)}

def basic_document_search(doc_name=None, limit=5):
    """
    A very basic function to just retrieve document content without any filtering
    
    Args:
        doc_name (str): Optional document name
        limit (int): Maximum number of results to return
    
    Returns:
        list: Document content
    """
    conn = get_db_connection()
    if not conn:
        logging.error("Failed to connect to database for basic search")
        return []
    
    try:
        cursor = conn.cursor()
        
        if doc_name:
            # Inspect table structure first to debug
            table_info = inspect_table_structure()
            logging.info(f"Table structure: {table_info}")
            
            # Try simplified query
            try:
                sql = "SELECT * FROM documents WHERE doc_name = %s LIMIT %s"
                cursor.execute(sql, (doc_name, limit))
                results = cursor.fetchall()
                logging.info(f"Basic document search returned {len(results)} results")
                
                # Get column names
                col_names = [desc[0] for desc in cursor.description]
                
                # Build Document objects
                documents = []
                for row in results:
                    # Map row to dict
                    row_dict = {}
                    for i, col_name in enumerate(col_names):
                        row_dict[col_name] = row[i]
                    
                    # Extract content and metadata
                    content = row_dict.get('content', '')
                    
                    # Build metadata
                    metadata = {}
                    if 'metadata' in row_dict and row_dict['metadata']:
                        metadata = row_dict['metadata']
                    elif 'doc_name' in row_dict:
                        metadata['source'] = row_dict['doc_name']
                    if 'page_number' in row_dict:
                        metadata['page'] = row_dict['page_number']
                    
                    # Create document
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
                
                return documents
                
            except Exception as e:
                logging.error(f"Error in basic document search: {str(e)}")
                
        cursor.close()
        conn.close()
        return []
        
    except Exception as e:
        logging.error(f"Error in basic document search: {str(e)}")
        if conn and not conn.closed:
            conn.close()
        return []

def sql_keyword_search(query, doc_name=None, limit=50):
    """
    Perform a SQL-based keyword search for when vector search fails.
    Enhanced to better handle names and phrases.
    
    Args:
        query (str): The search query
        doc_name (str): Optional document name to filter by
        limit (int): Maximum number of results to return
        
    Returns:
        list: List of Document objects with search results
    """
    conn = get_db_connection()
    if not conn:
        logging.error("Failed to connect to database for SQL search")
        return []
    
    try:
        # First, check if the document exists and log useful details
        doc_check = check_document_exists(doc_name)
        if "error" not in doc_check:
            logging.info(f"Database has {doc_check['total_docs']} total document chunks")
            logging.info(f"Available documents: {doc_check['doc_names']}")
            
            if doc_name and doc_check['specific_doc']:
                if doc_check['specific_doc']['exists']:
                    logging.info(f"Document '{doc_name}' exists with {doc_check['specific_doc']['chunk_count']} chunks")
                    logging.info(f"Sample content: {doc_check['specific_doc']['sample_content']}")
                    logging.info(f"Sample metadata: {doc_check['specific_doc']['sample_metadata']}")
                else:
                    logging.warning(f"Document '{doc_name}' NOT FOUND in database!")
        
        # Check table structure for debugging
        table_info = inspect_table_structure()
        
        cursor = conn.cursor()
        
        # Split query into keywords
        keywords = query.lower().split()
        
        # Check if this looks like a name search (e.g., "Brian Moore")
        is_name_search = len(keywords) >= 2 and all(len(word) > 2 for word in query.split())
        
        # Create ILIKE conditions for each keyword
        keyword_conditions = []
        
        # For possible names, try the exact phrase
        if is_name_search:
            full_phrase = " ".join(keywords)
            keyword_conditions.append(f"content ILIKE '%{full_phrase}%'")
            # Also search for the name in reverse order (e.g., "Moore, Brian")
            if len(keywords) == 2:
                reverse_phrase = f"{keywords[1]}, {keywords[0]}"
                keyword_conditions.append(f"content ILIKE '%{reverse_phrase}%'")
        
        # Always include individual keyword search
        for keyword in keywords:
            if len(keyword) > 3:  # Only use keywords with more than 3 characters
                keyword_conditions.append(f"content ILIKE '%{keyword}%'")
        
        # Combine with OR
        if keyword_conditions:
            keyword_sql = " OR ".join(keyword_conditions)
        else:
            # Fallback if no valid keywords
            keyword_sql = "1=1"
        
        logging.info(f"SQL search keywords: {keywords}")
        logging.info(f"SQL conditions: {keyword_sql}")
        
        # Try using the basic document search first
        basic_docs = basic_document_search(doc_name, limit=5)
        if basic_docs:
            logging.info(f"Basic document search returned {len(basic_docs)} results")
            for i, doc in enumerate(basic_docs):
                logging.info(f"Basic result {i+1}: {doc.page_content[:100]}...")
        
        # SIMPLIFIED VERSION: Just try to get any documents first
        if doc_name:
            try:
                # Try explicit query with direct column references
                sql_query = """
                    SELECT * FROM documents
                    WHERE doc_name = %s
                    LIMIT %s;
                """
                logging.info(f"Executing simplified SQL query for document: {doc_name}")
                cursor.execute(sql_query, (doc_name, limit))
                results = cursor.fetchall()
                logging.info(f"Simple SQL query returned {len(results)} results")
                
                # Get column names
                col_names = [desc[0] for desc in cursor.description]
                
                # Build Document objects
                documents = []
                for row in results:
                    # Map row to dict
                    row_dict = {}
                    for i, col_name in enumerate(col_names):
                        row_dict[col_name] = row[i]
                    
                    # Extract content and metadata
                    content = row_dict.get('content', '')
                    
                    # Build metadata
                    metadata = {}
                    if 'metadata' in row_dict and row_dict['metadata']:
                        metadata = row_dict['metadata']
                    elif 'doc_name' in row_dict:
                        metadata['source'] = row_dict['doc_name']
                    if 'page_number' in row_dict:
                        metadata['page'] = row_dict['page_number']
                    
                    # Create document
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
                
                logging.info(f"SQL keyword search found {len(documents)} results")
                cursor.close()
                conn.close()
                return documents
            
            except Exception as e:
                logging.error(f"Error in simplified SQL query: {str(e)}")
        
        cursor.close()
        conn.close()
        return []
    
    except Exception as e:
        logging.error(f"Error in SQL keyword search: {str(e)}")
        if conn and not conn.closed:
            conn.close()
        return []

def direct_document_search(query, doc_name=None, use_fulltext=True, limit=50):
    """
    Higher-level search function that uses direct SQL.
    Combines full-text search capabilities with keyword search as fallback.
    
    Args:
        query (str): The search query
        doc_name (str): Optional document name to filter by
        use_fulltext (bool): Whether to use PostgreSQL full-text search 
        limit (int): Maximum number of results to return
        
    Returns:
        list: List of Document objects with search results
    """
    # Start with the simpler keyword search
    try:
        return sql_keyword_search(query, doc_name, limit)
    except Exception as e:
        logging.error(f"SQL keyword search failed: {str(e)}")
        # Fall back to most basic search
        conn = get_db_connection()
        if not conn:
            logging.error("Failed to connect to database")
            return []
        
        try:
            cursor = conn.cursor()
            
            # Use super basic query that should always work
            if doc_name:
                sql = "SELECT id, doc_name, page_number, content, metadata FROM documents WHERE doc_name = %s LIMIT %s"
                cursor.execute(sql, (doc_name, limit))
            else:
                sql = "SELECT id, doc_name, page_number, content, metadata FROM documents LIMIT %s"
                cursor.execute(sql, (limit,))
                
            rows = cursor.fetchall()
            logging.info(f"Basic fallback search returned {len(rows)} results")
            
            # Create documents
            documents = []
            for row in rows:
                try:
                    doc_id, doc_name, page_num, content, meta = row
                    
                    # Basic metadata
                    metadata = {"source": doc_name, "page": page_num}
                    
                    # Add document
                    documents.append(Document(
                        page_content=content,
                        metadata=metadata
                    ))
                except Exception as inner_e:
                    logging.error(f"Error processing row: {str(inner_e)}")
                    
            cursor.close()
            conn.close()
            
            return documents
            
        except Exception as e:
            logging.error(f"Fallback database query failed: {str(e)}")
            if conn and not conn.closed:
                conn.close()
            return [] 