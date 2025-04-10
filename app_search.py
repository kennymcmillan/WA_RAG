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

def sql_keyword_search(query, doc_name=None, include_tables=True, table_boost=2.0, limit=50):
    """
    Perform a SQL-based keyword search.
    
    Args:
        query (str): Search query
        doc_name (str, optional): Filter by document name
        include_tables (bool): Whether to specially handle table data
        table_boost (float): Boost factor for results containing tables
        limit (int): Maximum number of results to return
        
    Returns:
        List[Document]: List of matching documents
    """
    from app_database import get_db_connection
    from langchain.schema import Document
    
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        
        # Create a safe version of the query for SQL ILIKE
        safe_query = query.replace("%", "\\%").replace("_", "\\_")
        terms = [term.strip() for term in safe_query.split() if term.strip()]
        
        # If no valid terms, return empty list
        if not terms:
            return []
            
        sql = """
            SELECT id, doc_name, page_number, content, metadata, 
                   CASE 
                     WHEN metadata->>'has_tables' = 'true' THEN %s
                     ELSE 1.0
                   END AS boost_factor
            FROM documents
            WHERE (
        """
        
        params = [table_boost]
        conditions = []
        
        # Build search conditions for each term (AND logic between terms)
        for term in terms:
            term_conditions = []
            
            # Basic content search
            term_conditions.append("content ILIKE %s")
            params.append(f"%{term}%")
            
            # Table-specific search if enabled
            if include_tables:
                # Search for terms in content containing table markers
                term_conditions.append("(content ILIKE %s AND content ILIKE %s)")
                params.append(f"%{term}%")
                params.append("%[TABLE%")
                
                # Search using table metadata
                term_conditions.append("(metadata->>'has_tables' = 'true' AND content ILIKE %s)")
                params.append(f"%{term}%")
            
            conditions.append("(" + " OR ".join(term_conditions) + ")")
        
        # Combine all term conditions with AND
        sql += " AND ".join(conditions)
        
        # Add document filter if specified
        if doc_name:
            sql += " AND (doc_name = %s OR metadata->>'source' = %s)"
            params.append(doc_name)
            params.append(doc_name)
            
        # Sort by relevance score (number of terms matched) and boost factor
        sql += """
            )
            ORDER BY (
                -- Count occurrences of each search term
                """ + " + ".join([f"CASE WHEN content ILIKE %s THEN 1 ELSE 0 END" for _ in terms]) + """
            ) * boost_factor DESC, 
            id ASC
            LIMIT %s
        """
        
        # Add params for the ILIKE conditions in ORDER BY
        for term in terms:
            params.append(f"%{term}%")
            
        # Add limit parameter
        params.append(limit)
        
        # Execute the query
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        logging.info(f"SQL keyword search found {len(rows)} results")
        
        # Process results
        documents = []
        for row in rows:
            doc_id, doc_name, page_num, content, meta, boost = row
            
            # Get metadata
            if meta:
                # Convert metadata from JSON
                try:
                    metadata = meta
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                except:
                    metadata = {"source": doc_name}
                    if page_num:
                        metadata["page"] = page_num
            else:
                metadata = {"source": doc_name}
                if page_num:
                    metadata["page"] = page_num
            
            # Check if this chunk contains tables
            has_tables = metadata.get('has_tables', False)
            if has_tables:
                logging.info(f"Found chunk with tables: {doc_id}")
                # Add a flag that can help the LLM identify table content
                metadata["contains_tables"] = True
                metadata["result_from_table_search"] = include_tables
            
            # Create document
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        cursor.close()
        conn.close()
        
        return documents
        
    except Exception as e:
        logging.error(f"SQL keyword search error: {str(e)}", exc_info=True)
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

def diagnose_vector_search_issues(doc_name=None, limit=5):
    """
    Diagnostic function to examine documents in the database and verify embedding presence
    
    Args:
        doc_name (str): Optional document name to filter by
        limit (int): Maximum number of records to examine
        
    Returns:
        dict: Diagnostic information
    """
    logging.info("Running vector search diagnostics")
    result = {
        "total_documents": 0,
        "documents_with_embeddings": 0,
        "document_names": [],
        "embedding_samples": [],
        "issues": []
    }
    
    # Connect to database
    conn = get_db_connection()
    if not conn:
        logging.error("Failed to connect to database for diagnostics")
        result["issues"].append("Database connection failed")
        return result
    
    try:
        cursor = conn.cursor()
        
        # Check total document count
        cursor.execute("SELECT COUNT(*) FROM documents")
        result["total_documents"] = cursor.fetchone()[0]
        
        # Get list of document names
        cursor.execute("SELECT DISTINCT doc_name FROM documents")
        result["document_names"] = [row[0] for row in cursor.fetchall()]
        
        # Check if langchain_pg_embedding table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'langchain_pg_embedding'
            )
        """)
        embedding_table_exists = cursor.fetchone()[0]
        
        if not embedding_table_exists:
            result["issues"].append("The langchain_pg_embedding table doesn't exist")
            return result
        
        # Count documents with embeddings
        cursor.execute("""
            SELECT COUNT(*) 
            FROM langchain_pg_embedding
        """)
        result["documents_with_embeddings"] = cursor.fetchone()[0]
        
        if result["documents_with_embeddings"] == 0:
            result["issues"].append("No documents have embeddings")
            return result
        
        # Check for specific document if provided
        if doc_name:
            # Try to find the document in the documents table
            cursor.execute("SELECT COUNT(*) FROM documents WHERE doc_name = %s", (doc_name,))
            doc_count = cursor.fetchone()[0]
            
            if doc_count == 0:
                result["issues"].append(f"Document '{doc_name}' not found in documents table")
            else:
                result["doc_count"] = doc_count
                
                # Check if this document has embeddings
                try:
                    cursor.execute("""
                        SELECT COUNT(*) 
                        FROM langchain_pg_embedding 
                        WHERE cmetadata->>'source' = %s
                    """, (doc_name,))
                    embedding_count = cursor.fetchone()[0]
                    result["doc_embedding_count"] = embedding_count
                    
                    if embedding_count == 0:
                        result["issues"].append(f"Document '{doc_name}' has no embeddings")
                except Exception as e:
                    logging.error(f"Error checking embeddings for '{doc_name}': {str(e)}")
                    result["issues"].append(f"Error checking embeddings: {str(e)}")
        
        # Sample some embeddings to verify they are non-empty
        try:
            if doc_name:
                cursor.execute("""
                    SELECT le.embedding, le.cmetadata, d.content
                    FROM langchain_pg_embedding le
                    JOIN documents d ON le.cmetadata->>'source' = d.doc_name
                    WHERE le.cmetadata->>'source' = %s
                    LIMIT %s
                """, (doc_name, limit))
            else:
                cursor.execute("""
                    SELECT le.embedding, le.cmetadata, d.content
                    FROM langchain_pg_embedding le
                    JOIN documents d ON le.cmetadata->>'source' = d.doc_name
                    LIMIT %s
                """, (limit,))
                
            samples = cursor.fetchall()
            
            for i, sample in enumerate(samples):
                embedding = sample[0]
                metadata = sample[1]
                content = sample[2]
                
                sample_info = {
                    "has_embedding": embedding is not None and len(embedding) > 0,
                    "embedding_dimensions": len(embedding) if embedding else 0,
                    "metadata": metadata,
                    "content_preview": content[:100] + "..." if content and len(content) > 100 else content
                }
                
                result["embedding_samples"].append(sample_info)
                
                # Check for potential issues
                if not sample_info["has_embedding"]:
                    result["issues"].append(f"Sample {i+1} has empty embedding")
                if not metadata or not isinstance(metadata, dict):
                    result["issues"].append(f"Sample {i+1} has invalid metadata")
                if "source" not in metadata:
                    result["issues"].append(f"Sample {i+1} missing 'source' in metadata")
                
        except Exception as e:
            logging.error(f"Error sampling embeddings: {str(e)}")
            result["issues"].append(f"Error sampling embeddings: {str(e)}")
        
        cursor.close()
        conn.close()
        
        # Overall assessment
        if not result["issues"]:
            if result["documents_with_embeddings"] < result["total_documents"]:
                result["issues"].append(f"Only {result['documents_with_embeddings']} out of {result['total_documents']} documents have embeddings")
            else:
                result["status"] = "OK - No issues detected"
        
        return result
        
    except Exception as e:
        logging.error(f"Error running vector search diagnostics: {str(e)}")
        if conn and not conn.closed:
            conn.close()
        result["issues"].append(f"Diagnostic error: {str(e)}")
        return result 