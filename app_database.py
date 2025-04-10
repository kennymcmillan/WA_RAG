"""
app_database.py

This module contains utilities for database operations.
Handles connections to PostgreSQL database with pgvector extension for vector similarity search.
Provides functions for:
- Creating database connection strings
- Establishing database connections
- Initializing pgvector extension and tables
- Saving document chunks and their embeddings to the database
"""

import os
import psycopg2
from psycopg2.extras import execute_values
import streamlit as st
import logging
import json
import math  # For NaN handling

# Helper function to make data JSON-safe (handle NaN, Infinity, etc.)
def make_json_safe(obj):
    """
    Convert an object to be JSON-safe, handling NaN, Infinity, and other values that don't serialize well
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    else:
        return obj

def get_connection_string():
    # Get database credentials from environment variables
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    
    # Ensure all parts are present before forming the string
    if all([DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT]):
        connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        logging.info("Database connection string configured.")
        return connection_string
    else:
        logging.error("Database connection details missing in .env file.")
        st.error("Database connection details missing in .env file.")
        return None

def get_db_connection():
    try:
        # Get connection parameters with explicit logging
        db_name = os.getenv("DB_NAME")
        db_user = os.getenv("DB_USER") 
        db_pwd = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        
        # Check if all required parameters are present
        if not all([db_name, db_user, db_pwd, db_host, db_port]):
            missing = []
            if not db_name: missing.append("DB_NAME")
            if not db_user: missing.append("DB_USER")
            if not db_pwd: missing.append("DB_PASSWORD")
            if not db_host: missing.append("DB_HOST")
            if not db_port: missing.append("DB_PORT")
            logging.error(f"Missing database connection parameters: {', '.join(missing)}")
            return None
        
        # Log the connection attempt without exposing credentials
        logging.info(f"Connecting to database {db_name} at {db_host}:{db_port}")
        
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_pwd,
            host=db_host,
            port=db_port
        )
        logging.info("Database connection successful.")
        return conn
    except Exception as e:
        logging.error(f"Error connecting to database: {str(e)}")
        return None

def check_pgvector_extension():
    """Check if pgvector extension is installed in the database"""
    conn = None
    try:
        # Get connection
        conn = get_db_connection()
        if not conn:
            logging.error("Failed to connect to database to check pgvector")
            return False
        
        # Check if pgvector extension exists
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector';
        """)
        result = cursor.fetchone()[0]
        
        # Check result
        if result > 0:
            logging.info("pgvector extension is installed")
            
            # Also check for langchain_pg_embedding table
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'langchain_pg_embedding'
                );
            """)
            table_exists = cursor.fetchone()[0]
            
            if table_exists:
                logging.info("langchain_pg_embedding table exists")
                return True
            else:
                logging.error("pgvector extension is installed but langchain_pg_embedding table is missing")
                return False
        else:
            logging.error("pgvector extension is NOT installed")
            return False
    except Exception as e:
        logging.error(f"Error checking pgvector extension: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def initialize_pgvector():
    """
    Initialize the database with pgvector extension and required tables.
    More forceful implementation that will attempt to create the extension
    and tables if they don't exist.
    """
    conn = None
    try:
        # First check existing connection
        conn = get_db_connection()
        if not conn:
            logging.error("Failed to connect to database for initialization")
            return False
        
        cursor = conn.cursor()
        
        # Step 1: Create pgvector extension if it doesn't exist
        logging.info("Checking for pgvector extension...")
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            logging.info("pgvector extension created or already exists")
        except Exception as e:
            logging.error(f"Failed to create pgvector extension: {str(e)}")
            logging.warning("You may need to install pgvector extension on your PostgreSQL server")
            return False
        
        # Step 2: Check if documents table exists and has proper structure
        logging.info("Setting up documents table...")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    doc_name TEXT NOT NULL,
                    page_number INTEGER,
                    content TEXT NOT NULL,
                    embedding vector(1536),
                    metadata JSONB
                );
            """)
            
            # Create index for documents table
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx
                ON documents
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            conn.commit()
            logging.info("Documents table created or verified")
        except Exception as e:
            logging.error(f"Error setting up documents table: {str(e)}")
        
        # Step 3: Check if langchain_pg_embedding table exists
        logging.info("Checking for langchain_pg_embedding table...")
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'langchain_pg_embedding'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        # Step 4: If table doesn't exist, create it manually
        if not table_exists:
            logging.info("Creating langchain_pg_embedding table...")
            try:
                # Create the table with required structure for PGVector
                cursor.execute("""
                    CREATE TABLE langchain_pg_embedding (
                        uuid UUID PRIMARY KEY,
                        cmetadata JSONB,
                        document TEXT,
                        embedding VECTOR(1536)
                    );
                """)
                
                # Create index for faster similarity search
                cursor.execute("""
                    CREATE INDEX ON langchain_pg_embedding USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                """)
                
                conn.commit()
                logging.info("Successfully created langchain_pg_embedding table and index")
            except Exception as e:
                logging.error(f"Failed to create langchain_pg_embedding table: {str(e)}")
                return False
        else:
            logging.info("langchain_pg_embedding table already exists")
        
        cursor.close()
        conn.close()
        
        # Verify if everything is properly set up
        if check_pgvector_extension():
            logging.info("pgvector setup complete and verified")
            return True
        else:
            logging.error("pgvector setup could not be verified")
            return False
        
    except Exception as e:
        logging.error(f"Error initializing pgvector: {str(e)}")
        if conn and not conn.closed:
            conn.close()
        return False

def save_document_to_db(doc_name, chunks, embeddings, metadata_list):
    conn = get_db_connection()
    if not conn:
        return False

    try:
        with conn.cursor() as cur:
            # Debug logging for document information
            logging.info(f"Saving document to database: {doc_name} with {len(chunks)} chunks")
            
            # Log a sample of the data being stored
            if chunks and len(chunks) > 0:
                logging.info(f"Sample chunk: {chunks[0][:100]}...")
                
            if metadata_list and len(metadata_list) > 0:
                # Check if there are tables in the first metadata item
                if metadata_list[0].get('has_tables', False):
                    logging.info(f"Document contains tables. First page has {metadata_list[0].get('tables_count', 0)} tables.")
                logging.info(f"Sample metadata: {str(metadata_list[0])[:200]}...")
            
            # Ensure metadata has the proper structure for JSONB querying
            # Metadata should have both 'source' and 'page' fields at a minimum
            validated_metadata_list = []
            for metadata in metadata_list:
                # Create a copy to avoid modifying the original metadata
                validated_metadata = dict(metadata)
                
                # Ensure 'source' field exists and equals doc_name
                if 'source' not in validated_metadata:
                    validated_metadata['source'] = doc_name
                
                # Ensure 'page' field exists
                if 'page' not in validated_metadata:
                    validated_metadata['page'] = None
                
                # Handle table metadata - validate and clean tables data
                if 'tables' in validated_metadata and isinstance(validated_metadata['tables'], list):
                    # Ensure tables data is compact but complete by keeping essential fields
                    for table in validated_metadata['tables']:
                        # We can remove the full table_text from metadata as it's already in the content
                        if 'table_text' in table:
                            del table['table_text']
                        
                        # Keep table_data and a compact version of table_html
                        # You could also remove table_html if space is a concern
                
                validated_metadata_list.append(validated_metadata)
                
            # Log first validated metadata for debugging
            if validated_metadata_list:
                logging.info(f"First validated metadata sample: {str(validated_metadata_list[0])[:200]}...")
            
            # Prepare data for batch insert
            data = []
            for i, (chunk, embedding, metadata) in enumerate(zip(chunks, embeddings, validated_metadata_list)):
                # Check if this chunk contains tables
                has_tables = metadata.get('has_tables', False)
                
                # Add table-specific tags to the metadata for enhanced searching
                if has_tables:
                    # Add a tag to make searching for tables easier
                    metadata['contains_table'] = True
                    metadata['searchable_table'] = True
                
                # Make metadata JSON-safe
                json_safe_metadata = make_json_safe(metadata)
                
                data.append((
                    doc_name,
                    metadata.get('page', None),
                    chunk,
                    embedding,
                    json.dumps(json_safe_metadata) # Serialize metadata dict to JSON string
                ))

            # Batch insert
            execute_values(
                cur,
                """
                INSERT INTO documents (doc_name, page_number, content, embedding, metadata)
                VALUES %s
                """,
                data
            )

            conn.commit()
            logging.info(f"Successfully saved {len(chunks)} chunks for document: {doc_name}")
        return True
    except Exception as e:
        st.error(f"Error saving to database: {str(e)}")
        logging.error(f"Error saving document to database: {str(e)}", exc_info=True)
        return False
    finally:
        conn.close()

def load_documents_from_database():
    """Load document names directly from the database"""
    try:
        # Connect to database using env variables
        db_name = os.getenv("DB_NAME")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        
        if not all([db_name, db_user, db_password, db_host, db_port]):
            logging.error("Database credentials not found in environment variables")
            return {}
        
        # Connect to the database
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        
        # Get unique document names
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT metadata->>'source' as doc_name, COUNT(*) as chunk_count
            FROM documents 
            WHERE metadata->>'source' IS NOT NULL
            GROUP BY metadata->>'source'
            ORDER BY doc_name;
        """)
        docs = cursor.fetchall()
        
        # Store both document names and chunk counts
        doc_info = {doc[0]: {"chunk_count": doc[1]} for doc in docs}
        
        # Close connection
        cursor.close()
        conn.close()
        
        logging.info(f"Loaded {len(doc_info)} documents from database")
        return doc_info
    except Exception as e:
        logging.error(f"Error loading documents from database: {str(e)}")
        return {}

def inspect_database_contents():
    """Retrieve and log sample documents from the database for debugging purposes"""
    try:
        # Connect to database using env variables
        conn = get_db_connection()
        if not conn:
            logging.error("Failed to connect to database for inspection")
            return
        
        cursor = conn.cursor()
        
        # Check total document count
        cursor.execute("SELECT COUNT(*) FROM documents;")
        count = cursor.fetchone()
        logging.info(f"Total document chunks in database: {count[0]}")
        
        # Check document names and counts
        cursor.execute("SELECT doc_name, COUNT(*) FROM documents GROUP BY doc_name;")
        doc_counts = cursor.fetchall()
        logging.info("Document counts by doc_name:")
        for doc in doc_counts:
            logging.info(f"- {doc[0]}: {doc[1]} chunks")
        
        # Check metadata structure for a few samples
        cursor.execute("SELECT id, doc_name, metadata FROM documents LIMIT 3;")
        sample_rows = cursor.fetchall()
        logging.info("Sample document rows:")
        for row in sample_rows:
            logging.info(f"ID: {row[0]}")
            logging.info(f"doc_name: {row[1]}")
            logging.info(f"metadata: {row[2]}")
            logging.info("---")
            
        # Close connection
        cursor.close()
        conn.close()
        
        return {
            "total_count": count[0] if count else 0,
            "doc_counts": doc_counts,
            "sample_rows": sample_rows
        }
    except Exception as e:
        logging.error(f"Error inspecting database: {str(e)}", exc_info=True)
        return None 