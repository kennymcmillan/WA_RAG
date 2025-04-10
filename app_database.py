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
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        logging.info("Database connection successful.") # Added logging
        return conn
    except Exception as e:
        logging.error(f"Error connecting to database: {str(e)}") # Added logging
        st.error(f"Error connecting to database: {str(e)}")
        return None

def initialize_pgvector():
    conn = get_db_connection()
    if not conn:
        return False

    try:
        with conn.cursor() as cur:
            # Enable pgvector extension if not already enabled
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create documents table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    doc_name TEXT NOT NULL,
                    page_number INTEGER,
                    content TEXT NOT NULL,
                    embedding vector(384),
                    metadata JSONB
                );
            """)

            # Create index for similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx
                ON documents
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)

            conn.commit()
        return True
    except Exception as e:
        st.error(f"Error initializing database: {str(e)}")
        return False
    finally:
        conn.close()

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
                logging.info(f"Sample metadata: {metadata_list[0]}")
            
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
                
                validated_metadata_list.append(validated_metadata)
                
            # Log first validated metadata for debugging
            if validated_metadata_list:
                logging.info(f"First validated metadata: {validated_metadata_list[0]}")
            
            # Prepare data for batch insert
            data = []
            for i, (chunk, embedding, metadata) in enumerate(zip(chunks, embeddings, validated_metadata_list)):
                data.append((
                    doc_name,
                    metadata.get('page', None), # Keep page number separate if desired, or remove if fully in metadata
                    chunk,
                    embedding,
                    json.dumps(metadata) # Serialize metadata dict to JSON string
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