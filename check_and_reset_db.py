import os
import psycopg2
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

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
        logging.info("Database connection successful.")
        return conn
    except Exception as e:
        logging.error(f"Error connecting to database: {str(e)}")
        return None

def check_tables():
    """Check what tables exist in the database"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            # List tables
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
            tables = cur.fetchall()
            
            logging.info("Tables in database:")
            for table in tables:
                logging.info(f"- {table[0]}")
            
            # Check if 'documents' table exists
            documents_exists = any(table[0] == 'documents' for table in tables)
            logging.info(f"Documents table exists: {documents_exists}")
            
            # Check if LangChain tables exist
            langchain_tables = [table[0] for table in tables if table[0].startswith('langchain_')]
            logging.info(f"LangChain tables: {langchain_tables}")
            
            return documents_exists, langchain_tables
    except Exception as e:
        logging.error(f"Error checking tables: {str(e)}")
        return False, []
    finally:
        conn.close()

def check_documents_structure():
    """Check the structure of the documents table if it exists"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            # Check if documents table exists first
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='public' AND table_name='documents');")
            if not cur.fetchone()[0]:
                logging.info("Documents table does not exist.")
                return False
            
            # Check table structure
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = 'documents';
            """)
            columns = cur.fetchall()
            
            logging.info("Documents table structure:")
            for col in columns:
                logging.info(f"- {col[0]}: {col[1]}")
            
            # Check for sample data
            cur.execute("SELECT COUNT(*) FROM documents;")
            count = cur.fetchone()[0]
            logging.info(f"Total documents: {count}")
            
            if count > 0:
                # Get a sample row
                cur.execute("SELECT * FROM documents LIMIT 1;")
                sample = cur.fetchone()
                column_names = [desc[0] for desc in cur.description]
                
                logging.info("Sample row:")
                for i, col in enumerate(column_names):
                    logging.info(f"- {col}: {sample[i]}")
                
                # Check metadata
                if 'metadata' in column_names:
                    metadata_index = column_names.index('metadata')
                    metadata = sample[metadata_index]
                    logging.info(f"Sample metadata: {metadata}")
            
            return True
    except Exception as e:
        logging.error(f"Error checking documents structure: {str(e)}")
        return False
    finally:
        conn.close()

def reset_documents_table():
    """Drop and recreate the documents table with correct structure"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            # Drop the documents table if it exists
            cur.execute("DROP TABLE IF EXISTS documents;")
            logging.info("Dropped documents table.")
            
            # Create the documents table with correct structure
            cur.execute("""
                CREATE TABLE documents (
                    id SERIAL PRIMARY KEY,
                    doc_name TEXT NOT NULL,
                    page_number INTEGER,
                    content TEXT NOT NULL,
                    embedding vector(384),
                    metadata JSONB
                );
            """)
            logging.info("Created new documents table.")
            
            # Create index for similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx
                ON documents
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            logging.info("Created vector index on documents table.")
            
            conn.commit()
            return True
    except Exception as e:
        logging.error(f"Error resetting documents table: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def clear_langchain_tables():
    """Clear or drop LangChain tables"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            # Get LangChain tables
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name LIKE 'langchain_%';")
            tables = [table[0] for table in cur.fetchall()]
            
            if tables:
                logging.info(f"Found LangChain tables: {tables}")
                
                # Drop each table
                for table in tables:
                    cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
                    logging.info(f"Dropped table: {table}")
                
                conn.commit()
                logging.info("All LangChain tables dropped.")
            else:
                logging.info("No LangChain tables found.")
            
            return True
    except Exception as e:
        logging.error(f"Error clearing LangChain tables: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def initialize_pgvector():
    """Initialize pgvector extension and required tables"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logging.info("Enabled pgvector extension.")
            
            conn.commit()
            return True
    except Exception as e:
        logging.error(f"Error initializing pgvector: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def main():
    # Check for PostgreSQL version
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            logging.info(f"PostgreSQL version: {version}")
        conn.close()
    
    # Check existing tables
    docs_exists, langchain_tables = check_tables()
    
    # If documents table exists, check its structure
    if docs_exists:
        check_documents_structure()
    
    # Ask user if they want to reset
    print("\nBased on the info above, would you like to:")
    print("1. Reset only the documents table")
    print("2. Reset both documents and LangChain tables")
    print("3. Don't reset anything (exit)")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        logging.info("Resetting documents table...")
        reset_documents_table()
    elif choice == '2':
        logging.info("Resetting all tables...")
        clear_langchain_tables()
        reset_documents_table()
        initialize_pgvector()
    else:
        logging.info("No changes made.")
    
    logging.info("Done!")

if __name__ == "__main__":
    main() 