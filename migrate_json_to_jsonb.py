import os
import psycopg2
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

def get_db_connection():
    # Get database credentials from environment variables
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    
    if not all([db_name, db_user, db_password, db_host, db_port]):
        logging.error('Database connection details missing in .env file')
        return None
    
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        return conn
    except Exception as e:
        logging.error(f'Error connecting to database: {e}')
        return None

def migrate_json_to_jsonb():
    logging.info('Starting migration from JSON to JSONB')
    conn = get_db_connection()
    
    if not conn:
        logging.error('Failed to connect to database')
        return False
    
    try:
        # Check if metadata column exists and get its type
        with conn.cursor() as cur:
            cur.execute("""
                SELECT data_type 
                FROM information_schema.columns 
                WHERE table_name = 'documents' AND column_name = 'metadata'
            """)
            result = cur.fetchone()
            
            if not result:
                logging.error('Metadata column not found in documents table')
                return False
                
            column_type = result[0].upper()
            logging.info(f'Current metadata column type: {column_type}')
            
            # If already JSONB, no need to migrate
            if column_type == 'JSONB':
                logging.info('Metadata column is already JSONB, no migration needed')
                return True
                
            # Create a backup of the documents table first
            logging.info('Creating backup of documents table')
            cur.execute('CREATE TABLE IF NOT EXISTS documents_backup AS SELECT * FROM documents')
            conn.commit()
            logging.info('Backup created successfully')
            
            # Alter the column type to JSONB
            logging.info('Altering metadata column to JSONB type')
            cur.execute('ALTER TABLE documents ALTER COLUMN metadata TYPE JSONB USING metadata::jsonb')
            conn.commit()
            logging.info('Successfully migrated metadata column to JSONB')
            
            return True
    except Exception as e:
        logging.error(f'Error during migration: {e}')
        return False
    finally:
        conn.close()

if __name__ == '__main__':
    if migrate_json_to_jsonb():
        logging.info('Migration completed successfully')
    else:
        logging.error('Migration failed')