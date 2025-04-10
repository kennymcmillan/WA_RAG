import os
import psycopg2
from dotenv import load_dotenv

def clear_documents_table():
    """Clear the documents table in the database"""
    load_dotenv()
    
    # Get database credentials from .env file
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    
    if not all([db_name, db_user, db_password, db_host, db_port]):
        print("Database credentials not found in .env file")
        return
    
    # Connect to the database
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        print(f"Successfully connected to database '{db_name}' on {db_host}")
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Check if the documents table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'documents'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            print("The 'documents' table does not exist.")
            return
        
        # Confirm with user
        confirm = input("⚠️ WARNING: This will delete ALL documents from the database. Are you sure? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return
        
        # Get count before deletion
        cursor.execute("SELECT COUNT(*) FROM documents;")
        count_before = cursor.fetchone()[0]
        print(f"Current document count: {count_before}")
        
        # Clear the table
        cursor.execute("DELETE FROM documents;")
        conn.commit()
        
        # Get count after deletion
        cursor.execute("SELECT COUNT(*) FROM documents;")
        count_after = cursor.fetchone()[0]
        
        print(f"Deleted {count_before - count_after} document chunks from the database.")
        print(f"Remaining document count: {count_after}")
        
        # Close the connection
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")

if __name__ == "__main__":
    clear_documents_table() 