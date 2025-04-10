import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

try:
    # Get database connection
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    
    # Create cursor
    cursor = conn.cursor()
    
    # Check if jsonpath extension exists
    cursor.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'jsonpath';")
    jsonpath_available = cursor.fetchone()
    
    if jsonpath_available:
        # Enable the jsonpath extension if available
        cursor.execute("CREATE EXTENSION IF NOT EXISTS jsonpath;")
        conn.commit()
        print("Successfully enabled jsonpath extension!")
    else:
        # Implement alternative SQL method for document filtering
        print("jsonpath extension not available. Creating custom metadata filtering functions...")
        
        # Create a custom function to match metadata fields (alternative to jsonb_path_match)
        custom_function_sql = """
        CREATE OR REPLACE FUNCTION custom_metadata_match(metadata JSONB, field_name TEXT, field_value TEXT)
        RETURNS BOOLEAN AS $$
        BEGIN
            RETURN metadata->>field_name = field_value;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        cursor.execute(custom_function_sql)
        conn.commit()
        print("Created custom metadata matching function!")
        
        # Test the function
        cursor.execute("SELECT custom_metadata_match('{\"source\": \"test.pdf\"}'::jsonb, 'source', 'test.pdf');")
        result = cursor.fetchone()
        print(f"Test result of custom function: {result[0]}")
    
    # Close connection
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"Error: {str(e)}") 