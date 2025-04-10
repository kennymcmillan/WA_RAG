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
    
    # Get PostgreSQL version
    cursor.execute("SELECT version();")
    version = cursor.fetchone()[0]
    print("PostgreSQL Version:", version)
    
    # Check pgvector extension version
    cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
    result = cursor.fetchone()
    if result:
        print("pgvector Extension Version:", result[0])
    else:
        print("pgvector Extension: Not installed")
    
    # Check PostgreSQL jsonb functions capability
    try:
        cursor.execute("SELECT count(*) FROM pg_proc WHERE proname = 'jsonb_path_match';")
        count = cursor.fetchone()[0]
        if count > 0:
            print("jsonb_path_match function: Available")
        else:
            print("jsonb_path_match function: Not available")
    except Exception as e:
        print(f"Error checking jsonb_path_match: {str(e)}")
    
    # Close connection
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"Error connecting to database: {str(e)}") 