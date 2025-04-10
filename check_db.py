#!/usr/bin/env python
"""
Simple script to check database connection
"""
import os
import sys
from dotenv import load_dotenv
from app_database import get_db_connection, get_connection_string, inspect_database_contents

# First, try to load the .env file directly
print("=== Loading Environment Variables ===")
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
print(f"Looking for .env file at: {dotenv_path}")
print(f".env file exists: {os.path.exists(dotenv_path)}")

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print("Loaded .env file")
else:
    print("ERROR: .env file not found!")
    sys.exit(1)

# Check environment variables
print("\n=== Database Environment Variables ===")
print(f"DB_NAME: {os.getenv('DB_NAME')}")
print(f"DB_USER: {os.getenv('DB_USER')}")
print(f"DB_PASSWORD: {'*****' if os.getenv('DB_PASSWORD') else 'Not Set'}")
print(f"DB_HOST: {os.getenv('DB_HOST')}")
print(f"DB_PORT: {os.getenv('DB_PORT')}")
print(f"DATABASE_URL: {os.getenv('DATABASE_URL')}")

# Get the connection string that would be used
print("\n=== Connection String ===")
conn_string = get_connection_string()
print(f"Connection string is set: {bool(conn_string)}")
if conn_string:
    # Don't print the actual string for security reasons
    print(f"Connection string type: {type(conn_string)}")
    print(f"Connection string length: {len(conn_string)}")

# Try to connect
print("\n=== Database Connection Test ===")
# Try using the DB_PORT from .env directly
try:
    import psycopg2
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    print("Direct connection successful!")
    conn.close()
except Exception as e:
    print(f"Direct connection failed: {str(e)}")

# Try using the app's connection function
conn = get_db_connection()
print(f"App connection function: {conn is not None}")

if conn:
    # Get some basic info
    cursor = conn.cursor()
    print("\nChecking database tables...")
    try:
        cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")
        tables = cursor.fetchall()
        print(f"Number of tables: {len(tables)}")
        print("Table names:")
        for table in tables:
            print(f"  - {table[0]}")
            
        # Check documents table if it exists
        if any(table[0] == 'documents' for table in tables):
            print("\nChecking documents table...")
            cursor.execute("SELECT COUNT(*) FROM documents;")
            doc_count = cursor.fetchone()[0]
            print(f"Number of document chunks: {doc_count}")
            
            # Get a sample of document names
            cursor.execute("SELECT DISTINCT doc_name FROM documents LIMIT 5;")
            doc_names = cursor.fetchall()
            print("Sample document names:")
            for doc in doc_names:
                print(f"  - {doc[0]}")
    except Exception as e:
        print(f"Error querying database: {str(e)}")
    finally:
        cursor.close()
        conn.close()
else:
    print("Failed to connect to the database")

# Also try the inspect function
print("\n=== Database Inspection ===")
try:
    results = inspect_database_contents()
    print(f"Inspection results available: {results is not None}")
    if results:
        print(f"Inspection found data: {bool(results)}")
except Exception as e:
    print(f"Error during inspection: {str(e)}") 