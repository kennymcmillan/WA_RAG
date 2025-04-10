import streamlit as st
import os
from dotenv import load_dotenv

# Try to load environment variables from .env
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path, override=True)
    st.success("Loaded environment variables from .env file")
else:
    st.error("No .env file found")

# Display database environment variables
st.header("Database Environment Variables")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Database Variables")
    st.code(f"""
DB_NAME: {os.getenv('DB_NAME')}
DB_USER: {os.getenv('DB_USER')}
DB_PASSWORD: {"*****" if os.getenv('DB_PASSWORD') else "Not Set"}
DB_HOST: {os.getenv('DB_HOST')}
DB_PORT: {os.getenv('DB_PORT')}
DATABASE_URL: {os.getenv('DATABASE_URL')}
    """)

# Test database connection
st.header("Database Connection Test")

try:
    import psycopg2
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    st.success("✅ Database connection successful!")
    
    # Display database info
    cursor = conn.cursor()
    cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")
    tables = cursor.fetchall()
    
    st.subheader(f"Database Tables ({len(tables)})")
    for table in tables:
        st.text(f"- {table[0]}")
        
    # Check documents table
    if any(table[0] == 'documents' for table in tables):
        st.subheader("Documents Table Summary")
        
        cursor.execute("SELECT COUNT(*) FROM documents;")
        doc_count = cursor.fetchone()[0]
        st.text(f"Total document chunks: {doc_count}")
        
        cursor.execute("SELECT DISTINCT doc_name FROM documents;")
        doc_names = cursor.fetchall()
        st.text(f"Number of unique documents: {len(doc_names)}")
        
        st.text("Documents in database:")
        for doc in doc_names:
            st.text(f"- {doc[0]}")
    
    conn.close()
except Exception as e:
    st.error(f"❌ Database connection failed: {str(e)}")
    
# Display environment summary
st.header("System Information")
st.text(f"Python working directory: {os.getcwd()}")
st.text(f".env file location: {dotenv_path}")
st.text(f".env file exists: {os.path.exists(dotenv_path)}") 