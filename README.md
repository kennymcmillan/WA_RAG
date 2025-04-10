# Aspire Academy Document Assistant

## Overview

The Aspire Academy Document Assistant is a powerful document retrieval and question-answering system built with Streamlit. The application allows users to upload PDF documents, process them for semantic search, and ask questions about their content using natural language.

![Aspire Academy Document Assistant](aspire_logo.png)

## Key Features

- **PDF Document Processing**: Upload and automatically process PDFs
- **Dropbox Integration**: Access, process, and save PDFs to/from Dropbox
- **Smart Chunking**: Adaptive document chunking based on content characteristics
- **Hybrid Vector Search**: Combines semantic and keyword search for optimal results
- **Question Answering**: Ask questions about document content with AI-powered answers
- **Citation Support**: View the exact sources used to generate answers
- **Auto-Processing**: Automatically process documents when uploaded
- **Modern UI**: Clean, responsive interface with Aspire Academy branding

## Technical Stack

- **Frontend**: Streamlit
- **Database**: PostgreSQL with pgvector extension
- **Embeddings**: HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **PDF Processing**: PyPDF2
- **LLM Integration**: OpenRouter API with DeepSeek R1 Distill Llama 70B (free, recommended)
  - **Other Free Alternatives**: Phi-3-mini-4k-instruct, Llama-3-8b-instruct, Gemma-7b-it, or MistralLite
- **Cloud Storage**: Dropbox API
- **Vector Search**: LangChain with pgvector
- **Text Analysis**: scikit-learn for TF-IDF based retrieval

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/aspire-academy-document-assistant.git
cd aspire-academy-document-assistant
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables by creating a `.env` file in the root directory with the following contents:
```
# Database configuration
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=your_database_host
DB_PORT=your_database_port

# OpenRouter API
OPENROUTER_API_KEY=your_openrouter_api_key

# Dropbox integration
DROPBOX_APPKEY=your_dropbox_app_key
DROPBOX_APPSECRET=your_dropbox_app_secret
DROPBOX_REFRESH_TOKEN=your_dropbox_refresh_token
DROPBOX_TOKEN=your_dropbox_access_token
```

Note that the database connection parameters use `DB_NAME`, `DB_USER`, etc. rather than `DATABASE_URL`. Make sure to use these exact variable names.

4. Set up PostgreSQL with pgvector extension:
   - Install PostgreSQL 13+ on your server
   - Install pgvector extension: `CREATE EXTENSION vector;`
   - Create a dedicated database for the application
   - Update your `.env` file with the correct database credentials

5. Set up Dropbox integration (optional):
```bash
python setup_dropbox.py
```
This script will guide you through the OAuth process to obtain the necessary tokens for Dropbox integration.

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Upload your PDFs through the web interface or set up the Dropbox integration for automatic syncing.

3. Select documents from the sidebar to include in your search.

4. Ask questions about your selected documents using the chat interface.

5. The application will display AI-generated answers based on the content of your documents, along with source citations.

## Database Configuration

The application requires a PostgreSQL database with the pgvector extension installed. Here's how to set it up:

### PostgreSQL Setup

1. Install PostgreSQL 13+ on your server
2. Create a database for the application:
```sql
CREATE DATABASE aspire_docs;
```

3. Install the pgvector extension:
```sql
\c aspire_docs
CREATE EXTENSION IF NOT EXISTS vector;
```

4. Create a dedicated user (optional):
```sql
CREATE USER aspire_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE aspire_docs TO aspire_user;
```

5. Update your `.env` file with these credentials:
```
DB_NAME=aspire_docs
DB_USER=aspire_user
DB_PASSWORD=your_secure_password
DB_HOST=localhost
DB_PORT=5432
```

### Database Initialization

The application will automatically initialize the necessary tables on first run, but you can also do this manually:

```bash
python -c "from app_database import initialize_pgvector; initialize_pgvector()"
```

### Verifying Database Connection

If you're having issues with database connection, run the included diagnostic tool:

```bash
python check_db.py
```

This will check your database connection and verify that all tables are properly set up.

## Dropbox Integration

The application supports integration with Dropbox, allowing you to:
- Automatically sync documents from your Dropbox account
- Upload files directly to your Dropbox from the application
- Maintain persistent authentication with Dropbox

To set up Dropbox integration:
1. Create a Dropbox app at https://www.dropbox.com/developers/apps
   - Choose "Scoped access" app type
   - Select "Full Dropbox" access
   - Give your app a name (e.g., "Aspire Document Assistant")
2. Note your app key and app secret from the app settings page
3. Run the setup script: `python setup_dropbox.py`
4. Follow the prompts to enter your Dropbox app key and secret
5. Open the provided URL in your browser and authorize the application
6. Copy the authorization code from the browser and paste it back in the terminal
7. You'll be asked to confirm your app key and secret for verification
8. The script will automatically update your `.env` file with all necessary tokens

The setup process handles all OAuth complexities and ensures your application maintains persistent access to your Dropbox files.

### Persistent Dropbox Authentication

The application implements OAuth 2.0 with refresh tokens for Dropbox integration:
- Long-lived access without manual token renewal
- Automatic token refreshing when expired
- Simple setup process with guided authentication flow

If you encounter any authentication issues with Dropbox:
1. Run `python setup_dropbox.py` again to generate new tokens
2. Follow the on-screen instructions
3. The script will update your `.env` file with fresh credentials

## Application Structure

- **app.py**: Main application file and UI
- **app_database.py**: Database connection and operations
- **app_documents.py**: PDF processing and smart chunking
- **app_embeddings.py**: Vector embedding functionality
- **app_llm.py**: LLM interaction for question answering
- **app_vector.py**: Vector store operations and search
- **app_rag.py**: Advanced RAG techniques including hybrid retrieval
- **app_search.py**: SQL and keyword search functionality
- **app_multi_search.py**: Functions for searching across multiple documents
- **app_dropbox.py**: Dropbox integration utilities
- **setup_dropbox.py**: Helper script for Dropbox OAuth setup
- **check_db.py**: Diagnostic script for database connection
- **check_streamlit_env.py**: Tool to verify Streamlit environment
- **clear_db.py**: Utility to clear database tables
- **deploy_secrets.py**: Helper for Streamlit Cloud deployment
- **env_to_streamlit.py**: Tool for converting .env to Streamlit secrets

## Advanced Features

### Hybrid Retrieval System

The application uses a sophisticated hybrid retrieval approach that combines multiple search methods:

1. **Vector Similarity Search**: 
   - Uses embeddings to find semantically similar content
   - Identifies relevant passages even when keywords don't match
   - Implemented with pgvector for efficient similarity queries

2. **Natural Language SQL Search**:
   - Uses SQL queries to find relevant content based on keywords
   - Particularly effective for technical terms and exact matches
   - Provides fast and reliable search results

3. **Table-Oriented Search**:
   - Automatically detects when a query is likely looking for tabular data
   - Boosts relevance of table chunks in search results
   - Provides enhanced results for data and statistics queries

The system automatically combines results from all methods, tracks metrics for each search type, and ranks them by relevance, providing comprehensive results.

### Advanced RAG Techniques

The system implements several advanced Retrieval-Augmented Generation techniques:

1. **Parent-Child Document Retrieval**:
   - Retrieves smaller chunks for precise matching
   - Includes parent documents for additional context
   - Preserves broader context while maintaining specificity

2. **DocumentCollection System**:
   - Custom extension of document lists that tracks search metrics
   - Maintains information about which search methods found which results
   - Enables diagnostic information and performance tracking

3. **Optimized Chunking**:
   - Adaptive chunk sizes based on document characteristics
   - Smart separator detection preserves natural document structure
   - Maintains semantic coherence in chunks

4. **Enhanced Result Ranking**:
   - Multi-factor ranking based on semantic relevance and keyword matches
   - Weights results based on various factors (keyword presence, position, etc.)
   - Provides more relevant results first

### Parallel Processing

For improved performance when processing multiple documents, the application uses parallel processing with ThreadPoolExecutor.

## Troubleshooting

### Database Connection Issues

If you see "Database connection parameters missing" or similar warnings:

1. **Check your .env file**: Make sure it exists and contains the correct database parameters:
   ```
   DB_NAME=your_database_name
   DB_USER=your_database_user
   DB_PASSWORD=your_database_password
   DB_HOST=your_database_host
   DB_PORT=your_database_port
   ```

2. **Verify environment variable loading**: Run the diagnostic script:
   ```bash
   python check_streamlit_env.py
   ```
   This will show if your environment variables are being loaded correctly.

3. **Check PostgreSQL connection**: Make sure PostgreSQL is running and accessible:
   ```bash
   python check_db.py
   ```
   This script will test the database connection and report any issues.

4. **Common problems**:
   - Incorrect port number (standard PostgreSQL port is 5432)
   - Missing pgvector extension
   - Database server not running
   - Incorrect host address (use 'localhost' for local development)

### Dropbox Authentication

- **Token Expired Error**: When you see "Dropbox token expired or invalid", run:
  ```bash
  python setup_dropbox.py
  ```
  to refresh your Dropbox tokens.

- **Missing Refresh Token**: If you see "Unable to refresh access token without refresh token", make sure your `.env` file has the `DROPBOX_REFRESH_TOKEN` variable.

### Other Common Issues

- **PDF Processing Errors**: Ensure PDFs are not corrupted and are text-based (not scanned images)
- **Vector Search Issues**: Make sure the pgvector extension is properly installed in your PostgreSQL database
- **Table Extraction Requirements**: For table extraction to work, you need Java Runtime Environment (JRE) installed as tabula-py depends on it

## Deployment to Streamlit Cloud

This application can be easily deployed to Streamlit Cloud for sharing with others.

### Transferring Environment Variables to Streamlit Cloud

For Streamlit Cloud deployment, you'll need to transfer your environment variables to Streamlit's secrets management system. Two utility scripts are provided to help with this process:

1. **For cloud deployment**: 
   ```bash
   python deploy_secrets.py
   ```
   This script will read your `.env` file and generate a formatted output that you can directly copy and paste into Streamlit Cloud's secrets management interface.

2. **For local development with secrets.toml**:
   ```bash
   python env_to_streamlit.py
   ```
   This script will read your `.env` file and create a `.streamlit/secrets.toml` file that can be used for local development with the Streamlit secrets API.

### Deployment Steps

1. Push your code to a GitHub repository
2. Sign in to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app" and select your repository
4. Set the main file path to `app.py`
5. Run `python deploy_secrets.py` locally to generate the secrets format
6. Copy the output and paste it into the Streamlit Cloud secrets management panel
7. Deploy your app!

The application is configured to automatically detect whether it's running locally or in Streamlit Cloud and will use the appropriate source for environment variables.

## License

[Specify your license]

## Contributors

[List contributors or contact information]
