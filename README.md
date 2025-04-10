# Aspire Academy Document Assistant

## Overview

The Aspire Academy Document Assistant is a powerful document retrieval and question-answering system built with Streamlit. The application allows users to upload PDF documents, process them for semantic search, and ask questions about their content using natural language.

![Aspire Academy Document Assistant](aspire_logo.png)

## Key Features

- **PDF Document Processing**: Upload and automatically process PDFs
- **Dropbox Integration**: Access, process, and save PDFs to/from Dropbox
- **Smart Chunking**: Adaptive document chunking based on content characteristics
- **Vector Search**: Semantic search using pgvector in PostgreSQL
- **Question Answering**: Ask questions about document content with AI-powered answers
- **Citation Support**: View the exact sources used to generate answers
- **Auto-Processing**: Automatically process documents when uploaded
- **Modern UI**: Clean, responsive interface with Aspire Academy branding

## Technical Stack

- **Frontend**: Streamlit
- **Database**: PostgreSQL with pgvector extension
- **Embeddings**: HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **PDF Processing**: PyPDF2
- **LLM Integration**: OpenAI API via OpenRouter
- **Cloud Storage**: Dropbox API
- **Vector Search**: LangChain with pgvector

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
DATABASE_NAME=aspire_docs_db
DATABASE_USER=postgres
DATABASE_PASSWORD=your_password
DATABASE_HOST=localhost
DATABASE_PORT=5432

# Dropbox integration
DROPBOX_APPKEY=your_dropbox_app_key
DROPBOX_APPSECRET=your_dropbox_app_secret
DROPBOX_REFRESH_TOKEN=your_dropbox_refresh_token
DROPBOX_TOKEN=your_dropbox_access_token
```

4. Set up Dropbox integration (optional):
```bash
python setup_dropbox.py
```
This script will guide you through the OAuth process to obtain the necessary tokens for Dropbox integration.

## Usage

1. Start the application:
```bash
python main.py
```

2. Upload your PDFs through the web interface or set up the Dropbox integration for automatic syncing.

3. Enter your query in the search bar to retrieve relevant information from your documents.

4. The application will display the most relevant passages from your documents, along with the source file information.

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
- **app_vector.py**: Vector store operations
- **app_dropbox.py**: Dropbox integration
- **setup_dropbox.py**: Helper script for Dropbox OAuth setup

## Advanced Features

### Optimized Chunking

The application uses adaptive chunking that automatically adjusts based on the content's characteristics:
- Shorter documents (< 1000 chars/page): 300 character chunks with 30 char overlap
- Average documents: 500 character chunks with 50 char overlap
- Longer documents (> 5000 chars/page): 1000 character chunks with 100 char overlap

### Parallel Processing

For improved performance when processing multiple documents, the application uses parallel processing with ThreadPoolExecutor.

### Smart Separators

The chunking system uses a nuanced set of text separators to maintain semantic boundaries:
- Paragraphs, lines, sentences, and other natural text divisions
- Ensures chunks maintain meaningful context

## Troubleshooting

- **Database Connection Issues**: Verify your database credentials and ensure pgvector is installed
- **Dropbox Authentication**: Run `python setup_dropbox.py` to generate new tokens if you encounter authentication errors
- **PDF Processing Errors**: Ensure PDFs are not corrupted and are text-based (not scanned images)
- **API Limits**: If you encounter rate limits with OpenRouter, consider upgrading your API tier

## License

[Specify your license]

## Contributors

[List contributors or contact information]
