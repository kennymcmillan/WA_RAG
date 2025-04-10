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
- **app_vector.py**: Vector store operations and search
- **app_rag.py**: Advanced RAG techniques including hybrid retrieval
- **app_dropbox.py**: Dropbox integration
- **setup_dropbox.py**: Helper script for Dropbox OAuth setup

## Advanced Features

### Hybrid Retrieval System

The application uses a sophisticated hybrid retrieval approach that combines multiple search methods:

1. **Vector Similarity Search**: 
   - Uses embeddings to find semantically similar content
   - Identifies relevant passages even when keywords don't match
   - Implemented with pgvector for efficient similarity queries

2. **TF-IDF Based Search**:
   - Uses statistical term importance to find relevant chunks
   - Particularly effective for technical terms and exact matches
   - Implements scikit-learn's TF-IDF vectorizer with cosine similarity

3. **Keyword Fallback Search**:
   - Simple keyword matching as a reliability fallback
   - Ensures results even if other methods fail

The hybrid approach combines results from all methods and ranks them by relevance, providing more comprehensive and accurate results than any single method alone.

### Advanced RAG Techniques

The system implements several advanced Retrieval-Augmented Generation techniques:

1. **Parent-Child Document Retrieval**:
   - Retrieves smaller chunks for precise matching
   - Includes parent documents for additional context
   - Preserves broader context while maintaining specificity

2. **Optimized Chunking**:
   - Adaptive chunk sizes based on document characteristics
   - Shorter documents: 300 character chunks with 30 char overlap
   - Average documents: 500 character chunks with 50 char overlap
   - Longer documents: 1000 character chunks with 100 char overlap

3. **Enhanced Result Ranking**:
   - Multi-factor ranking based on semantic relevance and keyword matches
   - Weights results based on various factors (keyword presence, position, etc.)
   - Provides more relevant results first

4. **Diagnostic Feedback**:
   - Transparency in the retrieval process
   - Shows number of results from each retrieval method
   - Helps diagnose and optimize retrieval performance

### LLM Performance

The application uses **DeepSeek R1 Distill Llama 70B** (free) through OpenRouter API. This model offers exceptional performance for document analysis and question answering:

- **High Reasoning Capabilities**: Excels at mathematical and logical reasoning (AIME 2024 pass@1: 70.0, MATH-500 pass@1: 94.5)
- **128,000 Token Context Window**: Can process extensive document chunks for comprehensive analysis
- **Zero Cost**: Completely free for both input and output tokens
- **Advanced Distillation**: Distilled from Llama-3.3-70B-Instruct using DeepSeek R1's outputs
- **Competitive Performance**: Comparable to larger frontier models

To use this model in your application, specify the following in your API calls:
```python
response = openrouter_client.chat.completions.create(
    model="deepseek/deepseek-r1-distill-llama-70b:free",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query_with_context}
    ]
)
```

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
- **Missing Dependencies**: If you encounter `scikit-learn` related errors, run `pip install scikit-learn numpy`
- **Vector Search Issues**: Make sure the pgvector extension is properly installed in your PostgreSQL database
- **API Limits**: If you encounter rate limits with OpenRouter, consider upgrading your API tier

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
