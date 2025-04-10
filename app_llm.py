"""
app_llm.py

This module handles interactions with large language models.
Provides:
- OpenAI client initialization using OpenRouter as a proxy
- Question answering functionality based on provided context
- Uses Claude-3-Haiku model via OpenRouter
- Structured prompting to ensure answers are grounded in the provided context
"""

import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get answer from OpenRouter
def get_answer(question, context, retrieved_docs=None):
    """
    Get an answer from a language model based on the question and context.
    
    Args:
        question (str): The user's question
        context (str): The relevant context for answering the question
        retrieved_docs (list, optional): List of retrieved document objects with metadata
        
    Returns:
        tuple: (answer, sources_text) where answer is the generated response and
              sources_text is formatted text listing the sources used
    """
    try:
        # Check for OpenRouter API key
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logging.error("OPENROUTER_API_KEY not found in environment variables")
            return "I'm sorry, but I can't answer that question because my API key is missing.", ""
        
        # Initialize OpenAI client with OpenRouter base URL
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        # Create better prompt with clear instructions and citation guidance
        prompt_text = f"""
        I need you to answer a question based on the provided context information.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        INSTRUCTIONS:
        1. Only answer what you can based on the context above.
        2. If you don't know or the context doesn't contain the answer, say "I don't have enough information to answer that question."
        3. Keep your answer concise and directly address the question.
        4. Always cite specific page numbers when referring to information from the context.
        5. Format your answer in clear, easy-to-read text.
        6. Do not include general knowledge not found in the context.
        7. Do not cite sources that weren't included in the context.
        """
        
        # Get response from OpenRouter using Claude-3-Haiku
        response = client.chat.completions.create(
            model="anthropic/claude-3-haiku",  # Use Claude-3-Haiku for best balance of performance and cost
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.1,  # Lower temperature for more factual answers
            max_tokens=1000   # Limit token usage
        )
        
        # Extract the answer from the response
        answer = response.choices[0].message.content.strip()
        
        # Process sources from retrieved_docs if available
        sources_text = ""
        if retrieved_docs and len(retrieved_docs) > 0:
            # Extract unique sources
            sources = set()
            for doc in retrieved_docs:
                source = doc.metadata.get('source', 'Unknown Source')
                page = doc.metadata.get('page', 'Unknown Page')
                sources.add(f"{source} (Page {page})")
            
            # Format sources list
            if sources:
                sources_list = sorted(list(sources))
                sources_text = "**Sources:**\n"
                for source in sources_list:
                    sources_text += f"- {source}\n"
        
        return answer, sources_text
        
    except Exception as e:
        logging.error(f"Error getting answer: {str(e)}")
        return f"I encountered an error while processing your question: {str(e)}", "" 