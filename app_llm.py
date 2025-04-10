"""
app_llm.py

This module handles interactions with large language models.
Provides:
- OpenAI client initialization using OpenRouter as a proxy
- Question answering functionality based on provided context
- Uses the deepseek-chat model via OpenRouter
- Structured prompting to ensure answers are grounded in the provided context
"""

import os
from openai import OpenAI
import streamlit as st
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client with OpenRouter
logging.info("Initializing OpenAI client...")
try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", "")
    )
    logging.info("OpenAI client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")
    st.error(f"Failed to initialize OpenAI client: {e}")

# Get answer from OpenRouter
def get_answer(question, context):
    try:
        prompt = f"""You are a helpful assistant for Aspire Academy. Answer the question based ONLY on the following context:

Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the information provided in the context.
2. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."
3. Be concise and clear in your response.
4. Do not make up information or use knowledge outside of the provided context.
"""

        response = client.chat.completions.create(
            model="deepseek/deepseek-chat",  # Use the free deepseek model via OpenRouter
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Aspire Academy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "I'm sorry, I encountered an error while generating an answer. Please try again." 