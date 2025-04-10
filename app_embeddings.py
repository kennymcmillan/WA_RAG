"""
app_embeddings.py

This module manages document embedding functionality.
Provides:
- Initialization of the HuggingFace embeddings model
- A pre-loaded embeddings model instance for use throughout the application
- Uses the all-MiniLM-L6-v2 model for creating 384-dimensional embeddings
"""

from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import logging

def load_embeddings_model():
    logging.info("Initializing HuggingFace embeddings model...")
    try:
        # Use the updated import path
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logging.info("HuggingFace embeddings model initialized successfully.")
        return embeddings
    except Exception as e:
        logging.error(f"Failed to initialize HuggingFace embeddings model: {e}")
        st.error(f"Failed to initialize HuggingFace embeddings model: {e}")
        return None

embeddings = load_embeddings_model() 