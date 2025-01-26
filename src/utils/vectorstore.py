from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import streamlit as st  # Optional for logging in Streamlit apps

def log_message(message, is_error=False):
    """Utility function to log messages with or without Streamlit."""
    if "streamlit" in globals():
        if is_error:
            st.error(message)
        else:
            st.write(message)
    else:
        if is_error:
            print(f"ERROR: {message}")
        else:
            print(message)

def initialize_pinecone():
    """Initialize Pinecone connection and create an index."""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        log_message("PINECONE_API_KEY is missing! Please check your .env file.", is_error=True)
        return

    # Initialize Pinecone client
    try:
        pc = Pinecone(api_key=api_key)
        log_message("Pinecone client initialized successfully.")
    except Exception as e:
        log_message(f"Failed to initialize Pinecone client: {str(e)}", is_error=True)
        return

    # Define index name and create/delete as needed
    index_name = "pdf-index"
    try:
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)
            log_message(f"Deleted existing index: {index_name}")

        pc.create_index(
            name=index_name,
            dimension=384,  # Match the embedding dimension
            metric="euclidean",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Adjust region as needed
        )
        log_message(f"Created index: {index_name}")
    except Exception as e:
        log_message(f"Error managing Pinecone index: {str(e)}", is_error=True)

def upload_to_pinecone(vectorstore):
    """Upload embeddings from FAISS to Pinecone."""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    host = os.getenv("PINECONE_HOST")

    if not api_key or not host:
        log_message("PINECONE_API_KEY or PINECONE_HOST is missing! Please check your .env file.", is_error=True)
        return

    # Initialize Pinecone client
    try:
        pc = Pinecone(api_key=api_key)
        index_name = "pdf-index"
        index = pc.Index(index_name)
        log_message(f"Connected to Pinecone index: {index_name}")
    except Exception as e:
        log_message(f"Failed to connect to Pinecone index: {str(e)}", is_error=True)
        return

    # Prepare and upload vectors
    try:
        vectors = []
        for doc_id, embedding in zip(vectorstore.docstore._dict.keys(), vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)):
            metadata = {"source": "pdf"}  # Example metadata
            vectors.append((doc_id, embedding, metadata))  # Format for Pinecone: (id, vector, metadata)

        # Upload to Pinecone
        index.upsert(vectors=vectors)
        log_message("Vectors uploaded to Pinecone successfully.")
    except Exception as e:
        log_message(f"Failed to upload vectors to Pinecone: {str(e)}", is_error=True)
