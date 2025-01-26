import os
import pickle
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

embedding_file = "embeddings.pkl"

def create_embeddings(splits):
    """Create and save embeddings."""
    if os.path.exists(embedding_file):
        print("Removing old embeddings file...")
        os.remove(embedding_file)

    model_name = "BAAI/bge-small-en"
    hf_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(splits, hf_embeddings)

    with open(embedding_file, "wb") as f:
        pickle.dump(vectorstore, f)

    return vectorstore

def load_embeddings():
    """Load saved embeddings."""
    if os.path.exists(embedding_file):
        with open(embedding_file, "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError("No embeddings file found.")
