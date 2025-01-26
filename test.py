from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Debugging (optional): Print values
api_key = os.getenv("PINECONE_API_KEY")
host = os.getenv("PINECONE_HOST")

if not api_key or not host:
    raise ValueError("Pinecone API key or host is missing!")

print(f"PINECONE_API_KEY: {api_key[:5]}******")  # Partial key for security
print(f"PINECONE_HOST: {host}")
