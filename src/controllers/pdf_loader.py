import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(uploaded_file):
    """Load the PDF file."""
    start_time = time.time()
    with open("temp_pdf_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    docs = PyPDFLoader("temp_pdf_file.pdf").load()
    end_time = time.time()
    print(f"PDF loading time: {end_time - start_time:.2f} seconds")
    return docs

def split_document(docs):
    """Split the document into chunks."""
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    end_time = time.time()
    print(f"Document splitting time: {end_time - start_time:.2f} seconds")
    return splits
