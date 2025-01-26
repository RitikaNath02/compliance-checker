import streamlit as st
from src.controllers.pdf_loader import load_pdf, split_document
from src.config.embeddings import create_embeddings, load_embeddings
from src.utils.vectorstore import initialize_pinecone, upload_to_pinecone
from src.models.llama_model import query_selected_models  # Import the model query function

st.title("PDF Ingestion and Query with LLaMA or Groq")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    docs = load_pdf(uploaded_file)
    splits = split_document(docs)
    vectorstore = create_embeddings(splits)
    st.write("PDF processed and embeddings created.")

    # Optional: Upload embeddings to Pinecone
    if st.checkbox("Upload embeddings to Pinecone"):
        initialize_pinecone()
        upload_to_pinecone(vectorstore)
        st.write("Embeddings uploaded to Pinecone.")

# Step 2: Model Selection
st.subheader("Query Models")
selected_models = st.multiselect(
    "Select models to query:", 
    options=["LLaMA", "Groq"], 
    default=["LLaMA"]
)

# Step 3: User Query
user_query = st.text_input("Ask a question to the selected model(s)")

if user_query and selected_models:
    try:
        vectorstore = load_embeddings()  # Load previously created embeddings
        # Query the selected models
        responses = query_selected_models(
            user_query, 
            vectorstore, 
            selected_models=[model.lower() for model in selected_models]
        )
        
        # Display responses
        for model, response in responses.items():
            st.subheader(f"{model} Response")
            st.write(response)

    except Exception as e:
        st.error(f"Error: {str(e)}")

