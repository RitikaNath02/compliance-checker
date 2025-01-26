import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def query_selected_models(query, vectorstore, selected_models=("llama", "groq")):
    """Query the selected models (LLaMA, Groq, or both) using context."""
    llm_llama = Ollama(model="llama3.1") if "llama" in selected_models else None

    try:
        # Retrieve relevant context from the vectorstore
        search_results = vectorstore.similarity_search(query, k=5)
        context = "\n".join([result.page_content for result in search_results])
    except Exception as e:
        context = "No relevant context from the uploaded document."

    prompt = f"""
    You are a helpful assistant. Use the following context to answer the user's question.

    Context:
    {context}

    User's question:
    {query}
    """

    responses = {}

    # Query LLaMA if selected
    if llm_llama:
        try:
            response = llm_llama(prompt)
            if isinstance(response, str):
                responses["LLaMA"] = response
            elif isinstance(response, dict) and "content" in response:
                responses["LLaMA"] = response["content"]
            else:
                responses["LLaMA"] = "Unexpected response format from the LLaMA model."
        except Exception as e:
            responses["LLaMA"] = f"Error querying LLaMA3.1 model: {str(e)}"

    # Query Groq if selected
    if "groq" in selected_models:
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt.strip(),
                    }
                ],
                model="llama3-8b-8192",  # Replace with the appropriate Groq model name
            )
            responses["Groq"] = chat_completion.choices[0].message.content
        except Exception as e:
            responses["Groq"] = f"Error querying Groq model: {str(e)}"

    return responses
