from datetime import datetime
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer
from timescale_vector import client
from fpdf import FPDF
import pandas as pd

# Initialize VectorStore
vec = VectorStore()

# --------------------------------------------------------------
# Shipping question
# --------------------------------------------------------------
relevant_question = "How can I track my order?"

# Perform the search on the vector store
results = vec.search(relevant_question, limit=3)

# Debugging: Print the results of the search
print("Search results:", results)

# Proceed only if results are not empty
if not results.empty:
    # Step 1: Create metadata dictionary (mocking metadata for testing)
    print("Length of results DataFrame:", len(results))
    print("Columns in results DataFrame:", results.columns)
    print("First few rows of results DataFrame:")
    print(results.head())  # Inspect the first few rows of the results

    # Assuming 'agreement_date', 'effective_date', 'expiration_date' are missing in your data,
    # replace them with mock values or skip this if not required.
    sample_metadata = results.apply(
        lambda x: {
            'agreement_date': x.get('agreement_date', 'N/A'),  # Use .get() to avoid key errors
            'effective_date': x.get('effective_date', 'N/A'),
            'expiration_date': x.get('expiration_date', 'N/A')
        },
        axis=1
    )
    
    print("Embedding for query:", vec.get_embedding(relevant_question))

    print("Length of metadata from apply:", len(sample_metadata))
    print("First few entries of metadata from apply:")
    print(sample_metadata.head())  # Inspect the first few entries

    # Apply function to create 'metadata' column
    results['metadata'] = sample_metadata

    # --------------------------------------------------------------
    # Step 2: Prepare the data to be inserted into the database
    # --------------------------------------------------------------
    embedding_data = results[['contents', 'embedding', 'metadata']]  # Ensure 'contents' column is used

    # Prepare DataFrame to upsert
    df = pd.DataFrame(embedding_data)

    # --------------------------------------------------------------
    # Step 3: Upsert into database
    # --------------------------------------------------------------
    if not df.empty:  # Only upsert if the DataFrame is not empty
        vec.upsert(df)
    else:
        print("No data to upsert into the database.")

    # --------------------------------------------------------------
    # Step 4: Generate the response from Synthesizer
    # --------------------------------------------------------------
    response = Synthesizer.generate_response(
        question=relevant_question,
        context=results[['contents', 'metadata']]
    )

    # --------------------------------------------------------------
    # Step 5: Create the PDF report
    # --------------------------------------------------------------
    def create_pdf_report(response, filename="report.pdf"):
        pdf = FPDF()
        pdf.add_page()

        # Set margins to give more space for content
        pdf.set_left_margin(15)
        pdf.set_right_margin(15)

        # Title
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Contract Analysis Report", ln=True, align='C')
        pdf.ln(10)

        # Main answer section
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Analysis Summary:", ln=True)
        pdf.set_font("Helvetica", "", 11)

        # Split answer into paragraphs and clean markdown
        paragraphs = response.answer.split('\n')
        for para in paragraphs:
            # Clean the paragraph of markdown characters
            cleaned_para = para.replace('**', '').replace('*', '').strip()
            if cleaned_para:  # Only process non-empty paragraphs
                # Check if it's a header (like "Compliance Report:", "Strengths:", etc.)
                if any(header in cleaned_para for header in ["Compliance Report:", "Strengths:", "Areas for Improvement:", "Reasoning:", "Additional Information:"]):
                    pdf.set_font("Helvetica", "B", 12)
                    pdf.ln(5)
                    pdf.cell(0, 10, cleaned_para, ln=True)
                    pdf.set_font("Helvetica", "", 11)
                else:
                    pdf.multi_cell(0, 7, cleaned_para)
                    pdf.ln(3)

        pdf.ln(10)

        # Thought process section
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Detailed Analysis:", ln=True)
        pdf.set_font("Helvetica", "", 11)

        for thought in response.thought_process:
            cleaned_thought = thought.replace('**', '').replace('*', '').strip()
            if cleaned_thought:  # Only process non-empty thoughts
                if cleaned_thought.endswith(':'):
                    pdf.set_font("Helvetica", "B", 11)
                    pdf.ln(5)
                    pdf.multi_cell(0, 7, cleaned_thought)
                else:
                    pdf.set_font("Helvetica", "", 11)
                    pdf.multi_cell(0, 7, cleaned_thought)
                    pdf.ln(3)

        # Output PDF
        pdf.output(filename)

else:
    print("No results returned from the vector search.")
