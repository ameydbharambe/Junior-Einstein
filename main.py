import os
from dotenv import load_dotenv
from google import genai
import pdfplumber
import pymupdf
import streamlit as st
import PIL.Image
from google.genai import types

#set up environment variables from .env file
load_dotenv() 

# Check if the API key is set in the environment variables
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY environment variable not found.") 
    raise SystemExit

# Initialize the GenAI client and create a chat session
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
chat = client.chats.create(model="gemini-2.5-flash")

#Extract text from a given pdf file
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

#Extract info from any diagrams in the pdf file
def extract_and_analyze_images(pdf_path):
    doc = pymupdf.open(pdf_path)
    image_descriptions = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Create a Part object for Gemini
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )

            # Ask Gemini to explain the specific image
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[image_part, "Explain what this research paper diagram or chart shows by breaking it into smaller bits and only using common vocabulary (i.e. not any technical terms)."]
            )
            image_descriptions.append(f"Page {page_index+1} Diagram: {response.text}")
            
    return image_descriptions

#Extract any information from tables in the pdf file
def extract_table_info(pdf_file):
    table_strings = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                # Convert list of lists into a simple text-based table
                formatted_table = "\n".join([" | ".join([str(cell) if cell else "" for cell in row]) for row in table])
                table_strings.append(formatted_table)
    return "\n\n".join(table_strings)

#Utilize gemini to summarize the extracted text, tables, and diagrams from the pdf file
def generate_final_summary(text, tables, images):
    # Constructing the prompt to guide Gemini's behavior
    prompt = f"""
    You are a world-class science communicator. Your goal is to explain this research 
    to someone with no background in the field.

    RULES:
    1. Use simple, everyday analogies for complex terms (e.g., 'Like a chef testing a soup...')
    2. Avoid all academic jargon. If a term is necessary, define it using common english.
    3. Begin by explaining the problem the research is trying to solve, and why it matters in the real world. Also compare it to any prior work in the field, if applicable.
    4. Provide a step by step workflow of the research process, with minimal technical details. Focus on the big picture of what was done and why, rather than the specific methods.
    5. If there are diagrams, break down what they show in simple terms with page numbers
    6. Clarify what exactly what was done in the research, and what the results were by providing a comparison to the real world (e.g., 'This is like a chef testing a new recipe and finding that it tastes better than the old one.')
    7. Add a sentence about scalability limits and any comparisons to prior work, if applicable.
    8. Keep the summary balanced, no positive or negative bias (address both strengths and weaknesses of the research).
    9. Keep the summary concise, no more than 500 words.
    INPUT DATA:
    TEXT: {text[:8000]}
    TABLES: {tables}
    DIAGRAMS: {images}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text



#Develop the User Interface using Streamlit
st.title("📄 Junior Einstein")
st.write("Upload a dense research paper and get an explanation even a teenager could understand.")

with st.sidebar:
    st.header("Settings")
    st.info("This app uses Gemini 2.5 Flash to analyze all content in a given paper.")
    if st.button("Clear Cache"):
        st.cache_data.clear()

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # Save temp file for path-based processing
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # STATUS CONTAINER (Modern UI element)
    with st.status("Analyzing your paper...", expanded=True) as status:
        st.write("🔍 Extracting text and metadata...")
        text = extract_text_from_pdf(temp_path)
        
        st.write("📊 Reading tables...")
        tables = extract_table_info(temp_path)
        
        st.write("🖼️ Analyzing diagrams...")
        images = extract_and_analyze_images(temp_path)
        
        st.write("📝 Crafting summary...")
        summary = generate_final_summary(text, tables, images)
        
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    # OUTPUT DISPLAY
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🌟 The Big Picture")
        st.markdown(summary)
    
    with col2:
        with st.expander("📝 View Raw Table Data"):
            st.text(tables)
        with st.expander("🖼️ Diagram Insights"):
            for desc in images:
                st.write(desc)

    # CLEANUP
    os.remove(temp_path)
