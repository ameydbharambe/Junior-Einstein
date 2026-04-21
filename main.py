import os
import io
from dotenv import load_dotenv
from google import genai
import pdfplumber
import pymupdf
import streamlit as st
from PIL import Image
from google.genai import types

# -----------------------------
# Setup
# -----------------------------
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY environment variable not found.")
    st.stop()

client = genai.Client(api_key=api_key)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Junior Einstein",
    page_icon="📘",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f7fbff 0%, #edf5ff 100%);
        }

        .main-title {
            font-size: 2.7rem;
            font-weight: 800;
            color: #0f3d91;
            margin-bottom: 0.2rem;
            letter-spacing: -0.02em;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #4267a3;
            margin-bottom: 1.8rem;
        }

        .hero-box {
            background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);
            border: 1px solid #bfdbfe;
            border-radius: 20px;
            padding: 1.4rem 1.4rem 1.2rem 1.4rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 10px 30px rgba(37, 99, 235, 0.08);
        }

        .section-card {
            background: white;
            border: 1px solid #dbeafe;
            border-radius: 18px;
            padding: 1.2rem;
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.06);
            margin-bottom: 1rem;
        }

        .section-title {
            color: #0f3d91;
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        .small-label {
            color: #5b7db3;
            font-size: 0.92rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.4rem;
        }

        .diagram-card {
            background: #ffffff;
            border: 1px solid #cfe3ff;
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.06);
        }

        .diagram-caption {
            color: #214d9c;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .footer-note {
            color: #6b85b6;
            font-size: 0.92rem;
            margin-top: 0.5rem;
        }

        [data-testid="stSidebar"] {
            background: #f4f9ff;
            border-right: 1px solid #dbeafe;
        }

        .stButton > button {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1rem;
            font-weight: 600;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
            color: white;
        }

        .stFileUploader {
            background: white;
            border-radius: 16px;
            padding: 0.6rem;
            border: 1px solid #dbeafe;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Helpers
# -----------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_table_info(pdf_file):
    table_strings = []
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for idx, table in enumerate(tables, start=1):
                formatted_table = "\n".join(
                    [" | ".join([str(cell) if cell else "" for cell in row]) for row in table]
                )
                table_strings.append(f"Page {page_num}, Table {idx}\n{formatted_table}")
    return "\n\n".join(table_strings)

def extract_and_analyze_images(pdf_path):
    """
    Returns a list of dictionaries:
    [
        {
            "page": 1,
            "index": 1,
            "image": PIL.Image,
            "description": "..."
        },
        ...
    ]
    """
    doc = pymupdf.open(pdf_path)
    diagram_data = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list, start=1):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image.get("ext", "png").lower()

                mime_type = f"image/{image_ext}"
                if image_ext == "jpg":
                    mime_type = "image/jpeg"

                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                image_part = types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
                )

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        image_part,
                        (
                            "Explain what this research paper figure, chart, or diagram shows. "
                            "Use simple language, break it into smaller parts, and focus on what "
                            "a student should notice from the visual."
                        )
                    ]
                )

                diagram_data.append({
                    "page": page_index + 1,
                    "index": img_index,
                    "image": pil_image,
                    "description": response.text.strip() if response.text else "No description generated."
                })

            except Exception as e:
                diagram_data.append({
                    "page": page_index + 1,
                    "index": img_index,
                    "image": None,
                    "description": f"Could not analyze this image: {e}"
                })

    return diagram_data

def generate_final_summary(text, tables, diagrams):
    diagram_text = "\n\n".join(
        [f"Page {d['page']} Figure {d['index']}: {d['description']}" for d in diagrams]
    )

    prompt = f"""
    You are a strong science communicator. Explain this paper in a way that a smart high school student can understand.

    Rules:
    1. Start with the main problem the paper is solving and why it matters.
    2. Use simple language and define difficult ideas in plain English.
    3. Give a step-by-step explanation of what the researchers did.
    4. Include what the visuals help show.
    5. Mention the results, strengths, and weaknesses in a balanced way.
    6. Keep it concise, organized, and under 500 words.
    7. If relevant, compare to older or standard approaches.

    Input:
    TEXT:
    {text[:8000]}

    TABLES:
    {tables[:4000] if tables else "No tables found."}

    DIAGRAM NOTES:
    {diagram_text[:4000] if diagram_text else "No diagrams found."}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text.strip() if response.text else "No summary generated."

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## Settings")
    st.write("This app analyzes research papers using Gemini 2.5 Flash.")
    st.write("It extracts text, tables, and figures, then explains them clearly.")
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared.")

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero-box">
        <div class="main-title">Junior Einstein</div>
        <div class="subtitle">
            Upload a research paper and get a cleaner, more visual explanation of the ideas, results, and diagrams.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# -----------------------------
# Main App
# -----------------------------
if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        with st.status("Analyzing paper...", expanded=True) as status:
            st.write("Extracting text...")
            text = extract_text_from_pdf(temp_path)

            st.write("Reading tables...")
            tables = extract_table_info(temp_path)

            st.write("Finding and analyzing figures...")
            diagrams = extract_and_analyze_images(temp_path)

            st.write("Writing summary...")
            summary = generate_final_summary(text, tables, diagrams)

            status.update(label="Analysis complete", state="complete", expanded=False)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Paper Summary</div>", unsafe_allow_html=True)
        st.markdown(summary)
        st.markdown("</div>", unsafe_allow_html=True)

        tabs = st.tabs(["Figures", "Tables", "Raw Text Preview"])

        with tabs[0]:
            st.markdown("<div class='small-label'>Visual Breakdown</div>", unsafe_allow_html=True)

            if diagrams:
                for d in diagrams:
                    st.markdown("<div class='diagram-card'>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='diagram-caption'>Page {d['page']} • Figure {d['index']}</div>",
                        unsafe_allow_html=True
                    )

                    if d["image"] is not None:
                        st.image(d["image"], use_container_width=True)

                    st.write(d["description"])
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No figures or diagrams were found in this PDF.")

        with tabs[1]:
            st.markdown("<div class='small-label'>Extracted Tables</div>", unsafe_allow_html=True)
            if tables.strip():
                st.text(tables)
            else:
                st.info("No tables were found in this PDF.")

        with tabs[2]:
            st.markdown("<div class='small-label'>Text Preview</div>", unsafe_allow_html=True)
            preview = text[:5000] if text else ""
            if preview.strip():
                st.text(preview)
            else:
                st.info("No readable text was extracted from this PDF.")

        st.markdown(
            "<div class='footer-note'>Tip: Papers with scanned pages or unusual figure formats may need extra handling for the best results.</div>",
            unsafe_allow_html=True
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
