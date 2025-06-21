import streamlit as st
import os
import json
import asyncio
from datetime import datetime
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import logging
from processors.document_loader import DocumentLoader
from processors.document_classifier import DocumentClassifier
from processors.data_extractor import DataExtractor
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
with open("config/config.json", "r") as f:
    config = json.load(f)

ui_config = config.get("ui", {})
openai_config = config.get("openai", {})

# Load environment variables
load_dotenv()

# Check for OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.warning("OpenAI API key not found in environment variables!")
    st.warning("OpenAI API key not set. Please add it to your .env file.")

# Initialize processors
document_loader = DocumentLoader()
document_classifier = DocumentClassifier()
data_extractor = DataExtractor()

# Set page config based on configuration
st.set_page_config(
    page_title=ui_config.get("page_title", "Document Processor"),
    page_icon=ui_config.get("page_icon", "📄"),
    layout=ui_config.get("layout", "wide")
)

# Add CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .info-text {
        color: #666;
        font-size: 0.9rem;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Document Classification & Data Extraction</h1>', unsafe_allow_html=True)

# User info (from prompt)
current_date_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
current_user = os.getenv("USER", "User")

st.markdown(f"""
<div class="info-text">
    📅 Current Date/Time (UTC): {current_date_utc}<br>
    👤 User: {current_user}
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.markdown("## Settings")
use_llm = st.sidebar.checkbox("Use LLM for Classification", value=ui_config.get("default_llm_enabled", True))

if not openai_api_key and use_llm:
    st.sidebar.warning("OpenAI API key not set but LLM is enabled!")

# File uploader
st.markdown('<h2 class="sub-header">Upload Document</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "png", "jpg", "jpeg", "txt"])

async def process_document(file_path):
    """Process the document asynchronously"""
    # Load and extract text
    text = document_loader.load_document(file_path)
    if not text or text.strip() == "":
        return {"error": "Could not extract text from document"}
    
    # Classify document
    classification_result = await document_classifier.classify(text, use_llm=use_llm)
    document_type = classification_result.get("category")
    
    if not document_type:
        return {
            "error": "Unable to classify document",
            "classification": classification_result
        }
    
    # Extract data
    extracted_data = await data_extractor.extract_data(text, document_type)
    
    # Return results
    return {
        "filename": os.path.basename(file_path),
        "text_length": len(text),
        "text_sample": text[:500] + "..." if len(text) > 500 else text,
        "classification": classification_result,
        "extracted_data": extracted_data
    }

# Main content
if uploaded_file is not None:
    # Create a temporary file
    with st.spinner("Processing document..."):
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Process the document
        result = asyncio.run(process_document(temp_file_path))
        
        # Clean up the temporary file
        os.remove(temp_file_path)
    
    # Display results
    if "error" in result:
        st.error(f"Error: {result['error']}")
        if "classification" in result:
            st.write("Classification attempt:", result["classification"])
    else:
        # Success
        st.success("Document processed successfully!")
        
        # Display document information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="sub-header">Document Information</h3>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write(f"**Filename:** {result['filename']}")
            st.write(f"**Document Type:** {result['classification']['category']}")
            st.write(f"**Confidence:** {result['classification']['confidence']:.2f}")
            st.write(f"**Classification Method:** {result['classification']['method']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<h3 class="sub-header">Classification Reasoning</h3>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write(result['classification']['reasoning'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<h3 class="sub-header">Extracted Data</h3>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            if "error" in result["extracted_data"]:
                st.error(f"Extraction error: {result['extracted_data']['error']}")
            else:
                # Convert to DataFrame for display
                extracted_items = []
                for k, v in result["extracted_data"].items():
                    # Ensure homogeneous (string) type for Arrow compatibility
                    if isinstance(v, (dict, list)):
                        safe_val = json.dumps(v, ensure_ascii=False)
                    elif v is None:
                        safe_val = ""
                    else:
                        safe_val = str(v)
                    extracted_items.append({"Field": k, "Value": safe_val})
                
                df = pd.DataFrame(extracted_items)
                st.table(df)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show text sample
        with st.expander("Document Text Sample"):
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            st.text(result["text_sample"])
            st.markdown('</div>', unsafe_allow_html=True)

# Demo section
if not uploaded_file:
    st.markdown('<h3 class="sub-header">How It Works</h3>', unsafe_allow_html=True)
    st.markdown("""
    This application can classify documents and extract structured data using AI:
    
    1. **Upload a document** - Supports PDF, images (JPG, PNG), and text files
    2. **Automatic classification** - Identifies if it's an invoice, resume, or contract
    3. **Data extraction** - Pulls out relevant fields based on document type
    
    The system uses:
    - **Document loading** with pypdf and EasyOCR
    - **Classification** with sentence transformers and OpenAI
    - **Data extraction** with LangChain and GPT models
    
    Toggle "Use LLM for Classification" in the sidebar to switch between methods.
    """)

# Footer
st.markdown('<div class="footer">Created with Streamlit and LangChain</div>', unsafe_allow_html=True)