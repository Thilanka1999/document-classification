import streamlit as st
import numpy as np
import cv2
import os
import tempfile
from PIL import Image
import json
from pdf2image import convert_from_bytes

from models.classifier import DocumentClassifier
from models.extractor import DocumentExtractor
from utils.preprocess import preprocess_for_classification
from utils.visualize import format_extracted_data

# Set page config
st.set_page_config(
    page_title="Document Classification and Data Extraction",
    page_icon="📄",
    layout="wide",
)

@st.cache_resource
def load_models():
    """Load classification and extraction models"""
    # Check if models directory exists
    if not os.path.exists("models"):
        os.makedirs("models")
        
    # Load document classifier
    model_path = "models/document_classifier.h5"
    if os.path.exists(model_path):
        classifier = DocumentClassifier(model_path=model_path)
    else:
        classifier = DocumentClassifier()
        
    # Load document extractor
    config_path = "models/extraction_config.json"
    if os.path.exists(config_path):
        extractor = DocumentExtractor(config_path=config_path)
    else:
        extractor = DocumentExtractor()
        
    return classifier, extractor

def process_document(file_bytes, file_type):
    """Process uploaded document"""
    # Load models
    classifier, extractor = load_models()
    
    # Convert document to image
    if file_type == "application/pdf":
        # Convert PDF to image
        images = convert_from_bytes(file_bytes)
        # Use first page for classification
        image = images[0]
    else:
        # Open image directly
        image = Image.open(tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name)
        image.paste(Image.open(tempfile.SpooledTemporaryFile()).convert('RGB'))
        
    # Convert to numpy array
    image_np = np.array(image)
    
    # Preprocess image for classification
    preprocessed = preprocess_for_classification(image_np)
    
    # Classify document
    doc_type, confidence, all_scores = classifier.predict(preprocessed)
    
    # Extract information based on document type
    extracted_info = extractor.extract_info(image_np, doc_type)
    
    return image, doc_type, confidence, extracted_info, all_scores

def main():
    """Main Streamlit application"""
    st.title("📄 Document Classification and Data Extraction")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This application classifies documents into different categories "
        "and extracts relevant information based on the document type."
    )
    
    st.sidebar.header("Supported Document Types")
    st.sidebar.write("- Invoice")
    st.sidebar.write("- Resume")
    st.sidebar.write("- Receipt")
    
    # File upload
    st.header("1. Upload a Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or image file",
        type=["pdf", "png", "jpg", "jpeg"],
    )
    
    if uploaded_file is not None:
        # Get file type
        file_type = uploaded_file.type
        
        # Read file bytes
        file_bytes = uploaded_file.read()
        
        # Process document
        with st.spinner("Processing document..."):
            try:
                image, doc_type, confidence, extracted_info, all_scores = process_document(file_bytes, file_type)
                
                # Display results in columns
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.header("2. Document Classification")
                    st.image(image, caption="Uploaded Document", use_column_width=True)
                    st.write(f"**Document Type:** {doc_type.title()} (Confidence: {confidence:.2f})")
                    
                    # Show all classification scores
                    st.write("Classification Scores:")
                    for doc_class, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"- {doc_class.title()}: {score:.2f}")
                
                with col2:
                    st.header("3. Extracted Information")
                    
                    # Display extracted information based on document type
                    if doc_type == "invoice":
                        st.write("### Invoice Details")
                        st.write(f"**Invoice Number:** {extracted_info.get('invoice_number', 'N/A')}")
                        st.write(f"**Date:** {extracted_info.get('date', 'N/A')}")
                        st.write(f"**Vendor:** {extracted_info.get('vendor', 'N/A')}")
                        st.write(f"**Total Amount:** {extracted_info.get('total_amount', 'N/A')}")
                    
                    elif doc_type == "resume":
                        st.write("### Resume Details")
                        st.write(f"**Name:** {extracted_info.get('name', 'N/A')}")
                        st.write(f"**Email:** {extracted_info.get('email', 'N/A')}")
                        st.write(f"**Phone:** {extracted_info.get('phone', 'N/A')}")
                        
                        st.write("**Education:**")
                        for edu in extracted_info.get('education', []):
                            st.write(f"- {edu}")
                            
                        st.write("**Experience:**")
                        for exp in extracted_info.get('experience', []):
                            st.write(f"- {exp}")
                            
                        st.write("**Skills:**")
                        for skill in extracted_info.get('skills', []):
                            st.write(f"- {skill}")
                    
                    elif doc_type == "receipt":
                        st.write("### Receipt Details")
                        st.write(f"**Merchant:** {extracted_info.get('merchant', 'N/A')}")
                        st.write(f"**Date:** {extracted_info.get('date', 'N/A')}")
                        st.write(f"**Time:** {extracted_info.get('time', 'N/A')}")
                        st.write(f"**Total:** {extracted_info.get('total', 'N/A')}")
                        
                        st.write("**Items:**")
                        for item in extracted_info.get('items', []):
                            st.write(f"- {item}")
                    
                    else:
                        st.write(format_extracted_data(extracted_info))
                    
                    # Download extracted data as JSON
                    st.download_button(
                        "Download Extracted Data",
                        data=json.dumps(extracted_info, indent=4),
                        file_name=f"{doc_type}_data.json",
                        mime="application/json",
                    )
                    
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
    
    # Add instructions when no file is uploaded
    else:
        st.info("👆 Upload a document to get started!")
        st.write(
            "This application will classify your document and extract relevant information based on its type. "
            "Currently, it supports invoices, resumes, and receipts."
        )
        
        # Sample images
        st.header("Examples")
        cols = st.columns(3)
        
        with cols[0]:
            st.write("#### Invoice")
            st.image("https://templates.invoicehome.com/invoice-template-us-neat-750px.png", width=200)
            
        with cols[1]:
            st.write("#### Resume")
            st.image("https://assets.website-files.com/5e9ac4e89ba5994a3ffa4d3e/5e9ebfd15b58b4f296be1acb_Resume%209.png", width=200)
            
        with cols[2]:
            st.write("#### Receipt")
            st.image("https://images.examples.com/wp-content/uploads/2017/05/Sample-Itemized-Receipt-Template.jpg", width=200)

if __name__ == "__main__":
    main() 