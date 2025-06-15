"""
Test script to demonstrate the document classification and data extraction functionality.
This script will create a simple demo using sample images.
"""

import os
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tempfile

# Create examples directory if it doesn't exist
os.makedirs('data/examples', exist_ok=True)

# Download sample images
sample_files = {
    'invoice': {
        'url': 'https://templates.invoicehome.com/invoice-template-us-neat-750px.png',
        'path': 'data/examples/sample_invoice.jpg'
    },
    'resume': {
        'url': 'https://assets.website-files.com/5e9ac4e89ba5994a3ffa4d3e/5e9ebfd15b58b4f296be1acb_Resume%209.png',
        'path': 'data/examples/sample_resume.jpg'
    },
    'receipt': {
        'url': 'https://images.examples.com/wp-content/uploads/2017/05/Sample-Itemized-Receipt-Template.jpg',
        'path': 'data/examples/sample_receipt.jpg'
    }
}

# Download sample images if they don't exist
for doc_type, file_info in sample_files.items():
    if not os.path.exists(file_info['path']):
        print(f"Downloading sample {doc_type} image...")
        try:
            response = requests.get(file_info['url'], stream=True)
            if response.status_code == 200:
                with open(file_info['path'], 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {doc_type} image to {file_info['path']}")
            else:
                print(f"Failed to download {doc_type} image. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error downloading {doc_type} image: {str(e)}")

# Import project modules
try:
    from models.classifier import DocumentClassifier
    from models.extractor import DocumentExtractor
    from utils.preprocess import preprocess_for_classification
    
    # Initialize models
    print("Initializing models...")
    classifier = DocumentClassifier()
    extractor = DocumentExtractor()
    
    # Test classification and extraction on each sample
    for doc_type, file_info in sample_files.items():
        if os.path.exists(file_info['path']):
            print(f"\nProcessing sample {doc_type}...")
            
            # Load image
            image = Image.open(file_info['path'])
            
            # Classify document
            image_np = np.array(image)
            preprocessed = preprocess_for_classification(image_np)
            predicted_type, confidence, all_scores = classifier.predict(preprocessed)
            
            print(f"Predicted document type: {predicted_type}")
            print(f"Confidence: {confidence:.2f}")
            
            # Extract information
            extracted_info = extractor.extract_info(image, predicted_type)
            print(f"Extracted information:")
            for key, value in extracted_info.items():
                if isinstance(value, list):
                    print(f"  {key}:")
                    for item in value[:3]:  # Limit to first 3 items
                        print(f"    - {item}")
                    if len(value) > 3:
                        print(f"    - ... ({len(value) - 3} more)")
                else:
                    print(f"  {key}: {value}")
    
    print("\nSetup complete! Run 'streamlit run app.py' to start the application.")
    
except ImportError as e:
    print(f"Error importing project modules: {str(e)}")
    print("Make sure to install all required dependencies from requirements.txt")
except Exception as e:
    print(f"Error: {str(e)}") 