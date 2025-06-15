# Document Classification and Data Extraction

## Overview
This project implements a document classification and data extraction system with a user-friendly Streamlit interface. The system can classify uploaded documents into predefined categories and extract relevant information from them based on document type.

## Features
- Document classification using a Convolutional Neural Network (CNN)
- Automatic data extraction from classified documents using OCR
- User-friendly interface for document upload and result display
- Support for multiple document types (Invoice, Resume, Receipt)

## Directory Structure
```
.
├── README.md
├── requirements.txt
├── app.py                  # Main Streamlit application
├── models/
│   ├── classifier.py       # Document classification model
│   └── extractor.py        # Data extraction module
├── utils/
│   ├── preprocess.py       # Image preprocessing utilities
│   └── visualize.py        # Visualization utilities
├── data/
│   ├── train/              # Training data for classification model
│   └── examples/           # Example documents for testing
└── notebooks/
    ├── train_classifier.ipynb  # Notebook for model training
    └── test_extraction.ipynb   # Notebook for testing extraction
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/document-classification-extraction.git
cd document-classification-extraction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application
1. Start the Streamlit app:
```bash
streamlit run app.py
```
2. Open your browser and go to `http://localhost:8501`

## Usage
1. Upload a document (PDF or image) using the file uploader
2. The system will classify the document type
3. Based on the classification, relevant information will be extracted
4. The extracted information will be displayed on the interface

## Model Training
To train the classification model on your own data:
1. Add your training data to the `data/train/` directory
2. Run the training notebook:
```bash
jupyter notebook notebooks/train_classifier.ipynb
```

## Technical Details
- Document classification: CNN-based model trained on document images
- Data extraction: Combination of OCR (Optical Character Recognition) and rule-based information extraction
- UI: Streamlit framework

## License
This project is licensed under the MIT License - see the LICENSE file for details. 