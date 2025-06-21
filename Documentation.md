# Document Classification & Data Extraction System

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Components](#components)
   - [Document Loader](#document-loader)
   - [Document Classifier](#document-classifier)
   - [Data Extractor](#data-extractor)
4. [Implementation Details](#implementation-details)
5. [Configuration Files](#configuration-files)
6. [User Interface](#user-interface)
7. [Installation](#installation)
8. [Dependencies](#dependencies)
9. [Running the Application](#running-the-application)

## Project Overview
This project implements an intelligent document processing system that can:
1. Load various document formats (PDF, images, text files)
2. Automatically classify documents into predefined categories
3. Extract structured data from documents based on their category

The system uses a hybrid approach combining embedding-based similarity metrics and Large Language Models (LLMs) for document classification and data extraction, making it both robust and effective across different document types.

## System Architecture
The application follows a modular architecture with three main processing components:

1. **Document Loader**: Handles the loading and text extraction from various file formats
2. **Document Classifier**: Identifies the document category using multiple classification methods
3. **Data Extractor**: Extracts structured information based on document type

These components work together in a pipeline, with the system first loading the document, then classifying it, and finally extracting relevant data based on the classification result.

## Components

### Document Loader
The Document Loader component (`processors/document_loader.py`) is responsible for:
- Loading documents from various file formats (PDF, images, text)
- Extracting text content using appropriate techniques:
  - Direct text extraction from PDFs using PyPDF
  - OCR for scanned documents or images using EasyOCR
  - Standard text loading for text-based formats

This component provides a unified interface for document loading regardless of format, abstracting away the complexities of different extraction methods.

### Document Classifier
The Document Classifier component (`processors/document_classifier.py`) uses an ensemble method that combines predictions from two different classifiers to improve accuracy and robustness:

1. **Embedding-based Classification**:
   - Uses the Sentence Transformers model (`paraphrase-MiniLM-L6-v2`)
   - Pre-computes embeddings for each category description
   - Calculates similarity scores between the document and each category
   - Provides a stable, fast, and reliable baseline classification.

2. **LLM-based Classification**:
   - Leverages OpenAI's GPT models through LangChain
   - Provides document text and category descriptions to the LLM
   - Returns a category prediction with a confidence score and detailed reasoning.

The system intelligently weighs the outputs from both methods to make a final, more accurate classification decision.

### Data Extractor
The Data Extractor component (`processors/data_extractor.py`) is responsible for:
- Loading schema definitions for each document type
- Creating appropriate prompts for extraction based on document type
- Using LLMs to extract structured data from document text
- Validating extraction results against the schema

The extraction is schema-driven, with each document type having its own set of fields and validation rules defined in the extraction schema configuration.

## Implementation Details

### Classification Approach
The system uses an intelligent ensemble strategy to classify documents, always evaluating both an LLM-based and an embedding-based classifier (unless the LLM is disabled via the UI). The final decision follows a clear set of rules:

1.  **Agreement**: If both classifiers agree on the category, the system uses this category. The final confidence score is a weighted average of the two, providing a more reliable measure. The classification method is marked as `ensemble_agreement`.

2.  **Disagreement**: If the classifiers disagree, the system resolves the conflict based on their confidence scores:
    - The LLM's prediction is chosen if its weighted confidence is higher than the embedding model's and exceeds a configurable threshold (e.g., 70%). In this case, the method is `llm`.
    - Otherwise, the system falls back to the embedding-based prediction, which provides a reliable baseline. The method is marked as `embeddings`.

This ensemble approach leverages the semantic understanding of LLMs while retaining the stability and efficiency of embedding-based methods, with clear rules for combining their strengths.

### Data Extraction Methodology
Data extraction follows these steps:

1. Load the appropriate schema for the identified document type
2. Create detailed field instructions for the LLM based on schema properties
3. Prompt the LLM to extract structured information from the document text
4. Validate the extracted data against required fields in the schema
5. Return the structured data or appropriate error information

The system uses document-specific schemas to ensure extraction is tailored to each document type, improving accuracy and relevance of extracted information.

## Configuration Files

### Categories Configuration (`config/categories.json`)
Defines the document categories the system can classify, with each category having:
- A unique name identifier
- A detailed description used for classification

The system currently supports three document types:
- Invoices
- Resumes
- Contracts

### Extraction Schema (`config/extraction_schema.json`)
Defines the structured data to extract for each document type, including:
- Field properties with type and description
- Required vs. optional fields
- Validation rules

### System Configuration (`config/config.json`)
Contains general settings for the application:
- OpenAI model configuration
- Classification settings (confidence thresholds, fallback behavior)
- Extraction settings

## User Interface
The application provides a Streamlit-based web interface with the following features:

- File upload for various document formats
- Settings panel for configuration options
- Results display showing:
  - Document classification results with confidence and reasoning
  - Extracted structured data in tabular format
  - Document text sample for verification
- Informational content explaining system capabilities

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Thilanka1999/document-classification.git
   cd document-classification
   ```

2. **Create and activate a virtual environment:**

   On Windows:
   ```bash
   python -m venv venv
   .\\venv\\Scripts\\activate
   ```

3. **Install dependencies from `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file** with your OpenAI API key. You can do this by running one of the following commands in your terminal or by creating the file manually.

   On Windows (Command Prompt):
   ```bash
   echo OPENAI_API_KEY=your_api_key_here > .env
   ```

   On Windows (PowerShell):
   ```powershell
   "OPENAI_API_KEY=your_api_key_here" | Out-File -Encoding utf8 .env
   ```

   On macOS/Linux:
   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

## Dependencies
Main dependencies include:
- Streamlit - Web interface
- PyPDF and PDF2Image - PDF processing
- EasyOCR and OpenCV - Image text extraction
- LangChain and OpenAI - LLM integration
- Sentence-Transformers - Text embedding
- Pandas - Data handling
- Dotenv - Environment configuration

Full dependencies are listed in the requirements file.

## Running the Application
To run the application:

1. Ensure all dependencies are installed
2. Make sure your OpenAI API key is set in the `.env` file
3. Execute: `streamlit run app.py`
4. Access the web interface at http://localhost:8501

The application will start in your default web browser, ready to process documents. 