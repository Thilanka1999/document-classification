# Document Classification & Data Extraction

A Streamlit application that classifies documents and extracts structured data using AI techniques. The application supports PDF, image, and text files.

## Features

- **Document Loading**: Extract text from PDF documents and images using OCR.
- **Ensemble Classification**: Intelligent combination of LLM (OpenAI GPT) and embedding-based similarity to classify documents into predefined categories (invoice, resume, contract) with higher accuracy.
- **Data Extraction**: Extract structured data fields based on document type.
- **Configurable**: Easily adjust settings through JSON configuration files.

## Project Structure

```
document-classification/
  ├── app.py               # Main Streamlit application
  ├── config/              # Configuration directory
  │   ├── categories.json  # Document categories & descriptions
  │   ├── config.json      # Main configuration file
  │   └── extraction_schema.json  # Data extraction schemas
  ├── processors/          # Core processing modules
  │   ├── data_extractor.py      # Extract structured data from documents
  │   ├── document_classifier.py # Ensemble document classification logic
  │   └── document_loader.py     # Load & parse different document formats
  └── requirements.txt     # Python dependencies
```

## Configuration

The application uses a hierarchical configuration system with the main settings in `config/config.json`:

- **OpenAI Settings**: Configure model, temperature, and token limits
- **Classification Settings**: Ensemble weights and confidence thresholds
- **Extraction Settings**: Data extraction options
- **Document Loading Settings**: File format support and OCR options
- **UI Settings**: Streamlit interface configuration

## Installation

### Prerequisites
- Python 3.7+
- Git
- An OpenAI API key

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
   .\venv\Scripts\activate
   ```

   On macOS/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your OpenAI API key** to a `.env` file:

   Windows (Command Prompt):
   ```bash
   echo OPENAI_API_KEY=your_api_key_here > .env
   ```

   Windows (PowerShell):
   ```powershell
   "OPENAI_API_KEY=your_api_key_here" | Out-File -Encoding utf8 .env
   ```

   macOS/Linux:
   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then upload a document through the web interface. The system will:
1. Load and parse the document
2. Classify the document type using the ensemble method
3. Extract relevant structured data based on the detected type
4. Display results including confidence scores, reasoning, and extracted fields

## Extending the System

### Adding New Document Categories
1. Add new categories to `config/categories.json` with name and description.
2. Add corresponding extraction schemas to `config/extraction_schema.json`.

### Customizing Configuration
Edit `config/config.json` to adjust model parameters, ensemble weights, confidence thresholds, and other settings. 