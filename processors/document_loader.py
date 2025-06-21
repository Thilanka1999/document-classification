import os
import pypdf
import cv2
import easyocr
import numpy as np
import logging
from pdf2image import convert_from_path
from PIL import Image
from typing import Optional, List, Dict, Any
import json

# Set up logger for this module
logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self, config_path: str = "config/config.json"):
        # Load main configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.loading_config = self.config["document_loading"]
        
        # Initialize OCR reader for image-based documents
        ocr_languages = self.loading_config.get("ocr_languages", ["en"])
        self.reader = easyocr.Reader(ocr_languages)
        
        # Set up supported file extensions
        self.supported_extensions = {
            "document": self.loading_config.get("supported_extensions", {}).get("document", 
                [".pdf", ".txt", ".md", ".csv"]),
            "image": self.loading_config.get("supported_extensions", {}).get("image", 
                [".jpg", ".jpeg", ".png", ".tiff", ".bmp"])
        }
    
    def load_document(self, file_path: str) -> Optional[str]:
        """Load document and extract text from various file formats"""
        if not os.path.exists(file_path):
            logger.error(f"File not found - {file_path}")
            return None
            
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            # PDF files
            if file_ext in self.supported_extensions["document"] and file_ext == '.pdf':
                return self._extract_text_from_pdf(file_path)
                
            # Image files
            elif file_ext in self.supported_extensions["image"]:
                return self._extract_text_from_image(file_path)
                
            # Text files
            elif file_ext in self.supported_extensions["document"] and file_ext != '.pdf':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"Error reading text file {file_path}: {str(e)}", exc_info=True)
                    return None
                    
            # Unsupported format
            else:
                logger.warning(f"Unsupported file format: {file_ext}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}", exc_info=True)
            return None
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file using pypdf"""
        text = ""
        try:
            # Try direct text extraction first
            pdf_reader = pypdf.PdfReader(file_path)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # If no text was extracted, the PDF might be scanned
            if not text.strip() and self.loading_config.get("pdf_extract_images_if_no_text", True):
                logger.info("PDF appears to be scanned, using OCR...")
                images = convert_from_path(file_path)
                for img in images:
                    img_np = np.array(img)
                    result = self.reader.readtext(img_np)
                    for res in result:
                        text += res[1] + " "
                    text += "\n"
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}", exc_info=True)
            return ""
    
    def _extract_text_from_image(self, file_path: str) -> str:
        """Extract text from image file using EasyOCR"""
        try:
            img = cv2.imread(file_path)
            result = self.reader.readtext(img)
            
            text = ""
            for res in result:
                text += res[1] + " "
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}", exc_info=True)
            return ""