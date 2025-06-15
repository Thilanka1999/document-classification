import cv2
import numpy as np
import pytesseract
import easyocr
import re
from PIL import Image
import os
import json
from utils.preprocess import (
    process_image_for_ocr, 
    parse_invoice, 
    parse_resume, 
    parse_receipt
)

class DocumentExtractor:
    def __init__(self, ocr_engine='tesseract', config_path=None):
        """
        Document information extractor
        
        Args:
            ocr_engine: OCR engine to use ('tesseract' or 'easyocr')
            config_path: Path to extraction configuration file
        """
        self.ocr_engine = ocr_engine
        
        # Initialize OCR engines
        if ocr_engine == 'easyocr':
            self.reader = easyocr.Reader(['en'])
        
        # Load extraction configuration if provided
        self.config = None
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
    def extract_text(self, image):
        """
        Extract text from image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Extracted text
        """
        # Process image for better OCR results
        processed_img = process_image_for_ocr(image)
        
        # Extract text using specified OCR engine
        if self.ocr_engine == 'tesseract':
            text = pytesseract.image_to_string(processed_img)
            return text
        elif self.ocr_engine == 'easyocr':
            result = self.reader.readtext(processed_img)
            text = ' '.join([item[1] for item in result])
            return text
        else:
            raise ValueError(f"Unsupported OCR engine: {self.ocr_engine}")
    
    def extract_info(self, image, doc_type):
        """
        Extract structured information from document
        
        Args:
            image: PIL Image or numpy array
            doc_type: Document type (invoice, resume, receipt)
            
        Returns:
            Dict of extracted information
        """
        # Extract text from image
        text = self.extract_text(image)
        
        # Extract information based on document type
        if doc_type == 'invoice':
            return parse_invoice(text)
        elif doc_type == 'resume':
            return parse_resume(text)
        elif doc_type == 'receipt':
            return parse_receipt(text)
        else:
            return {'text': text}
    
    def extract_fields_by_config(self, image, doc_type):
        """
        Extract specific fields from document based on configuration
        
        Args:
            image: PIL Image or numpy array
            doc_type: Document type
            
        Returns:
            Dict of extracted fields
        """
        # Check if configuration is available
        if not self.config or doc_type not in self.config:
            return self.extract_info(image, doc_type)
        
        # Extract text from image
        text = self.extract_text(image)
        
        # Get field extraction rules for document type
        field_rules = self.config[doc_type]
        
        # Extract fields based on rules
        extracted_fields = {}
        
        for field_name, rule in field_rules.items():
            if rule['type'] == 'regex':
                # Extract field using regex
                matches = re.findall(rule['pattern'], text)
                if matches:
                    extracted_fields[field_name] = matches[0] if rule.get('first_match_only', True) else matches
            elif rule['type'] == 'region':
                # Extract field from specific image region
                if isinstance(image, Image.Image):
                    img_array = np.array(image)
                else:
                    img_array = image.copy()
                
                x, y, w, h = rule['coordinates']
                region = img_array[y:y+h, x:x+w]
                region_text = pytesseract.image_to_string(region)
                extracted_fields[field_name] = region_text.strip()
        
        return extracted_fields
    
    def save_config(self, config_path):
        """
        Save extraction configuration to file
        
        Args:
            config_path: Path to save configuration
        """
        if self.config:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
                
    def add_field_extraction_rule(self, doc_type, field_name, rule_type, pattern=None, coordinates=None, first_match_only=True):
        """
        Add field extraction rule to configuration
        
        Args:
            doc_type: Document type
            field_name: Field name to extract
            rule_type: Rule type ('regex' or 'region')
            pattern: Regex pattern (for 'regex' rule type)
            coordinates: (x, y, w, h) coordinates (for 'region' rule type)
            first_match_only: Whether to return only first match (for 'regex' rule type)
        """
        # Initialize configuration if needed
        if not self.config:
            self.config = {}
            
        # Initialize document type configuration if needed
        if doc_type not in self.config:
            self.config[doc_type] = {}
            
        # Add field extraction rule
        if rule_type == 'regex':
            self.config[doc_type][field_name] = {
                'type': 'regex',
                'pattern': pattern,
                'first_match_only': first_match_only
            }
        elif rule_type == 'region':
            self.config[doc_type][field_name] = {
                'type': 'region',
                'coordinates': coordinates
            }
        else:
            raise ValueError(f"Unsupported rule type: {rule_type}")
            
    def get_text_regions(self, image):
        """
        Detect text regions in image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            List of (x, y, w, h) tuples for detected text regions
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        min_area = 50  # Minimum area to consider as text
        text_regions = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if area > min_area:
                text_regions.append((x, y, w, h))
                
        return text_regions
        
    def custom_extract(self, image, doc_type, custom_rules):
        """
        Extract information using custom rules
        
        Args:
            image: PIL Image or numpy array
            doc_type: Document type
            custom_rules: Dict of custom extraction rules
            
        Returns:
            Dict of extracted information
        """
        # Extract text from image
        text = self.extract_text(image)
        
        # Extract information based on custom rules
        extracted_info = {}
        
        for field_name, rule in custom_rules.items():
            if rule['type'] == 'regex':
                matches = re.findall(rule['pattern'], text)
                if matches:
                    extracted_info[field_name] = matches[0] if rule.get('first_match_only', True) else matches
            elif rule['type'] == 'keyword':
                keyword = rule['keyword']
                if keyword in text:
                    lines = text.split('\n')
                    for i, line in enumerate(lines):
                        if keyword in line:
                            # Get value from the same line or next line
                            if ':' in line:
                                extracted_info[field_name] = line.split(':', 1)[1].strip()
                            elif i + 1 < len(lines):
                                extracted_info[field_name] = lines[i + 1].strip()
                            break
                    
        return extracted_info 