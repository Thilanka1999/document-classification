import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes

def convert_pdf_to_image(pdf_file_bytes):
    """
    Convert PDF file to a list of images
    
    Args:
        pdf_file_bytes: PDF file as bytes
        
    Returns:
        List of PIL Images
    """
    images = convert_from_bytes(pdf_file_bytes)
    return images

def process_image_for_ocr(image):
    """
    Preprocess image for better OCR results
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        Preprocessed numpy array
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Denoise image
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    return denoised

def preprocess_for_classification(image, target_size=(224, 224)):
    """
    Preprocess image for document classification
    
    Args:
        image: PIL Image or numpy array
        target_size: Target size for model input
        
    Returns:
        Preprocessed numpy array
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize image
    resized = cv2.resize(image, target_size)
    
    # Convert to RGB if grayscale
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    
    # Normalize pixel values to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized

def extract_text_from_image(image):
    """
    Extract text from image using OCR
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        Extracted text as string
    """
    # Preprocess image for OCR
    processed_img = process_image_for_ocr(image)
    
    # Extract text using Tesseract
    text = pytesseract.image_to_string(processed_img)
    
    return text

def extract_structured_info(image, data_type):
    """
    Extract structured information from image based on document type
    
    Args:
        image: PIL Image or numpy array
        data_type: Type of document
        
    Returns:
        Dict of extracted information
    """
    # Extract text from image
    text = extract_text_from_image(image)
    
    # Parse text based on document type
    if data_type == 'invoice':
        return parse_invoice(text)
    elif data_type == 'resume':
        return parse_resume(text)
    elif data_type == 'receipt':
        return parse_receipt(text)
    else:
        return {'text': text}
    
def parse_invoice(text):
    """
    Parse invoice text to extract structured information
    
    Args:
        text: Extracted text from OCR
        
    Returns:
        Dict of extracted information
    """
    lines = text.split('\n')
    
    # Initialize extracted data
    data = {
        'invoice_number': None,
        'date': None,
        'total_amount': None,
        'vendor': None,
        'items': []
    }
    
    # Simple rule-based extraction
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Extract invoice number
        if 'invoice' in line.lower() and '#' in line:
            parts = line.split('#')
            if len(parts) > 1:
                data['invoice_number'] = parts[1].strip()
        elif 'invoice number' in line.lower() or 'invoice no' in line.lower():
            if i + 1 < len(lines):
                data['invoice_number'] = lines[i + 1].strip()
        
        # Extract date
        if 'date' in line.lower():
            parts = line.split(':')
            if len(parts) > 1:
                data['date'] = parts[1].strip()
        
        # Extract total amount
        if 'total' in line.lower() and ('$' in line or '£' in line or '€' in line):
            import re
            amounts = re.findall(r'\$\d+\.\d+|\£\d+\.\d+|\€\d+\.\d+|\d+\.\d+', line)
            if amounts:
                data['total_amount'] = amounts[0]
        
        # Extract vendor name (usually at the top)
        if i < 5 and not data['vendor']:
            if len(line) > 3 and not any(keyword in line.lower() for keyword in ['invoice', 'date', 'bill', 'receipt']):
                data['vendor'] = line
    
    return data

def parse_resume(text):
    """
    Parse resume text to extract structured information
    
    Args:
        text: Extracted text from OCR
        
    Returns:
        Dict of extracted information
    """
    lines = text.split('\n')
    
    # Initialize extracted data
    data = {
        'name': None,
        'email': None,
        'phone': None,
        'education': [],
        'experience': [],
        'skills': []
    }
    
    current_section = None
    
    # Simple rule-based extraction
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Extract email
        if '@' in line and '.' in line:
            import re
            emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', line)
            if emails:
                data['email'] = emails[0]
        
        # Extract phone
        if any(char.isdigit() for char in line):
            import re
            phones = re.findall(r'\+?[\d\s()-]{10,}', line)
            if phones and not data['phone']:
                data['phone'] = phones[0]
        
        # Identify sections
        if 'education' in line.lower():
            current_section = 'education'
            continue
        elif 'experience' in line.lower() or 'employment' in line.lower() or 'work' in line.lower():
            current_section = 'experience'
            continue
        elif 'skill' in line.lower():
            current_section = 'skills'
            continue
        
        # Extract data based on current section
        if current_section == 'education' and len(line) > 10:
            data['education'].append(line)
        elif current_section == 'experience' and len(line) > 10:
            data['experience'].append(line)
        elif current_section == 'skills':
            data['skills'].extend([skill.strip() for skill in line.split(',')])
        
        # Extract name (usually at the top)
        if i < 3 and not data['name'] and len(line) > 3:
            if not any(keyword in line.lower() for keyword in ['resume', 'cv', 'curriculum']):
                data['name'] = line
    
    return data

def parse_receipt(text):
    """
    Parse receipt text to extract structured information
    
    Args:
        text: Extracted text from OCR
        
    Returns:
        Dict of extracted information
    """
    lines = text.split('\n')
    
    # Initialize extracted data
    data = {
        'merchant': None,
        'date': None,
        'time': None,
        'total': None,
        'items': []
    }
    
    item_section = False
    
    # Simple rule-based extraction
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Extract date
        if 'date' in line.lower():
            import re
            dates = re.findall(r'\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}', line)
            if dates:
                data['date'] = dates[0]
        
        # Extract time
        if 'time' in line.lower():
            import re
            times = re.findall(r'\d{1,2}:\d{2}', line)
            if times:
                data['time'] = times[0]
        
        # Extract total
        if 'total' in line.lower():
            import re
            amounts = re.findall(r'\$\d+\.\d+|\£\d+\.\d+|\€\d+\.\d+|\d+\.\d+', line)
            if amounts:
                data['total'] = amounts[0]
        
        # Extract items
        if 'item' in line.lower() or 'qty' in line.lower() or 'quantity' in line.lower():
            item_section = True
            continue
        
        if item_section and len(line) > 5:
            # Try to identify if this line contains an item
            if re.search(r'\d+\s+[\w\s]+\s+[\$\£\€]?\d+\.\d+', line):
                data['items'].append(line)
        
        # Extract merchant name (usually at the top)
        if i < 3 and not data['merchant']:
            if len(line) > 3 and not any(keyword in line.lower() for keyword in ['receipt', 'invoice', 'date', 'time']):
                data['merchant'] = line
    
    return data 