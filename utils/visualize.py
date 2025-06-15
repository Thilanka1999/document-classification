import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def display_image(image):
    """
    Display an image using Matplotlib
    
    Args:
        image: PIL Image or numpy array
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def highlight_regions(image, regions, color=(0, 255, 0), thickness=2):
    """
    Highlight regions on an image
    
    Args:
        image: PIL Image or numpy array
        regions: List of (x, y, w, h) tuples
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Image with highlighted regions
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    # Make a copy of the image
    result = image.copy()
    
    # Draw rectangles around regions
    for x, y, w, h in regions:
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        
    return result

def visualize_classification(image, label, confidence, top_n=None):
    """
    Visualize classification result
    
    Args:
        image: PIL Image or numpy array
        label: Predicted label
        confidence: Confidence score
        top_n: Dict of top N predictions {label: score}
        
    Returns:
        Visualization image
    """
    # Convert numpy array to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
    # Create a copy of the image
    result = image.copy()
    draw = ImageDraw.Draw(result)
    
    # Add classification result
    text = f"Predicted: {label} ({confidence:.2f})"
    draw.text((10, 10), text, fill=(0, 255, 0))
    
    # Add top N predictions if provided
    if top_n:
        y_pos = 40
        for label, score in top_n.items():
            text = f"{label}: {score:.2f}"
            draw.text((10, y_pos), text, fill=(0, 200, 0))
            y_pos += 20
            
    return result

def format_extracted_data(data):
    """
    Format extracted data for display
    
    Args:
        data: Dict of extracted data
        
    Returns:
        Formatted string
    """
    result = []
    
    for key, value in data.items():
        if isinstance(value, list):
            if value:
                result.append(f"{key.replace('_', ' ').title()}:")
                for item in value:
                    result.append(f"  - {item}")
            else:
                result.append(f"{key.replace('_', ' ').title()}: None")
        else:
            result.append(f"{key.replace('_', ' ').title()}: {value}")
            
    return "\n".join(result) 