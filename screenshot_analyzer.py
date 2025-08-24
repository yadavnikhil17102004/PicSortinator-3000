"""
Screenshot Analyzer - Digital Investigation Tool
Initial Version: Basic Image Detection

This module processes screenshot images and detects whether they contain text, UI elements,
or graphics. It returns a JSON object with detection results.
"""

import os
import sys
import json
import uuid
import cv2
import numpy as np
from PIL import Image
import pytesseract

# Tesseract path setup (Windows users need to set this)
# For Windows: Replace with your Tesseract installation path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class ScreenshotAnalyzer:
    """Class for analyzing screenshot images for digital investigations."""
    
    def __init__(self):
        """Initialize the Screenshot Analyzer."""
        pass
        
    def process_image(self, image_path):
        """
        Process the input image and detect if it contains text or UI elements.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            dict: JSON-compatible dictionary with detection results.
        """
        # Check if the file exists and is an image
        if not os.path.exists(image_path):
            return self._create_error_response("File not found")
            
        if not self._is_valid_image(image_path):
            return self._create_error_response("Invalid image file")
        
        # Generate a unique ID for the image
        image_id = str(uuid.uuid4())
        
        try:
            # Load the image for processing
            img = cv2.imread(image_path)
            
            # Check if the image was loaded successfully
            if img is None:
                return self._create_error_response("Failed to load image")
            
            # Basic image processing to detect text
            is_processable = self._detect_processable_content(img)
            
            # Create the response
            response = {
                "image_id": image_id,
                "processable": is_processable,
                "detected_text": "",  # Empty for now, will be implemented in future versions
                "tags": [],  # Empty for now, will be implemented in future versions
                "category": "uncategorized"
            }
            
            return response
            
        except Exception as e:
            return self._create_error_response(f"Processing error: {str(e)}")
    
    def _detect_processable_content(self, img):
        """
        Detect if the image contains text or UI elements.
        
        Args:
            img: OpenCV image object.
            
        Returns:
            bool: True if the image contains text or UI elements, False otherwise.
        """
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If there are contours, it likely contains text or UI elements
        if len(contours) > 10:  # Arbitrary threshold, can be adjusted
            return True
            
        # Apply edge detection for UI elements
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = cv2.countNonZero(edges)
        
        # If there are enough edge pixels, it likely contains UI elements
        if edge_pixels > (img.shape[0] * img.shape[1] * 0.01):  # At least 1% of pixels are edges
            return True
            
        return False
    
    def _is_valid_image(self, image_path):
        """
        Check if the file is a valid image.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            bool: True if the file is a valid image, False otherwise.
        """
        try:
            img = Image.open(image_path)
            img.verify()  # Verify it's an image
            return True
        except:
            return False
    
    def _create_error_response(self, error_message):
        """
        Create an error response.
        
        Args:
            error_message (str): Error message.
            
        Returns:
            dict: Error response as a dictionary.
        """
        return {
            "image_id": "error",
            "processable": False,
            "detected_text": "",
            "tags": [],
            "category": "error",
            "error": error_message
        }


def main():
    """Main function to run the Screenshot Analyzer."""
    if len(sys.argv) < 2:
        print("Usage: python screenshot_analyzer.py <image_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    analyzer = ScreenshotAnalyzer()
    result = analyzer.process_image(image_path)
    
    # Print the result as formatted JSON
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
