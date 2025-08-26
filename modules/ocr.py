"""
OCR module for extracting text from images.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TextExtractor:
    """Extracts text from images using OCR."""
    
    def __init__(self):
        """Initialize the text extractor."""
        self.tesseract_available = False
        self.easyocr_available = False
        self.easyocr_reader = None
        
        # Check for Tesseract
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("Tesseract OCR is available")
        except (ImportError, Exception) as e:
            logger.warning(f"Tesseract OCR not available: {e}")
            
        # Check for EasyOCR (as fallback)
        try:
            import easyocr
            # Skip actual loading in test environment
            if "PYTEST_CURRENT_TEST" in os.environ:
                self.easyocr_available = True
            else:
                self.easyocr_reader = easyocr.Reader(['en'])
                self.easyocr_available = True
            logger.info("EasyOCR is available")
        except ImportError as e:
            logger.warning(f"EasyOCR not available: {e}")
    
    def extract_text(self, image_path):
        """Extract text from an image using available OCR tools."""
        try:
            # Try Tesseract first
            if self.tesseract_available:
                return self._extract_with_tesseract(image_path)
            
            # Fall back to EasyOCR if available
            if self.easyocr_available:
                return self._extract_with_easyocr(image_path)
                
            # No OCR available
            logger.warning("No OCR engines available")
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            return ""
    
    def _extract_with_tesseract(self, image_path):
        """Extract text using Tesseract OCR."""
        try:
            import pytesseract
            from PIL import Image
            import cv2
            import numpy as np
            
            # Load image
            img = cv2.imread(str(image_path))
            
            if img is None:
                return ""
                
            # Preprocessing to improve OCR accuracy
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Apply dilation and erosion to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Save processed image to a temporary file
            temp_file = "temp_ocr.jpg"
            cv2.imwrite(temp_file, opening)
            
            # Use Tesseract on the processed image
            text = pytesseract.image_to_string(Image.open(temp_file))
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            return text.strip()
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            
            # Try EasyOCR as fallback if available
            if self.easyocr_available:
                return self._extract_with_easyocr(image_path)
            return ""
    
    def _extract_with_easyocr(self, image_path):
        """Extract text using EasyOCR."""
        try:
            if self.easyocr_reader is None:
                import easyocr
                self.easyocr_reader = easyocr.Reader(['en'])
                
            # Read the image
            result = self.easyocr_reader.readtext(str(image_path))
            
            # Extract text from results
            text_parts = [item[1] for item in result]
            
            return " ".join(text_parts)
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return ""
