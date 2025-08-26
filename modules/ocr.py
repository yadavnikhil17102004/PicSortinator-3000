#!/usr/bin/env python3
"""
PicSortinator 3000 - OCR Module
==============================

Extracts text from images with surgical precision.
Because your screenshots deserve to be searchable.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Optional import for duplicate detection
try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

logger = logging.getLogger(__name__)

class TextExtractor:
    """Advanced OCR with preprocessing and attitude."""
    
    def __init__(self):
        """Initialize the text extractor with OCR superpowers."""
        self.tesseract_available = False
        
        # Sarcastic responses for different scenarios
        self.funny_responses = {
            'no_tesseract': [
                "🤖 Tesseract is missing. It's like trying to read with your eyes closed.",
                "📖 OCR engine not found. Did you forget to install Tesseract?",
                "🔍 Can't read text without OCR tools. Install Tesseract first!"
            ],
            'no_text': [
                "📄 No text found. Either this image is text-free or my reading glasses need cleaning.",
                "🔍 Searched everywhere, found nothing. This image is as text-free as your browser history.",
                "📖 Zero readable text detected. Maybe it's in a language I don't speak?"
            ],
            'lots_of_text': [
                "📚 Found enough text to write a novel! Someone was chatty.",
                "💬 Text extraction complete. This image had more words than your last conversation.",
                "📖 Successfully extracted text. Your screenshot game is strong."
            ],
            'low_quality': [
                "😵‍💫 Image quality makes my OCR dizzy. Next time try less potato, more camera.",
                "🔍 Text confidence is lower than my faith in your photography skills.",
                "📷 Image preprocessing required. Did you take this while riding a roller coaster?"
            ]
        }
        
        # Try to import and configure pytesseract
        try:
            import pytesseract
            from PIL import Image
            
            # Set the exact path to Tesseract since it's not in PATH
            tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                logger.info(f"✅ Found Tesseract at: {tesseract_path}")
            
            # Try a simple test to see if tesseract is available
            test_result = pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info(f"✅ Tesseract OCR available: {test_result}")
            
        except Exception as e:
            logger.warning(f"❌ Tesseract not available: {e}")
            logger.warning("📖 Text extraction will be limited")
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from an image using advanced OCR preprocessing."""
        if not self.tesseract_available:
            return np.random.choice(self.funny_responses['no_tesseract'])
        
        try:
            import pytesseract
            
            # Load and preprocess the image
            processed_image = self._preprocess_image(image_path)
            if processed_image is None:
                return "Could not load or process image"
            
            # Configure tesseract for better accuracy
            config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?@#$%^&*()_+-=[]{}|;:"<>?/~`'
            
            # Extract text
            text = pytesseract.image_to_string(processed_image, config=config)
            
            # Clean up the text
            text = self._clean_extracted_text(text)
            
            if not text or len(text.strip()) < 3:
                return np.random.choice(self.funny_responses['no_text'])
            
            # Add some personality based on text length
            if len(text) > 500:
                logger.info("📚 Extracted lots of text - this was probably a document")
            
            return text
            
        except Exception as e:
            logger.error(f"💥 Error extracting text from {image_path}: {e}")
            return "Text extraction failed"
    
    def _preprocess_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Preprocess image for better OCR accuracy.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed PIL Image or None if failed
        """
        try:
            # Load image with OpenCV for advanced preprocessing
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply multiple preprocessing techniques
            
            # 1. Noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # 2. Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast_enhanced = clahe.apply(denoised)
            
            # 3. Adaptive thresholding (works better than simple thresholding)
            thresh = cv2.adaptiveThreshold(
                contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # 4. Morphological operations to clean up
            kernel = np.ones((1,1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL Image
            pil_image = Image.fromarray(cleaned)
            
            # 5. Additional PIL enhancements
            enhancer = ImageEnhance.Sharpness(pil_image)
            sharpened = enhancer.enhance(1.5)
            
            return sharpened
            
        except Exception as e:
            logger.error(f"💥 Failed to preprocess image {image_path}: {e}")
            return None
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean up extracted text by removing noise and formatting.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace and line breaks
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Filter out lines that are likely OCR artifacts
        cleaned_lines = []
        for line in lines:
            # Skip very short lines (likely artifacts)
            if len(line) < 2:
                continue
            
            # Skip lines with mostly special characters
            special_char_ratio = sum(1 for c in line if not c.isalnum() and not c.isspace()) / len(line)
            if special_char_ratio > 0.7:
                continue
            
            cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)
    
    def extract_detailed_text(self, image_path: str) -> Dict[str, any]:
        """Extract text with detailed confidence and position information."""
        if not self.tesseract_available:
            return {
                'error': 'Tesseract not available',
                'message': np.random.choice(self.funny_responses['no_tesseract'])
            }
        
        try:
            import pytesseract
            
            # Load and preprocess image
            processed_image = self._preprocess_image(image_path)
            if processed_image is None:
                return {'error': 'Could not process image'}
            
            # Get data with confidence scores and positions
            data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
            
            results = {
                'text': '',
                'words': [],
                'lines': [],
                'average_confidence': 0,
                'image_analysis': {},
                'ai_comment': ''
            }
            
            # Process word-level data
            words = []
            confidences = []
            
            for i, word in enumerate(data['text']):
                if word.strip():
                    confidence = int(data['conf'][i])
                    if confidence > 30:  # Filter very low confidence
                        word_info = {
                            'word': word.strip(),
                            'confidence': confidence,
                            'bbox': {
                                'x': data['left'][i],
                                'y': data['top'][i],
                                'width': data['width'][i],
                                'height': data['height'][i]
                            }
                        }
                        words.append(word_info)
                        confidences.append(confidence)
            
            results['words'] = words
            results['text'] = ' '.join([w['word'] for w in words])
            results['average_confidence'] = sum(confidences) / len(confidences) if confidences else 0
            
            # Add image analysis
            results['image_analysis'] = self._analyze_text_image(image_path, processed_image)
            
            # Add personality based on results
            if results['average_confidence'] < 50:
                results['ai_comment'] = np.random.choice(self.funny_responses['low_quality'])
            elif len(results['text']) > 500:
                results['ai_comment'] = np.random.choice(self.funny_responses['lots_of_text'])
            elif not results['text']:
                results['ai_comment'] = np.random.choice(self.funny_responses['no_text'])
            else:
                results['ai_comment'] = "📖 Text extracted successfully. Quality seems decent."
            
            return results
            
        except Exception as e:
            logger.error(f"💥 Detailed text extraction failed for {image_path}: {e}")
            return {'error': f'Extraction failed: {e}'}
    
    def _analyze_text_image(self, image_path: str, processed_image: Image.Image) -> Dict[str, any]:
        """
        Analyze the image to understand text characteristics.
        
        Args:
            image_path: Original image path
            processed_image: Preprocessed image
            
        Returns:
            Analysis results
        """
        analysis = {}
        
        try:
            # Original image analysis
            with Image.open(image_path) as original:
                analysis['original_size'] = original.size
                analysis['mode'] = original.mode
                analysis['format'] = original.format
            
            # Processed image analysis
            analysis['processed_size'] = processed_image.size
            
            # Calculate image hash for duplicate detection (if available)
            if IMAGEHASH_AVAILABLE:
                analysis['image_hash'] = str(imagehash.phash(processed_image))
            else:
                analysis['image_hash'] = 'unavailable'
            
            # Estimate text density (rough heuristic)
            img_array = np.array(processed_image)
            white_pixels = np.sum(img_array == 255)
            total_pixels = img_array.size
            analysis['text_density'] = 1.0 - (white_pixels / total_pixels)
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
        
        return analysis
    
    def is_text_heavy_image(self, image_path: str, confidence_threshold: int = 60) -> bool:
        """
        Determine if an image contains significant amounts of readable text.
        
        Args:
            image_path: Path to image
            confidence_threshold: Minimum confidence for text detection
            
        Returns:
            True if image contains significant text
        """
        try:
            result = self.extract_detailed_text(image_path)
            
            if 'error' in result:
                return False
            
            # Multiple criteria for text-heavy images
            high_confidence_words = [w for w in result.get('words', []) 
                                   if w['confidence'] > confidence_threshold]
            
            # Criteria 1: More than 10 high-confidence words
            if len(high_confidence_words) > 10:
                return True
            
            # Criteria 2: More than 100 characters total
            if len(result.get('text', '')) > 100:
                return True
            
            # Criteria 3: High text density in image
            text_density = result.get('image_analysis', {}).get('text_density', 0)
            if text_density > 0.3:  # More than 30% non-white pixels
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"💥 Error checking text content for {image_path}: {e}")
            return False
    
    def extract_keywords(self, image_path: str, min_word_length: int = 3) -> List[str]:
        """
        Extract keywords from image text for better searchability.
        
        Args:
            image_path: Path to image
            min_word_length: Minimum length for keywords
            
        Returns:
            List of extracted keywords
        """
        try:
            text = self.extract_text(image_path)
            
            if not text or len(text) < 10:
                return []
            
            # Simple keyword extraction
            words = text.lower().split()
            
            # Filter and clean words
            keywords = []
            common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'do', 'does', 'did', 'a', 'an'}
            
            for word in words:
                # Clean the word
                clean_word = ''.join(c for c in word if c.isalnum())
                
                # Filter criteria
                if (len(clean_word) >= min_word_length and 
                    clean_word not in common_words and 
                    clean_word not in keywords):
                    keywords.append(clean_word)
            
            return keywords[:10]  # Return top 10 keywords
            
        except Exception as e:
            logger.error(f"💥 Keyword extraction failed for {image_path}: {e}")
            return []
