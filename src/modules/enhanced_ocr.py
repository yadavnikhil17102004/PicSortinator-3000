"""
Enhanced OCR Module for ForenSnap
Supports multiple languages, automatic language detection, and improved accuracy
"""

import os
import cv2
import numpy as np
import easyocr
import pytesseract
from PIL import Image
from langdetect import detect, LangDetectError
import re
from typing import List, Dict, Any, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedOCR:
    """Enhanced OCR class with multi-language support."""
    
    def __init__(self):
        """Initialize Enhanced OCR with support for multiple languages."""
        # Initialize EasyOCR readers for different language groups
        self.readers = {}
        self.supported_languages = {
            'en': ['en'],  # English
            'es': ['es'],  # Spanish
            'fr': ['fr'],  # French
            'de': ['de'],  # German
            'it': ['it'],  # Italian
            'pt': ['pt'],  # Portuguese
            'ru': ['ru'],  # Russian
            'ar': ['ar'],  # Arabic
            'hi': ['hi'],  # Hindi
            'zh': ['ch_sim', 'ch_tra'],  # Chinese (Simplified and Traditional)
            'ja': ['ja'],  # Japanese
            'ko': ['ko'],  # Korean
            'multi': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ar', 'hi']  # Multi-language
        }
        
        # Initialize default English reader
        try:
            self.readers['en'] = easyocr.Reader(['en'], gpu=False)
            logger.info("EasyOCR English reader initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR English reader: {e}")
            self.readers['en'] = None
        
        # Tesseract language codes
        self.tesseract_langs = {
            'en': 'eng',
            'es': 'spa',
            'fr': 'fra',
            'de': 'deu',
            'it': 'ita',
            'pt': 'por',
            'ru': 'rus',
            'ar': 'ara',
            'hi': 'hin',
            'zh': 'chi_sim+chi_tra',
            'ja': 'jpn',
            'ko': 'kor'
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image (np.ndarray): Input image array
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised)
        
        # Apply sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
        
        # Threshold to get binary image
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of extracted text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            str: Detected language code
        """
        if not text or len(text.strip()) < 3:
            return 'en'  # Default to English for short texts
        
        try:
            detected_lang = detect(text)
            # Map some common language codes
            lang_mapping = {
                'ca': 'es',  # Catalan -> Spanish
                'nl': 'de',  # Dutch -> German
                'sv': 'en',  # Swedish -> English
                'no': 'en',  # Norwegian -> English
                'da': 'en',  # Danish -> English
            }
            return lang_mapping.get(detected_lang, detected_lang)
        except LangDetectError:
            return 'en'  # Default to English if detection fails
    
    def extract_text_tesseract(self, image: np.ndarray, lang: str = 'eng') -> Dict[str, Any]:
        """
        Extract text using Tesseract OCR.
        
        Args:
            image (np.ndarray): Input image
            lang (str): Language code for Tesseract
            
        Returns:
            Dict[str, Any]: OCR results with text and confidence
        """
        try:
            # Configure Tesseract
            custom_config = f'--oem 3 --psm 6 -l {lang}'
            
            # Extract text with confidence scores
            data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Filter out low-confidence results
            text_parts = []
            confidences = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:  # Minimum confidence threshold
                    text = data['text'][i].strip()
                    if text:
                        text_parts.append(text)
                        confidences.append(int(data['conf'][i]))
            
            full_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': full_text,
                'confidence': avg_confidence,
                'method': 'tesseract',
                'language': lang,
                'word_count': len(text_parts)
            }
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return {
                'text': '',
                'confidence': 0,
                'method': 'tesseract',
                'language': lang,
                'word_count': 0
            }
    
    def extract_text_easyocr(self, image: np.ndarray, languages: List[str]) -> Dict[str, Any]:
        """
        Extract text using EasyOCR.
        
        Args:
            image (np.ndarray): Input image
            languages (List[str]): List of language codes for EasyOCR
            
        Returns:
            Dict[str, Any]: OCR results with text and confidence
        """
        try:
            # Get or create reader for the language combination
            lang_key = '_'.join(sorted(languages))
            
            if lang_key not in self.readers:
                self.readers[lang_key] = easyocr.Reader(languages, gpu=False)
            
            reader = self.readers[lang_key]
            if reader is None:
                raise Exception("EasyOCR reader not available")
            
            # Perform OCR
            results = reader.readtext(image, detail=1, paragraph=True)
            
            # Extract text and calculate average confidence
            text_parts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Minimum confidence threshold
                    text_parts.append(text.strip())
                    confidences.append(confidence)
            
            full_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': full_text,
                'confidence': avg_confidence * 100,  # Convert to percentage
                'method': 'easyocr',
                'languages': languages,
                'word_count': len(text_parts)
            }
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return {
                'text': '',
                'confidence': 0,
                'method': 'easyocr',
                'languages': languages,
                'word_count': 0
            }
    
    def extract_text_hybrid(self, image_path: str, target_languages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract text using hybrid approach (both Tesseract and EasyOCR).
        
        Args:
            image_path (str): Path to the image file
            target_languages (Optional[List[str]]): Specific languages to use
            
        Returns:
            Dict[str, Any]: Combined OCR results with best text extraction
        """
        # Load and preprocess image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            pil_image = Image.open(image_path).convert('RGB')
        else:
            # Assume it's already an image array
            image = image_path
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if image is None:
            return {
                'text': '',
                'confidence': 0,
                'method': 'error',
                'error': 'Could not load image'
            }
        
        preprocessed = self.preprocess_image(image)
        
        # Try multiple OCR approaches
        results = []
        
        # 1. Try EasyOCR with auto-detected languages or specified languages
        if target_languages:
            easy_languages = []
            for lang in target_languages:
                if lang in self.supported_languages:
                    easy_languages.extend(self.supported_languages[lang])
        else:
            easy_languages = self.supported_languages['multi']  # Use multi-language by default
        
        try:
            easyocr_result = self.extract_text_easyocr(preprocessed, easy_languages[:3])  # Limit to 3 languages for performance
            results.append(easyocr_result)
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")
        
        # 2. Try Tesseract with English first
        tesseract_result = self.extract_text_tesseract(preprocessed, 'eng')
        results.append(tesseract_result)
        
        # 3. If we have text from initial OCR, detect language and try again with specific language
        best_text = max([r['text'] for r in results], key=len)
        if best_text and len(best_text) > 10:
            detected_lang = self.detect_language(best_text)
            if detected_lang in self.tesseract_langs and detected_lang != 'en':
                tesseract_lang_result = self.extract_text_tesseract(preprocessed, self.tesseract_langs[detected_lang])
                results.append(tesseract_lang_result)
        
        # Choose the best result based on confidence and text length
        if not results:
            return {
                'text': '',
                'confidence': 0,
                'method': 'no_results',
                'detected_language': 'unknown'
            }
        
        # Score results based on confidence and text length
        scored_results = []
        for result in results:
            if result['text']:
                # Normalize confidence to 0-100 scale
                conf = result['confidence']
                if conf > 1:  # EasyOCR returns 0-100, Tesseract varies
                    normalized_conf = min(conf, 100)
                else:
                    normalized_conf = conf * 100
                
                text_length_score = min(len(result['text']) / 100, 1.0) * 10  # Up to 10 points for text length
                total_score = normalized_conf * 0.7 + text_length_score * 0.3
                
                scored_results.append((total_score, result))
        
        if not scored_results:
            return results[0] if results else {'text': '', 'confidence': 0, 'method': 'no_text'}
        
        # Get the best result
        best_score, best_result = max(scored_results, key=lambda x: x[0])
        
        # Add additional metadata
        detected_lang = self.detect_language(best_result['text']) if best_result['text'] else 'unknown'
        
        final_result = best_result.copy()
        final_result.update({
            'detected_language': detected_lang,
            'all_results': results,
            'final_score': best_score,
            'text_preprocessing': 'applied'
        })
        
        return final_result
    
    def extract_structured_data(self, text: str) -> Dict[str, List[str]]:
        """
        Extract structured data from OCR text (phone numbers, emails, etc.).
        
        Args:
            text (str): OCR extracted text
            
        Returns:
            Dict[str, List[str]]: Structured data extracted from text
        """
        structured_data = {
            'phone_numbers': [],
            'email_addresses': [],
            'urls': [],
            'currency_amounts': [],
            'dates': [],
            'times': [],
            'credit_cards': [],
            'social_handles': []
        }
        
        if not text:
            return structured_data
        
        # Phone number patterns (international)
        phone_patterns = [
            r'(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{4}',  # US/Canada
            r'(?:\+?91[-.\s]?)?[6-9]\d{9}',  # India
            r'(?:\+?44[-.\s]?)?(?:0)?[1-9]\d{8,9}',  # UK
            r'(?:\+?49[-.\s]?)?(?:0)?[1-9]\d{9,10}',  # Germany
            r'(?:\+?33[-.\s]?)?(?:0)?[1-9]\d{8}',  # France
            r'(?:\+?86[-.\s]?)?1[3-9]\d{9}',  # China
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            structured_data['phone_numbers'].extend(matches)
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        structured_data['email_addresses'] = re.findall(email_pattern, text)
        
        # URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        structured_data['urls'] = re.findall(url_pattern, text)
        
        # Currency amounts
        currency_patterns = [
            r'(?:[$€£¥₹])\s*\d+(?:,\d{3})*(?:\.\d{2})?',  # Symbols before
            r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY|INR|CAD|AUD)',  # Currency codes after
        ]
        for pattern in currency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            structured_data['currency_amounts'].extend(matches)
        
        # Dates (various formats)
        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',   # YYYY/MM/DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            structured_data['dates'].extend(matches)
        
        # Times
        time_pattern = r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp][Mm])?\b'
        structured_data['times'] = re.findall(time_pattern, text)
        
        # Credit card numbers (masked for security)
        cc_pattern = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        cc_matches = re.findall(cc_pattern, text)
        # Mask all but last 4 digits
        for cc in cc_matches:
            masked = '*' * (len(cc.replace('-', '').replace(' ', '')) - 4) + cc[-4:]
            structured_data['credit_cards'].append(masked)
        
        # Social media handles
        social_pattern = r'@[A-Za-z0-9_]+'
        structured_data['social_handles'] = re.findall(social_pattern, text)
        
        # Remove duplicates
        for key in structured_data:
            structured_data[key] = list(set(structured_data[key]))
        
        return structured_data

# Example usage and testing
if __name__ == "__main__":
    ocr = EnhancedOCR()
    
    # Test with a sample image (you would replace this with an actual image path)
    # result = ocr.extract_text_hybrid("test_image.jpg")
    # print(json.dumps(result, indent=2))
