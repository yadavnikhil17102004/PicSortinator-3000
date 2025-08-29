#!/usr/bin/env python3
"""
PicSortinator 3000 - OCR Module
==============================

Extracts text from images with surgical precision.
Because your screenshots deserve to be searchable.

🔍 Historical Note: This module has been through more transformations than a Transformer model!
We've learned that sometimes less preprocessing is more... who knew? 🤷‍♂️

⚠️  Warning: May occasionally read "COFFEE" as "C0FF33" - we're working on it!
🎯 Pro Tip: If it looks like gibberish, it probably was gibberish to begin with.
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
        """Extract text from an image using conservative OCR approach."""        
        if not self.tesseract_available:
            return np.random.choice(self.funny_responses['no_tesseract'])
        
        try:
            import pytesseract
            
            # Step 1: Try original image with minimal config (most conservative)
            try:
                with Image.open(image_path) as img:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Try very simple OCR first
                    simple_config = '--oem 3 --psm 3'
                    simple_text = pytesseract.image_to_string(img, config=simple_config, lang='eng')
                    
                    if self._is_meaningful_text(simple_text):
                        cleaned = self._aggressive_clean_text(simple_text)
                        if cleaned:
                            return cleaned
            except Exception as e:
                logger.debug(f"Simple OCR failed: {e}")
            
            # Step 2: Try with very light preprocessing if simple OCR failed
            try:
                cv_image = cv2.imread(image_path)
                if cv_image is not None:
                    # Convert to grayscale
                    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                    
                    # Only apply CLAHE for contrast enhancement (no other processing)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)
                    
                    # Convert back to PIL
                    pil_image = Image.fromarray(enhanced)
                    
                    # Try OCR on enhanced image
                    enhanced_text = pytesseract.image_to_string(pil_image, config='--oem 3 --psm 3', lang='eng')
                    
                    if self._is_meaningful_text(enhanced_text):
                        cleaned = self._aggressive_clean_text(enhanced_text)
                        if cleaned:
                            return cleaned
            except Exception as e:
                logger.debug(f"Enhanced OCR failed: {e}")
            
            # Step 3: If still no good text, return "no text found"
            return "No readable text found"
            
        except Exception as e:
            logger.error(f"💥 Error extracting text from {image_path}: {e}")
            return "Text extraction failed"
    
    def _is_meaningful_text(self, text: str) -> bool:
        """
        Very strict check for meaningful text (no random character strings).
        
        🤖 This function has seen things... terrible, terrible OCR things.
        Like "Hello World" becoming "H3||0 W0r1d" or worse: "SOOTAEINSBET f Biles"
        We're the last line of defense against gibberish! 🛡️
        """
        if not text or len(text.strip()) < 3:
            return False
        
        # Remove whitespace and newlines
        clean_text = ''.join(text.split())
        
        # Must be at least 3 characters
        if len(clean_text) < 3:
            return False
        
        # Check for meaningful English words pattern
        import re
        
        # Look for actual words (3+ letters)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        
        # Must have at least 1 meaningful word
        if len(words) < 1:
            return False
        
        # Check for dictionary-like words (common English patterns)
        common_patterns = [
            r'\b(the|and|for|are|but|not|you|all|can|had|her|was|one|our|out|day|get|has|him|his|how|its|may|new|now|old|see|two|way|who|boy|did|man|men|put|say|she|too|use)\b',
            r'\b[A-Z][a-z]{2,}\b',  # Capitalized words
            r'\b\d+\b'  # Numbers
        ]
        
        # At least one pattern should match
        for pattern in common_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _aggressive_clean_text(self, text: str) -> str:
        """Aggressively clean text to remove OCR artifacts."""
        if not text:
            return ""
        
        import re
        
        # Remove obvious OCR artifacts
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip lines that are mostly special characters or single chars
            if len(line) < 3:
                continue
            
            # Skip lines with too many consecutive consonants (OCR errors)
            if re.search(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{4,}', line):
                continue
            
            # Skip lines with too many numbers mixed with letters randomly
            if re.search(r'[a-zA-Z]\d[a-zA-Z]\d[a-zA-Z]', line):
                continue
            
            # Only keep lines with recognizable word patterns
            words = re.findall(r'\b[a-zA-Z]{2,}\b', line)
            if len(words) >= 1:  # At least 1 word
                clean_lines.append(line)
        
        result = ' '.join(clean_lines)
        
        # Final cleanup
        result = re.sub(r'\s+', ' ', result)  # Multiple spaces to single
        result = result.strip()
        
        return result if len(result) >= 3 else ""
        """Try extracting text with minimal preprocessing first."""
        try:
            import pytesseract
            from PIL import Image
            
            # Load image with minimal processing
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Simple config for clear text
                config = '--oem 3 --psm 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?@#$%^&*()_+-=[]{}|;:"<>?/~`"'
                text = pytesseract.image_to_string(img, config=config, lang='eng')
                
                return text
                
        except Exception as e:
            logger.debug(f"Minimal preprocessing failed: {e}")
            return ""
    
    def _preprocess_gently(self, image_path: str) -> Optional[Image.Image]:
        """Apply gentle preprocessing that preserves text quality."""
        try:
            # Load image with OpenCV
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Very light denoising (preserve text details)
            denoised = cv2.fastNlMeansDenoising(gray, h=5)
            
            # Light CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
            enhanced = clahe.apply(denoised)
            
            # Convert back to PIL Image
            pil_image = Image.fromarray(enhanced)
            
            return pil_image
            
        except Exception as e:
            logger.debug(f"Gentle preprocessing failed: {e}")
            return None
    
    def _is_good_text(self, text: str) -> bool:
        """Check if extracted text is meaningful and not random characters."""
        if not text or len(text.strip()) < 2:
            return False
        
        # Count alphanumeric characters vs total characters
        alpha_num = sum(1 for c in text if c.isalnum())
        total_chars = len(text)
        
        if total_chars == 0:
            return False
        
        # Lower threshold for alphanumeric ratio (was 0.3, now 0.2)
        if (alpha_num / total_chars) < 0.2:
            return False
        
        # Check for meaningful words (at least 2 letters, was 3 letters)
        words = [word for word in text.split() if len(word) >= 2 and word.isalpha()]
        if len(words) < 1:  # Was 2, now 1
            return False
        
        # Additional check: if we have at least 3 alphanumeric characters in a row, it's probably text
        import re
        if re.search(r'[a-zA-Z0-9]{3,}', text):
            return True
        
        return True  # More permissive
    
    def _calculate_text_confidence(self, text: str) -> float:
        """Calculate a confidence score for extracted text."""
        if not text:
            return 0.0
        
        # Get word-level confidence data
        try:
            import pytesseract
            # This is a simplified confidence calculation
            words = text.split()
            if not words:
                return 0.0
            
            # Simple heuristic: longer meaningful words = higher confidence
            meaningful_words = [w for w in words if len(w) >= 3 and w.isalpha()]
            confidence = min(len(meaningful_words) / len(words), 1.0)
            
            return confidence
            
        except Exception:
            return 0.5  # Default confidence
    
    def _preprocess_image_smart(self, image_path: str) -> Optional[Image.Image]:
        """
        Smart preprocessing that adapts to different image types.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed PIL Image or None if failed
        """
        try:
            # Load image with OpenCV
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                return None
            
            # Get image characteristics
            height, width = cv_image.shape[:2]
            aspect_ratio = width / height
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Analyze image to determine best preprocessing approach
            image_analysis = self._analyze_image_for_ocr(gray)
            
            # Adaptive preprocessing based on image characteristics
            if image_analysis['is_text_document']:
                # For documents: mild preprocessing
                processed = self._preprocess_document(gray)
            elif image_analysis['is_low_contrast']:
                # For low contrast images: enhance contrast
                processed = self._preprocess_low_contrast(gray)
            elif image_analysis['has_noise']:
                # For noisy images: denoise
                processed = self._preprocess_noisy(gray)
            else:
                # Default preprocessing
                processed = self._preprocess_standard(gray)
            
            # Convert back to PIL Image
            pil_image = Image.fromarray(processed)
            
            return pil_image
            
        except Exception as e:
            logger.error(f"💥 Failed to preprocess image {image_path}: {e}")
            return None
    
    def _analyze_image_for_ocr(self, gray_image: np.ndarray) -> Dict[str, bool]:
        """
        Analyze image characteristics to determine best preprocessing approach.
        
        Args:
            gray_image: Grayscale image array
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'is_text_document': False,
            'is_low_contrast': False,
            'has_noise': False,
            'is_high_resolution': False
        }
        
        try:
            # Check contrast
            min_val, max_val = np.min(gray_image), np.max(gray_image)
            contrast_ratio = (max_val - min_val) / 255.0
            analysis['is_low_contrast'] = contrast_ratio < 0.3
            
            # Check for noise (using Laplacian variance)
            laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            analysis['has_noise'] = laplacian_var < 100
            
            # Check resolution
            height, width = gray_image.shape
            total_pixels = height * width
            analysis['is_high_resolution'] = total_pixels > 1000000  # > 1MP
            
            # Simple heuristic for text documents (high contrast, uniform background)
            if contrast_ratio > 0.5 and laplacian_var > 200:
                analysis['is_text_document'] = True
                
        except Exception as e:
            logger.debug(f"Image analysis failed: {e}")
        
        return analysis
    
    def _preprocess_document(self, gray_image: np.ndarray) -> np.ndarray:
        """Preprocessing optimized for text documents."""
        try:
            # Mild denoising
            denoised = cv2.fastNlMeansDenoising(gray_image, h=10)
            
            # CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Light Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            return thresh
            
        except Exception as e:
            logger.debug(f"Document preprocessing failed: {e}")
            return gray_image
    
    def _preprocess_low_contrast(self, gray_image: np.ndarray) -> np.ndarray:
        """Preprocessing for low contrast images."""
        try:
            # Stronger contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray_image)
            
            # Sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            return thresh
            
        except Exception as e:
            logger.debug(f"Low contrast preprocessing failed: {e}")
            return gray_image
    
    def _preprocess_noisy(self, gray_image: np.ndarray) -> np.ndarray:
        """Preprocessing for noisy images."""
        try:
            # Strong denoising
            denoised = cv2.fastNlMeansDenoising(gray_image, h=15)
            
            # Bilateral filter to preserve edges
            filtered = cv2.bilateralFilter(denoised, 9, 75, 75)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            return thresh
            
        except Exception as e:
            logger.debug(f"Noisy image preprocessing failed: {e}")
            return gray_image
    
    def _preprocess_standard(self, gray_image: np.ndarray) -> np.ndarray:
        """Standard preprocessing for general images."""
        try:
            # Moderate denoising
            denoised = cv2.fastNlMeansDenoising(gray_image, h=10)
            
            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            return thresh
            
        except Exception as e:
            logger.debug(f"Standard preprocessing failed: {e}")
            return gray_image
    
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
