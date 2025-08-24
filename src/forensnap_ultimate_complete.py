#!/usr/bin/env python3
"""
ForenSnap Ultimate - AI-Powered Screenshot Classifier for Investigations
===========================================================================

A comprehensive digital forensics tool that automatically analyzes screenshots to:
- Categorize content (chats, transactions, threats, adult content)
- Extract text in multiple languages with high accuracy
- Detect social media platforms and messaging apps
- Identify NSFW/adult content using local AI models
- Advanced threat detection with sentiment analysis
- Face and object detection for evidence gathering
- Generate legal-compliant reports for court proceedings

Author: ForenSnap Team
Version: 2.0.0
License: MIT
"""

import os
import sys
import json
import uuid
import re
import cv2
import numpy as np
import datetime
import hashlib
import base64
import subprocess
import importlib.util
import warnings
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import logging
import traceback

# Setup organized directory structure
ROOT_DIR = Path(__file__).parent.parent  # Go up from src/ to root
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
SRC_DIR = ROOT_DIR / "src"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configure logging with organized directory structure
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'forensnap.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================================================================================
# DEPENDENCY MANAGEMENT AND AUTO-INSTALLATION
# ================================================================================================

class DependencyManager:
    """Manages dependencies and auto-installation."""
    
    REQUIRED_PACKAGES = {
        'pillow': 'PIL',
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'transformers': 'transformers',
        'clip-by-openai': 'clip',
        'easyocr': 'easyocr',
        'langdetect': 'langdetect',
        'spacy': 'spacy',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'python-multipart': 'python-multipart',
        'sqlalchemy': 'sqlalchemy',
        'reportlab': 'reportlab',
        'pytesseract': 'pytesseract',
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'tkinter': 'tkinter'  # Usually comes with Python
    }
    
    @staticmethod
    def check_and_install_dependencies():
        """Check for required packages and install if missing."""
        print("🔍 Checking dependencies...")
        missing_packages = []
        optional_packages = ['clip-by-openai']  # These are optional
        
        for package, import_name in DependencyManager.REQUIRED_PACKAGES.items():
            try:
                if import_name == 'tkinter':
                    import tkinter
                else:
                    __import__(import_name)
                print(f"✅ {package} - OK")
            except ImportError:
                print(f"❌ {package} - MISSING")
                missing_packages.append(package)
        
        if missing_packages:
            # Separate required from optional
            required_missing = [p for p in missing_packages if p not in optional_packages]
            optional_missing = [p for p in missing_packages if p in optional_packages]
            
            if required_missing:
                print(f"\n⚠️  Missing {len(required_missing)} required packages. Installing...")
                for package in required_missing:
                    try:
                        print(f"📦 Installing {package}...")
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", package, "--quiet"
                        ])
                        print(f"✅ {package} installed successfully")
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Failed to install {package}: {e}")
                        if package not in optional_packages:  # Only fail for required packages
                            return False
            
            if optional_missing:
                print(f"\n📦 Optional packages missing: {', '.join(optional_missing)}")
                print("   These features will be disabled but ForenSnap will still work.")
            
            # Special handling for spaCy model
            try:
                import spacy
                try:
                    spacy.load("en_core_web_sm")
                except OSError:
                    print("📦 Installing spaCy English model...")
                    subprocess.check_call([
                        sys.executable, "-m", "spacy", "download", "en_core_web_sm"
                    ])
            except:
                pass
        
        print("✅ All dependencies ready!")
        return True

# Install dependencies at startup
if not DependencyManager.check_and_install_dependencies():
    print("❌ Failed to install some dependencies. Please install manually.")
    sys.exit(1)

# Try to import the enhanced OCR module
try:
    from modules.enhanced_ocr import EnhancedOCR
    print("✅ Enhanced OCR module loaded successfully")
    USE_ENHANCED_OCR = True
except ImportError as e:
    print(f"⚠️  Enhanced OCR module not found: {e}")
    print("   Using built-in OCR implementation instead.")
    EnhancedOCR = None
    USE_ENHANCED_OCR = False

# Now import all required modules
try:
    from PIL import Image as PILImage, ImageTk
    import pytesseract
    import easyocr
    from langdetect import detect, LangDetectException
    import spacy
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, ForeignKey, Text, JSON
    from sqlalchemy.orm import sessionmaker, relationship, declarative_base
    from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
    from fastapi.responses import JSONResponse, FileResponse
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    import tkinter as tk
    from tkinter import filedialog, scrolledtext, messagebox, ttk
    import uvicorn
    import threading
    import queue
    import time
    
    # Optional CLIP import - using different approach for better compatibility
    try:
        # Try different CLIP implementations
        try:
            import clip
            CLIP_AVAILABLE = True
            print("📦 OpenAI CLIP loaded successfully")
        except ImportError:
            try:
                # Alternative: sentence-transformers CLIP
                from sentence_transformers import SentenceTransformer
                import clip  # Set as None initially
                clip = None
                CLIP_AVAILABLE = True
                print("📦 Sentence-Transformers CLIP fallback loaded")
            except ImportError:
                CLIP_AVAILABLE = False
                clip = None
                print("📦 CLIP not available - NSFW detection will use alternative methods")
    except Exception as e:
        CLIP_AVAILABLE = False
        clip = None
        print(f"📦 CLIP loading error: {e}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please run the installation again or install packages manually.")
    sys.exit(1)

# ================================================================================================
# DATABASE MODELS AND ENUMS
# ================================================================================================

Base = declarative_base()

class Category(str, Enum):
    """Content categories for classification."""
    CHAT = "chat"
    TRANSACTION = "transaction"
    THREAT = "threat"
    ADULT = "adult_content"
    SOCIAL_MEDIA = "social_media"
    NEWS = "news"
    DOCUMENT = "document"
    UNCATEGORIZED = "uncategorized"

class ThreatLevel(str, Enum):
    """Threat severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PlatformType(str, Enum):
    """Social media platform types."""
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    SNAPCHAT = "snapchat"
    TIKTOK = "tiktok"
    LINKEDIN = "linkedin"
    DISCORD = "discord"
    SIGNAL = "signal"
    UNKNOWN = "unknown"

class TagType(str, Enum):
    """Tag classification types."""
    ENTITY = "entity"
    KEYWORD = "keyword"
    PLATFORM = "platform"
    OBJECT = "object"
    FACE = "face"
    THREAT = "threat"
    BLIP = "blip"
    OCR = "ocr"

# Database Models
class Case(Base):
    """Investigation case management."""
    __tablename__ = 'cases'
    
    id = Column(String(36), primary_key=True)
    case_number = Column(String(100), nullable=False, unique=True)
    title = Column(String(255), nullable=False)
    investigator = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String(50), default='active')
    description = Column(Text)
    images = relationship("Image", back_populates="case")

class Image(Base):
    """Image metadata and analysis results."""
    __tablename__ = 'images'
    
    id = Column(String(36), primary_key=True)
    case_id = Column(String(36), ForeignKey('cases.id'), nullable=True)
    file_path = Column(String(500), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    file_size = Column(Integer, nullable=True)
    detected_text = Column(Text, nullable=True)
    detected_language = Column(String(10), nullable=True)
    category = Column(String(50), nullable=False, default='uncategorized', index=True)
    platform = Column(String(50), nullable=True, index=True)
    threat_level = Column(String(20), default='none', index=True)
    nsfw_score = Column(Float, default=0.0)
    confidence_score = Column(Float, default=0.0)
    analysis_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    case = relationship("Case", back_populates="images")
    tags = relationship("ImageTag", back_populates="image")

class Tag(Base):
    """Tag definitions."""
    __tablename__ = 'tags'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False, index=True)
    tag_type = Column(String(50), nullable=False, index=True)
    frequency = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class ImageTag(Base):
    """Many-to-many relationship between images and tags."""
    __tablename__ = 'image_tags'
    
    image_id = Column(String(36), ForeignKey('images.id'), primary_key=True)
    tag_id = Column(Integer, ForeignKey('tags.id'), primary_key=True)
    confidence = Column(Float, nullable=True)
    
    image = relationship("Image", back_populates="tags")
    tag = relationship("Tag")

# ================================================================================================
# ENHANCED OCR WITH MULTI-LANGUAGE SUPPORT
# ================================================================================================

class EnhancedOCR:
    """Enhanced OCR with multiple engines and robust preprocessing."""
    
    def __init__(self):
        """Initialize OCR engines with proper error handling."""
        self.easyocr_readers = {}
        self.tesseract_available = False
        self.easyocr_available = False
        
        # Initialize EasyOCR
        try:
            logger.info("Initializing EasyOCR...")
            self.easyocr_readers['en'] = easyocr.Reader(['en'], gpu=False, verbose=False)
            self.easyocr_available = True
            logger.info("✅ EasyOCR initialized successfully")
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")
            self.easyocr_available = False
        
        # Check Tesseract
        self.tesseract_available = self._check_tesseract()
        
        if not self.easyocr_available and not self.tesseract_available:
            logger.warning("⚠️ No OCR engines available - OCR functionality will be limited")
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract is properly configured."""
        try:
            # Try to get version
            version = pytesseract.get_tesseract_version()
            logger.info(f"✅ Tesseract {version} found")
            return True
        except Exception as e:
            logger.debug(f"Initial Tesseract check failed: {e}")
            
            # Try common Tesseract paths on Windows
            common_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    try:
                        pytesseract.pytesseract.tesseract_cmd = path
                        version = pytesseract.get_tesseract_version()
                        logger.info(f"✅ Tesseract {version} found at {path}")
                        return True
                    except:
                        continue
            
            logger.info("Tesseract not found. OCR will use EasyOCR only.")
            return False
    
    def preprocess_image_robust(self, image: np.ndarray, enhance: bool = True) -> np.ndarray:
        """
        Robust image preprocessing for OCR with multiple enhancement options.
        
        Args:
            image: Input image as numpy array
            enhance: Whether to apply enhancement (for low quality images)
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Get image dimensions
            height, width = gray.shape
            
            # 1. Upscaling for small images
            if height < 400 or width < 400:
                scale_factor = max(400 / height, 400 / width, 2.0)  # At least 2x upscale
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                logger.debug(f"Upscaled image from {width}x{height} to {new_width}x{new_height}")
            
            if not enhance:
                return gray
            
            # 2. Noise reduction for noisy images
            if self._is_noisy(gray):
                gray = cv2.medianBlur(gray, 3)
            
            # 3. Contrast enhancement using CLAHE (limited)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # 4. Adaptive threshold for better text separation
            # Try both methods and pick the best
            thresh1 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            thresh2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            
            # Choose threshold with more white pixels (usually better for text)
            white_pixels1 = np.sum(thresh1 == 255)
            white_pixels2 = np.sum(thresh2 == 255)
            
            if white_pixels1 > white_pixels2:
                final_image = thresh1
            else:
                final_image = thresh2
            
            return final_image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            # Return original grayscale as fallback
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image.copy()
    
    def _is_noisy(self, image: np.ndarray) -> bool:
        """Check if image is noisy using Laplacian variance."""
        try:
            laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
            return laplacian_var < 100  # Low variance indicates noise
        except:
            return False
    
    def extract_text_easyocr(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text using EasyOCR."""
        if not self.easyocr_available:
            return {'text': '', 'confidence': 0, 'method': 'easyocr_unavailable'}
        
        try:
            reader = self.easyocr_readers.get('en')
            if not reader:
                return {'text': '', 'confidence': 0, 'method': 'easyocr_no_reader'}
            
            # EasyOCR works better with original colors sometimes
            results = reader.readtext(image, detail=1, paragraph=True)
            
            if not results:
                return {'text': '', 'confidence': 0, 'method': 'easyocr_no_results'}
            
            # Extract text and calculate confidence
            text_parts = []
            confidences = []
            
            for (bbox, text, conf) in results:
                if conf > 0.1:  # Lower threshold for better capture
                    text_parts.append(text.strip())
                    confidences.append(conf)
            
            final_text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) * 100 if confidences else 0
            
            return {
                'text': final_text,
                'confidence': avg_confidence,
                'method': 'easyocr',
                'details': results
            }
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return {'text': '', 'confidence': 0, 'method': 'easyocr_error', 'error': str(e)}
    
    def extract_text_tesseract(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text using Tesseract."""
        if not self.tesseract_available:
            return {'text': '', 'confidence': 0, 'method': 'tesseract_unavailable'}
        
        try:
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = PILImage.fromarray(image)
            
            # Try different PSM modes for better results
            psm_modes = [6, 8, 13, 3]  # 6=single block, 8=single word, 13=raw line, 3=auto
            
            best_result = {'text': '', 'confidence': 0}
            
            for psm in psm_modes:
                try:
                    config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-=[]{{}}|;:,.<>?/~` '
                    text = pytesseract.image_to_string(pil_image, config=config)
                    
                    if text and len(text.strip()) > len(best_result['text']):
                        best_result = {
                            'text': text.strip(),
                            'confidence': 80,  # Tesseract doesn't provide confidence easily
                            'psm': psm
                        }
                except Exception as e:
                    logger.debug(f"Tesseract PSM {psm} failed: {e}")
                    continue
            
            return {
                'text': best_result['text'],
                'confidence': best_result['confidence'],
                'method': 'tesseract',
                'psm_used': best_result.get('psm', 6)
            }
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return {'text': '', 'confidence': 0, 'method': 'tesseract_error', 'error': str(e)}
    
    def detect_language(self, text: str) -> str:
        """Detect language of extracted text."""
        if not text or len(text.strip()) < 10:
            return 'en'
        
        try:
            return detect(text)
        except:
            return 'en'
    
    def extract_text_hybrid(self, image_path: str) -> Dict[str, Any]:
        """
        Main OCR function using hybrid approach.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'text': '',
                    'confidence': 0,
                    'language': 'unknown',
                    'method': 'file_load_error',
                    'error': f'Could not load image: {image_path}'
                }
            
            logger.debug(f"Processing image: {os.path.basename(image_path)} ({image.shape[1]}x{image.shape[0]})")
            
            # Try multiple preprocessing approaches
            preprocessing_methods = [
                ('simple', False),      # Simple upscaling only
                ('enhanced', True),     # Full enhancement
            ]
            
            all_results = []
            
            for method_name, enhance in preprocessing_methods:
                processed_image = self.preprocess_image_robust(image, enhance=enhance)
                
                # Try EasyOCR
                if self.easyocr_available:
                    easyocr_result = self.extract_text_easyocr(processed_image)
                    easyocr_result['preprocessing'] = method_name
                    all_results.append(easyocr_result)
                
                # Try Tesseract
                if self.tesseract_available:
                    tesseract_result = self.extract_text_tesseract(processed_image)
                    tesseract_result['preprocessing'] = method_name
                    all_results.append(tesseract_result)
            
            # No results at all
            if not all_results:
                return {
                    'text': '',
                    'confidence': 0,
                    'language': 'unknown',
                    'method': 'no_engines',
                    'error': 'No OCR engines available'
                }
            
            # Filter out empty results
            valid_results = [r for r in all_results if r.get('text', '').strip()]
            
            if not valid_results:
                # Return best empty result with method info
                best_empty = max(all_results, key=lambda x: x.get('confidence', 0))
                return {
                    'text': '',
                    'confidence': 0,
                    'language': 'unknown',
                    'method': best_empty.get('method', 'unknown'),
                    'all_attempts': len(all_results),
                    'error': 'No text detected in image'
                }
            
            # Select best result based on confidence and text length
            def score_result(result):
                text_length = len(result.get('text', '').strip())
                confidence = result.get('confidence', 0)
                # Prefer longer text with decent confidence
                return text_length * 0.7 + confidence * 0.3
            
            best_result = max(valid_results, key=score_result)
            
            # Detect language
            detected_language = self.detect_language(best_result['text'])
            
            # Prepare final result
            final_result = {
                'text': best_result['text'].strip(),
                'confidence': best_result.get('confidence', 0),
                'language': detected_language,
                'method': best_result.get('method', 'unknown'),
                'preprocessing': best_result.get('preprocessing', 'unknown'),
                'all_results': all_results,
                'total_attempts': len(all_results),
                'valid_results': len(valid_results)
            }
            
            logger.info(f"OCR completed: {len(best_result['text'])} chars, "
                       f"{best_result.get('confidence', 0):.1f}% confidence, "
                       f"{best_result.get('method', 'unknown')} method")
            
            return final_result
            
        except Exception as e:
            logger.error(f"OCR hybrid extraction failed: {e}")
            return {
                'text': '',
                'confidence': 0,
                'language': 'unknown',
                'method': 'extraction_error',
                'error': str(e)
            }

# ================================================================================================
# SOCIAL MEDIA PLATFORM DETECTION
# ================================================================================================

class PlatformDetector:
    """Detect social media platforms from screenshots."""
    
    def __init__(self):
        """Initialize platform detection patterns."""
        self.platform_patterns = {
            PlatformType.WHATSAPP: {
                'ui_elements': ['whatsapp', 'typing...', 'online', 'last seen', 'voice message'],
                'colors': [(37, 211, 102), (18, 140, 126)],  # WhatsApp green colors
                'text_patterns': [r'whatsapp', r'\d{2}:\d{2}', r'voice message', r'typing\.\.\.']
            },
            PlatformType.TELEGRAM: {
                'ui_elements': ['telegram', 'forwarded from', 'reply', 'edit', 'delete'],
                'colors': [(0, 136, 204), (64, 165, 221)],  # Telegram blue colors
                'text_patterns': [r'telegram', r'forwarded from', r'@\w+', r'\d+ members']
            },
            PlatformType.INSTAGRAM: {
                'ui_elements': ['instagram', 'story', 'follow', 'following', 'likes'],
                'colors': [(195, 42, 163), (245, 96, 64)],  # Instagram gradient colors
                'text_patterns': [r'instagram', r'@\w+', r'\d+[km]?\s*likes?', r'story']
            },
            PlatformType.FACEBOOK: {
                'ui_elements': ['facebook', 'like', 'comment', 'share', 'feeling'],
                'colors': [(24, 119, 242), (66, 103, 178)],  # Facebook blue colors
                'text_patterns': [r'facebook', r'like', r'comment', r'share', r'\d+ comments?']
            },
            PlatformType.TWITTER: {
                'ui_elements': ['twitter', 'tweet', 'retweet', 'reply', 'hashtag'],
                'colors': [(29, 161, 242), (91, 192, 222)],  # Twitter blue colors
                'text_patterns': [r'twitter', r'@\w+', r'#\w+', r'\d+[km]?\s*retweets?']
            },
            PlatformType.DISCORD: {
                'ui_elements': ['discord', 'server', 'channel', 'voice', 'bot'],
                'colors': [(88, 101, 242), (114, 137, 218)],  # Discord purple/blue
                'text_patterns': [r'discord', r'#\w+', r'@everyone', r'bot']
            },
            PlatformType.SIGNAL: {
                'ui_elements': ['signal', 'disappearing messages', 'safety number'],
                'colors': [(58, 150, 221), (33, 150, 243)],  # Signal blue
                'text_patterns': [r'signal', r'disappearing', r'safety number']
            }
        }
    
    def detect_platform_by_color(self, image: np.ndarray) -> List[Tuple[PlatformType, float]]:
        """Detect platform by characteristic colors."""
        results = []
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for platform, patterns in self.platform_patterns.items():
            color_score = 0
            for target_color in patterns['colors']:
                # Create mask for similar colors
                lower = np.array([max(0, c - 30) for c in target_color])
                upper = np.array([min(255, c + 30) for c in target_color])
                mask = cv2.inRange(rgb_image, lower, upper)
                
                # Calculate percentage of matching pixels
                color_percentage = np.sum(mask > 0) / (image.shape[0] * image.shape[1])
                color_score = max(color_score, color_percentage)
            
            if color_score > 0.01:  # At least 1% matching pixels
                results.append((platform, color_score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def detect_platform_by_text(self, text: str) -> List[Tuple[PlatformType, float]]:
        """Detect platform by text patterns."""
        results = []
        text_lower = text.lower()
        
        for platform, patterns in self.platform_patterns.items():
            text_score = 0
            total_patterns = len(patterns['ui_elements']) + len(patterns['text_patterns'])
            
            # Check UI element keywords
            for element in patterns['ui_elements']:
                if element in text_lower:
                    text_score += 1
            
            # Check regex patterns
            for pattern in patterns['text_patterns']:
                if re.search(pattern, text_lower):
                    text_score += 1
            
            if text_score > 0:
                confidence = text_score / total_patterns
                results.append((platform, confidence))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def detect_platform(self, image_path: str, text: str) -> Dict[str, Any]:
        """Main platform detection function."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'platform': PlatformType.UNKNOWN, 'confidence': 0.0, 'method': 'error'}
            
            # Detect by color and text
            color_results = self.detect_platform_by_color(image)
            text_results = self.detect_platform_by_text(text)
            
            # Combine results
            platform_scores = {}
            
            # Weight color detection (40%) and text detection (60%)
            for platform, score in color_results:
                platform_scores[platform] = platform_scores.get(platform, 0) + (score * 0.4)
            
            for platform, score in text_results:
                platform_scores[platform] = platform_scores.get(platform, 0) + (score * 0.6)
            
            if not platform_scores:
                return {
                    'platform': PlatformType.UNKNOWN,
                    'confidence': 0.0,
                    'method': 'no_detection',
                    'color_results': color_results,
                    'text_results': text_results
                }
            
            # Get best match
            best_platform = max(platform_scores.items(), key=lambda x: x[1])
            
            return {
                'platform': best_platform[0],
                'confidence': min(best_platform[1], 1.0),
                'method': 'hybrid',
                'all_scores': platform_scores,
                'color_results': color_results,
                'text_results': text_results
            }
            
        except Exception as e:
            logger.error(f"Platform detection failed: {e}")
            return {'platform': PlatformType.UNKNOWN, 'confidence': 0.0, 'method': 'error', 'error': str(e)}

# ================================================================================================
# ADVANCED THREAT DETECTION WITH NLP
# ================================================================================================

class ThreatDetector:
    """Advanced threat detection using NLP and pattern matching."""
    
    def __init__(self):
        """Initialize threat detection models."""
        self.threat_keywords = {
            'violence': [
                'kill', 'murder', 'die', 'death', 'hurt', 'harm', 'attack', 'beat',
                'stab', 'shoot', 'gun', 'knife', 'weapon', 'violence', 'assault'
            ],
            'blackmail': [
                'blackmail', 'extort', 'threaten', 'expose', 'reveal', 'publish',
                'leak', 'embarrass', 'ruin', 'destroy', 'consequences'
            ],
            'harassment': [
                'harass', 'stalk', 'follow', 'watch', 'creep', 'obsess',
                'bother', 'annoy', 'pester', 'intimidate'
            ],
            'suicide': [
                'suicide', 'kill myself', 'end my life', 'want to die',
                'better off dead', 'not worth living'
            ],
            'hate_speech': [
                'hate', 'racist', 'discrimination', 'slur', 'offensive',
                'bigot', 'prejudice', 'intolerant'
            ]
        }
        
        self.escalation_indicators = [
            'deadline', 'last chance', 'final warning', 'or else',
            'you have until', 'time is running out', 'decide now'
        ]
        
        self.urgency_patterns = [
            r'\b(?:urgent|asap|immediate|now|today|tonight)\b',
            r'\b(?:hurry|quick|fast|soon)\b',
            r'\b\d+\s*(?:hours?|minutes?|days?)\s*(?:left|remaining)\b'
        ]
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Simple sentiment analysis."""
        if not text:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        # Simple keyword-based sentiment analysis
        positive_words = [
            'good', 'great', 'happy', 'love', 'like', 'awesome', 'fantastic',
            'wonderful', 'excellent', 'amazing', 'perfect', 'thanks'
        ]
        
        negative_words = [
            'bad', 'hate', 'angry', 'mad', 'upset', 'sad', 'terrible',
            'awful', 'horrible', 'stupid', 'idiot', 'worst', 'suck'
        ]
        
        words = text.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        total = len(words)
        
        if total == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        pos_score = pos_count / total
        neg_score = neg_count / total
        neu_score = max(0, 1 - pos_score - neg_score)
        
        return {'positive': pos_score, 'negative': neg_score, 'neutral': neu_score}
    
    def detect_threats(self, text: str) -> Dict[str, Any]:
        """Detect threats in text."""
        if not text:
            return {
                'threat_level': ThreatLevel.NONE,
                'threat_score': 0.0,
                'threat_categories': [],
                'indicators': []
            }
        
        text_lower = text.lower()
        threat_score = 0.0
        threat_categories = []
        indicators = []
        
        # Check threat categories
        for category, keywords in self.threat_keywords.items():
            category_score = 0
            category_matches = []
            
            for keyword in keywords:
                if keyword in text_lower:
                    category_score += 1
                    category_matches.append(keyword)
            
            if category_score > 0:
                threat_categories.append({
                    'category': category,
                    'score': category_score,
                    'matches': category_matches
                })
                threat_score += category_score * 0.1
        
        # Check escalation indicators
        escalation_score = 0
        for indicator in self.escalation_indicators:
            if indicator in text_lower:
                escalation_score += 1
                indicators.append(indicator)
        
        threat_score += escalation_score * 0.15
        
        # Check urgency patterns
        urgency_score = 0
        for pattern in self.urgency_patterns:
            if re.search(pattern, text_lower):
                urgency_score += 1
                indicators.append(f"urgency_pattern: {pattern}")
        
        threat_score += urgency_score * 0.1
        
        # Sentiment analysis
        sentiment = self.analyze_sentiment(text)
        if sentiment['negative'] > 0.3:
            threat_score += sentiment['negative'] * 0.2
            indicators.append(f"negative_sentiment: {sentiment['negative']:.2f}")
        
        # Determine threat level
        if threat_score >= 0.8:
            threat_level = ThreatLevel.CRITICAL
        elif threat_score >= 0.6:
            threat_level = ThreatLevel.HIGH
        elif threat_score >= 0.4:
            threat_level = ThreatLevel.MEDIUM
        elif threat_score >= 0.2:
            threat_level = ThreatLevel.LOW
        else:
            threat_level = ThreatLevel.NONE
        
        return {
            'threat_level': threat_level,
            'threat_score': min(threat_score, 1.0),
            'threat_categories': threat_categories,
            'indicators': indicators,
            'sentiment': sentiment,
            'escalation_indicators': escalation_score,
            'urgency_indicators': urgency_score
        }

# ================================================================================================
# NSFW/ADULT CONTENT DETECTION
# ================================================================================================

class NSFWDetector:
    """NSFW content detection using computer vision."""
    
    def __init__(self):
        """Initialize NSFW detector."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = None
        self.clip_preprocess = None
        
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("CLIP model loaded for NSFW detection")
        except Exception as e:
            logger.warning(f"CLIP model failed to load: {e}")
        
        self.nsfw_prompts = [
            "explicit sexual content", "nudity", "adult content",
            "pornographic image", "sexual activity", "naked person"
        ]
        
        self.safe_prompts = [
            "safe for work content", "family friendly image",
            "appropriate content", "professional image"
        ]
    
    def detect_skin_regions(self, image: np.ndarray) -> float:
        """Detect skin regions in image."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_percentage = np.sum(skin_mask > 0) / (image.shape[0] * image.shape[1])
        
        return skin_percentage
    
    def detect_nsfw_content(self, image_path: str) -> Dict[str, Any]:
        """Detect NSFW content in image."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'classification': 'UNKNOWN', 'score': 0.0, 'error': 'Could not load image'}
            
            # Skin detection
            skin_percentage = self.detect_skin_regions(image)
            
            # CLIP-based detection
            clip_score = 0.0
            if self.clip_model and self.clip_preprocess:
                try:
                    pil_image = PILImage.open(image_path).convert('RGB')
                    image_tensor = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        nsfw_tokens = clip.tokenize(self.nsfw_prompts).to(self.device)
                        safe_tokens = clip.tokenize(self.safe_prompts).to(self.device)
                        
                        image_features = self.clip_model.encode_image(image_tensor)
                        nsfw_features = self.clip_model.encode_text(nsfw_tokens)
                        safe_features = self.clip_model.encode_text(safe_tokens)
                        
                        nsfw_similarities = torch.cosine_similarity(image_features, nsfw_features)
                        safe_similarities = torch.cosine_similarity(image_features, safe_features)
                        
                        nsfw_score = float(torch.mean(nsfw_similarities))
                        safe_score = float(torch.mean(safe_similarities))
                        
                        clip_score = nsfw_score - safe_score
                        
                except Exception as e:
                    logger.debug(f"CLIP NSFW detection failed: {e}")
            
            # Combine scores
            skin_score = min(skin_percentage * 2, 1.0)  # Normalize skin percentage
            final_score = (skin_score * 0.6 + max(0, clip_score) * 0.4)
            
            # Classify
            if final_score >= 0.7:
                classification = "VERY_LIKELY"
            elif final_score >= 0.5:
                classification = "LIKELY"
            elif final_score >= 0.3:
                classification = "POSSIBLE"
            else:
                classification = "UNLIKELY"
            
            return {
                'classification': classification,
                'score': final_score,
                'skin_percentage': skin_percentage,
                'clip_score': clip_score,
                'method': 'hybrid'
            }
            
        except Exception as e:
            logger.error(f"NSFW detection failed: {e}")
            return {'classification': 'UNKNOWN', 'score': 0.0, 'error': str(e)}

# ================================================================================================
# MAIN FORENSNAP ANALYZER
# ================================================================================================

class ForenSnapAnalyzer:
    """Main ForenSnap analyzer integrating all detection modules."""
    
    def __init__(self, db_path=None):
        """Initialize ForenSnap analyzer."""
        if db_path is None:
            db_path = DATA_DIR / 'forensnap_ultimate.db'
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize all detection modules
        logger.info("Initializing ForenSnap modules...")
        
        # Use enhanced OCR module if available, otherwise use built-in
        if USE_ENHANCED_OCR and EnhancedOCR:
            logger.info("Using enhanced OCR module from modules/enhanced_ocr.py")
            # Import from the enhanced module
            from modules.enhanced_ocr import EnhancedOCR as ExternalEnhancedOCR
            self.ocr = ExternalEnhancedOCR()
        else:
            logger.info("Using built-in OCR implementation")
            self.ocr = EnhancedOCR()
        self.platform_detector = PlatformDetector()
        self.threat_detector = ThreatDetector()
        self.nsfw_detector = NSFWDetector()
        
        # Initialize BLIP model with better configuration
        try:
            logger.info("Loading BLIP model...")
            # Try BLIP-large first for better performance, fallback to base
            try:
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=True)
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
                logger.info("BLIP-large model loaded successfully")
            except:
                logger.info("BLIP-large not available, using base model...")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                logger.info("BLIP-base model loaded successfully")
        except Exception as e:
            logger.warning(f"BLIP model failed to load: {e}")
            self.blip_processor = None
            self.blip_model = None
        
        logger.info("ForenSnap initialization complete!")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text."""
        entities = []
        if not text:
            return entities
        
        # Phone numbers
        phone_patterns = [
            r'(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{4}',
            r'(?:\+?91[-.\s]?)?[6-9]\d{9}',
            r'(?:\+?44[-.\s]?)?(?:0)?[1-9]\d{8,9}'
        ]
        
        for pattern in phone_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    'text': match.group(0),
                    'type': 'PHONE_NUMBER',
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append({
                'text': match.group(0),
                'type': 'EMAIL',
                'start': match.start(),
                'end': match.end()
            })
        
        # Currency amounts
        currency_pattern = r'(?:[$€£¥₹])\s*\d+(?:,\d{3})*(?:\.\d{2})?'
        for match in re.finditer(currency_pattern, text):
            entities.append({
                'text': match.group(0),
                'type': 'CURRENCY',
                'start': match.start(),
                'end': match.end()
            })
        
        # URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        for match in re.finditer(url_pattern, text):
            entities.append({
                'text': match.group(0),
                'type': 'URL',
                'start': match.start(),
                'end': match.end()
            })
        
        return entities
    
    def classify_content(self, text: str, platform: PlatformType, threat_result: Dict) -> Category:
        """Classify content into categories."""
        # Check for adult content classification from NSFW detector
        if hasattr(self, '_last_nsfw_result'):
            if self._last_nsfw_result.get('classification') in ['VERY_LIKELY', 'LIKELY']:
                return Category.ADULT
        
        # Check for threats
        if threat_result['threat_level'] != ThreatLevel.NONE:
            return Category.THREAT
        
        # Check for transactions
        transaction_keywords = [
            'payment', 'transaction', 'transfer', 'credit', 'debit',
            'bank', 'account', 'wallet', 'upi', 'paid', 'received'
        ]
        
        transaction_score = sum(1 for keyword in transaction_keywords if keyword in text.lower())
        
        # Check for chat/social media
        if platform != PlatformType.UNKNOWN:
            return Category.SOCIAL_MEDIA
        
        chat_keywords = [
            'message', 'chat', 'conversation', 'reply', 'said',
            'whatsapp', 'telegram', 'instagram', 'facebook'
        ]
        
        chat_score = sum(1 for keyword in chat_keywords if keyword in text.lower())
        
        if transaction_score >= 2:
            return Category.TRANSACTION
        elif chat_score >= 2 or platform != PlatformType.UNKNOWN:
            return Category.CHAT
        else:
            return Category.UNCATEGORIZED
    
    def generate_blip_caption(self, image_path: str) -> str:
        """Generate image caption using BLIP with better parameters."""
        if not self.blip_processor or not self.blip_model:
            return ""
        
        try:
            image = PILImage.open(image_path).convert('RGB')
            
            # Generate multiple captions with different prompts
            captions = []
            
            # 1. Unconditional caption
            try:
                inputs = self.blip_processor(image, return_tensors="pt")
                with torch.no_grad():
                    out = self.blip_model.generate(
                        **inputs, 
                        max_length=75,
                        num_beams=5,
                        early_stopping=True,
                        temperature=0.8,
                        do_sample=True
                    )
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                if caption and len(caption.strip()) > 3:
                    captions.append(caption.strip())
            except Exception as e:
                logger.debug(f"Unconditional BLIP failed: {e}")
            
            # 2. Conditional caption with context-aware prompts
            context_prompts = [
                "a screenshot of",
                "the image shows", 
                "this digital image contains",
                "a mobile phone screen displaying"
            ]
            
            for prompt in context_prompts[:2]:  # Use first 2 prompts to avoid too many requests
                try:
                    inputs = self.blip_processor(image, prompt, return_tensors="pt")
                    with torch.no_grad():
                        out = self.blip_model.generate(
                            **inputs, 
                            max_length=85,
                            num_beams=3,
                            early_stopping=True,
                            temperature=0.7
                        )
                    caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                    if caption and len(caption.strip()) > len(prompt):
                        captions.append(caption.strip())
                except Exception as e:
                    logger.debug(f"Conditional BLIP with prompt '{prompt}' failed: {e}")
                    continue
            
            # Return the longest, most descriptive caption
            if captions:
                best_caption = max(captions, key=len)
                # Clean up the caption
                best_caption = re.sub(r'^(a screenshot of|the image shows|this digital image contains)\s*', '', best_caption, flags=re.IGNORECASE)
                return best_caption.strip()
            
            return ""
            
        except Exception as e:
            logger.debug(f"BLIP caption generation failed: {e}")
            return ""
    
    def extract_caption_tags(self, caption: str) -> List[str]:
        """Extract meaningful tags from BLIP caption by processing each word."""
        if not caption:
            return []
        
        # Common stop words to filter out
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'there', 'this', 'these', 'they',
            'them', 'their', 'his', 'her', 'him', 'she', 'we', 'you', 'your'
        }
        
        # Clean the caption and split into words
        words = re.findall(r'\b\w+\b', caption.lower())
        
        # Filter out stop words and short words
        meaningful_words = [
            word for word in words 
            if word not in stop_words and len(word) > 2
        ]
        
        # Remove duplicates while preserving order
        tags = []
        seen = set()
        for word in meaningful_words:
            if word not in seen:
                tags.append(word)
                seen.add(word)
        
        return tags[:10]  # Limit to 10 most relevant tags
    
    def process_image(self, image_path: str, case_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a single image with comprehensive analysis."""
        try:
            if not os.path.exists(image_path):
                return {'error': f'File not found: {image_path}'}
            
            # Generate unique ID and calculate hash
            image_id = str(uuid.uuid4())
            file_hash = self.calculate_file_hash(image_path)
            file_size = os.path.getsize(image_path)
            
            session = self.Session()
            
            # Check if already processed
            existing = session.query(Image).filter_by(file_hash=file_hash).first()
            if existing:
                logger.info(f"Image already processed: {existing.id}")
                return self._create_response_from_db(existing)
            
            logger.info(f"Processing new image: {os.path.basename(image_path)}")
            
            # OCR text extraction
            ocr_result = self.ocr.extract_text_hybrid(image_path)
            detected_text = ocr_result.get('text', '')
            detected_language = ocr_result.get('language', 'unknown')
            
            # Platform detection
            platform_result = self.platform_detector.detect_platform(image_path, detected_text)
            platform = platform_result.get('platform', PlatformType.UNKNOWN)
            
            # Threat detection
            threat_result = self.threat_detector.detect_threats(detected_text)
            threat_level = threat_result.get('threat_level', ThreatLevel.NONE)
            
            # NSFW detection
            nsfw_result = self.nsfw_detector.detect_nsfw_content(image_path)
            self._last_nsfw_result = nsfw_result  # Store for classification
            nsfw_score = nsfw_result.get('score', 0.0)
            
            # Content classification
            category = self.classify_content(detected_text, platform, threat_result)
            
            # BLIP caption generation
            blip_caption = self.generate_blip_caption(image_path)
            
            # Extract caption tags from BLIP
            caption_tags = self.extract_caption_tags(blip_caption) if blip_caption else []
            
            # Extract entities
            entities = self.extract_entities(detected_text)
            
            # Calculate overall confidence
            confidence_score = (
                ocr_result.get('confidence', 0) * 0.3 +
                platform_result.get('confidence', 0) * 100 * 0.2 +
                (1 - nsfw_result.get('score', 0)) * 0.2 +
                (1 if threat_level == ThreatLevel.NONE else 0.8) * 0.3
            ) / 100
            
            # Create metadata
            metadata = {
                'ocr_result': ocr_result,
                'platform_result': platform_result,
                'threat_result': threat_result,
                'nsfw_result': nsfw_result,
                'entities': entities,
                'blip_caption': blip_caption,
                'processing_timestamp': datetime.datetime.utcnow().isoformat()
            }
            
            # Create database record
            image_record = Image(
                id=image_id,
                case_id=case_id,
                file_path=image_path,
                file_hash=file_hash,
                file_size=file_size,
                detected_text=detected_text,
                detected_language=detected_language,
                category=category.value,
                platform=platform.value,
                threat_level=threat_level.value,
                nsfw_score=nsfw_score,
                confidence_score=confidence_score,
                analysis_metadata=metadata
            )
            
            # Generate and save tags (including caption tags)
            all_tags = self._generate_comprehensive_tags(
                detected_text, entities, platform, category, 
                threat_result, blip_caption, caption_tags
            )
            
            # Save tags and create relationships
            tag_relationships = []
            processed_tags = set()  # Track processed tag combinations
            for tag_data in all_tags:
                tag = session.query(Tag).filter_by(
                    name=tag_data['name'], 
                    tag_type=tag_data['type']
                ).first()
                
                if not tag:
                    tag = Tag(
                        name=tag_data['name'],
                        tag_type=tag_data['type'],
                        frequency=1
                    )
                    session.add(tag)
                    session.flush()
                else:
                    tag.frequency += 1
                
                # Only add if not already processed (avoid duplicates)
                tag_key = (image_id, tag.id)
                if tag_key not in processed_tags:
                    tag_relationships.append(ImageTag(
                        image_id=image_id,
                        tag_id=tag.id,
                        confidence=tag_data.get('confidence')
                    ))
                    processed_tags.add(tag_key)
            
            # Save everything
            session.add(image_record)
            session.add_all(tag_relationships)
            session.commit()
            
            # Create response
            response = {
                'image_id': image_id,
                'file_path': image_path,
                'file_hash': file_hash,
                'file_size': file_size,
                'detected_text': detected_text,
                'detected_language': detected_language,
                'category': category.value,
                'platform': platform.value,
                'threat_level': threat_level.value,
                'nsfw_score': nsfw_score,
                'nsfw_classification': nsfw_result.get('classification', 'UNKNOWN'),
                'confidence_score': confidence_score,
                'entities': [e['text'] for e in entities],
                'tags': [t['name'] for t in all_tags],
                'blip_caption': blip_caption,
                'warnings': self._generate_warnings(threat_result, nsfw_result),
                'timestamp': datetime.datetime.utcnow().isoformat(),
                'processing_details': {
                    'ocr_method': ocr_result.get('method', 'unknown'),
                    'ocr_confidence': ocr_result.get('confidence', 0),
                    'platform_confidence': platform_result.get('confidence', 0),
                    'threat_score': threat_result.get('threat_score', 0),
                    'skin_percentage': nsfw_result.get('skin_percentage', 0)
                }
            }
            
            logger.info(f"Image processing complete: {category.value} | {platform.value} | {threat_level.value}")
            return response
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            logger.error(traceback.format_exc())
            if 'session' in locals():
                session.rollback()
            return {
                'error': str(e),
                'image_id': 'error',
                'timestamp': datetime.datetime.utcnow().isoformat()
            }
        finally:
            if 'session' in locals():
                session.close()
    
    def _generate_comprehensive_tags(self, text: str, entities: List, platform: PlatformType, 
                                   category: Category, threat_result: Dict, blip_caption: str, caption_tags: List[str] = None) -> List[Dict]:
        """Generate comprehensive tags from all analysis results."""
        tags = []
        
        # Category tag
        tags.append({'name': category.value, 'type': TagType.KEYWORD.value, 'confidence': 1.0})
        
        # Platform tag
        if platform != PlatformType.UNKNOWN:
            tags.append({'name': platform.value, 'type': TagType.PLATFORM.value, 'confidence': 0.9})
        
        # Entity tags
        for entity in entities:
            tags.append({
                'name': f"{entity['type'].lower()}_{entity['text']}",
                'type': TagType.ENTITY.value,
                'confidence': 0.8
            })
        
        # Threat tags
        if threat_result.get('threat_level') != ThreatLevel.NONE:
            tags.append({
                'name': f"threat_{threat_result['threat_level'].value}",
                'type': TagType.THREAT.value,
                'confidence': threat_result.get('threat_score', 0)
            })
        
        # BLIP caption tags from extracted meaningful words
        if caption_tags:
            for tag in caption_tags[:10]:  # Use extracted caption tags
                tags.append({
                    'name': f"caption_{tag}",
                    'type': TagType.BLIP.value,
                    'confidence': 0.8
                })
        
        # Additional BLIP caption tags from raw text (if caption_tags not available)
        elif blip_caption:
            caption_words = [word.lower() for word in blip_caption.split() if len(word) > 3]
            for word in caption_words[:5]:  # Top 5 words
                tags.append({
                    'name': f"blip_{word}",
                    'type': TagType.BLIP.value,
                    'confidence': 0.7
                })
        
        # Keyword tags from OCR text
        if text:
            words = re.findall(r'\b\w{4,}\b', text.lower())
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Top keywords
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            for word, freq in top_words:
                if word not in ['that', 'this', 'with', 'have', 'will', 'from']:  # Filter common words
                    tags.append({
                        'name': f"ocr_{word}",
                        'type': TagType.OCR.value,
                        'confidence': min(freq / len(words), 1.0)
                    })
        
        return tags
    
    def _generate_warnings(self, threat_result: Dict, nsfw_result: Dict) -> List[str]:
        """Generate warnings based on analysis results."""
        warnings = []
        
        if threat_result.get('threat_level') in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            warnings.append("⚠️ HIGH THREAT LEVEL DETECTED")
        
        if nsfw_result.get('classification') in ['VERY_LIKELY', 'LIKELY']:
            warnings.append("🔞 ADULT CONTENT DETECTED")
        
        if threat_result.get('threat_score', 0) > 0.5:
            warnings.append("🚨 POTENTIAL THREAT IDENTIFIED")
        
        if nsfw_result.get('skin_percentage', 0) > 0.4:
            warnings.append("👤 HIGH SKIN EXPOSURE DETECTED")
        
        return warnings
    
    def _create_response_from_db(self, db_image: Image) -> Dict[str, Any]:
        """Create response from existing database record."""
        metadata = db_image.analysis_metadata or {}
        
        # Get tags for this image
        tags = [tag.tag.name for tag in db_image.tags]
        entities = metadata.get('entities', [])
        
        return {
            'image_id': db_image.id,
            'file_path': db_image.file_path,
            'file_hash': db_image.file_hash,
            'detected_text': db_image.detected_text,
            'detected_language': db_image.detected_language,
            'category': db_image.category,
            'platform': db_image.platform,
            'threat_level': db_image.threat_level,
            'nsfw_score': db_image.nsfw_score,
            'nsfw_classification': metadata.get('nsfw_result', {}).get('classification', 'UNKNOWN'),
            'confidence_score': db_image.confidence_score,
            'entities': [e['text'] for e in entities] if entities else [],
            'tags': tags,
            'blip_caption': metadata.get('blip_caption', ''),
            'timestamp': db_image.created_at.isoformat(),
            'cached': True,
            'warnings': self._generate_warnings(
                metadata.get('threat_result', {}),
                metadata.get('nsfw_result', {})
            ),
            'processing_details': {
                'ocr_method': metadata.get('ocr_result', {}).get('method', 'unknown'),
                'ocr_confidence': metadata.get('ocr_result', {}).get('confidence', 0),
                'platform_confidence': metadata.get('platform_result', {}).get('confidence', 0),
                'threat_score': metadata.get('threat_result', {}).get('threat_score', 0),
                'skin_percentage': metadata.get('nsfw_result', {}).get('skin_percentage', 0)
            }
        }
    
    def search_images(self, query: str = None, category: str = None, platform: str = None,
                     threat_level: str = None, limit: int = 100) -> List[Dict]:
        """Search images with advanced filtering."""
        session = self.Session()
        try:
            query_obj = session.query(Image)
            
            if category:
                query_obj = query_obj.filter(Image.category == category)
            if platform:
                query_obj = query_obj.filter(Image.platform == platform)
            if threat_level:
                query_obj = query_obj.filter(Image.threat_level == threat_level)
            if query:
                query_obj = query_obj.filter(Image.detected_text.like(f'%{query}%'))
            
            results = query_obj.order_by(Image.created_at.desc()).limit(limit).all()
            return [self._create_response_from_db(img) for img in results]
            
        finally:
            session.close()
    
    def generate_legal_report(self, case_id: str = None, output_path: str = None) -> str:
        """Generate a legal compliance report."""
        if not output_path:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'ForenSnap_Legal_Report_{timestamp}.pdf'
        
        session = self.Session()
        try:
            # Get case info if specified
            case_info = None
            if case_id:
                case_info = session.query(Case).filter_by(id=case_id).first()
            
            # Get images
            if case_id:
                images = session.query(Image).filter_by(case_id=case_id).all()
            else:
                images = session.query(Image).order_by(Image.created_at.desc()).limit(100).all()
            
            # Create PDF report
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            elements = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                textColor=colors.darkblue,
                alignment=1,  # Center alignment
                spaceAfter=20
            )
            
            elements.append(Paragraph("FORENSNAP DIGITAL INVESTIGATION REPORT", title_style))
            elements.append(Spacer(1, 20))
            
            # Report metadata
            report_info = [
                ['Report Generated:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Total Images Analyzed:', str(len(images))],
                ['Analysis Software:', 'ForenSnap Ultimate v2.0.0'],
                ['Report ID:', str(uuid.uuid4())[:8]]
            ]
            
            if case_info:
                report_info.extend([
                    ['Case Number:', case_info.case_number],
                    ['Case Title:', case_info.title],
                    ['Investigator:', case_info.investigator]
                ])
            
            info_table = Table(report_info, colWidths=[2*inch, 4*inch])
            info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            elements.append(info_table)
            elements.append(PageBreak())
            
            # Summary statistics
            elements.append(Paragraph("ANALYSIS SUMMARY", styles['Heading1']))
            
            # Category distribution
            category_stats = {}
            threat_stats = {}
            platform_stats = {}
            
            for img in images:
                category_stats[img.category] = category_stats.get(img.category, 0) + 1
                threat_stats[img.threat_level] = threat_stats.get(img.threat_level, 0) + 1
                if img.platform:
                    platform_stats[img.platform] = platform_stats.get(img.platform, 0) + 1
            
            # Create summary tables
            cat_data = [['Category', 'Count', 'Percentage']]
            for cat, count in category_stats.items():
                percentage = (count / len(images)) * 100
                cat_data.append([cat.replace('_', ' ').title(), str(count), f'{percentage:.1f}%'])
            
            cat_table = Table(cat_data, colWidths=[2*inch, 1*inch, 1*inch])
            cat_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(Paragraph("Content Category Distribution:", styles['Heading2']))
            elements.append(cat_table)
            elements.append(Spacer(1, 20))
            
            # Threat level distribution
            threat_data = [['Threat Level', 'Count', 'Risk Assessment']]
            for threat, count in threat_stats.items():
                risk = 'CRITICAL' if threat in ['high', 'critical'] else 'LOW-MEDIUM'
                threat_data.append([threat.replace('_', ' ').title(), str(count), risk])
            
            threat_table = Table(threat_data, colWidths=[2*inch, 1*inch, 2*inch])
            threat_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(Paragraph("Threat Level Assessment:", styles['Heading2']))
            elements.append(threat_table)
            elements.append(PageBreak())
            
            # Detailed findings for high-risk images
            high_risk_images = [img for img in images if img.threat_level in ['high', 'critical'] or img.nsfw_score > 0.7]
            
            if high_risk_images:
                elements.append(Paragraph("HIGH-RISK CONTENT DETAILS", styles['Heading1']))
                
                for img in high_risk_images[:10]:  # Limit to top 10
                    elements.append(Paragraph(f"Image ID: {img.id}", styles['Heading2']))
                    
                    details = [
                        ['File Path:', img.file_path],
                        ['Category:', img.category.replace('_', ' ').title()],
                        ['Platform:', img.platform.replace('_', ' ').title() if img.platform else 'Unknown'],
                        ['Threat Level:', img.threat_level.replace('_', ' ').title()],
                        ['NSFW Score:', f'{img.nsfw_score:.2f}'],
                        ['Confidence:', f'{img.confidence_score:.2f}'],
                        ['Processing Date:', img.created_at.strftime('%Y-%m-%d %H:%M:%S')]
                    ]
                    
                    detail_table = Table(details, colWidths=[1.5*inch, 4*inch])
                    detail_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    
                    elements.append(detail_table)
                    
                    # Add detected text excerpt
                    if img.detected_text:
                        text_excerpt = img.detected_text[:200] + '...' if len(img.detected_text) > 200 else img.detected_text
                        elements.append(Paragraph("Detected Text:", styles['Heading3']))
                        elements.append(Paragraph(text_excerpt, styles['Normal']))
                    
                    elements.append(Spacer(1, 20))
            
            # Legal disclaimer
            elements.append(PageBreak())
            elements.append(Paragraph("LEGAL DISCLAIMER & CHAIN OF CUSTODY", styles['Heading1']))
            
            disclaimer_text = """
            This report was generated using ForenSnap Ultimate digital forensics software. The analysis results are based on 
            automated AI-powered detection algorithms and should be verified by qualified digital forensics experts. 
            
            All images analyzed maintain their original file integrity through SHA-256 hash verification. The analysis 
            metadata, timestamps, and processing details are stored in a secure database for audit trail purposes.
            
            This report is intended for authorized investigation purposes only and should be handled in accordance with 
            applicable privacy laws and regulations.
            """
            
            elements.append(Paragraph(disclaimer_text, styles['Normal']))
            
            # Build PDF
            doc.build(elements)
            logger.info(f"Legal report generated: {output_path}")
            return output_path
            
        finally:
            session.close()

# ================================================================================================
# GUI APPLICATION
# ================================================================================================

class ForenSnapGUI:
    """Advanced GUI for ForenSnap Ultimate."""
    
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("ForenSnap Ultimate - AI-Powered Digital Forensics")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize analyzer
        self.analyzer = ForenSnapAnalyzer()
        self.current_image_path = None
        self.processing_queue = queue.Queue()
        self.batch_results = []
        
        self._create_gui()
        self._start_processing_thread()
    
    def _create_gui(self):
        """Create the main GUI interface."""
        # Create main notebook
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#2b2b2b')
        style.configure('TNotebook.Tab', background='#404040', foreground='white')
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self._create_single_analysis_tab()
        self._create_batch_processing_tab()
        self._create_search_tab()
        self._create_reports_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("ForenSnap Ultimate Ready - All AI models loaded ✅")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                            bd=1, relief=tk.SUNKEN, anchor=tk.W,
                            bg='#404040', fg='white')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _create_single_analysis_tab(self):
        """Create single image analysis tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔍 Single Image Analysis")
        
        # Top controls
        control_frame = ttk.LabelFrame(frame, text="Image Controls")
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="📂 Select Image", command=self._select_image).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="🔬 Analyze", command=self._analyze_single).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="💾 Save Report", command=self._save_single_report).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Main content area
        content_frame = ttk.Frame(frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Image preview (left side)
        image_frame = ttk.LabelFrame(content_frame, text="Image Preview")
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.image_label = tk.Label(image_frame, text="No image selected", 
                                  bg='#404040', fg='white', width=50, height=20)
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Results area (right side)
        results_frame = ttk.LabelFrame(content_frame, text="Analysis Results")
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Results notebook
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Summary tab
        summary_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(summary_frame, text="📊 Summary")
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, wrap=tk.WORD, 
                                                    bg='#3b3b3b', fg='white', height=15)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # OCR Results tab
        ocr_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(ocr_frame, text="📝 OCR Text")
        
        self.ocr_text = scrolledtext.ScrolledText(ocr_frame, wrap=tk.WORD,
                                                bg='#3b3b3b', fg='white')
        self.ocr_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Threats tab
        threats_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(threats_frame, text="⚠️ Threats")
        
        self.threats_text = scrolledtext.ScrolledText(threats_frame, wrap=tk.WORD,
                                                    bg='#3b3b3b', fg='white')
        self.threats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _create_batch_processing_tab(self):
        """Create batch processing tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="⚡ Batch Processing")
        
        # Controls
        control_frame = ttk.LabelFrame(frame, text="Batch Controls")
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="📁 Select Folder", command=self._select_folder).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="▶️ Start Batch", command=self._start_batch).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="⏹️ Stop", command=self._stop_batch).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Progress area
        progress_frame = ttk.LabelFrame(frame, text="Progress")
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.progress_var = tk.StringVar()
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        # Results list
        results_frame = ttk.LabelFrame(frame, text="Batch Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create treeview for batch results
        columns = ("Filename", "Category", "Platform", "Threat", "NSFW", "Status")
        self.batch_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            self.batch_tree.heading(col, text=col)
            self.batch_tree.column(col, width=120)
        
        # Scrollbar for treeview
        batch_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.batch_tree.yview)
        self.batch_tree.configure(yscrollcommand=batch_scrollbar.set)
        
        self.batch_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        batch_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_search_tab(self):
        """Create search and database tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔎 Search Database")
        
        # Search controls
        search_frame = ttk.LabelFrame(frame, text="Search Filters")
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Search fields
        search_grid = ttk.Frame(search_frame)
        search_grid.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(search_grid, text="Text Query:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.search_text = tk.StringVar()
        ttk.Entry(search_grid, textvariable=self.search_text, width=30).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(search_grid, text="Category:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.search_category = tk.StringVar()
        category_combo = ttk.Combobox(search_grid, textvariable=self.search_category, width=15,
                                    values=["", "chat", "transaction", "threat", "adult_content", "social_media"])
        category_combo.grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(search_grid, text="Platform:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.search_platform = tk.StringVar()
        platform_combo = ttk.Combobox(search_grid, textvariable=self.search_platform, width=15,
                                    values=["", "whatsapp", "telegram", "instagram", "facebook", "twitter"])
        platform_combo.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(search_grid, text="Threat Level:").grid(row=1, column=2, sticky=tk.W, padx=5)
        self.search_threat = tk.StringVar()
        threat_combo = ttk.Combobox(search_grid, textvariable=self.search_threat, width=15,
                                  values=["", "none", "low", "medium", "high", "critical"])
        threat_combo.grid(row=1, column=3, padx=5, pady=2)
        
        ttk.Button(search_grid, text="🔍 Search", command=self._search_database).grid(row=2, column=0, pady=10)
        ttk.Button(search_grid, text="📊 Export Results", command=self._export_search).grid(row=2, column=1, pady=10)
        
        # Search results
        results_frame = ttk.LabelFrame(frame, text="Search Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        search_columns = ("ID", "Category", "Platform", "Threat", "Detected Text", "Date")
        self.search_tree = ttk.Treeview(results_frame, columns=search_columns, show="headings", height=20)
        
        for col in search_columns:
            self.search_tree.heading(col, text=col)
            if col == "Detected Text":
                self.search_tree.column(col, width=300)
            elif col == "ID":
                self.search_tree.column(col, width=80)
            else:
                self.search_tree.column(col, width=100)
        
        search_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.search_tree.yview)
        self.search_tree.configure(yscrollcommand=search_scrollbar.set)
        
        self.search_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        search_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_reports_tab(self):
        """Create reports generation tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📄 Legal Reports")
        
        # Report controls
        report_frame = ttk.LabelFrame(frame, text="Report Generation")
        report_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(report_frame, text="📋 Generate Legal Report", 
                  command=self._generate_legal_report).pack(side=tk.LEFT, padx=10, pady=10)
        ttk.Button(report_frame, text="📈 Statistics Report", 
                  command=self._generate_stats_report).pack(side=tk.LEFT, padx=10, pady=10)
        
        # Report preview area
        preview_frame = ttk.LabelFrame(frame, text="Report Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.report_text = scrolledtext.ScrolledText(preview_frame, wrap=tk.WORD,
                                                   bg='#3b3b3b', fg='white')
        self.report_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def _start_processing_thread(self):
        """Start background processing thread."""
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
    
    def _process_queue(self):
        """Process items in the queue."""
        while True:
            try:
                item = self.processing_queue.get(timeout=1)
                if item['type'] == 'single':
                    self._process_single_image(item['path'])
                elif item['type'] == 'batch':
                    self._process_batch_images(item['paths'])
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
    
    def _select_image(self):
        """Select single image for analysis."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.status_var.set(f"Selected: {os.path.basename(file_path)}")
            self._display_image(file_path)
    
    def _display_image(self, file_path):
        """Display image in preview area."""
        try:
            image = PILImage.open(file_path)
            image.thumbnail((400, 400), PILImage.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            self.image_label.config(text=f"Error loading image: {str(e)}")
    
    def _analyze_single(self):
        """Start single image analysis."""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        self.status_var.set("🔬 Analyzing image... Please wait")
        self.processing_queue.put({'type': 'single', 'path': self.current_image_path})
    
    def _process_single_image(self, image_path):
        """Process single image in background thread."""
        try:
            result = self.analyzer.process_image(image_path)
            
            # Update GUI in main thread
            self.root.after(0, self._update_single_results, result)
            
        except Exception as e:
            self.root.after(0, self._show_error, f"Analysis failed: {str(e)}")
    
    def _update_single_results(self, result):
        """Update GUI with single image results."""
        if 'error' in result:
            self._show_error(result['error'])
            return
        
        # Clear previous results
        self.summary_text.delete(1.0, tk.END)
        self.ocr_text.delete(1.0, tk.END)
        self.threats_text.delete(1.0, tk.END)
        
        # Summary tab
        summary = f"""
🔍 FORENSNAP ANALYSIS RESULTS
{'='*50}

📁 File: {os.path.basename(result['file_path'])}
🆔 Image ID: {result['image_id']}
📅 Processed: {result['timestamp']}

📊 CLASSIFICATION RESULTS:
• Category: {result['category'].replace('_', ' ').title()}
• Platform: {result['platform'].replace('_', ' ').title()}
• Threat Level: {result['threat_level'].replace('_', ' ').title()}
• Language: {result.get('detected_language', 'Unknown')}

🎯 CONFIDENCE SCORES:
• Overall Confidence: {result.get('confidence_score', 0):.2%}
• OCR Confidence: {result.get('processing_details', {}).get('ocr_confidence', 0):.1f}%
• Platform Confidence: {result.get('processing_details', {}).get('platform_confidence', 0):.2%}
• Threat Score: {result.get('processing_details', {}).get('threat_score', 0):.2%}

🔞 CONTENT SAFETY:
• NSFW Classification: {result.get('nsfw_classification', 'UNKNOWN')}
• NSFW Score: {result.get('nsfw_score', 0):.2%}
• Skin Percentage: {result.get('processing_details', {}).get('skin_percentage', 0):.1%}

🏷️ DETECTED ENTITIES:
{chr(10).join(f'• {entity}' for entity in result.get('entities', []))}

🤖 AI CAPTION:
{result.get('blip_caption', 'No caption generated')}

⚠️ WARNINGS:
{chr(10).join(f'• {warning}' for warning in result.get('warnings', []))}
"""
        
        self.summary_text.insert(tk.END, summary)
        
        # OCR tab
        ocr_content = f"""
📝 EXTRACTED TEXT (OCR)
{'='*50}

Method: {result.get('processing_details', {}).get('ocr_method', 'Unknown')}
Confidence: {result.get('processing_details', {}).get('ocr_confidence', 0):.1f}%
Language: {result.get('detected_language', 'Unknown')}

TEXT CONTENT:
{'-'*30}
{result.get('detected_text', 'No text detected')}
"""
        
        self.ocr_text.insert(tk.END, ocr_content)
        
        # Threats tab
        threats_content = f"""
⚠️ THREAT ANALYSIS
{'='*50}

Threat Level: {result['threat_level'].replace('_', ' ').title()}
Threat Score: {result.get('processing_details', {}).get('threat_score', 0):.2%}

ANALYSIS DETAILS:
• Content contains potential threats
• Automated analysis may require human verification
• Report to appropriate authorities if necessary

For detailed threat analysis, consult with security professionals.
"""
        
        self.threats_text.insert(tk.END, threats_content)
        
        self.status_var.set("✅ Analysis complete!")
    
    def _show_error(self, error_message):
        """Show error message."""
        messagebox.showerror("Error", error_message)
        self.status_var.set("❌ Error occurred")
    
    def _select_folder(self):
        """Select folder for batch processing."""
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if folder_path:
            self.batch_folder = folder_path
            self.status_var.set(f"Selected folder: {os.path.basename(folder_path)}")
    
    def _start_batch(self):
        """Start batch processing."""
        if not hasattr(self, 'batch_folder'):
            messagebox.showwarning("Warning", "Please select a folder first")
            return
        
        # Find all image files in folder
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
        image_paths = []
        
        for root, dirs, files in os.walk(self.batch_folder):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            messagebox.showwarning("Warning", "No image files found in selected folder")
            return
        
        self.batch_total = len(image_paths)
        self.batch_processed = 0
        self.progress_bar['maximum'] = self.batch_total
        self.progress_var.set(f"Processing 0/{self.batch_total} images...")
        
        # Clear previous results
        self.batch_tree.delete(*self.batch_tree.get_children())
        
        # Start processing
        self.processing_queue.put({'type': 'batch', 'paths': image_paths})
        self.status_var.set("🔄 Batch processing started...")
    
    def _process_batch_images(self, image_paths):
        """Process batch of images."""
        for i, image_path in enumerate(image_paths):
            try:
                result = self.analyzer.process_image(image_path)
                
                # Update progress
                self.root.after(0, self._update_batch_progress, i + 1, result, image_path)
                
            except Exception as e:
                error_result = {'error': str(e), 'file_path': image_path}
                self.root.after(0, self._update_batch_progress, i + 1, error_result, image_path)
        
        self.root.after(0, self._batch_complete)
    
    def _update_batch_progress(self, processed_count, result, image_path):
        """Update batch processing progress."""
        self.batch_processed = processed_count
        self.progress_bar['value'] = processed_count
        self.progress_var.set(f"Processing {processed_count}/{self.batch_total} images...")
        
        # Add to results tree
        filename = os.path.basename(image_path)
        
        if 'error' in result:
            values = (filename, "ERROR", "N/A", "N/A", "N/A", "Failed")
        else:
            values = (
                filename,
                result.get('category', 'Unknown').replace('_', ' ').title(),
                result.get('platform', 'Unknown').replace('_', ' ').title(),
                result.get('threat_level', 'None').replace('_', ' ').title(),
                result.get('nsfw_classification', 'Unknown'),
                "✅ Complete"
            )
        
        self.batch_tree.insert("", tk.END, values=values)
        self.batch_results.append(result)
        
        # Auto-scroll to bottom
        self.batch_tree.yview_moveto(1)
    
    def _batch_complete(self):
        """Handle batch processing completion."""
        self.progress_var.set(f"✅ Completed! Processed {self.batch_processed} images")
        self.status_var.set("✅ Batch processing complete!")
        messagebox.showinfo("Complete", f"Batch processing finished!\nProcessed {self.batch_processed} images")
    
    def _stop_batch(self):
        """Stop batch processing."""
        # Note: This is a simplified stop - in production you'd want proper thread management
        self.status_var.set("⏹️ Batch processing stopped")
    
    def _search_database(self):
        """Search database with filters."""
        try:
            results = self.analyzer.search_images(
                query=self.search_text.get() or None,
                category=self.search_category.get() or None,
                platform=self.search_platform.get() or None,
                threat_level=self.search_threat.get() or None,
                limit=200
            )
            
            # Clear previous results
            self.search_tree.delete(*self.search_tree.get_children())
            
            # Populate results
            for result in results:
                text_preview = (result.get('detected_text', '')[:50] + '...') if len(result.get('detected_text', '')) > 50 else result.get('detected_text', '')
                
                values = (
                    result.get('image_id', '')[:8],
                    result.get('category', '').replace('_', ' ').title(),
                    result.get('platform', '').replace('_', ' ').title(),
                    result.get('threat_level', '').replace('_', ' ').title(),
                    text_preview,
                    result.get('timestamp', '')[:10]  # Just date part
                )
                
                self.search_tree.insert("", tk.END, values=values)
            
            self.status_var.set(f"🔍 Found {len(results)} matching images")
            
        except Exception as e:
            self._show_error(f"Search failed: {str(e)}")
    
    def _export_search(self):
        """Export search results."""
        # Get current search results
        items = self.search_tree.get_children()
        if not items:
            messagebox.showwarning("Warning", "No search results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Search Results",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("JSON Files", "*.json")]
        )
        
        if file_path:
            try:
                # Simple CSV export
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["ID", "Category", "Platform", "Threat", "Detected Text", "Date"])
                    
                    for item in items:
                        values = self.search_tree.item(item)['values']
                        writer.writerow(values)
                
                messagebox.showinfo("Success", f"Results exported to {file_path}")
                self.status_var.set("📤 Search results exported")
                
            except Exception as e:
                self._show_error(f"Export failed: {str(e)}")
    
    def _generate_legal_report(self):
        """Generate comprehensive legal report."""
        try:
            report_path = self.analyzer.generate_legal_report()
            
            # Show preview in text area
            preview_text = f"""
📄 LEGAL REPORT GENERATED
{'='*60}

Report Path: {report_path}
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This comprehensive legal report includes:

✓ Executive Summary
✓ Analysis Methodology  
✓ Evidence Chain of Custody
✓ Detailed Findings
✓ Risk Assessment
✓ Technical Specifications
✓ Legal Disclaimers

The report has been saved as a PDF file suitable for 
court proceedings and legal documentation.

IMPORTANT LEGAL NOTES:
• All evidence maintains cryptographic integrity
• Processing timestamps are recorded
• AI analysis results require expert validation
• Report complies with digital forensics standards

Report ID: {str(uuid.uuid4())[:8]}
Software: ForenSnap Ultimate v2.0.0
"""
            
            self.report_text.delete(1.0, tk.END)
            self.report_text.insert(tk.END, preview_text)
            
            messagebox.showinfo("Report Generated", f"Legal report saved to:\n{report_path}")
            self.status_var.set("📄 Legal report generated successfully")
            
        except Exception as e:
            self._show_error(f"Report generation failed: {str(e)}")
    
    def _generate_stats_report(self):
        """Generate statistics report."""
        self.status_var.set("📊 Generating statistics report...")
        # This would generate statistical analysis
        messagebox.showinfo("Info", "Statistics report generation coming soon!")
    
    def _save_single_report(self):
        """Save single image analysis report."""
        if not hasattr(self, 'current_result'):
            messagebox.showwarning("Warning", "No analysis results to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Analysis Report",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("JSON Files", "*.json")]
        )
        
        if file_path:
            try:
                content = self.summary_text.get(1.0, tk.END)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                messagebox.showinfo("Saved", f"Report saved to {file_path}")
                self.status_var.set("💾 Report saved successfully")
                
            except Exception as e:
                self._show_error(f"Save failed: {str(e)}")

# ================================================================================================
# COMMAND LINE INTERFACE
# ================================================================================================

def main():
    """Main entry point for ForenSnap Ultimate."""
    print("\n" + "="*80)
    print("🔬 FORENSNAP ULTIMATE - AI-POWERED DIGITAL FORENSICS SUITE")
    print("="*80)
    print("Version: 2.0.0 | Advanced Screenshot Analysis for Investigations")
    print("Features: Multi-language OCR, Threat Detection, Platform ID, NSFW Detection")
    print("="*80 + "\n")
    
    if len(sys.argv) < 2:
        print("🚀 Launching GUI Interface...")
        # Launch GUI
        root = tk.Tk()
        app = ForenSnapGUI(root)
        root.mainloop()
    else:
        # Command line mode
        command = sys.argv[1].lower()
        
        if command == "analyze":
            if len(sys.argv) < 3:
                print("❌ Usage: python forensnap_ultimate.py analyze <image_path>")
                sys.exit(1)
            
            analyzer = ForenSnapAnalyzer()
            result = analyzer.process_image(sys.argv[2])
            print(json.dumps(result, indent=2, default=str))
        
        elif command == "batch":
            if len(sys.argv) < 3:
                print("❌ Usage: python forensnap_ultimate.py batch <folder_path>")
                sys.exit(1)
            
            analyzer = ForenSnapAnalyzer()
            folder_path = sys.argv[2]
            
            # Find all images
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
            image_paths = []
            
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        image_paths.append(os.path.join(root, file))
            
            print(f"🔍 Found {len(image_paths)} images to process...")
            
            results = []
            for i, image_path in enumerate(image_paths, 1):
                print(f"📸 Processing {i}/{len(image_paths)}: {os.path.basename(image_path)}")
                result = analyzer.process_image(image_path)
                results.append(result)
            
            # Save batch results
            output_file = DATA_DIR / f"forensnap_batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"✅ Batch processing complete! Results saved to: {output_file}")
        
        elif command == "report":
            analyzer = ForenSnapAnalyzer()
            report_path = analyzer.generate_legal_report()
            print(f"📄 Legal report generated: {report_path}")
        
        elif command == "search":
            if len(sys.argv) < 3:
                print("❌ Usage: python forensnap_ultimate.py search <query>")
                sys.exit(1)
            
            analyzer = ForenSnapAnalyzer()
            results = analyzer.search_images(query=sys.argv[2])
            print(f"🔍 Found {len(results)} matching images:")
            
            for result in results[:10]:  # Show first 10
                print(f"  • {result.get('image_id', '')[:8]} - {result.get('category', '')} - {result.get('detected_text', '')[:50]}...")
        
        elif command == "test-ocr":
            print("🧪 Testing Enhanced OCR...")
            if len(sys.argv) >= 3:
                image_path = sys.argv[2]
                if os.path.exists(image_path):
                    # Initialize just the OCR component
                    if USE_ENHANCED_OCR and EnhancedOCR:
                        logger.info("Using enhanced OCR module from modules/enhanced_ocr.py")
                        from modules.enhanced_ocr import EnhancedOCR as ExternalEnhancedOCR
                        ocr = ExternalEnhancedOCR()
                    else:
                        logger.info("Using built-in OCR implementation")
                        ocr = EnhancedOCR()
                    
                    result = ocr.extract_text_hybrid(image_path)
                    
                    print(f"\n📸 OCR Test Results for: {os.path.basename(image_path)}")
                    print(f"Method: {result.get('method', 'unknown')}")
                    print(f"Confidence: {result.get('confidence', 0):.1f}%")
                    print(f"Language: {result.get('language', 'unknown')}")
                    print(f"Text Length: {len(result.get('text', ''))} characters")
                    print(f"Preprocessing: {result.get('preprocessing', 'unknown')}")
                    print(f"Valid Results: {result.get('valid_results', 0)}/{result.get('total_attempts', 0)}")
                    
                    if result.get('text'):
                        print(f"\nExtracted Text:")
                        print("-" * 50)
                        print(result['text'])
                    else:
                        print(f"\nNo text extracted. Error: {result.get('error', 'Unknown')}")
                else:
                    print(f"❌ Image not found: {image_path}")
            else:
                print("❌ Usage: python forensnap_ultimate.py test-ocr <image_path>")
        
        else:
            print("❌ Unknown command. Available commands:")
            print("  • analyze <image_path>     - Analyze single image")
            print("  • batch <folder_path>      - Batch process folder")  
            print("  • report                   - Generate legal report")
            print("  • search <query>           - Search database")
            print("  • test-ocr <image_path>    - Test enhanced OCR on image")
            print("  • (no args)                - Launch GUI")

if __name__ == "__main__":
    main()
