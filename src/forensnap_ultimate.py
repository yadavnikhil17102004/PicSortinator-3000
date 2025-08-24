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
Version: 2.1.0 - Enhanced Edition
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/forensnap.log'),
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

# Now import the enhanced OCR module
try:
    from modules.enhanced_ocr import EnhancedOCR
    print("✅ Enhanced OCR module loaded successfully")
except ImportError as e:
    print(f"⚠️  Enhanced OCR module not found: {e}")
    # Fall back to using the built-in version
    EnhancedOCR = None

# Import all other required modules
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

# Import rest of the application (database models, analyzers, etc.)
from forensnap_ultimate import *

# If we couldn't import the enhanced OCR, use the built-in version
if EnhancedOCR is None:
    # Use the enhanced version from the main file
    pass

def main():
    """Main entry point for ForenSnap Ultimate."""
    print("\n" + "="*80)
    print("🔬 FORENSNAP ULTIMATE - ENHANCED AI-POWERED DIGITAL FORENSICS SUITE")
    print("="*80)
    print("Version: 2.1.0 | Advanced Screenshot Analysis for Investigations")
    print("Features: Enhanced Multi-language OCR, Threat Detection, Platform ID, NSFW Detection")
    print("="*80 + "\n")
    
    if len(sys.argv) < 2:
        print("🚀 Launching Enhanced GUI Interface...")
        # Launch GUI
        root = tk.Tk()
        app = ForenSnapGUI(root)
        root.mainloop()
    else:
        # Command line mode
        command = sys.argv[1].lower()
        
        if command == "analyze":
            if len(sys.argv) < 3:
                print("❌ Usage: python run_forensnap_ultimate.py analyze <image_path>")
                sys.exit(1)
            
            analyzer = ForenSnapAnalyzer()
            result = analyzer.process_image(sys.argv[2])
            print(json.dumps(result, indent=2, default=str))
        
        elif command == "batch":
            if len(sys.argv) < 3:
                print("❌ Usage: python run_forensnap_ultimate.py batch <folder_path>")
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
            output_file = f"data/forensnap_batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"✅ Batch processing complete! Results saved to: {output_file}")
        
        elif command == "report":
            analyzer = ForenSnapAnalyzer()
            report_path = analyzer.generate_legal_report()
            print(f"📄 Legal report generated: {report_path}")
        
        elif command == "search":
            if len(sys.argv) < 3:
                print("❌ Usage: python run_forensnap_ultimate.py search <query>")
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
                print("❌ Usage: python run_forensnap_ultimate.py test-ocr <image_path>")
        
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
