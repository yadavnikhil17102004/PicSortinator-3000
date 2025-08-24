#!/usr/bin/env python3
"""
ForenSnap Ultimate Configuration
===============================

Configuration settings for ForenSnap Ultimate application.
"""

import os
import logging
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
RESOURCES_DIR = BASE_DIR / "resources"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
RESOURCES_DIR.mkdir(exist_ok=True)

# Database settings
DATABASE_URL = f"sqlite:///{DATA_DIR}/forensnap_ultimate.db"

# OCR Settings
OCR_SETTINGS = {
    "confidence_threshold": 0.1,
    "max_image_size": 4096,  # Maximum image dimension
    "supported_formats": [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"],
    "tesseract_config": "--oem 3 --psm 6",
    "easyocr_languages": ["en"],
    "preprocessing_methods": ["simple", "enhanced"]
}

# BLIP Settings
BLIP_SETTINGS = {
    "model_preference": ["Salesforce/blip-image-captioning-large", "Salesforce/blip-image-captioning-base"],
    "max_length": 75,
    "num_beams": 5,
    "temperature": 0.8
}

# CLIP Settings  
CLIP_SETTINGS = {
    "model": "ViT-B/32",
    "nsfw_threshold": 0.5,
    "batch_size": 1
}

# Threat Detection Settings
THREAT_SETTINGS = {
    "confidence_threshold": 0.2,
    "escalation_threshold": 0.6,
    "urgency_threshold": 0.4
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'detailed',
            'class': 'logging.FileHandler',
            'filename': str(LOGS_DIR / 'forensnap.log'),
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# GUI Settings
GUI_SETTINGS = {
    "window_size": "1400x900",
    "theme": "clam",
    "colors": {
        "bg_primary": "#2b2b2b",
        "bg_secondary": "#404040", 
        "fg_primary": "white",
        "accent": "#4CAF50"
    }
}

# Performance Settings
PERFORMANCE_SETTINGS = {
    "use_gpu": True,  # Use GPU if available
    "max_workers": 4,  # For batch processing
    "cache_models": True,
    "memory_limit_gb": 8
}
