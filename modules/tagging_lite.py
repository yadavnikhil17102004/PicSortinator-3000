#!/usr/bin/env python3
"""
PicSortinator 3000 - Lightweight Tagging Module
===============================================

Computer vision without the headaches!
Uses OpenCV and basic algorithms for image analysis.
"""

import os
import cv2
import numpy as np
from PIL import Image
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class ImageTagger:
    """Lightweight image content tagger using OpenCV."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize the image tagger."""
        self.models_dir = models_dir
        self.basic_tags = {
            'colorful': self._is_colorful,
            'bright': self._is_bright,
            'dark': self._is_dark,
            'landscape': self._is_landscape,
            'portrait': self._is_portrait,
            'large': self._is_large,
            'small': self._is_small,
            'outdoor': self._likely_outdoor,
            'indoor': self._likely_indoor
        }
        
    def tag_image(self, image_path: str) -> List[str]:
        """
        Generate tags for an image using basic computer vision.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of tags for the image
        """
        try:
            # Load image with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Could not load image: {image_path}")
                return ['unknown']
            
            # Get basic image properties
            tags = []
            
            # Check each basic property
            for tag_name, check_func in self.basic_tags.items():
                if check_func(img):
                    tags.append(tag_name)
            
            # Add file-based tags
            filename = os.path.basename(image_path).lower()
            file_tags = self._get_filename_tags(filename)
            tags.extend(file_tags)
            
            # Add format tag
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in ['.jpg', '.jpeg']:
                tags.append('jpeg')
            elif file_ext == '.png':
                tags.append('png')
            elif file_ext == '.gif':
                tags.append('gif')
            
            # Ensure we always have at least one tag
            if not tags:
                tags = ['photo']
                
            return tags[:20]  # Limit to 20 tags
            
        except Exception as e:
            logger.error(f"Error tagging image {image_path}: {e}")
            return ['error']
    
    def _is_colorful(self, img: np.ndarray) -> bool:
        """Check if image is colorful."""
        if len(img.shape) != 3:
            return False
        
        # Convert to HSV and check saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        mean_saturation = np.mean(saturation)
        
        return mean_saturation > 50  # Threshold for colorfulness
    
    def _is_bright(self, img: np.ndarray) -> bool:
        """Check if image is bright."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        mean_brightness = np.mean(gray)
        return mean_brightness > 120
    
    def _is_dark(self, img: np.ndarray) -> bool:
        """Check if image is dark."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        mean_brightness = np.mean(gray)
        return mean_brightness < 80
    
    def _is_landscape(self, img: np.ndarray) -> bool:
        """Check if image is in landscape orientation."""
        height, width = img.shape[:2]
        return width > height * 1.2
    
    def _is_portrait(self, img: np.ndarray) -> bool:
        """Check if image is in portrait orientation."""
        height, width = img.shape[:2]
        return height > width * 1.2
    
    def _is_large(self, img: np.ndarray) -> bool:
        """Check if image is large."""
        height, width = img.shape[:2]
        return width > 1920 or height > 1080
    
    def _is_small(self, img: np.ndarray) -> bool:
        """Check if image is small."""
        height, width = img.shape[:2]
        return width < 640 and height < 480
    
    def _likely_outdoor(self, img: np.ndarray) -> bool:
        """Guess if image is likely outdoor (very basic heuristic)."""
        # Check for blue/green dominance (sky/grass)
        if len(img.shape) != 3:
            return False
            
        b, g, r = cv2.split(img)
        
        # Calculate color ratios
        blue_ratio = np.mean(b) / 255.0
        green_ratio = np.mean(g) / 255.0
        
        # Simple heuristic: outdoor images often have more blue/green
        return (blue_ratio + green_ratio) > 0.6
    
    def _likely_indoor(self, img: np.ndarray) -> bool:
        """Guess if image is likely indoor."""
        return not self._likely_outdoor(img)
    
    def _get_filename_tags(self, filename: str) -> List[str]:
        """Extract tags from filename."""
        tags = []
        
        # Common filename patterns
        filename_keywords = {
            'screenshot': ['screenshot', 'screen'],
            'selfie': ['selfie', 'self'],
            'photo': ['photo', 'pic', 'img'],
            'document': ['doc', 'document', 'scan'],
            'meme': ['meme', 'funny'],
            'wallpaper': ['wallpaper', 'background'],
            'profile': ['profile', 'avatar'],
            'thumbnail': ['thumb', 'thumbnail'],
            'cover': ['cover', 'banner'],
            'logo': ['logo', 'icon']
        }
        
        for tag, keywords in filename_keywords.items():
            if any(keyword in filename for keyword in keywords):
                tags.append(tag)
        
        # Check for date patterns
        import re
        if re.search(r'\d{4}[-_]\d{2}[-_]\d{2}', filename):
            tags.append('dated')
        
        return tags

    def get_confidence_score(self, image_path: str) -> float:
        """
        Get confidence score for image processing.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        try:
            # For our basic tagger, confidence is based on file accessibility
            if os.path.exists(image_path) and os.path.getsize(image_path) > 1024:
                return 0.8  # Good confidence for basic analysis
            return 0.3
        except:
            return 0.1

# For backward compatibility
def load_image_model():
    """Load the lightweight image tagging model."""
    return ImageTagger()

def classify_image(image_path: str) -> Tuple[List[str], float]:
    """
    Classify image and return tags with confidence.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Tuple of (tags_list, confidence_score)
    """
    tagger = ImageTagger()
    tags = tagger.tag_image(image_path)
    confidence = tagger.get_confidence_score(image_path)
    return tags, confidence
