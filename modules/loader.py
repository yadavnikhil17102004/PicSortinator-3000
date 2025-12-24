#!/usr/bin/env python3
"""
PicSortinator 3000 - Image Loader Module
========================================

Handles scanning directories for images and extracting basic metadata.

📁 This module is like a digital bloodhound - it WILL find your images!
🔍 Warning: May discover embarrassing photos you forgot you had.
"""

import os
import sqlite3
from pathlib import Path
from datetime import datetime
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ImageLoader:
    """Handles loading and scanning images from directories."""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
    
    def __init__(self):
        """Initialize the ImageLoader."""
        pass
    
    def scan_directory(self, directory_path):
        """
        Scan a directory for supported image files.
        
        Args:
            directory_path (str): Path to directory to scan
            
        Returns:
            list: List of Path objects for found images
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        image_files = []
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                # We skip img.verify() here for performance.
                # Validation happens during metadata extraction.
                image_files.append(file_path)
                    
        return sorted(image_files)
    
    def extract_metadata(self, image_path):
        """
        Extract basic metadata from an image file.
        
        Args:
            image_path (Path): Path to image file
            
        Returns:
            dict: Metadata dictionary
        """
        try:
            with Image.open(image_path) as img:
                # Basic file info
                file_stats = os.stat(image_path)
                metadata = {
                    'filename': image_path.name,
                    'path': str(image_path),
                    'size': file_stats.st_size,
                    'creation_date': datetime.fromtimestamp(file_stats.st_ctime),
                    'modified_date': datetime.fromtimestamp(file_stats.st_mtime),
                    'width': img.width,
                    'height': img.height,
                    'format': img.format,
                    'mode': img.mode
                }
                
                # EXIF data if available
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    if exif:
                        metadata['has_exif'] = True
                        # Extract some common EXIF tags
                        for tag_id, tag_name in {
                            271: 'camera_make',
                            272: 'camera_model', 
                            306: 'datetime_taken',
                            274: 'orientation'
                        }.items():
                            if tag_id in exif:
                                metadata[tag_name] = exif[tag_id]
                else:
                    metadata['has_exif'] = False
                    
                return metadata
                
        except Exception as e:
            logger.error(f"Error extracting metadata from {image_path}: {e}")
            return None
    
    def get_file_hash(self, image_path):
        """
        Calculate MD5 hash of image file for duplicate detection.
        
        Args:
            image_path (Path): Path to image file
            
        Returns:
            str: MD5 hash of file
        """
        import hashlib
        
        try:
            hash_md5 = hashlib.md5()
            with open(image_path, "rb") as f:
                # Use 64KB chunks for better performance on large files
                for chunk in iter(lambda: f.read(65536), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {image_path}: {e}")
            return None
