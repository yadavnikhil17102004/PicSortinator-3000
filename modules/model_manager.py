#!/usr/bin/env python3
"""
PicSortinator 3000 - Model Manager
=================================

Handles downloading, caching, and loading of ML models for image classification.
Keeps models offline after first download for true offline operation.
"""

import os
import requests
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import tensorflow as tf

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages ML model downloads and caching for offline operation."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model manager.
        
        Args:
            models_dir: Directory to store cached models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self._models = {}
        
        # Model configurations
        self.MODEL_CONFIGS = {
            'mobilenet_v2': {
                'url': None,  # Will use tf.keras.applications
                'filename': 'mobilenet_v2_imagenet.h5',
                'input_size': (224, 224),
                'preprocessing': 'tf.keras.applications.mobilenet_v2.preprocess_input',
                'labels_url': 'https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_labels.txt'
            },
            'imagenet_labels': {
                'url': 'https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_lables.txt',
                'filename': 'imagenet_labels.txt'
            }
        }
    
    def get_model_path(self, model_name: str) -> Path:
        """Get the local path for a model."""
        config = self.MODEL_CONFIGS.get(model_name)
        if not config:
            raise ValueError(f"Unknown model: {model_name}")
        return self.models_dir / config['filename']
    
    def is_model_cached(self, model_name: str) -> bool:
        """Check if a model is already cached locally."""
        model_path = self.get_model_path(model_name)
        return model_path.exists()
    
    def download_file(self, url: str, filepath: Path, description: str = "file") -> bool:
        """
        Download a file with progress indication.
        
        Args:
            url: URL to download from
            filepath: Local path to save to
            description: Description for progress messages
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading {description} from {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024) == 0:  # Every MB
                                logger.info(f"Downloaded {percent:.1f}% ({downloaded // (1024*1024)}MB)")
            
            logger.info(f"Successfully downloaded {description} to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {description}: {e}")
            if filepath.exists():
                filepath.unlink()  # Clean up partial download
            return False
    
    def load_imagenet_labels(self) -> Dict[int, str]:
        """
        Load ImageNet class labels.
        
        Returns:
            Dict mapping class indices to human-readable labels
        """
        labels_path = self.models_dir / "imagenet_labels.txt"
        
        # If not cached, create a basic set of common labels
        if not labels_path.exists():
            logger.info("Creating basic ImageNet labels")
            basic_labels = self._get_basic_imagenet_labels()
            with open(labels_path, 'w') as f:
                for i, label in basic_labels.items():
                    f.write(f"{label}\n")
        
        # Load labels
        labels = {}
        try:
            with open(labels_path, 'r') as f:
                for i, line in enumerate(f):
                    label = line.strip()
                    if label:
                        # Clean up label (remove technical suffixes)
                        clean_label = label.split(',')[0].strip()
                        labels[i] = clean_label
            
            logger.info(f"Loaded {len(labels)} ImageNet labels")
            return labels
            
        except Exception as e:
            logger.error(f"Failed to load ImageNet labels: {e}")
            return self._get_basic_imagenet_labels()
    
    def _get_basic_imagenet_labels(self) -> Dict[int, str]:
        """Get a basic set of common ImageNet labels for offline use."""
        return {
            0: "object", 1: "person", 2: "animal", 3: "vehicle", 4: "building",
            5: "food", 6: "furniture", 7: "plant", 8: "document", 9: "landscape",
            10: "indoor", 11: "outdoor", 12: "cat", 13: "dog", 14: "car",
            15: "house", 16: "tree", 17: "flower", 18: "book", 19: "computer",
            20: "phone", 21: "television", 22: "kitchen", 23: "bedroom", 24: "bathroom"
        }
    
    def load_mobilenet_model(self) -> tf.keras.Model:
        """
        Load MobileNetV2 model for image classification.
        
        Returns:
            Loaded TensorFlow model
        """
        if 'mobilenet_v2' in self._models:
            return self._models['mobilenet_v2']
        
        model_path = self.get_model_path('mobilenet_v2')
        
        try:
            # Try to load cached model
            if model_path.exists():
                logger.info(f"Loading cached MobileNetV2 from {model_path}")
                model = tf.keras.models.load_model(model_path)
            else:
                # Download pre-trained model
                logger.info("Loading MobileNetV2 from Keras applications")
                model = tf.keras.applications.MobileNetV2(
                    input_shape=(224, 224, 3),
                    alpha=1.0,
                    include_top=True,
                    weights='imagenet',
                    classes=1000
                )
                
                # Cache the model
                logger.info(f"Caching model to {model_path}")
                model.save(model_path)
            
            self._models['mobilenet_v2'] = model
            logger.info("MobileNetV2 loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load MobileNetV2: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available and cached models."""
        info = {
            'models_directory': str(self.models_dir),
            'available_models': list(self.MODEL_CONFIGS.keys()),
            'cached_models': []
        }
        
        for model_name in self.MODEL_CONFIGS:
            if self.is_model_cached(model_name):
                model_path = self.get_model_path(model_name)
                info['cached_models'].append({
                    'name': model_name,
                    'path': str(model_path),
                    'size_mb': round(model_path.stat().st_size / (1024 * 1024), 2)
                })
        
        return info
    
    def cleanup_models(self, keep_recent: int = 2):
        """
        Clean up old model files to save space.
        
        Args:
            keep_recent: Number of recent models to keep
        """
        logger.info(f"Cleaning up models directory, keeping {keep_recent} most recent")
        
        # This is a placeholder - implement based on your needs
        # Could sort by modification time and remove oldest
        pass
