"""
Image tagging module for identifying objects and scenes in images.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ImageTagger:
    """Tags images based on their content using pre-trained ML models."""
    
    def __init__(self):
        """Initialize the image tagger with appropriate models."""
        self.model_loaded = False
        self.device = None
        self.model = None
        self.processor = None
        
        # Try to load the tagging model
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Check if we're in a development environment
            if "PYTEST_CURRENT_TEST" in os.environ:
                logger.info("Running in test mode, skipping model loading")
                return
                
            # Load model and processor
            logger.info("Loading image tagging model...")
            model_name = "openai/clip-vit-base-patch32"
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForZeroShotImageClassification.from_pretrained(model_name)
            self.model.to(self.device)
            
            self.model_loaded = True
            logger.info("Image tagging model loaded successfully")
            
        except ImportError as e:
            logger.warning(f"Could not load tagging model: {e}")
            logger.warning("Will use fallback tagging mechanism")
        except Exception as e:
            logger.error(f"Error loading tagging model: {e}")
    
    def generate_tags(self, image_path):
        """Generate tags for an image using ML model or fallback approach."""
        try:
            # Check if model is loaded
            if self.model_loaded:
                return self._generate_tags_with_model(image_path)
            else:
                return self._generate_tags_fallback(image_path)
        except Exception as e:
            logger.error(f"Error generating tags for {image_path}: {e}")
            return ["error"]
    
    def _generate_tags_with_model(self, image_path):
        """Generate tags using the loaded ML model."""
        try:
            from PIL import Image
            
            # Define candidate labels
            candidate_labels = [
                "person", "people", "animal", "cat", "dog", "bird", 
                "landscape", "beach", "mountain", "city", "building",
                "food", "selfie", "party", "document", "screenshot",
                "meme", "funny", "sunset", "car", "vehicle", "nature",
                "indoor", "outdoor", "text", "artwork", "drawing"
            ]
            
            # Load and process image
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt", text=candidate_labels)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits_per_image[0]
                scores = torch.nn.functional.softmax(scores, dim=0)
            
            # Get top 5 tags with scores above threshold
            top_scores, top_indices = scores.topk(5)
            threshold = 0.1  # Minimum confidence threshold
            
            tags = []
            for score, idx in zip(top_scores, top_indices):
                if score > threshold:
                    tags.append(candidate_labels[idx])
            
            return tags if tags else ["unclassified"]
            
        except Exception as e:
            logger.error(f"Model-based tagging failed: {e}")
            return self._generate_tags_fallback(image_path)
    
    def _generate_tags_fallback(self, image_path):
        """Generate tags using a simple fallback approach when ML is unavailable."""
        try:
            import cv2
            import numpy as np
            from PIL import Image
            
            # Load image
            img = cv2.imread(str(image_path))
            
            if img is None:
                return ["invalid_image"]
            
            tags = []
            
            # Get basic image properties
            height, width, channels = img.shape
            
            # Check if it's likely a screenshot based on dimensions
            common_screen_resolutions = [
                (1920, 1080), (1280, 720), (1366, 768),
                (2560, 1440), (3840, 2160)
            ]
            
            if any(abs(width - w) < 5 and abs(height - h) < 5 for w, h in common_screen_resolutions):
                tags.append("screenshot")
            
            # Check for faces (very basic)
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    tags.append("person")
                    if len(faces) > 3:
                        tags.append("group")
            except:
                pass
            
            # Check color properties
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Check if image is dark
            if np.mean(v) < 50:
                tags.append("dark")
            
            # Check if image is bright/colorful
            if np.mean(s) > 100:
                tags.append("colorful")
                
            # Check if image has significant green (potentially nature)
            green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
            if np.count_nonzero(green_mask) > (width * height * 0.3):
                tags.append("nature")
                
            # Check if image has significant blue (potentially sky/water)
            blue_mask = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
            if np.count_nonzero(blue_mask) > (width * height * 0.3):
                tags.append("sky")
                
            # Add aspect ratio related tags
            aspect = width / height
            if abs(aspect - 1) < 0.1:
                tags.append("square")
            elif aspect > 2:
                tags.append("panorama")
                
            if not tags:
                tags = ["unclassified"]
                
            return tags
                
        except Exception as e:
            logger.error(f"Fallback tagging failed: {e}")
            return ["unclassified"]
