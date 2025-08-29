#!/usr/bin/env python3
"""
PicSortinator 3000 - ML Tagging Module
=====================================

Military-grade image classification using MobileNetV2.
Because your photos deserve better than guesswork.

🎯 Fun Fact: This AI has seen more images than a social media influencer!
🤖 Warning: May develop strong opinions about your photography skills.
🏷️  Easter Egg: The confidence threshold is like a bouncer at an exclusive club - very picky!
"""

import os
import cv2
import numpy as np
from PIL import Image
import logging
from typing import List, Dict, Tuple, Optional
import tensorflow as tf
from .model_manager import ModelManager

logger = logging.getLogger(__name__)

class ImageTagger:
    """ML-powered image content tagger with personality."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize the image tagger with ML models."""
        self.model_manager = ModelManager(models_dir)
        self.model = None
        self.labels = None
        self.confidence_threshold = 0.2  # Reduced from 0.3 to get more tags
        self.input_size = (224, 224)  # MobileNetV2 input size
        
        # Enhanced useful categories with more specific terms
        self.useful_categories = {
            # People
            'person', 'people', 'man', 'woman', 'child', 'baby', 'face', 'portrait', 'selfie',
            'human', 'boy', 'girl', 'adult', 'family',
            # Clothing & Fashion
            'shirt', 'dress', 'jacket', 'coat', 'sweater', 'sweatshirt', 'hoodie', 'jeans',
            'pants', 'skirt', 'shoes', 'sneakers', 'boots', 'hat', 'cap', 'glasses',
            'fashion', 'clothing', 'outfit', 'style',
            # Vehicles
            'car', 'automobile', 'truck', 'bus', 'motorcycle', 'bicycle', 'bike', 'vehicle',
            'taxi', 'van', 'suv', 'sedan', 'convertible',
            # Animals
            'dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'pig', 'chicken', 'duck',
            'pet', 'animal', 'wildlife', 'mammal',
            # Food & Drinks
            'food', 'pizza', 'cake', 'sandwich', 'burger', 'bread', 'fruit', 'vegetable',
            'apple', 'banana', 'orange', 'salad', 'soup', 'meal', 'dinner', 'lunch',
            'breakfast', 'coffee', 'tea', 'drink', 'beer', 'wine', 'juice',
            # Architecture & Places
            'building', 'house', 'home', 'church', 'castle', 'bridge', 'tower',
            'architecture', 'city', 'town', 'street', 'road', 'highway',
            # Nature & Outdoors
            'tree', 'flower', 'plant', 'mountain', 'beach', 'ocean', 'sea', 'lake',
            'sky', 'sunset', 'sunrise', 'cloud', 'nature', 'landscape', 'garden',
            'forest', 'park', 'outdoor', 'scenery',
            # Technology
            'phone', 'computer', 'laptop', 'keyboard', 'screen', 'tv', 'television',
            'camera', 'technology', 'electronic', 'device',
            # Objects & Items
            'book', 'newspaper', 'document', 'paper', 'letter', 'text', 'magazine',
            'furniture', 'chair', 'table', 'bed', 'sofa', 'desk',
            'kitchen', 'bathroom', 'bedroom', 'living_room', 'office', 'room',
            # Activities & Sports
            'sports', 'game', 'ball', 'tennis', 'football', 'basketball', 'soccer',
            'baseball', 'swimming', 'running', 'exercise', 'gym',
            # Entertainment
            'music', 'guitar', 'piano', 'instrument', 'concert', 'performance',
            'art', 'painting', 'drawing', 'sculpture', 'museum',
            # Events & Occasions
            'party', 'wedding', 'birthday', 'celebration', 'event', 'holiday',
            'christmas', 'halloween', 'vacation', 'travel', 'trip'
        }
        
        # Sarcastic responses for different scenarios
        self.funny_responses = {
            'loading_model': [
                "🤖 Teaching AI to judge your photo collection...",
                "📸 Initializing the 'Is this actually art?' detector...",
                "🧠 Loading models that understand your aesthetic choices better than you do..."
            ],
            'processing': [
                "🔍 Analyzing... yep, that's definitely a thing.",
                "🎯 Found: objects, questionable composition, infinite possibilities.",
                "📊 Confidence level: Higher than your selfie game."
            ],
            'high_confidence': [
                "💯 I'm absolutely certain this is what I think it is.",
                "🎯 My neural networks are vibing with this image.",
                "✅ This classification is more confident than your dating profile."
            ],
            'low_confidence': [
                "🤔 I'm as confused about this image as you are about your life choices.",
                "❓ Either this is abstract art, or I need new glasses.",
                "🤷 My best guess is... *gestures vaguely*"
            ]
        }
        
        logger.info("🚀 ImageTagger initialized with ML superpowers")
    
    def _load_model(self):
        """Load the MobileNetV2 model and ImageNet labels."""
        if self.model is not None:
            return
        
        logger.info("📦 Loading MobileNetV2 model...")
        
        try:
            self.model = self.model_manager.load_mobilenet_model()
            self.labels = self.model_manager.load_imagenet_labels()
            logger.info(f"✅ Model loaded with {len(self.labels)} classes")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def generate_tags(self, image_path: str, max_tags: int = 5) -> List[str]:
        """
        Generate ML-powered content tags for an image.
        
        Args:
            image_path: Path to the image file
            max_tags: Maximum number of tags to return
            
        Returns:
            List of content tags with confidence scores
        """
        self._load_model()
        
        try:
            # Load and preprocess image
            image = self._load_and_preprocess_image(image_path)
            if image is None:
                return ["unreadable_image"]
            
            # Run ML inference
            predictions = self.model.predict(image, verbose=0)
            
            # Get top predictions with higher confidence
            top_indices = np.argsort(predictions[0])[::-1][:20]  # Get more candidates
            
            tags = []
            for idx in top_indices:
                confidence = predictions[0][idx]
                if confidence >= self.confidence_threshold:
                    label = self.labels.get(idx, f"class_{idx}")
                    # Clean up the label
                    clean_label = self._clean_label(label)
                    
                    # Only add if it's a useful category and not already in tags
                    if (clean_label and 
                        self._is_useful_tag(clean_label) and 
                        clean_label not in tags):
                        tags.append(clean_label)
                        if len(tags) >= max_tags:
                            break
            
            # If we don't have enough meaningful tags, try with lower confidence
            if len(tags) < 2:
                for idx in top_indices:
                    confidence = predictions[0][idx]
                    if confidence >= 0.1:  # Very low threshold for backup tags
                        label = self.labels.get(idx, f"class_{idx}")
                        clean_label = self._clean_label(label)
                        
                        if (clean_label and 
                            self._is_useful_tag(clean_label) and 
                            clean_label not in tags):
                            tags.append(clean_label)
                            if len(tags) >= max_tags:
                                break
            
            # Add basic image properties
            basic_tags = self._analyze_image_properties(image_path)
            tags.extend([tag for tag in basic_tags if tag not in tags])
            
            return tags[:max_tags] if tags else ["unclear_content"]
            
        except Exception as e:
            logger.error(f"💥 Error generating tags for {image_path}: {e}")
            return ["processing_error"]
    
    def _is_useful_tag(self, tag: str) -> bool:
        """
        Check if a tag is useful and relevant for photo organization.
        
        Args:
            tag: Cleaned tag to evaluate
            
        Returns:
            True if tag is useful, False otherwise
        """
        if not tag or len(tag) < 2:
            return False
        
        # Check if tag contains useful keywords
        tag_lower = tag.lower()
        
        # Filter out generic/unuseful tags
        useless_tags = {
            'artifact', 'noise', 'blur', 'distortion', 'pattern', 'texture',
            'background', 'foreground', 'object', 'thing', 'item', 'stuff',
            'color', 'shape', 'form', 'structure', 'material', 'substance',
            'n02', 'n03', 'n04', 'n01', 'class'  # ImageNet technical suffixes
        }
        
        # Skip if tag contains useless words
        for useless in useless_tags:
            if useless in tag_lower:
                return False
        
        # Check if tag contains any useful category words
        useful_keywords = [
            'person', 'people', 'man', 'woman', 'child', 'baby', 'face', 'portrait', 'selfie',
            'car', 'automobile', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle',
            'dog', 'cat', 'bird', 'horse', 'cow', 'animal', 'pet',
            'food', 'pizza', 'cake', 'sandwich', 'fruit', 'vegetable', 'meal',
            'building', 'house', 'church', 'castle', 'bridge', 'architecture',
            'tree', 'flower', 'plant', 'mountain', 'beach', 'ocean', 'sky', 'sunset', 'nature', 'landscape',
            'phone', 'computer', 'laptop', 'keyboard', 'screen', 'tv', 'technology',
            'book', 'newspaper', 'document', 'paper', 'letter', 'text',
            'kitchen', 'bathroom', 'bedroom', 'living_room', 'office', 'indoor',
            'car_interior', 'street', 'road', 'highway', 'parking_lot', 'outdoor',
            'sports', 'game', 'ball', 'tennis', 'football', 'basketball',
            'music', 'guitar', 'piano', 'instrument',
            'art', 'painting', 'drawing', 'sculpture',
            'fashion', 'clothing', 'shirt', 'dress', 'shoes',
            'travel', 'vacation', 'holiday', 'trip'
        ]
        
        for keyword in useful_keywords:
            if keyword in tag_lower:
                return True
        
        # Allow specific object names (longer than 4 chars and likely real objects)
        if len(tag) > 4 and tag_lower.replace(' ', '').replace('-', '').isalnum():
            # Check if it looks like a real word/phrase
            words = tag_lower.split()
            if all(len(word) >= 3 for word in words):
                return True
        
        return False
    
    def _load_and_preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess an image for ML inference.
        
        Args:
            image_path: Path to image
            
        Returns:
            Preprocessed image array ready for model input
        """
        try:
            # Handle Unicode file paths by using raw byte reading
            try:
                # Method 1: Direct PIL opening (works for most cases)
                with Image.open(image_path) as img:
                    return self._process_image(img)
            except (UnicodeDecodeError, OSError) as e:
                logger.warning(f"Unicode/path issue with {image_path}: {e}")
                
                # Method 2: Read as bytes first, then open
                try:
                    with open(image_path, 'rb') as f:
                        img = Image.open(f)
                        img.load()  # Force load the image data
                        return self._process_image(img)
                except Exception as e2:
                    logger.warning(f"Byte reading failed for {image_path}: {e2}")
                    
                    # Method 3: Use OpenCV then convert to PIL
                    try:
                        # OpenCV can often handle problematic paths better
                        cv_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if cv_image is not None:
                            # Convert BGR to RGB
                            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(rgb_image)
                            return self._process_image(img)
                    except Exception as e3:
                        logger.error(f"All image loading methods failed for {image_path}: {e3}")
                        return None
                        
        except Exception as e:
            logger.error(f"💥 Failed to load/preprocess image {image_path}: {e}")
            return None
    
    def _process_image(self, img: Image.Image) -> np.ndarray:
        """Process a PIL image into model input format."""
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize(self.input_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(img, dtype=np.float32)
        
        # Expand dimensions for batch processing
        image_array = np.expand_dims(image_array, axis=0)
        
        # Preprocess for MobileNetV2
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
        
        return image_array
    
    def _clean_label(self, label: str) -> str:
        """
        Clean up ImageNet labels to be more user-friendly.
        
        Args:
            label: Raw ImageNet label
            
        Returns:
            Cleaned, user-friendly label
        """
        if not label:
            return ""
        
        # Remove technical suffixes and clean up
        clean = label.split(',')[0].strip().lower()
        
        # Remove common ImageNet artifacts
        artifacts_to_remove = ['n02', 'n03', 'n04', 'n01']  # WordNet synset IDs
        for artifact in artifacts_to_remove:
            if clean.startswith(artifact):
                return ""
        
        # Replace underscores and hyphens with spaces
        clean = clean.replace('_', ' ').replace('-', ' ')
        
        # Remove extra whitespace
        clean = ' '.join(clean.split())
        
        return clean
    
    def _analyze_image_properties(self, image_path: str) -> List[str]:
        """
        Analyze basic image properties for additional tags.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of property-based tags
        """
        tags = []
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                aspect_ratio = width / height
                
                # Orientation tags
                if aspect_ratio > 1.3:
                    tags.append("landscape_orientation")
                elif aspect_ratio < 0.8:
                    tags.append("portrait_orientation")
                else:
                    tags.append("square_format")
                
                # Resolution tags
                total_pixels = width * height
                if total_pixels > 8000000:  # > 8MP
                    tags.append("high_resolution")
                elif total_pixels < 500000:  # < 0.5MP
                    tags.append("low_resolution")
                
                # File format
                format_tag = f"{img.format.lower()}_format" if img.format else "unknown_format"
                tags.append(format_tag)
        
        except Exception as e:
            logger.error(f"Failed to analyze properties for {image_path}: {e}")
        
        return tags
    
    def get_detailed_analysis(self, image_path: str) -> Dict[str, any]:
        """
        Get detailed analysis including confidence scores and metadata.
        
        Args:
            image_path: Path to image
            
        Returns:
            Detailed analysis results
        """
        self._load_model()
        
        analysis = {
            'image_path': image_path,
            'tags': [],
            'confidence_scores': {},
            'top_predictions': [],
            'image_properties': {},
            'processing_notes': []
        }
        
        try:
            # Load and preprocess image
            image = self._load_and_preprocess_image(image_path)
            if image is None:
                analysis['processing_notes'].append("Failed to load image")
                return analysis
            
            # Run ML inference
            predictions = self.model.predict(image, verbose=0)
            
            # Get top 10 predictions with confidence
            top_indices = np.argsort(predictions[0])[::-1][:10]
            
            for idx in top_indices:
                confidence = float(predictions[0][idx])
                label = self.labels.get(idx, f"class_{idx}")
                clean_label = self._clean_label(label)
                
                analysis['top_predictions'].append({
                    'label': clean_label or label,
                    'confidence': confidence,
                    'class_id': int(idx)
                })
                
                if confidence >= self.confidence_threshold and clean_label:
                    analysis['tags'].append(clean_label)
                    analysis['confidence_scores'][clean_label] = confidence
            
            # Add image properties
            with Image.open(image_path) as img:
                analysis['image_properties'] = {
                    'width': img.width,
                    'height': img.height,
                    'format': img.format,
                    'mode': img.mode,
                    'aspect_ratio': round(img.width / img.height, 2)
                }
            
            # Add a sarcastic comment based on confidence
            max_confidence = max([p['confidence'] for p in analysis['top_predictions']])
            if max_confidence > 0.8:
                analysis['ai_comment'] = np.random.choice(self.funny_responses['high_confidence'])
            else:
                analysis['ai_comment'] = np.random.choice(self.funny_responses['low_confidence'])
                
        except Exception as e:
            analysis['processing_notes'].append(f"Error during analysis: {e}")
            logger.error(f"💥 Detailed analysis failed for {image_path}: {e}")
        
        return analysis
    
    def tag_image(self, image_path: str) -> List[str]:
        """
        Generate tags for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of content tags
        """
        logger.info(f"Tagging image: {image_path}")
        
        try:
            # Use the existing generate_tags method
            tags = self.generate_tags(image_path)
            if tags:
                logger.info(f"Generated {len(tags)} tags for {image_path}")
                return tags
            else:
                logger.warning(f"No tags generated for {image_path}")
                return []
        except Exception as e:
            logger.error(f"💥 Error tagging image {image_path}: {e}")
            return ["error"]
    
    def batch_tag_images(self, image_paths: List[str], progress_callback=None) -> Dict[str, List[str]]:
        """
        Tag multiple images efficiently.
        
        Args:
            image_paths: List of image paths
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping image paths to tags
        """
        results = {}
        total = len(image_paths)
        
        logger.info(f"🚀 Starting batch tagging of {total} images...")
        print(np.random.choice(self.funny_responses['loading_model']))
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                tags = self.generate_tags(image_path)
                results[image_path] = tags
                
                if progress_callback:
                    progress_callback(i, total, image_path)
                elif i % 10 == 0:  # Progress update every 10 images
                    progress = (i / total) * 100
                    print(f"📊 Progress: {progress:.1f}% ({i}/{total}) - {os.path.basename(image_path)}")
                    
            except Exception as e:
                logger.error(f"💥 Failed to tag {image_path}: {e}")
                results[image_path] = ["error"]
        
        logger.info(f"✅ Batch tagging completed! Tagged {len(results)} images.")
        print("🎉 Tagging complete! Your photos have been judged and categorized.")
        return results
