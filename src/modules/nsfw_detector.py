"""
NSFW/Adult Content Detection Module for ForenSnap
Implements local detection without relying on cloud services
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import clip
from typing import Dict, Any, Tuple, Optional, List
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NSFWDetector:
    """NSFW content detection using multiple approaches for high accuracy."""
    
    def __init__(self):
        """Initialize NSFW detector with multiple models."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_loaded = False
        
        # CLIP model for semantic understanding
        self.clip_model = None
        self.clip_preprocess = None
        
        # Nudity detection keywords for CLIP
        self.nsfw_text_prompts = [
            "explicit sexual content",
            "nudity",
            "adult content",
            "pornographic image",
            "sexual activity",
            "naked person",
            "intimate body parts",
            "sexual organs",
            "erotic content"
        ]
        
        self.safe_text_prompts = [
            "safe for work content",
            "family friendly image",
            "appropriate content",
            "professional image",
            "educational content",
            "news content",
            "social media post",
            "normal photograph"
        ]
        
        # Skin tone detection parameters
        self.skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all detection models."""
        try:
            # Load CLIP model
            logger.info("Loading CLIP model for semantic content analysis...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.models_loaded = True
            logger.info("NSFW detection models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load NSFW detection models: {e}")
            self.models_loaded = False
    
    def detect_skin_regions(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect skin regions in the image using color-based segmentation.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            Dict[str, Any]: Skin detection results
        """
        # Convert BGR to HSV for better skin detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for skin tones
        skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
        
        # Calculate skin percentage
        total_pixels = image.shape[0] * image.shape[1]
        skin_pixels = np.sum(skin_mask > 0)
        skin_percentage = (skin_pixels / total_pixels) * 100
        
        # Find contours to analyze skin regions
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze largest skin regions
        skin_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                region_percentage = (area / total_pixels) * 100
                aspect_ratio = w / h if h > 0 else 0
                
                skin_regions.append({
                    'area': area,
                    'percentage': region_percentage,
                    'bbox': (x, y, w, h),
                    'aspect_ratio': aspect_ratio
                })
        
        # Sort regions by area (largest first)
        skin_regions.sort(key=lambda x: x['area'], reverse=True)
        
        return {
            'skin_percentage': skin_percentage,
            'skin_regions': skin_regions[:5],  # Top 5 regions
            'total_regions': len(skin_regions),
            'mask': skin_mask
        }
    
    def analyze_image_composition(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image composition for NSFW indicators.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Dict[str, Any]: Composition analysis results
        """
        height, width = image.shape[:2]
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Blur detection (NSFW images are often deliberately blurred)
        blur_measure = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Color histogram analysis
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
        
        # Calculate color diversity (lower diversity might indicate skin-toned images)
        color_diversity = np.std(hist_b) + np.std(hist_g) + np.std(hist_r)
        
        # Brightness analysis
        brightness = np.mean(gray)
        
        return {
            'edge_density': edge_density,
            'blur_measure': blur_measure,
            'color_diversity': color_diversity,
            'brightness': brightness,
            'resolution': (width, height),
            'aspect_ratio': width / height if height > 0 else 1
        }
    
    def clip_semantic_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """
        Use CLIP model for semantic content analysis.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            Dict[str, Any]: CLIP analysis results
        """
        if not self.models_loaded or self.clip_model is None:
            return {
                'nsfw_probability': 0.0,
                'safe_probability': 0.0,
                'error': 'CLIP model not loaded'
            }
        
        try:
            # Preprocess image for CLIP
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize text prompts
            nsfw_tokens = clip.tokenize(self.nsfw_text_prompts).to(self.device)
            safe_tokens = clip.tokenize(self.safe_text_prompts).to(self.device)
            
            with torch.no_grad():
                # Get image features
                image_features = self.clip_model.encode_image(image_tensor)
                
                # Get text features
                nsfw_features = self.clip_model.encode_text(nsfw_tokens)
                safe_features = self.clip_model.encode_text(safe_tokens)
                
                # Calculate similarities
                nsfw_similarities = torch.cosine_similarity(image_features, nsfw_features)
                safe_similarities = torch.cosine_similarity(image_features, safe_features)
                
                # Get probabilities
                nsfw_scores = torch.softmax(nsfw_similarities, dim=0)
                safe_scores = torch.softmax(safe_similarities, dim=0)
                
                # Aggregate scores
                nsfw_probability = float(torch.mean(nsfw_scores))
                safe_probability = float(torch.mean(safe_scores))
                
                # Detailed analysis per prompt
                nsfw_details = {}
                for i, prompt in enumerate(self.nsfw_text_prompts):
                    nsfw_details[prompt] = float(nsfw_scores[i])
                
                safe_details = {}
                for i, prompt in enumerate(self.safe_text_prompts):
                    safe_details[prompt] = float(safe_scores[i])
            
            return {
                'nsfw_probability': nsfw_probability,
                'safe_probability': safe_probability,
                'nsfw_details': nsfw_details,
                'safe_details': safe_details,
                'confidence': abs(nsfw_probability - safe_probability)
            }
            
        except Exception as e:
            logger.error(f"CLIP analysis failed: {e}")
            return {
                'nsfw_probability': 0.0,
                'safe_probability': 0.0,
                'error': str(e)
            }
    
    def calculate_nsfw_score(self, 
                           skin_analysis: Dict[str, Any], 
                           composition_analysis: Dict[str, Any],
                           clip_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate final NSFW score based on all analyses.
        
        Args:
            skin_analysis: Skin detection results
            composition_analysis: Image composition analysis
            clip_analysis: CLIP semantic analysis
            
        Returns:
            Dict[str, Any]: Final NSFW assessment
        """
        score_components = {
            'skin_score': 0.0,
            'composition_score': 0.0,
            'semantic_score': 0.0,
            'final_score': 0.0
        }
        
        # Skin-based scoring (0-1 scale)
        skin_percentage = skin_analysis.get('skin_percentage', 0)
        if skin_percentage > 40:
            score_components['skin_score'] = min(1.0, (skin_percentage - 40) / 40)
        
        # Large skin regions are more suspicious
        large_regions = [r for r in skin_analysis.get('skin_regions', []) if r['percentage'] > 10]
        if large_regions:
            score_components['skin_score'] = min(1.0, score_components['skin_score'] + 0.3)
        
        # Composition-based scoring
        edge_density = composition_analysis.get('edge_density', 0)
        blur_measure = composition_analysis.get('blur_measure', 0)
        color_diversity = composition_analysis.get('color_diversity', 0)
        
        # Low edge density + low color diversity might indicate skin-heavy images
        if edge_density < 0.1 and color_diversity < 50000:
            score_components['composition_score'] += 0.3
        
        # Very blurry images might be deliberately obscured
        if blur_measure < 100:
            score_components['composition_score'] += 0.2
        
        score_components['composition_score'] = min(1.0, score_components['composition_score'])
        
        # Semantic scoring from CLIP
        nsfw_prob = clip_analysis.get('nsfw_probability', 0)
        safe_prob = clip_analysis.get('safe_probability', 0)
        confidence = clip_analysis.get('confidence', 0)
        
        # Use CLIP probability but adjust based on confidence
        if confidence > 0.1:  # High confidence in classification
            score_components['semantic_score'] = nsfw_prob
        else:
            # Low confidence, reduce the impact
            score_components['semantic_score'] = nsfw_prob * 0.5
        
        # Calculate weighted final score
        weights = {
            'skin': 0.4,        # Skin detection weight
            'composition': 0.2,  # Composition analysis weight
            'semantic': 0.4      # Semantic analysis weight
        }
        
        final_score = (
            score_components['skin_score'] * weights['skin'] +
            score_components['composition_score'] * weights['composition'] +
            score_components['semantic_score'] * weights['semantic']
        )
        
        score_components['final_score'] = final_score
        
        # Determine classification
        if final_score >= 0.7:
            classification = "VERY_LIKELY"
        elif final_score >= 0.5:
            classification = "LIKELY"
        elif final_score >= 0.3:
            classification = "POSSIBLE"
        else:
            classification = "UNLIKELY"
        
        return {
            'score_components': score_components,
            'final_score': final_score,
            'classification': classification,
            'confidence': max(confidence, 0.1),  # Minimum confidence
            'analysis_method': 'hybrid_local'
        }
    
    def detect_nsfw_content(self, image_path: str) -> Dict[str, Any]:
        """
        Main function to detect NSFW content in an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Dict[str, Any]: Complete NSFW analysis results
        """
        try:
            # Load image
            cv_image = cv2.imread(image_path)
            pil_image = Image.open(image_path).convert('RGB')
            
            if cv_image is None:
                return {
                    'error': 'Could not load image',
                    'classification': 'UNKNOWN',
                    'final_score': 0.0
                }
            
            # Perform all analyses
            skin_analysis = self.detect_skin_regions(cv_image)
            composition_analysis = self.analyze_image_composition(cv_image)
            clip_analysis = self.clip_semantic_analysis(pil_image)
            
            # Calculate final score
            nsfw_assessment = self.calculate_nsfw_score(
                skin_analysis, composition_analysis, clip_analysis
            )
            
            # Compile comprehensive results
            results = {
                'image_path': image_path,
                'classification': nsfw_assessment['classification'],
                'final_score': nsfw_assessment['final_score'],
                'confidence': nsfw_assessment['confidence'],
                'analysis_details': {
                    'skin_analysis': {
                        'skin_percentage': skin_analysis['skin_percentage'],
                        'total_regions': skin_analysis['total_regions'],
                        'largest_region_percentage': skin_analysis['skin_regions'][0]['percentage'] if skin_analysis['skin_regions'] else 0
                    },
                    'composition_analysis': composition_analysis,
                    'semantic_analysis': {
                        'nsfw_probability': clip_analysis.get('nsfw_probability', 0),
                        'safe_probability': clip_analysis.get('safe_probability', 0),
                        'method': 'CLIP'
                    }
                },
                'score_breakdown': nsfw_assessment['score_components'],
                'warnings': []
            }
            
            # Add warnings based on analysis
            if nsfw_assessment['final_score'] > 0.5:
                results['warnings'].append("High probability of adult content detected")
            
            if skin_analysis['skin_percentage'] > 50:
                results['warnings'].append("High skin exposure detected")
            
            if not self.models_loaded:
                results['warnings'].append("Some analysis models unavailable - limited accuracy")
            
            return results
            
        except Exception as e:
            logger.error(f"NSFW detection failed for {image_path}: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'classification': 'UNKNOWN',
                'final_score': 0.0,
                'confidence': 0.0
            }
    
    def batch_detect(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Detect NSFW content in multiple images.
        
        Args:
            image_paths (List[str]): List of image file paths
            
        Returns:
            List[Dict[str, Any]]: Results for all images
        """
        results = []
        total_images = len(image_paths)
        
        logger.info(f"Starting batch NSFW detection for {total_images} images")
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing image {i}/{total_images}: {os.path.basename(image_path)}")
            result = self.detect_nsfw_content(image_path)
            results.append(result)
        
        # Calculate batch statistics
        successful_results = [r for r in results if 'error' not in r]
        if successful_results:
            avg_score = sum(r['final_score'] for r in successful_results) / len(successful_results)
            high_risk_count = sum(1 for r in successful_results if r['final_score'] > 0.7)
            
            logger.info(f"Batch processing complete:")
            logger.info(f"  Average NSFW score: {avg_score:.3f}")
            logger.info(f"  High-risk images: {high_risk_count}/{len(successful_results)}")
        
        return results

# Example usage
if __name__ == "__main__":
    detector = NSFWDetector()
    
    # Test with a sample image (replace with actual image path)
    # result = detector.detect_nsfw_content("test_image.jpg")
    # print(json.dumps(result, indent=2, default=str))
