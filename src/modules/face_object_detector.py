"""
Advanced Face and Object Detection Module for ForenSnap Ultimate
Specialized for digital forensics investigations
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage
from typing import List, Dict, Any, Tuple, Optional
import logging
import os
import json
from pathlib import Path
import dlib
import face_recognition

# Configure logging
logger = logging.getLogger(__name__)

class FaceObjectDetector:
    """Advanced face and object detection for forensic investigations."""
    
    def __init__(self):
        """Initialize face and object detection models."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Face detection models
        self.face_cascade = None
        self.face_encodings_db = {}
        self.dlib_detector = None
        self.shape_predictor = None
        
        # Object detection categories for investigations
        self.evidence_objects = {
            'weapons': [
                'knife', 'gun', 'pistol', 'rifle', 'sword', 'blade',
                'weapon', 'firearm', 'ammunition', 'bullet'
            ],
            'drugs': [
                'syringe', 'needle', 'pills', 'powder', 'cannabis',
                'marijuana', 'cocaine', 'drug', 'substance'
            ],
            'documents': [
                'passport', 'license', 'id', 'card', 'certificate',
                'document', 'paper', 'form', 'contract'
            ],
            'technology': [
                'phone', 'computer', 'laptop', 'tablet', 'camera',
                'hard drive', 'usb', 'memory', 'device'
            ],
            'currency': [
                'money', 'cash', 'bills', 'coins', 'credit card',
                'bank card', 'currency', 'payment'
            ]
        }
        
        # Initialize detection models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all detection models."""
        try:
            # OpenCV face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("OpenCV face cascade loaded successfully")
            
            # Try to initialize dlib face detection
            try:
                self.dlib_detector = dlib.get_frontal_face_detector()
                
                # Try to load shape predictor (if available)
                predictor_path = "shape_predictor_68_face_landmarks.dat"
                if os.path.exists(predictor_path):
                    self.shape_predictor = dlib.shape_predictor(predictor_path)
                    logger.info("Dlib face detection with landmarks loaded")
                else:
                    logger.info("Dlib face detection loaded (no landmarks)")
                    
            except Exception as e:
                logger.warning(f"Dlib face detection not available: {e}")
            
            logger.info("Face and object detection models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize detection models: {e}")
    
    def detect_faces(self, image_path: str) -> Dict[str, Any]:
        """
        Detect faces in image with multiple methods.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            Dict[str, Any]: Face detection results
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'faces': [], 'error': 'Could not load image'}
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            faces_detected = []
            
            # Method 1: OpenCV Haar Cascades
            if self.face_cascade is not None:
                opencv_faces = self.face_cascade.detectMultiScale(
                    gray_image, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                
                for (x, y, w, h) in opencv_faces:
                    faces_detected.append({
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'confidence': 0.8,  # Default confidence for Haar cascades
                        'method': 'opencv',
                        'area': int(w * h),
                        'center': [int(x + w/2), int(y + h/2)]
                    })
            
            # Method 2: Dlib face detection
            if self.dlib_detector is not None:
                dlib_faces = self.dlib_detector(gray_image)
                
                for face in dlib_faces:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    
                    face_info = {
                        'bbox': [x, y, w, h],
                        'confidence': 0.85,  # Default confidence for dlib
                        'method': 'dlib',
                        'area': w * h,
                        'center': [x + w//2, y + h//2]
                    }
                    
                    # Add landmarks if available
                    if self.shape_predictor is not None:
                        landmarks = self.shape_predictor(gray_image, face)
                        landmark_points = []
                        for i in range(68):
                            point = landmarks.part(i)
                            landmark_points.append([point.x, point.y])
                        face_info['landmarks'] = landmark_points
                    
                    faces_detected.append(face_info)
            
            # Method 3: face_recognition library (if available)
            try:
                face_locations = face_recognition.face_locations(rgb_image)
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    faces_detected.append({
                        'bbox': [left, top, right - left, bottom - top],
                        'confidence': 0.9,
                        'method': 'face_recognition',
                        'area': (right - left) * (bottom - top),
                        'center': [(left + right)//2, (top + bottom)//2],
                        'encoding': face_encodings[i].tolist() if i < len(face_encodings) else None
                    })
                        
            except Exception as e:
                logger.debug(f"face_recognition method failed: {e}")
            
            # Remove duplicate faces (same face detected by multiple methods)
            unique_faces = self._remove_duplicate_faces(faces_detected)
            
            # Analyze face characteristics
            for face in unique_faces:
                face.update(self._analyze_face_characteristics(image, face['bbox']))
            
            result = {
                'total_faces': len(unique_faces),
                'faces': unique_faces,
                'image_dimensions': [image.shape[1], image.shape[0]],
                'detection_methods': list(set([f['method'] for f in unique_faces])),
                'analysis_timestamp': str(np.datetime64('now'))
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return {'faces': [], 'error': str(e)}
    
    def _remove_duplicate_faces(self, faces: List[Dict]) -> List[Dict]:
        """Remove duplicate face detections from multiple methods."""
        if len(faces) <= 1:
            return faces
        
        unique_faces = []
        
        for face in faces:
            is_duplicate = False
            face_center = face['center']
            
            for unique_face in unique_faces:
                unique_center = unique_face['center']
                
                # Calculate distance between face centers
                distance = np.sqrt(
                    (face_center[0] - unique_center[0])**2 + 
                    (face_center[1] - unique_center[1])**2
                )
                
                # If faces are close, consider them duplicates
                if distance < 50:  # Adjust threshold as needed
                    is_duplicate = True
                    # Keep the face with higher confidence
                    if face['confidence'] > unique_face['confidence']:
                        unique_faces.remove(unique_face)
                        unique_faces.append(face)
                    break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def _analyze_face_characteristics(self, image: np.ndarray, bbox: List[int]) -> Dict[str, Any]:
        """Analyze characteristics of detected face."""
        x, y, w, h = bbox
        
        # Extract face region
        face_region = image[y:y+h, x:x+w]
        if face_region.size == 0:
            return {}
        
        # Basic characteristics
        characteristics = {
            'size_category': 'small' if w * h < 5000 else 'medium' if w * h < 20000 else 'large',
            'aspect_ratio': w / h if h > 0 else 1,
            'position': {
                'left_third': x < image.shape[1] // 3,
                'center_third': image.shape[1] // 3 <= x < 2 * image.shape[1] // 3,
                'right_third': x >= 2 * image.shape[1] // 3,
                'top_third': y < image.shape[0] // 3,
                'middle_third': image.shape[0] // 3 <= y < 2 * image.shape[0] // 3,
                'bottom_third': y >= 2 * image.shape[0] // 3
            }
        }
        
        # Color analysis
        try:
            face_hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            avg_hue = np.mean(face_hsv[:, :, 0])
            avg_saturation = np.mean(face_hsv[:, :, 1])
            avg_brightness = np.mean(face_hsv[:, :, 2])
            
            characteristics['color_analysis'] = {
                'avg_hue': float(avg_hue),
                'avg_saturation': float(avg_saturation),
                'avg_brightness': float(avg_brightness),
                'skin_tone_category': self._categorize_skin_tone(avg_hue, avg_saturation)
            }
        except Exception as e:
            logger.debug(f"Face color analysis failed: {e}")
        
        return characteristics
    
    def _categorize_skin_tone(self, hue: float, saturation: float) -> str:
        """Categorize skin tone for demographic analysis."""
        # This is a simplified categorization for forensic purposes
        if hue < 15 or hue > 170:
            if saturation < 50:
                return "light"
            else:
                return "medium"
        elif 15 <= hue <= 25:
            return "medium"
        else:
            return "varied"  # Non-typical skin tone (could be lighting, etc.)
    
    def detect_objects(self, image_path: str) -> Dict[str, Any]:
        """
        Detect evidence-relevant objects in image.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            Dict[str, Any]: Object detection results
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'objects': [], 'error': 'Could not load image'}
            
            # For now, we'll use a simplified approach with template matching
            # In a full implementation, you'd use YOLO, R-CNN, or similar models
            detected_objects = []
            
            # Use edge detection to find object-like structures
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours (potential objects)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and shape
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum object size
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Extract object region for analysis
                    object_region = image[y:y+h, x:x+w]
                    
                    # Basic object classification (simplified)
                    object_type = self._classify_object_region(object_region)
                    
                    if object_type:
                        detected_objects.append({
                            'bbox': [x, y, w, h],
                            'type': object_type,
                            'confidence': 0.6,  # Conservative confidence
                            'area': int(area),
                            'aspect_ratio': w / h if h > 0 else 1,
                            'center': [x + w//2, y + h//2]
                        })
            
            # Sort by confidence and area
            detected_objects.sort(key=lambda x: (x['confidence'], x['area']), reverse=True)
            
            # Group by evidence categories
            evidence_summary = {}
            for category in self.evidence_objects:
                category_objects = [obj for obj in detected_objects if obj['type'] in self.evidence_objects[category]]
                if category_objects:
                    evidence_summary[category] = len(category_objects)
            
            result = {
                'total_objects': len(detected_objects),
                'objects': detected_objects[:20],  # Limit to top 20 objects
                'evidence_summary': evidence_summary,
                'image_dimensions': [image.shape[1], image.shape[0]],
                'analysis_timestamp': str(np.datetime64('now'))
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return {'objects': [], 'error': str(e)}
    
    def _classify_object_region(self, region: np.ndarray) -> Optional[str]:
        """
        Classify object region using basic computer vision techniques.
        This is a simplified version - real implementation would use trained models.
        """
        if region.size == 0:
            return None
        
        # Basic shape and color analysis
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Edge density analysis
        edges = cv2.Canny(gray_region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Color analysis
        avg_color = np.mean(region, axis=(0, 1))
        
        # Aspect ratio
        h, w = region.shape[:2]
        aspect_ratio = w / h if h > 0 else 1
        
        # Simple classification rules (would be replaced by ML models)
        if edge_density > 0.3:
            if aspect_ratio > 3:  # Long thin objects
                return 'knife'
            elif aspect_ratio < 0.7:  # Tall objects
                return 'phone'
        
        if avg_color[0] > 100 and avg_color[1] < 50:  # Reddish objects
            return 'substance'
        
        if edge_density > 0.2 and 0.8 < aspect_ratio < 1.2:  # Square-ish objects
            if np.mean(avg_color) > 150:  # Light colored
                return 'document'
            else:
                return 'device'
        
        return 'unknown_object'
    
    def analyze_person_characteristics(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze characteristics of people in image for investigation purposes.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            Dict[str, Any]: Person analysis results
        """
        try:
            # Get face detection results
            face_results = self.detect_faces(image_path)
            
            # Load image for body analysis
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image'}
            
            persons = []
            
            for i, face in enumerate(face_results.get('faces', [])):
                person = {
                    'person_id': f"person_{i+1}",
                    'face_info': face,
                    'estimated_characteristics': {}
                }
                
                # Estimate additional characteristics based on face
                if 'bbox' in face:
                    x, y, w, h = face['bbox']
                    
                    # Estimate age group (very basic)
                    face_area = w * h
                    if face_area < 3000:
                        age_estimate = "young"
                    elif face_area > 10000:
                        age_estimate = "adult"
                    else:
                        age_estimate = "unknown"
                    
                    person['estimated_characteristics'] = {
                        'age_group': age_estimate,
                        'face_size': face.get('size_category', 'unknown'),
                        'visibility': 'clear' if face.get('confidence', 0) > 0.8 else 'partial'
                    }
                
                persons.append(person)
            
            result = {
                'total_persons': len(persons),
                'persons': persons,
                'group_analysis': {
                    'multiple_persons': len(persons) > 1,
                    'crowd_scene': len(persons) > 5,
                    'identifiable_faces': len([p for p in persons if p['face_info'].get('confidence', 0) > 0.7])
                },
                'investigation_notes': self._generate_investigation_notes(persons)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Person analysis failed: {e}")
            return {'error': str(e)}
    
    def _generate_investigation_notes(self, persons: List[Dict]) -> List[str]:
        """Generate investigative notes based on person analysis."""
        notes = []
        
        if len(persons) == 0:
            notes.append("No persons clearly visible in image")
        elif len(persons) == 1:
            notes.append("Single person detected in image")
            person = persons[0]
            if person['face_info'].get('confidence', 0) > 0.8:
                notes.append("Face is clearly visible and suitable for identification")
        else:
            notes.append(f"Multiple persons detected ({len(persons)} total)")
            clear_faces = len([p for p in persons if p['face_info'].get('confidence', 0) > 0.7])
            notes.append(f"{clear_faces} persons have clearly visible faces")
        
        # Check for investigation-relevant characteristics
        large_faces = [p for p in persons if p['face_info'].get('size_category') == 'large']
        if large_faces:
            notes.append(f"{len(large_faces)} person(s) prominently featured in image")
        
        return notes
    
    def create_face_database_entry(self, image_path: str, person_name: str) -> bool:
        """
        Create a face database entry for person identification.
        
        Args:
            image_path (str): Path to reference image
            person_name (str): Name/ID of the person
            
        Returns:
            bool: Success status
        """
        try:
            # Load image and detect faces
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get face encodings
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if face_encodings:
                # Store the first (best) encoding
                self.face_encodings_db[person_name] = {
                    'encoding': face_encodings[0],
                    'reference_image': image_path,
                    'added_timestamp': str(np.datetime64('now'))
                }
                
                # Save to file
                db_path = "face_database.json"
                if os.path.exists(db_path):
                    with open(db_path, 'r') as f:
                        db = json.load(f)
                else:
                    db = {}
                
                db[person_name] = {
                    'encoding': face_encodings[0].tolist(),
                    'reference_image': image_path,
                    'added_timestamp': str(np.datetime64('now'))
                }
                
                with open(db_path, 'w') as f:
                    json.dump(db, f, indent=2)
                
                logger.info(f"Face database entry created for {person_name}")
                return True
            else:
                logger.warning(f"No faces found in reference image for {person_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create face database entry: {e}")
            return False
    
    def identify_known_faces(self, image_path: str) -> Dict[str, Any]:
        """
        Identify known faces in image using face database.
        
        Args:
            image_path (str): Path to image to analyze
            
        Returns:
            Dict[str, Any]: Face identification results
        """
        try:
            # Load face database
            db_path = "face_database.json"
            if not os.path.exists(db_path):
                return {'identifications': [], 'message': 'No face database found'}
            
            with open(db_path, 'r') as f:
                face_db = json.load(f)
            
            if not face_db:
                return {'identifications': [], 'message': 'Face database is empty'}
            
            # Load and analyze image
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find faces in the image
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            identifications = []
            
            for i, face_encoding in enumerate(face_encodings):
                best_match_name = "Unknown"
                best_match_distance = float('inf')
                
                # Compare with known faces
                for person_name, person_data in face_db.items():
                    known_encoding = np.array(person_data['encoding'])
                    distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
                    
                    if distance < best_match_distance and distance < 0.6:  # Threshold for match
                        best_match_distance = distance
                        best_match_name = person_name
                
                # Add identification result
                top, right, bottom, left = face_locations[i]
                identifications.append({
                    'bbox': [left, top, right - left, bottom - top],
                    'identified_as': best_match_name,
                    'confidence': max(0, 1 - best_match_distance) if best_match_name != "Unknown" else 0,
                    'match_distance': float(best_match_distance),
                    'face_number': i + 1
                })
            
            result = {
                'total_faces_found': len(face_locations),
                'identifications': identifications,
                'known_persons_detected': len([i for i in identifications if i['identified_as'] != "Unknown"]),
                'database_size': len(face_db)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Face identification failed: {e}")
            return {'error': str(e)}

# Example usage and testing
if __name__ == "__main__":
    detector = FaceObjectDetector()
    
    # Test face detection
    # result = detector.detect_faces("test_image.jpg")
    # print(json.dumps(result, indent=2, default=str))
    
    # Test object detection
    # result = detector.detect_objects("test_image.jpg")
    # print(json.dumps(result, indent=2, default=str))
