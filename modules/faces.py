"""
Face detection and recognition module.
"""

import os
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class FaceDetector:
    """Detects and recognizes faces in images."""
    
    def __init__(self, encoding_file=None):
        """Initialize the face detector."""
        self.face_recognition_available = False
        self.cv2_available = False
        self.known_encodings = {}
        
        # Try to load face_recognition
        try:
            import face_recognition
            self.face_recognition_available = True
            logger.info("face_recognition module is available")
        except ImportError:
            logger.warning("face_recognition module not available. Face detection will be limited.")
            
        # Check for OpenCV
        try:
            import cv2
            self.cv2_available = True
            # Load the face cascade classifier
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            logger.info("OpenCV face detection is available")
        except (ImportError, Exception) as e:
            logger.warning(f"OpenCV face detection not available: {e}")
            
        # Load known face encodings if provided
        if encoding_file and os.path.exists(encoding_file):
            try:
                with open(encoding_file, 'rb') as f:
                    self.known_encodings = pickle.load(f)
                logger.info(f"Loaded {len(self.known_encodings)} known face encodings")
            except Exception as e:
                logger.error(f"Failed to load face encodings: {e}")
    
    def detect_faces(self, image_path):
        """Detect faces in an image."""
        try:
            # Use face_recognition if available
            if self.face_recognition_available:
                return self._detect_with_face_recognition(image_path)
            
            # Fall back to OpenCV
            if self.cv2_available:
                return self._detect_with_opencv(image_path)
                
            # No face detection available
            logger.warning("No face detection methods available")
            return []
            
        except Exception as e:
            logger.error(f"Error detecting faces in {image_path}: {e}")
            return []
    
    def _detect_with_face_recognition(self, image_path):
        """Detect faces using the face_recognition library."""
        try:
            import face_recognition
            import numpy as np
            
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                return []
                
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            faces = []
            for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                # Check if face matches any known faces
                match_found = False
                
                if self.known_encodings:
                    for name, known_encodings in self.known_encodings.items():
                        matches = face_recognition.compare_faces(known_encodings, face_encoding)
                        if True in matches:
                            faces.append({
                                "id": name,
                                "location": face_location,
                                "encoding": face_encoding
                            })
                            match_found = True
                            break
                
                # If no match found, add as new face
                if not match_found:
                    face_id = f"person_{i}"
                    faces.append({
                        "id": face_id,
                        "location": face_location,
                        "encoding": face_encoding
                    })
            
            return faces
            
        except Exception as e:
            logger.error(f"face_recognition detection failed: {e}")
            
            # Try OpenCV as fallback if available
            if self.cv2_available:
                return self._detect_with_opencv(image_path)
            return []
    
    def _detect_with_opencv(self, image_path):
        """Detect faces using OpenCV's cascade classifier."""
        try:
            import cv2
            
            # Load image
            image = cv2.imread(str(image_path))
            
            if image is None:
                return []
                
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Format results
            results = []
            for i, (x, y, w, h) in enumerate(faces):
                face_id = f"person_{i}"
                results.append({
                    "id": face_id,
                    "location": (y, x + w, y + h, x),  # Convert to face_recognition format (top, right, bottom, left)
                    "encoding": None  # No encoding available with OpenCV
                })
                
            return results
            
        except Exception as e:
            logger.error(f"OpenCV face detection failed: {e}")
            return []
    
    def cluster_faces(self, face_encodings, tolerance=0.6):
        """Cluster similar faces together."""
        if not self.face_recognition_available or not face_encodings:
            return {}
            
        try:
            import face_recognition
            import numpy as np
            from sklearn.cluster import DBSCAN
            
            # Extract the actual encodings from the face data
            encodings = [face["encoding"] for face in face_encodings if face["encoding"] is not None]
            
            if not encodings:
                return {}
                
            # Cluster the faces
            clustering = DBSCAN(eps=tolerance, min_samples=1, metric="euclidean").fit(encodings)
            
            # Group faces by cluster
            face_clusters = {}
            for i, label in enumerate(clustering.labels_):
                cluster_id = f"person_{label}" if label >= 0 else "outlier"
                
                if cluster_id not in face_clusters:
                    face_clusters[cluster_id] = []
                    
                face_clusters[cluster_id].append(face_encodings[i])
                
            return face_clusters
            
        except Exception as e:
            logger.error(f"Face clustering failed: {e}")
            return {}
    
    def save_encodings(self, file_path):
        """Save known face encodings to a file."""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.known_encodings, f)
            logger.info(f"Saved {len(self.known_encodings)} face encodings to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save face encodings: {e}")
            return False
