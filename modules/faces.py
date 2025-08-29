"""
Face detection and recognition module using modern deep learning approaches.

🤖 Fun Fact: This module has evolved from detecting faces in everything (including your breakfast)
to actually knowing what a face is! We've come a long way from the Haar cascade days of 2001.

💡 Easter Egg: If you're reading this, you're probably debugging why your sandwich was tagged as "3 people"
"""

import os
import logging
import pickle
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

class FaceDetector:
    """Detects and recognizes faces in images using modern DNN models."""
    
    def __init__(self, encoding_file=None):
        """Initialize the face detector with modern DNN models."""
        self.face_recognition_available = False
        self.cv2_available = False
        self.mediapipe_available = False
        self.known_encodings = {}
        self.dnn_net = None
        self.mp_face_detection = None
        
        # Try to load face_recognition (still useful for encoding/recognition)
        try:
            import face_recognition
            self.face_recognition_available = True
            logger.info("face_recognition module is available")
        except ImportError:
            logger.warning("face_recognition module not available. Face recognition will be limited.")
            
        # Check for OpenCV with DNN support
        try:
            import cv2
            self.cv2_available = True
            
            # Initialize DNN face detector (much better than Haar cascades)
            self._load_dnn_models()
            
            # Also keep the old cascade as ultra-fallback
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            except:
                self.face_cascade = None
                
            logger.info("OpenCV with DNN face detection is available")
        except (ImportError, Exception) as e:
            logger.warning(f"OpenCV face detection not available: {e}")
            
        # Try MediaPipe (Google's fast and accurate solution)
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # 0 for close-range, 1 for full-range detection
                min_detection_confidence=0.7  # Conservative threshold
            )
            self.mediapipe_available = True
            logger.info("MediaPipe face detection is available")
        except ImportError:
            logger.warning("MediaPipe not available. Install with: pip install mediapipe")
            
        # Load known face encodings if provided
        if encoding_file and os.path.exists(encoding_file):
            try:
                with open(encoding_file, 'rb') as f:
                    self.known_encodings = pickle.load(f)
                logger.info(f"Loaded {len(self.known_encodings)} known face encodings")
            except Exception as e:
                logger.error(f"Failed to load face encodings: {e}")
    
    def _load_dnn_models(self):
        """Load pre-trained DNN models for face detection."""
        try:
            import cv2
            
            # Create models directory if it doesn't exist
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Paths for DNN models
            prototxt_path = models_dir / "deploy.prototxt"
            model_path = models_dir / "res10_300x300_ssd_iter_140000.caffemodel"
            
            # Download models if they don't exist
            if not prototxt_path.exists():
                logger.info("Downloading DNN face detection prototxt...")
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                    prototxt_path
                )
            
            if not model_path.exists():
                logger.info("Downloading DNN face detection model (this may take a moment)...")
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                    model_path
                )
            
            # Load the DNN model
            self.dnn_net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
            logger.info("DNN face detection model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load DNN models: {e}")
            self.dnn_net = None
    
    def detect_faces(self, image_path):
        """Detect faces using modern deep learning approaches with fallbacks."""
        try:
            # Quick check: if filename suggests it's not likely to contain faces, skip
            filename = os.path.basename(image_path).lower()
            food_keywords = ['curry', 'food', 'dish', 'meal', 'recipe', 'cooking', 'kitchen', 'plate', 'bowl', 'soup']
            if any(keyword in filename for keyword in food_keywords):
                logger.debug(f"Skipping face detection for likely food image: {filename}")
                # 🍛 Fun fact: We used to think curry had faces. It was a dark time in AI history.
                return []
            
            # Method 1: MediaPipe (fastest and most accurate for general use)
            if self.mediapipe_available:
                faces = self._detect_with_mediapipe(image_path)
                if faces is not None:  # None means error, empty list means no faces
                    logger.debug(f"MediaPipe found {len(faces)} faces")
                    return faces
            
            # Method 2: OpenCV DNN (excellent accuracy, good speed)
            if self.dnn_net is not None:
                faces = self._detect_with_dnn(image_path)
                if faces is not None:
                    logger.debug(f"DNN found {len(faces)} faces")
                    return faces
            
            # Method 3: face_recognition library (good for recognition tasks)
            if self.face_recognition_available:
                faces = self._detect_with_face_recognition(image_path)
                if faces:
                    logger.debug(f"face_recognition found {len(faces)} faces")
                    return faces
            
            # Method 4: Improved OpenCV Haar cascades (last resort)
            if self.cv2_available and self.face_cascade is not None:
                faces = self._detect_with_opencv(image_path)
                if faces:
                    logger.debug(f"OpenCV Haar found {len(faces)} faces after filtering")
                    return faces
                
            # No face detection available
            if not any([self.mediapipe_available, self.dnn_net, self.face_recognition_available, self.cv2_available]):
                logger.warning("No face detection methods available")
            
            return []
            
        except Exception as e:
            logger.error(f"Error detecting faces in {image_path}: {e}")
            return []
    
    def _detect_with_mediapipe(self, image_path):
        """Detect faces using Google's MediaPipe (most accurate and fast)."""
        try:
            import mediapipe as mp
            import cv2
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
                
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.mp_face_detection.process(rgb_image)
            
            if not results.detections:
                return []
            
            faces = []
            h, w, _ = image.shape
            
            for i, detection in enumerate(results.detections):
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Confidence check
                confidence = detection.score[0]
                if confidence < 0.7:  # Conservative threshold
                    continue
                
                # Additional validation
                if not self._validate_detection(image, x, y, width, height):
                    continue
                
                face_id = f"person_{i}"
                faces.append({
                    "id": face_id,
                    "location": (y, x + width, y + height, x),  # top, right, bottom, left
                    "confidence": confidence,
                    "encoding": None
                })
            
            return faces
            
        except Exception as e:
            logger.error(f"MediaPipe face detection failed: {e}")
            return None
    
    def _detect_with_dnn(self, image_path):
        """Detect faces using OpenCV DNN with pre-trained model."""
        try:
            import cv2
            import numpy as np
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
                
            (h, w) = image.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0)
            )
            
            # Pass blob through network
            self.dnn_net.setInput(blob)
            detections = self.dnn_net.forward()
            
            faces = []
            
            # Loop over detections
            for i in range(0, detections.shape[2]):
                # Extract confidence
                confidence = detections[0, 0, i, 2]
                
                # Filter weak detections
                if confidence < 0.7:  # Conservative threshold
                    continue
                
                # Compute coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                
                # Ensure coordinates are valid
                x = max(0, x)
                y = max(0, y)
                x1 = min(w, x1)
                y1 = min(h, y1)
                
                width = x1 - x
                height = y1 - y
                
                # Skip if dimensions are too small
                if width < 30 or height < 30:
                    continue
                
                # Additional validation
                if not self._validate_detection(image, x, y, width, height):
                    continue
                
                face_id = f"person_{len(faces)}"
                faces.append({
                    "id": face_id,
                    "location": (y, x1, y1, x),  # top, right, bottom, left
                    "confidence": float(confidence),
                    "encoding": None
                })
            
            return faces
            
        except Exception as e:
            logger.error(f"DNN face detection failed: {e}")
            return None
    
    def _validate_detection(self, image, x, y, width, height):
        """Validate a face detection to reduce false positives."""
        try:
            # Check aspect ratio
            aspect_ratio = width / height
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                return False
            
            # Check size relative to image
            h, w = image.shape[:2]
            face_area = width * height
            image_area = w * h
            area_ratio = face_area / image_area
            
            # Too large faces are often false positives
            if area_ratio > 0.6:
                return False
            
            # Too small faces are often noise
            if area_ratio < 0.001:
                return False
            
            # Check position (avoid extreme edges)
            center_x = x + width // 2
            center_y = y + height // 2
            
            edge_threshold = 0.05  # 5% from edge
            if (center_x < w * edge_threshold or 
                center_x > w * (1 - edge_threshold) or
                center_y < h * edge_threshold or 
                center_y > h * (1 - edge_threshold)):
                return False
            
            return True
            
        except Exception:
            return True  # If validation fails, be permissive
    
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
        """Detect faces using OpenCV's cascade classifier with strict filtering."""
        try:
            import cv2
            import numpy as np
            
            # Load image
            image = cv2.imread(str(image_path))
            
            if image is None:
                return []
                
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use more conservative parameters to reduce false positives
            # scaleFactor: 1.3 (larger = faster but less accurate)
            # minNeighbors: 8 (higher = more strict, fewer false positives)
            # minSize: (60, 60) (minimum face size to detect)
            # maxSize: () (maximum face size, empty = no limit)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.3, 
                minNeighbors=8,  # Very strict - real faces typically have many neighbors
                minSize=(60, 60),  # Ignore very small detections
                maxSize=(int(gray.shape[1]*0.8), int(gray.shape[0]*0.8))  # Ignore huge detections
            )
            
            if len(faces) == 0:
                return []
            
            # Additional filtering to reduce false positives
            valid_faces = []
            height, width = gray.shape
            
            for (x, y, w, h) in faces:
                # Filter by aspect ratio - faces should be roughly rectangular
                aspect_ratio = w / h
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    logger.debug(f"Rejected face due to aspect ratio: {aspect_ratio}")
                    continue
                
                # Filter by size relative to image - very large faces are often false positives
                face_area = w * h
                image_area = width * height
                area_ratio = face_area / image_area
                
                if area_ratio > 0.5:  # Face takes up more than 50% of image
                    logger.debug(f"Rejected face due to large area ratio: {area_ratio}")
                    continue
                
                # Filter by position - faces at extreme edges are often false positives
                center_x = x + w // 2
                center_y = y + h // 2
                
                edge_threshold = 0.1  # 10% from edge
                if (center_x < width * edge_threshold or 
                    center_x > width * (1 - edge_threshold) or
                    center_y < height * edge_threshold or 
                    center_y > height * (1 - edge_threshold)):
                    logger.debug(f"Rejected face too close to edge at ({center_x}, {center_y})")
                    continue
                
                # Additional check: analyze the detected region for face-like features
                face_region = gray[y:y+h, x:x+w]
                if not self._validate_face_region(face_region):
                    logger.debug(f"Rejected face due to failed region validation")
                    continue
                
                valid_faces.append((x, y, w, h))
            
            # Format results
            results = []
            for i, (x, y, w, h) in enumerate(valid_faces):
                face_id = f"person_{i}"
                results.append({
                    "id": face_id,
                    "location": (y, x + w, y + h, x),  # Convert to face_recognition format (top, right, bottom, left)
                    "encoding": None  # No encoding available with OpenCV
                })
            
            logger.debug(f"OpenCV detected {len(faces)} initial faces, filtered to {len(valid_faces)} valid faces")
            return results
            
        except Exception as e:
            logger.error(f"OpenCV face detection failed: {e}")
            return []
    
    def _validate_face_region(self, face_region):
        """Additional validation to check if detected region looks like a face."""
        try:
            import cv2
            import numpy as np
            
            if face_region.size == 0:
                return False
            
            # Check for reasonable contrast and variation
            # Real faces have variation in pixel values, unlike uniform objects
            std_dev = np.std(face_region)
            if std_dev < 10:  # Very low variation suggests uniform object
                return False
            
            # Check histogram distribution
            hist = cv2.calcHist([face_region], [0], None, [256], [0, 256])
            
            # Real faces typically have a more distributed histogram
            # Flat objects (like plates, circular patterns) have peaks
            hist_peak = np.max(hist)
            hist_mean = np.mean(hist)
            
            if hist_peak > hist_mean * 10:  # Very peaky histogram suggests non-face
                return False
            
            # Use eye cascade for additional validation if available
            try:
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=3)
                
                # Real faces should have at least one eye detected
                # But we're lenient since eye detection is also imperfect
                if len(eyes) == 0:
                    # Additional check: look for horizontal lines that might be eyes
                    # Apply edge detection
                    edges = cv2.Canny(face_region, 50, 150)
                    horizontal_lines = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
                    horizontal_detected = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_lines)
                    
                    # Count horizontal features
                    horizontal_pixels = np.sum(horizontal_detected > 0)
                    if horizontal_pixels < 5:  # Very few horizontal features
                        return False
                        
            except Exception:
                # If eye detection fails, don't reject - just continue
                pass
            
            return True
            
        except Exception as e:
            logger.debug(f"Face region validation failed: {e}")
            return True  # If validation fails, be permissive
    
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
