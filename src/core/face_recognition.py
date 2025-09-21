from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import dlib
from ..utils.config import get_config
from ..utils.helpers import memoize, safe_execute
import logging

@memoize
def load_hog_face_detector():
    """Load HOG face detector with memoization."""
    logging.info("Loading HOG face detector")
    return dlib.get_frontal_face_detector()

@memoize 
def load_face_predictor(predictor_path: Optional[str] = None):
    """Load facial landmark predictor."""
    if predictor_path is None:
        config = get_config("face_recognition")
        predictor_path = config['face_predictor_path']
    
    logging.info(f"Loading face predictor: {predictor_path}")
    return dlib.shape_predictor(predictor_path)

@memoize
def load_face_recognition_model(model_path: Optional[str] = None):
    """Load face recognition model."""
    if model_path is None:
        config = get_config("face_recognition")
        model_path = config['face_recognition_model']
    
    logging.info(f"Loading face recognition model: {model_path}")
    return dlib.face_recognition_model_v1(model_path)

def detect_faces_hog(image: np.ndarray, detector=None) -> List[Dict[str, Any]]:
    """
    Detect faces in image using HOG detector.
    
    Args:
        image: Input image as numpy array
        detector: HOG face detector (loads default if None)
        
    Returns:
        List of detected faces with bounding boxes
    """
    if detector is None:
        detector = load_hog_face_detector()
    
    # Convert to grayscale for HOG detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detect faces
    face_rects = detector(gray, 1)
    
    faces = []
    for i, rect in enumerate(face_rects):
        face = {
            'face_id': i,
            'bbox': [rect.left(), rect.top(), rect.right(), rect.bottom()],
            'center': [(rect.left() + rect.right()) // 2, (rect.top() + rect.bottom()) // 2],
            'width': rect.width(),
            'height': rect.height(),
            'area': rect.width() * rect.height()
        }
        faces.append(face)
    
    return faces

def extract_face_landmarks(image: np.ndarray, face_rect, predictor=None) -> np.ndarray:
    """
    Extract 68 facial landmarks from detected face.
    
    Args:
        image: Input image as numpy array
        face_rect: dlib rectangle object for face location
        predictor: Facial landmark predictor
        
    Returns:
        Array of 68 (x, y) landmark coordinates
    """
    if predictor is None:
        predictor = load_face_predictor()
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Get landmarks
    landmarks = predictor(gray, face_rect)
    
    # Convert to numpy array
    coords = np.zeros((68, 2), dtype=int)
    for i in range(68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    
    return coords

def compute_face_encoding(image: np.ndarray, face_rect, landmarks=None, 
                         face_rec_model=None) -> np.ndarray:
    """
    Compute 128-dimensional face encoding for recognition.
    
    Args:
        image: Input image as numpy array
        face_rect: dlib rectangle object for face location
        landmarks: Facial landmarks (computed if None)
        face_rec_model: Face recognition model
        
    Returns:
        128-dimensional face encoding
    """
    if face_rec_model is None:
        face_rec_model = load_face_recognition_model()
    
    if landmarks is None:
        predictor = load_face_predictor()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        landmarks = predictor(gray, face_rect)
    
    # Compute face encoding
    face_encoding = face_rec_model.compute_face_descriptor(image, landmarks)
    return np.array(face_encoding)

def compare_face_encodings(encoding1: np.ndarray, encoding2: np.ndarray, 
                          tolerance: Optional[float] = None) -> Tuple[float, bool]:
    """
    Compare two face encodings and determine if they match.
    
    Args:
        encoding1: First face encoding
        encoding2: Second face encoding
        tolerance: Distance threshold for matching
        
    Returns:
        Tuple of (distance, is_match)
    """
    if tolerance is None:
        config = get_config("face_recognition")
        tolerance = config['tolerance']
    
    # Calculate Euclidean distance
    distance = np.linalg.norm(encoding1 - encoding2)
    is_match = distance <= tolerance
    
    return float(distance), is_match

def detect_and_encode_faces(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect faces and compute encodings in one step.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        List of face dictionaries with encodings
    """
    detector = load_hog_face_detector()
    predictor = load_face_predictor()
    face_rec_model = load_face_recognition_model()
    
    # Detect faces
    faces = detect_faces_hog(image, detector)
    
    # Add encodings to each face
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    for face in faces:
        # Create dlib rectangle
        rect = dlib.rectangle(face['bbox'][0], face['bbox'][1], 
                             face['bbox'][2], face['bbox'][3])
        
        # Get landmarks and encoding
        landmarks = predictor(gray, rect)
        encoding = face_rec_model.compute_face_descriptor(image, landmarks)
        
        face['landmarks'] = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
        face['encoding'] = np.array(encoding)
    
    return faces

def count_unique_faces(face_encodings: List[np.ndarray], tolerance: Optional[float] = None) -> int:
    """
    Count unique faces by clustering similar encodings.
    
    Args:
        face_encodings: List of face encodings
        tolerance: Distance threshold for considering faces the same
        
    Returns:
        Number of unique faces
    """
    if not face_encodings:
        return 0
    
    if tolerance is None:
        config = get_config("face_recognition")
        tolerance = config['tolerance']
    
    unique_faces = []
    
    for encoding in face_encodings:
        is_unique = True
        for unique_encoding in unique_faces:
            distance = np.linalg.norm(encoding - unique_encoding)
            if distance <= tolerance:
                is_unique = False
                break
        
        if is_unique:
            unique_faces.append(encoding)
    
    return len(unique_faces)

def create_face_signature(faces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a signature representation of detected faces.
    
    Args:
        faces: List of face dictionaries with encodings
        
    Returns:
        Face signature dictionary
    """
    if not faces:
        return {'face_count': 0, 'unique_faces': 0, 'face_areas': []}
    
    encodings = [face['encoding'] for face in faces if 'encoding' in face]
    unique_count = count_unique_faces(encodings)
    
    return {
        'face_count': len(faces),
        'unique_faces': unique_count,
        'face_areas': [face['area'] for face in faces],
        'avg_face_size': np.mean([face['area'] for face in faces]),
        'face_positions': [face['center'] for face in faces]
    }

# Safe versions of functions
safe_detect_faces_hog = safe_execute(detect_faces_hog, default=[])
safe_detect_and_encode_faces = safe_execute(detect_and_encode_faces, default=[])
safe_create_face_signature = safe_execute(create_face_signature, default={})
