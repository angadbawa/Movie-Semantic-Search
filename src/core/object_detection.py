from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO
import logging
from ..utils.config import get_detection_config
from ..utils.helpers import memoize, safe_execute

@memoize
def load_yolov8_model(model_name: Optional[str] = None) -> YOLO:
    """
    Load YOLOv8 model with memoization for performance.
    
    Args:
        model_name: YOLOv8 model variant (yolov8n.pt, yolov8s.pt, etc.)
        
    Returns:
        Loaded YOLO model
    """
    if model_name is None:
        config = get_detection_config()
        model_name = config['model_name']
    
    logging.info(f"Loading YOLOv8 model: {model_name}")
    return YOLO(model_name)

def detect_objects_in_frame(frame: np.ndarray, model: Optional[YOLO] = None, 
                           confidence_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Detect objects in a single frame using YOLOv8.
    
    Args:
        frame: Input frame as numpy array
        model: YOLOv8 model (loads default if None)
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        List of detected objects with metadata
    """
    if model is None:
        model = load_yolov8_model()
    
    if confidence_threshold is None:
        config = get_detection_config()
        confidence_threshold = config['confidence_threshold']
    
    # Run inference
    results = model(frame, verbose=False)
    
    # Extract detections
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                confidence = float(boxes.conf[i])
                if confidence >= confidence_threshold:
                    detection = {
                        'class_id': int(boxes.cls[i]),
                        'class_name': model.names[int(boxes.cls[i])],
                        'confidence': confidence,
                        'bbox': boxes.xyxy[i].tolist(),  # [x1, y1, x2, y2]
                        'center': [(boxes.xyxy[i][0] + boxes.xyxy[i][2]) / 2, 
                                  (boxes.xyxy[i][1] + boxes.xyxy[i][3]) / 2]
                    }
                    detections.append(detection)
    
    return detections

def extract_object_names(detections: List[Dict[str, Any]]) -> List[str]:
    """
    Extract unique object class names from detections.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        List of unique object class names
    """
    return list(set(detection['class_name'] for detection in detections))

def filter_detections_by_confidence(detections: List[Dict[str, Any]], 
                                   min_confidence: float) -> List[Dict[str, Any]]:
    """
    Filter detections by minimum confidence threshold.
    
    Args:
        detections: List of detection dictionaries
        min_confidence: Minimum confidence threshold
        
    Returns:
        Filtered list of detections
    """
    return [det for det in detections if det['confidence'] >= min_confidence]

def filter_detections_by_classes(detections: List[Dict[str, Any]], 
                                allowed_classes: List[str]) -> List[Dict[str, Any]]:
    """
    Filter detections to only include specified classes.
    
    Args:
        detections: List of detection dictionaries
        allowed_classes: List of allowed class names
        
    Returns:
        Filtered list of detections
    """
    return [det for det in detections if det['class_name'] in allowed_classes]

def count_objects_by_class(detections: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Count detected objects by class name.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Dictionary mapping class names to counts
    """
    counts = {}
    for detection in detections:
        class_name = detection['class_name']
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts

def detect_objects_in_video_frames(frames: List[np.ndarray], 
                                  model: Optional[YOLO] = None) -> List[List[Dict[str, Any]]]:
    """
    Detect objects in multiple video frames.
    
    Args:
        frames: List of frame arrays
        model: YOLOv8 model (loads default if None)
        
    Returns:
        List of detection lists for each frame
    """
    if model is None:
        model = load_yolov8_model()
    
    all_detections = []
    for i, frame in enumerate(frames):
        detections = detect_objects_in_frame(frame, model)
        all_detections.append(detections)
        
        if (i + 1) % 100 == 0:
            logging.info(f"Processed {i + 1}/{len(frames)} frames")
    
    return all_detections

def create_object_signature(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a signature representation of detected objects for comparison.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Object signature dictionary
    """
    object_names = extract_object_names(detections)
    object_counts = count_objects_by_class(detections)
    
    return {
        'unique_objects': sorted(object_names),
        'object_counts': object_counts,
        'total_objects': len(detections),
        'confidence_scores': [det['confidence'] for det in detections]
    }

# Safe versions of functions
safe_detect_objects_in_frame = safe_execute(detect_objects_in_frame, default=[])
safe_extract_object_names = safe_execute(extract_object_names, default=[])
safe_create_object_signature = safe_execute(create_object_signature, default={})
