from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from ..utils.config import get_config
from ..utils.helpers import memoize, safe_execute
import logging

try:
    from hsemotion.facial_emotions import HSEmotionRecognizer
    _HSEMOTION_AVAILABLE = True
except ImportError:
    _HSEMOTION_AVAILABLE = False
    logging.warning("HSEmotion not available. Install with: pip install hsemotion")

@memoize
def load_emotion_model(model_name: Optional[str] = None):
    """
    Load HSEmotion model for facial emotion recognition.
    
    Args:
        model_name: Name of the emotion model
        
    Returns:
        Loaded emotion recognition model
    """
    if not _HSEMOTION_AVAILABLE:
        raise ImportError("HSEmotion not available")
    
    if model_name is None:
        config = get_config("emotion_detection")
        model_name = config['model_name']
    
    logging.info(f"Loading HSEmotion model: {model_name}")
    return HSEmotionRecognizer(model_name=model_name)

def detect_emotions_in_faces(image: np.ndarray, 
                           face_bboxes: List[List[int]], 
                           model=None) -> List[Dict[str, Any]]:
    """
    Detect emotions in detected faces.
    
    Args:
        image: Input image
        face_bboxes: List of face bounding boxes [x1, y1, x2, y2]
        model: HSEmotion model
        
    Returns:
        List of emotion predictions for each face
    """
    if not _HSEMOTION_AVAILABLE:
        return []
    
    if model is None:
        model = load_emotion_model()
    
    emotions = []
    
    for i, bbox in enumerate(face_bboxes):
        try:
            x1, y1, x2, y2 = bbox
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue
            
            # Predict emotions
            emotion_scores = model.predict_emotions(face_crop, logits=False)
            
            # Get top emotion
            top_emotion_idx = np.argmax(emotion_scores)
            top_emotion = model.get_labels()[top_emotion_idx]
            top_confidence = float(emotion_scores[top_emotion_idx])
            
            # Create emotion dictionary
            emotion_dict = {}
            for j, label in enumerate(model.get_labels()):
                emotion_dict[label] = float(emotion_scores[j])
            
            emotions.append({
                'face_id': i,
                'bbox': bbox,
                'top_emotion': top_emotion,
                'confidence': top_confidence,
                'all_emotions': emotion_dict
            })
            
        except Exception as e:
            logging.warning(f"Emotion detection failed for face {i}: {e}")
            continue
    
    return emotions

def analyze_frame_emotions(image: np.ndarray, 
                          faces: List[Dict[str, Any]], 
                          model=None) -> Dict[str, Any]:
    """
    Analyze emotions in a frame with detected faces.
    
    Args:
        image: Input frame
        faces: List of face detection results
        model: HSEmotion model
        
    Returns:
        Frame emotion analysis
    """
    if not faces:
        return {'emotions': [], 'dominant_emotion': None, 'emotion_counts': {}}
    
    # Extract face bboxes
    face_bboxes = [face['bbox'] for face in faces]
    
    # Detect emotions
    emotions = detect_emotions_in_faces(image, face_bboxes, model)
    
    # Analyze overall frame emotions
    all_emotions = [emotion['top_emotion'] for emotion in emotions]
    emotion_counts = {}
    for emotion in all_emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # Find dominant emotion
    dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None
    
    return {
        'emotions': emotions,
        'dominant_emotion': dominant_emotion,
        'emotion_counts': emotion_counts,
        'num_faces': len(emotions),
        'avg_confidence': np.mean([e['confidence'] for e in emotions]) if emotions else 0.0
    }

def analyze_shot_emotions(shot_frames: List[np.ndarray], 
                         shot_faces: List[List[Dict[str, Any]]], 
                         model=None,
                         sample_rate: int = 5) -> Dict[str, Any]:
    """
    Analyze emotions throughout a shot.
    
    Args:
        shot_frames: All frames in the shot
        shot_faces: Face detections for each frame
        model: HSEmotion model
        sample_rate: Sample every Nth frame
        
    Returns:
        Shot emotion analysis
    """
    if not shot_frames or not shot_faces:
        return {}
    
    # Sample frames for analysis
    sampled_indices = range(0, len(shot_frames), sample_rate)
    
    frame_emotions = []
    all_emotions = []
    emotion_timeline = []
    
    for i in sampled_indices:
        if i < len(shot_faces):
            frame_emotion = analyze_frame_emotions(shot_frames[i], shot_faces[i], model)
            frame_emotions.append(frame_emotion)
            
            # Collect all emotions
            for emotion_data in frame_emotion['emotions']:
                all_emotions.append(emotion_data['top_emotion'])
            
            # Add to timeline
            emotion_timeline.append({
                'frame_index': i,
                'dominant_emotion': frame_emotion['dominant_emotion'],
                'num_faces': frame_emotion['num_faces'],
                'confidence': frame_emotion['avg_confidence']
            })
    
    # Aggregate shot-level emotions
    emotion_counts = {}
    for emotion in all_emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # Calculate emotion percentages
    total_detections = len(all_emotions)
    emotion_percentages = {}
    if total_detections > 0:
        for emotion, count in emotion_counts.items():
            emotion_percentages[emotion] = count / total_detections
    
    # Find dominant emotion for the shot
    shot_dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None
    
    return {
        'shot_dominant_emotion': shot_dominant_emotion,
        'emotion_counts': emotion_counts,
        'emotion_percentages': emotion_percentages,
        'emotion_timeline': emotion_timeline,
        'total_emotion_detections': total_detections,
        'avg_faces_per_frame': np.mean([fe['num_faces'] for fe in frame_emotions]) if frame_emotions else 0
    }

def create_emotion_signature(shot_emotions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create emotion signature for shot similarity comparison.
    
    Args:
        shot_emotions: Shot emotion analysis results
        
    Returns:
        Emotion signature
    """
    if not shot_emotions:
        return {}
    
    return {
        'dominant_emotion': shot_emotions.get('shot_dominant_emotion'),
        'emotion_distribution': shot_emotions.get('emotion_percentages', {}),
        'has_emotions': shot_emotions.get('total_emotion_detections', 0) > 0,
        'emotion_diversity': len(shot_emotions.get('emotion_counts', {})),
        'avg_faces': shot_emotions.get('avg_faces_per_frame', 0)
    }

def calculate_emotion_similarity(emotions1: Dict[str, Any], emotions2: Dict[str, Any]) -> float:
    """
    Calculate similarity between two emotion signatures.
    
    Args:
        emotions1: First emotion signature
        emotions2: Second emotion signature
        
    Returns:
        Similarity score between 0 and 1
    """
    if not emotions1 or not emotions2:
        return 0.0
    
    # Dominant emotion similarity
    dom1 = emotions1.get('dominant_emotion')
    dom2 = emotions2.get('dominant_emotion')
    dominant_similarity = 1.0 if dom1 == dom2 else 0.0
    
    # Emotion distribution similarity (using cosine similarity)
    dist1 = emotions1.get('emotion_distribution', {})
    dist2 = emotions2.get('emotion_distribution', {})
    
    if not dist1 and not dist2:
        distribution_similarity = 1.0
    elif not dist1 or not dist2:
        distribution_similarity = 0.0
    else:
        # Get all unique emotions
        all_emotions = set(dist1.keys()) | set(dist2.keys())
        
        # Create vectors
        vec1 = np.array([dist1.get(emotion, 0) for emotion in all_emotions])
        vec2 = np.array([dist2.get(emotion, 0) for emotion in all_emotions])
        
        # Cosine similarity
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            distribution_similarity = 0.0
        else:
            distribution_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # Face count similarity
    faces1 = emotions1.get('avg_faces', 0)
    faces2 = emotions2.get('avg_faces', 0)
    max_faces = max(faces1, faces2)
    face_similarity = 1.0 - abs(faces1 - faces2) / max_faces if max_faces > 0 else 1.0
    
    # Weighted combination
    return 0.5 * dominant_similarity + 0.3 * distribution_similarity + 0.2 * face_similarity

def detect_emotional_scenes(shots_emotions: List[Dict[str, Any]], 
                           target_emotion: str = "happy") -> List[Dict[str, Any]]:
    """
    Find shots/scenes with specific emotional content.
    
    Args:
        shots_emotions: List of emotion analyses for shots
        target_emotion: Target emotion to search for
        
    Returns:
        List of shots matching the emotional criteria
    """
    matching_shots = []
    
    for i, shot_emotion in enumerate(shots_emotions):
        if not shot_emotion:
            continue
        
        dominant = shot_emotion.get('shot_dominant_emotion')
        percentages = shot_emotion.get('emotion_percentages', {})
        
        # Check if target emotion is dominant or significant
        target_percentage = percentages.get(target_emotion, 0)
        
        if dominant == target_emotion or target_percentage > 0.3:
            matching_shots.append({
                'shot_index': i,
                'dominant_emotion': dominant,
                'target_emotion_percentage': target_percentage,
                'emotion_data': shot_emotion
            })
    
    return matching_shots

# Safe versions of functions
safe_detect_emotions_in_faces = safe_execute(detect_emotions_in_faces, default=[])
safe_analyze_frame_emotions = safe_execute(analyze_frame_emotions, default={})
safe_analyze_shot_emotions = safe_execute(analyze_shot_emotions, default={})