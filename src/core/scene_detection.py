from typing import List, Dict, Any, Set, Tuple
import numpy as np
from ..utils.config import get_scene_config
from ..utils.helpers import safe_execute, chunk_list
import logging

def calculate_object_similarity(objects1: Dict[str, Any], objects2: Dict[str, Any]) -> float:
    """
    Calculate similarity between two object signatures using set theory.
    
    Args:
        objects1: First object signature
        objects2: Second object signature
        
    Returns:
        Similarity score between 0 and 1
    """
    if not objects1 or not objects2:
        return 0.0
    
    # Get unique objects as sets
    set1 = set(objects1.get('unique_objects', []))
    set2 = set(objects2.get('unique_objects', []))
    
    if not set1 and not set2:
        return 1.0  # Both empty
    
    if not set1 or not set2:
        return 0.0  # One empty, one not
    
    # Calculate Jaccard similarity (intersection over union)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    jaccard_similarity = intersection / union if union > 0 else 0.0
    
    # Also consider object counts similarity
    counts1 = objects1.get('object_counts', {})
    counts2 = objects2.get('object_counts', {})
    
    # Calculate normalized count difference for common objects
    common_objects = set1.intersection(set2)
    count_similarity = 0.0
    
    if common_objects:
        count_diffs = []
        for obj in common_objects:
            count1 = counts1.get(obj, 0)
            count2 = counts2.get(obj, 0)
            max_count = max(count1, count2)
            if max_count > 0:
                diff = 1.0 - abs(count1 - count2) / max_count
                count_diffs.append(diff)
        
        count_similarity = np.mean(count_diffs) if count_diffs else 0.0
    
    # Combine Jaccard and count similarities
    return 0.7 * jaccard_similarity + 0.3 * count_similarity

def calculate_face_similarity(faces1: Dict[str, Any], faces2: Dict[str, Any]) -> float:
    """
    Calculate similarity between two face signatures.
    
    Args:
        faces1: First face signature
        faces2: Second face signature
        
    Returns:
        Similarity score between 0 and 1
    """
    if not faces1 or not faces2:
        return 0.0
    
    count1 = faces1.get('unique_faces', 0)
    count2 = faces2.get('unique_faces', 0)
    
    if count1 == 0 and count2 == 0:
        return 1.0  # Both have no faces
    
    if count1 == 0 or count2 == 0:
        return 0.0  # One has faces, one doesn't
    
    # Simple similarity based on face count difference
    max_count = max(count1, count2)
    similarity = 1.0 - abs(count1 - count2) / max_count
    
    return similarity

def calculate_actor_similarity(actors1: Dict[str, Any], actors2: Dict[str, Any]) -> float:
    """Calculate similarity between actor signatures."""
    if not actors1 or not actors2:
        return 0.0
    
    actors_set1 = set(actors1.get('actors_present', []))
    actors_set2 = set(actors2.get('actors_present', []))
    
    if not actors_set1 and not actors_set2:
        return 1.0
    if not actors_set1 or not actors_set2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(actors_set1.intersection(actors_set2))
    union = len(actors_set1.union(actors_set2))
    return intersection / union if union > 0 else 0.0

def calculate_action_similarity(actions1: Dict[str, Any], actions2: Dict[str, Any]) -> float:
    """Calculate similarity between action signatures."""
    if not actions1 or not actions2:
        return 0.0
    
    # Compare dancing presence
    dancing1 = actions1.get('has_dancing', False)
    dancing2 = actions2.get('has_dancing', False)
    
    return 1.0 if dancing1 == dancing2 else 0.0

def calculate_emotion_similarity(emotions1: Dict[str, Any], emotions2: Dict[str, Any]) -> float:
    """Calculate similarity between emotion signatures."""
    if not emotions1 or not emotions2:
        return 0.0
    
    # Compare dominant emotions
    dom1 = emotions1.get('dominant_emotion')
    dom2 = emotions2.get('dominant_emotion')
    
    return 1.0 if dom1 == dom2 else 0.0

def calculate_dialogue_similarity(dialogue1: Dict[str, Any], dialogue2: Dict[str, Any]) -> float:
    """Calculate similarity between dialogue signatures."""
    if not dialogue1 or not dialogue2:
        return 0.0
    
    # Compare sentiment
    sentiment1 = dialogue1.get('sentiment', {}).get('sentiment', 'neutral')
    sentiment2 = dialogue2.get('sentiment', {}).get('sentiment', 'neutral')
    
    return 1.0 if sentiment1 == sentiment2 else 0.0

def calculate_shot_similarity(shot1: Dict[str, Any], shot2: Dict[str, Any]) -> float:
    """
    Calculate multimodal similarity between two shots using all 6 modalities.
    
    Args:
        shot1: First shot metadata with multimodal analysis
        shot2: Second shot metadata with multimodal analysis
        
    Returns:
        Combined similarity score between 0 and 1
    """
    similarities = {}
    
    # 1. Object similarity (YOLOv8)
    similarities['objects'] = calculate_object_similarity(
        shot1.get('objects', {}), 
        shot2.get('objects', {})
    )
    
    # 2. Face similarity (HOG)
    similarities['faces'] = calculate_face_similarity(
        shot1.get('faces', {}), 
        shot2.get('faces', {})
    )
    
    # 3. Actor similarity (Actor Recognition)
    similarities['actors'] = calculate_actor_similarity(
        shot1.get('multimodal_analysis', {}).get('actors', {}),
        shot2.get('multimodal_analysis', {}).get('actors', {})
    )
    
    # 4. Action similarity (MMAction2)
    similarities['actions'] = calculate_action_similarity(
        shot1.get('multimodal_analysis', {}).get('actions', {}),
        shot2.get('multimodal_analysis', {}).get('actions', {})
    )
    
    # 5. Emotion similarity (HSEmotion)
    similarities['emotions'] = calculate_emotion_similarity(
        shot1.get('multimodal_analysis', {}).get('emotions', {}),
        shot2.get('multimodal_analysis', {}).get('emotions', {})
    )
    
    # 6. Dialogue similarity (Whisper + Sentiment)
    similarities['dialogue'] = calculate_dialogue_similarity(
        shot1.get('multimodal_analysis', {}).get('dialogue', {}),
        shot2.get('multimodal_analysis', {}).get('dialogue', {})
    )
    
    # Weighted combination of all 6 modalities
    weights = {
        'objects': 0.2,    # YOLOv8 objects
        'faces': 0.15,     # HOG face detection
        'actors': 0.2,     # Actor identification  
        'actions': 0.2,    # MMAction2 actions
        'emotions': 0.15,  # HSEmotion emotions
        'dialogue': 0.1    # Whisper + sentiment
    }
    
    # Calculate weighted similarity
    weighted_similarity = sum(
        similarities[modality] * weights[modality] 
        for modality in similarities
    )
    
    return weighted_similarity

def sliding_window_scene_detection(shots: List[Dict[str, Any]], 
                                 window_size: Optional[int] = None,
                                 similarity_threshold: Optional[float] = None) -> List[List[int]]:
    """
    Detect scene boundaries using sliding window approach with set theory.
    
    Args:
        shots: List of shot metadata dictionaries
        window_size: Size of sliding window
        similarity_threshold: Minimum similarity to group shots
        
    Returns:
        List of scenes, where each scene is a list of shot indices
    """
    if len(shots) < 2:
        return [[0]] if shots else []
    
    config = get_scene_config()
    if window_size is None:
        window_size = config['sliding_window_size']
    if similarity_threshold is None:
        similarity_threshold = config['similarity_threshold']
    
    scenes = []
    current_scene = [0]  # Start with first shot
    
    for i in range(1, len(shots)):
        # Calculate similarity with shots in current scene
        similarities = []
        
        # Look at recent shots in current scene (sliding window)
        recent_shots = current_scene[-window_size:] if len(current_scene) >= window_size else current_scene
        
        for shot_idx in recent_shots:
            similarity = calculate_shot_similarity(shots[i], shots[shot_idx])
            similarities.append(similarity)
        
        # Use maximum similarity to recent shots
        max_similarity = max(similarities) if similarities else 0.0
        
        if max_similarity >= similarity_threshold:
            # Add to current scene
            current_scene.append(i)
        else:
            # Start new scene
            if len(current_scene) >= config.get('min_scene_length', 1):
                scenes.append(current_scene)
            else:
                # If current scene is too short, merge with previous scene
                if scenes:
                    scenes[-1].extend(current_scene)
                else:
                    scenes.append(current_scene)
            
            current_scene = [i]
    
    # Add the last scene
    if current_scene:
        if len(current_scene) >= config.get('min_scene_length', 1):
            scenes.append(current_scene)
        elif scenes:
            scenes[-1].extend(current_scene)
        else:
            scenes.append(current_scene)
    
    logging.info(f"Detected {len(scenes)} scenes from {len(shots)} shots")
    return scenes

def create_scene_metadata(shots: List[Dict[str, Any]], 
                         scene_shot_indices: List[int]) -> Dict[str, Any]:
    """
    Create metadata for a scene from its constituent shots.
    
    Args:
        shots: List of all shot metadata
        scene_shot_indices: Indices of shots in this scene
        
    Returns:
        Scene metadata dictionary
    """
    if not scene_shot_indices:
        return {}
    
    scene_shots = [shots[i] for i in scene_shot_indices]
    
    # Aggregate object information
    all_objects = []
    all_faces = []
    total_frames = 0
    total_duration = 0.0
    
    for shot in scene_shots:
        objects = shot.get('objects', {})
        faces = shot.get('faces', {})
        
        all_objects.extend(objects.get('unique_objects', []))
        total_frames += shot.get('frame_count', 0)
        total_duration += shot.get('duration_seconds', 0.0)
    
    # Create scene signature
    unique_objects = list(set(all_objects))
    
    # Count object occurrences across shots
    object_counts = {}
    for obj in unique_objects:
        count = sum(1 for shot in scene_shots 
                   if obj in shot.get('objects', {}).get('unique_objects', []))
        object_counts[obj] = count
    
    # Face information
    total_faces = sum(shot.get('faces', {}).get('face_count', 0) for shot in scene_shots)
    unique_faces = sum(shot.get('faces', {}).get('unique_faces', 0) for shot in scene_shots)
    
    return {
        'shot_count': len(scene_shot_indices),
        'shot_indices': scene_shot_indices,
        'frame_count': total_frames,
        'duration_seconds': total_duration,
        'unique_objects': unique_objects,
        'object_counts': object_counts,
        'total_faces': total_faces,
        'estimated_unique_faces': unique_faces,
        'start_frame': scene_shots[0].get('start_frame', 0),
        'end_frame': scene_shots[-1].get('end_frame', 0)
    }

def process_shots_to_scenes(shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process shots into scenes with complete metadata.
    
    Args:
        shots: List of shot metadata dictionaries
        
    Returns:
        List of scene metadata dictionaries
    """
    if not shots:
        return []
    
    # Detect scene boundaries
    scene_boundaries = sliding_window_scene_detection(shots)
    
    # Create scene metadata
    scenes = []
    for i, scene_shot_indices in enumerate(scene_boundaries):
        scene_metadata = create_scene_metadata(shots, scene_shot_indices)
        scene_metadata['scene_id'] = f"scene_{i + 1:03d}"
        scene_metadata['scene_index'] = i
        
        scenes.append(scene_metadata)
        
        logging.info(f"Scene {i + 1}: {len(scene_shot_indices)} shots, "
                    f"{scene_metadata['duration_seconds']:.1f}s, "
                    f"{len(scene_metadata['unique_objects'])} unique objects")
    
    return scenes

def create_scene_similarity_matrix(shots: List[Dict[str, Any]]) -> np.ndarray:
    """
    Create similarity matrix between all shots for analysis.
    
    Args:
        shots: List of shot metadata dictionaries
        
    Returns:
        Similarity matrix as numpy array
    """
    n_shots = len(shots)
    similarity_matrix = np.zeros((n_shots, n_shots))
    
    for i in range(n_shots):
        for j in range(i, n_shots):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                similarity = calculate_shot_similarity(shots[i], shots[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Symmetric matrix
    
    return similarity_matrix

# Safe versions of functions
safe_calculate_shot_similarity = safe_execute(calculate_shot_similarity, default=0.0)
safe_sliding_window_scene_detection = safe_execute(sliding_window_scene_detection, default=[])
safe_process_shots_to_scenes = safe_execute(process_shots_to_scenes, default=[])
