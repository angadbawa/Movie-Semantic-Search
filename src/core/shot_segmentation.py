from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from pathlib import Path
from .similarity_metrics import calculate_similarity_metrics, is_shot_boundary
from .object_detection import detect_objects_in_frame, create_object_signature
from .face_recognition import detect_and_encode_faces, create_face_signature
from ..utils.config import get_video_config, get_path
from ..utils.helpers import safe_execute, compose, pipe
import logging

def extract_frames_from_video(video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of frame arrays
    """
    if max_frames is None:
        config = get_video_config()
        max_frames = config['max_frames']
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        frame_count += 1
        
        if frame_count % 1000 == 0:
            logging.info(f"Extracted {frame_count} frames")
    
    cap.release()
    logging.info(f"Total frames extracted: {len(frames)}")
    return frames

def detect_shot_boundaries(frames: List[np.ndarray]) -> List[int]:
    """
    Detect shot boundaries based on visual similarity metrics.
    
    Args:
        frames: List of frame arrays
        
    Returns:
        List of frame indices where shot boundaries occur
    """
    if len(frames) < 2:
        return []
    
    boundaries = [0]  # First frame is always a boundary
    
    for i in range(len(frames) - 1):
        metrics = calculate_similarity_metrics(frames[i], frames[i + 1])
        
        if is_shot_boundary(metrics):
            boundaries.append(i + 1)
            logging.debug(f"Shot boundary detected at frame {i + 1}")
    
    logging.info(f"Detected {len(boundaries)} shot boundaries")
    return boundaries

def create_shots_from_boundaries(frames: List[np.ndarray], 
                                boundaries: List[int]) -> List[List[np.ndarray]]:
    """
    Split frames into shots based on detected boundaries.
    
    Args:
        frames: List of frame arrays
        boundaries: List of boundary frame indices
        
    Returns:
        List of shots, where each shot is a list of frames
    """
    shots = []
    
    for i in range(len(boundaries)):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1] if i + 1 < len(boundaries) else len(frames)
        
        shot_frames = frames[start_idx:end_idx]
        if shot_frames:  # Only add non-empty shots
            shots.append(shot_frames)
    
    logging.info(f"Created {len(shots)} shots from boundaries")
    return shots

def save_shot_as_video(shot_frames: List[np.ndarray], output_path: str, 
                      fps: Optional[int] = None) -> bool:
    """
    Save a shot as a video file.
    
    Args:
        shot_frames: List of frames in the shot
        output_path: Output video file path
        fps: Frames per second for output video
        
    Returns:
        True if successful, False otherwise
    """
    if not shot_frames:
        return False
    
    if fps is None:
        config = get_video_config()
        fps = config['fps']
    
    # Get frame dimensions
    height, width = shot_frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    try:
        for frame in shot_frames:
            out.write(frame)
        
        out.release()
        logging.info(f"Saved shot video: {output_path}")
        return True
    
    except Exception as e:
        logging.error(f"Failed to save shot video {output_path}: {e}")
        out.release()
        return False

def analyze_shot_content(shot_frames: List[np.ndarray]) -> Dict[str, Any]:
    """
    Analyze the content of a shot (objects, faces, etc.).
    
    Args:
        shot_frames: List of frames in the shot
        
    Returns:
        Dictionary containing shot analysis
    """
    if not shot_frames:
        return {}
    
    # Sample frames for analysis (use every 5th frame to reduce computation)
    sample_frames = shot_frames[::5] if len(shot_frames) > 5 else shot_frames
    
    # Analyze objects and faces
    all_objects = []
    all_faces = []
    
    for frame in sample_frames:
        # Object detection
        objects = detect_objects_in_frame(frame)
        all_objects.extend(objects)
        
        # Face detection
        faces = detect_and_encode_faces(frame)
        all_faces.extend(faces)
    
    # Create signatures
    object_signature = create_object_signature(all_objects)
    face_signature = create_face_signature(all_faces)
    
    return {
        'frame_count': len(shot_frames),
        'duration_seconds': len(shot_frames) / get_video_config()['fps'],
        'objects': object_signature,
        'faces': face_signature,
        'sample_frame_count': len(sample_frames)
    }

def process_video_to_shots(video_path: str, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Complete pipeline to process video into analyzed shots.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save shot videos
        
    Returns:
        List of shot metadata dictionaries
    """
    if output_dir is None:
        output_dir = get_path('shots_dir')
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames
    logging.info(f"Processing video: {video_path}")
    frames = extract_frames_from_video(video_path)
    
    if not frames:
        logging.error("No frames extracted from video")
        return []
    
    # Detect shot boundaries
    boundaries = detect_shot_boundaries(frames)
    
    # Create shots
    shots = create_shots_from_boundaries(frames, boundaries)
    
    # Process each shot
    shot_metadata = []
    
    for i, shot_frames in enumerate(shots):
        shot_id = f"shot_{i + 1:03d}"
        shot_video_path = output_dir / f"{shot_id}.mp4"
        
        # Save shot video
        success = save_shot_as_video(shot_frames, str(shot_video_path))
        
        if success:
            # Analyze shot content
            analysis = analyze_shot_content(shot_frames)
            
            metadata = {
                'shot_id': shot_id,
                'shot_index': i,
                'video_path': str(shot_video_path),
                'start_frame': boundaries[i] if i < len(boundaries) else 0,
                'end_frame': boundaries[i + 1] if i + 1 < len(boundaries) else len(frames),
                **analysis
            }
            
            shot_metadata.append(metadata)
            logging.info(f"Processed {shot_id}: {analysis['frame_count']} frames")
    
    logging.info(f"Successfully processed {len(shot_metadata)} shots")
    return shot_metadata

# Functional composition for shot processing pipeline
shot_processing_pipeline = compose(
    lambda video_path: extract_frames_from_video(video_path),
    lambda frames: (frames, detect_shot_boundaries(frames)),
    lambda data: create_shots_from_boundaries(data[0], data[1])
)

# Safe versions of functions
safe_extract_frames_from_video = safe_execute(extract_frames_from_video, default=[])
safe_detect_shot_boundaries = safe_execute(detect_shot_boundaries, default=[])
safe_analyze_shot_content = safe_execute(analyze_shot_content, default={})
safe_process_video_to_shots = safe_execute(process_video_to_shots, default=[])
