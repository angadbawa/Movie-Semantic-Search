from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import cv2
from ..utils.config import get_config
from ..utils.helpers import memoize, safe_execute, batch_process
import logging

try:
    from mmaction.apis import inference_recognizer, init_recognizer
    _MMACTION_AVAILABLE = True
except ImportError:
    _MMACTION_AVAILABLE = False
    logging.warning("MMAction2 not available. Install with: pip install mmaction2")

@memoize
def load_mmaction2_model(config_path: Optional[str] = None, checkpoint_path: Optional[str] = None):
    """Load MMAction2 model for action recognition."""
    if not _MMACTION_AVAILABLE:
        raise ImportError("MMAction2 not available")
    
    if config_path is None or checkpoint_path is None:
        config = get_config("action_detection")
        config_path = config_path or config['model_config']
        checkpoint_path = checkpoint_path or config['checkpoint']
    
    logging.info(f"Loading MMAction2 model: {config_path}")
    device = get_config("action_detection.device")
    return init_recognizer(config_path, checkpoint_path, device=device)

def detect_dancing_action(video_frames: List[np.ndarray], 
                         model=None,
                         confidence_threshold: float = 0.3) -> Dict[str, Any]:
    """Specifically detect dancing actions in video frames."""
    if not _MMACTION_AVAILABLE:
        return {'dancing_detected': False, 'confidence': 0.0}
    
    if model is None:
        model = load_mmaction2_model()
    
    try:
        # Run inference on video frames
        results = inference_recognizer(model, video_frames)
        
        # Look for dance-related actions
        dance_keywords = ['dance', 'dancing', 'ballet', 'choreography', 'performance']
        
        dancing_detected = False
        max_confidence = 0.0
        detected_action = None
        
        # Check predictions for dance-related activities
        for i, (label_idx, score) in enumerate(zip(results['pred_labels'], results['pred_scores'])):
            action_label = model.CLASSES[label_idx].lower()
            confidence = float(score)
            
            if any(keyword in action_label for keyword in dance_keywords):
                if confidence >= confidence_threshold and confidence > max_confidence:
                    dancing_detected = True
                    max_confidence = confidence
                    detected_action = {
                        'action_label': model.CLASSES[label_idx],
                        'confidence': confidence,
                        'rank': i + 1
                    }
        
        return {
            'dancing_detected': dancing_detected,
            'confidence': max_confidence,
            'action_details': detected_action
        }
        
    except Exception as e:
        logging.error(f"Action recognition failed: {e}")
        return {'dancing_detected': False, 'confidence': 0.0}

def analyze_shot_actions(shot_frames: List[np.ndarray], 
                        model=None,
                        sample_rate: int = 5) -> Dict[str, Any]:
    """Analyze actions in a complete shot by sampling frames."""
    if not shot_frames:
        return {}
    
    # Sample frames to reduce computation
    sampled_frames = shot_frames[::sample_rate]
    
    # Analyze actions in chunks (MMAction2 typically needs sequences)
    chunk_size = 16  # Typical temporal window for action recognition
    
    dancing_segments = []
    
    for i in range(0, len(sampled_frames), chunk_size):
        chunk = sampled_frames[i:i + chunk_size]
        if len(chunk) < 8:  # Skip if chunk too small
            continue
        
        # Check for dancing
        dance_result = detect_dancing_action(chunk, model)
        
        if dance_result['dancing_detected']:
            segment_info = {
                'start_frame': i * sample_rate,
                'end_frame': min((i + chunk_size) * sample_rate, len(shot_frames)),
                'dancing': dance_result
            }
            dancing_segments.append(segment_info)
    
    return {
        'dancing_segments': dancing_segments,
        'has_dancing': len(dancing_segments) > 0
    }

def create_action_signature(shot_actions: Dict[str, Any]) -> Dict[str, Any]:
    """Create a compact signature for shot actions."""
    if not shot_actions:
        return {}
    
    return {
        'has_dancing': shot_actions.get('has_dancing', False),
        'num_dancing_segments': len(shot_actions.get('dancing_segments', []))
    }

# Safe versions of functions
safe_detect_dancing_action = safe_execute(detect_dancing_action, default={'dancing_detected': False, 'confidence': 0.0})
safe_analyze_shot_actions = safe_execute(analyze_shot_actions, default={})