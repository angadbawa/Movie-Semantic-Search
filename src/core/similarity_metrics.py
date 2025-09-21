from typing import Dict, Tuple
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from ..utils.config import get_shot_config
from ..utils.helpers import safe_execute

def calculate_ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        image_a: First image as numpy array
        image_b: Second image as numpy array
        
    Returns:
        SSIM score between 0 and 1
    """
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    
    score, _ = structural_similarity(gray_a, gray_b, full=True)
    return float(score)

def extract_hsv_histogram(image: np.ndarray, bins: Tuple[int, int, int] = (70, 70, 70)) -> np.ndarray:
    """
    Extract normalized HSV histogram from image.
    
    Args:
        image: Input image as numpy array
        bins: Histogram bins for H, S, V channels
        
    Returns:
        Normalized histogram
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

def calculate_histogram_distance(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """
    Calculate Bhattacharyya distance between histograms of two images.
    
    Args:
        image_a: First image as numpy array
        image_b: Second image as numpy array
        
    Returns:
        Histogram distance (0 = identical, 1 = completely different)
    """
    config = get_shot_config()
    bins = tuple(config['histogram_bins'])
    
    hist_a = extract_hsv_histogram(image_a, bins)
    hist_b = extract_hsv_histogram(image_b, bins)
    
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))

def calculate_similarity_metrics(image_a: np.ndarray, image_b: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive similarity metrics between two images.
    
    Args:
        image_a: First image as numpy array
        image_b: Second image as numpy array
        
    Returns:
        Dictionary containing SSIM and histogram distance
    """
    return {
        'ssim': calculate_ssim(image_a, image_b),
        'histogram_distance': calculate_histogram_distance(image_a, image_b)
    }

def is_shot_boundary(similarity_metrics: Dict[str, float]) -> bool:
    """
    Determine if two frames represent a shot boundary based on similarity metrics.
    
    Args:
        similarity_metrics: Dictionary with 'ssim' and 'histogram_distance' keys
        
    Returns:
        True if frames represent a shot boundary, False otherwise
    """
    config = get_shot_config()
    
    return (similarity_metrics['histogram_distance'] >= config['histogram_threshold'] or 
            similarity_metrics['ssim'] <= config['ssim_threshold'])

def calculate_frame_differences(frames: list) -> list:
    """
    Calculate similarity metrics between consecutive frames.
    
    Args:
        frames: List of frame arrays
        
    Returns:
        List of similarity metrics for each frame pair
    """
    if len(frames) < 2:
        return []
    
    differences = []
    for i in range(len(frames) - 1):
        metrics = calculate_similarity_metrics(frames[i], frames[i + 1])
        differences.append(metrics)
    
    return differences

# Safe versions of functions
safe_calculate_ssim = safe_execute(calculate_ssim, default=0.0)
safe_calculate_histogram_distance = safe_execute(calculate_histogram_distance, default=1.0)
safe_calculate_similarity_metrics = safe_execute(calculate_similarity_metrics, default={'ssim': 0.0, 'histogram_distance': 1.0})
