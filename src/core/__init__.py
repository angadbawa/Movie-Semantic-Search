from .similarity_metrics import calculate_ssim, calculate_histogram_similarity
from .shot_segmentation import process_video_to_shots, detect_shot_boundaries
from .scene_detection import process_shots_to_scenes, detect_scene_boundaries
from .object_detection import detect_objects_in_frame, create_object_signature
from .face_recognition import detect_and_encode_faces, create_face_signature
from .action_recognition import analyze_shot_actions, create_action_signature
from .emotion_detection import analyze_shot_emotions, create_emotion_signature
from .actor_recognition import analyze_shot_actors, create_actor_signature, find_actors_dancing_together
from .audio_analysis import analyze_shot_audio, create_dialogue_signature
from .multimodal_embeddings import create_multimodal_embedding, process_natural_language_query
from .multimodal_pipeline import MultimodalMovieAnalyzer

__all__ = [
    "calculate_ssim",
    "calculate_histogram_similarity", 
    "process_video_to_shots",
    "detect_shot_boundaries",
    "process_shots_to_scenes",
    "detect_scene_boundaries",
    "detect_objects_in_frame",
    "create_object_signature",
    "detect_and_encode_faces",
    "create_face_signature",
    "analyze_shot_actions",
    "create_action_signature",
    "analyze_shot_emotions", 
    "create_emotion_signature",
    "analyze_shot_actors",
    "create_actor_signature",
    "find_actors_dancing_together",
    "analyze_shot_audio",
    "create_dialogue_signature",
    "create_multimodal_embedding",
    "process_natural_language_query",
    "MultimodalMovieAnalyzer"
]
