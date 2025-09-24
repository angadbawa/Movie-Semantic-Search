from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from ..utils.config import get_config
from ..utils.helpers import memoize, safe_execute
import logging

@memoize
def load_text_encoder(model_name: str = "all-MiniLM-L6-v2"):
    """Load sentence transformer for text embeddings."""
    logging.info(f"Loading text encoder: {model_name}")
    return SentenceTransformer(model_name)

@memoize  
def load_clip_model():
    """Load CLIP model for multimodal embeddings."""
    try:
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        logging.info(f"Loaded CLIP model on {device}")
        return model, preprocess, device
    except ImportError:
        logging.error("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
        return None, None, None

def create_scene_text_description(shot_data: Dict[str, Any]) -> str:
    """
    Create a natural language description of a scene/shot.
    
    Args:
        shot_data: Combined shot analysis data
        
    Returns:
        Text description of the scene
    """
    description_parts = []
    
    # Objects
    objects = shot_data.get('objects', {}).get('unique_objects', [])
    if objects:
        obj_str = ", ".join(objects[:5])  # Top 5 objects
        description_parts.append(f"Scene contains: {obj_str}")
    
    # Actors
    actors = shot_data.get('actors', {}).get('actors_present', [])
    if actors:
        if len(actors) == 1:
            description_parts.append(f"{actors[0]} is present")
        elif len(actors) == 2:
            description_parts.append(f"{actors[0]} and {actors[1]} are together")
        else:
            description_parts.append(f"{', '.join(actors[:-1])} and {actors[-1]} are present")
    
    # Actions
    actions = shot_data.get('actions', {})
    if actions.get('has_dancing'):
        description_parts.append("dancing is happening")
    
    dominant_action = actions.get('dominant_action', {}).get('action_label')
    if dominant_action and 'danc' not in dominant_action.lower():
        description_parts.append(f"{dominant_action} action is occurring")
    
    # Emotions
    emotions = shot_data.get('emotions', {})
    dominant_emotion = emotions.get('dominant_emotion')
    if dominant_emotion:
        description_parts.append(f"the mood is {dominant_emotion}")
    
    # Faces
    faces = shot_data.get('faces', {})
    face_count = faces.get('face_count', 0)
    if face_count > 0:
        description_parts.append(f"{face_count} people visible")
    
    return ". ".join(description_parts) if description_parts else "Scene with no specific details detected"

def create_visual_embedding(image: np.ndarray, clip_model=None) -> Optional[np.ndarray]:
    """
    Create visual embedding using CLIP.
    
    Args:
        image: Input image
        clip_model: CLIP model tuple (model, preprocess, device)
        
    Returns:
        Visual embedding vector
    """
    if clip_model is None:
        clip_model = load_clip_model()
    
    model, preprocess, device = clip_model
    if model is None:
        return None
    
    try:
        import clip
        from PIL import Image
        
        # Convert numpy to PIL
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        
        # Preprocess and encode
        image_input = preprocess(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten()
        
    except Exception as e:
        logging.error(f"Visual embedding failed: {e}")
        return None

def create_text_embedding(text: str, text_encoder=None) -> np.ndarray:
    """
    Create text embedding using sentence transformer.
    
    Args:
        text: Input text
        text_encoder: Sentence transformer model
        
    Returns:
        Text embedding vector
    """
    if text_encoder is None:
        text_encoder = load_text_encoder()
    
    try:
        embedding = text_encoder.encode(text, convert_to_numpy=True)
        return embedding
    except Exception as e:
        logging.error(f"Text embedding failed: {e}")
        return np.zeros(384)  # Default embedding size

def create_multimodal_embedding(shot_data: Dict[str, Any], 
                               representative_frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Create multimodal embedding combining visual and textual information.
    
    Args:
        shot_data: Combined shot analysis data
        representative_frame: Representative frame from the shot
        
    Returns:
        Multimodal embedding data
    """
    # Create text description
    text_description = create_scene_text_description(shot_data)
    
    # Create text embedding
    text_embedding = create_text_embedding(text_description)
    
    # Create visual embedding if frame is provided
    visual_embedding = None
    if representative_frame is not None:
        visual_embedding = create_visual_embedding(representative_frame)
    
    # Create structured embedding from metadata
    metadata_embedding = create_metadata_embedding(shot_data)
    
    return {
        'text_description': text_description,
        'text_embedding': text_embedding,
        'visual_embedding': visual_embedding,
        'metadata_embedding': metadata_embedding,
        'combined_embedding': combine_embeddings(text_embedding, visual_embedding, metadata_embedding)
    }

def create_metadata_embedding(shot_data: Dict[str, Any]) -> np.ndarray:
    """
    Create structured embedding from shot metadata.
    
    Args:
        shot_data: Shot analysis data
        
    Returns:
        Metadata embedding vector
    """
    # Create feature vector from structured data
    features = []
    
    # Object features (binary presence of common objects)
    common_objects = ['person', 'car', 'building', 'tree', 'chair', 'table', 'phone', 'book']
    objects = set(shot_data.get('objects', {}).get('unique_objects', []))
    for obj in common_objects:
        features.append(1.0 if obj in objects else 0.0)
    
    # Actor count
    num_actors = shot_data.get('actors', {}).get('num_actors', 0)
    features.append(min(num_actors / 5.0, 1.0))  # Normalize to 0-1
    
    # Action features
    actions = shot_data.get('actions', {})
    features.append(1.0 if actions.get('has_dancing') else 0.0)
    
    # Emotion features (one-hot for dominant emotion)
    emotions = ['happy', 'sad', 'angry', 'surprised', 'fear', 'disgust', 'neutral']
    dominant_emotion = shot_data.get('emotions', {}).get('dominant_emotion', '').lower()
    for emotion in emotions:
        features.append(1.0 if emotion == dominant_emotion else 0.0)
    
    # Face count
    face_count = shot_data.get('faces', {}).get('face_count', 0)
    features.append(min(face_count / 10.0, 1.0))  # Normalize to 0-1
    
    return np.array(features, dtype=np.float32)

def combine_embeddings(text_emb: np.ndarray, 
                      visual_emb: Optional[np.ndarray], 
                      metadata_emb: np.ndarray,
                      weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)) -> np.ndarray:
    """
    Combine different embedding modalities.
    
    Args:
        text_emb: Text embedding
        visual_emb: Visual embedding (optional)
        metadata_emb: Metadata embedding
        weights: Weights for (text, visual, metadata)
        
    Returns:
        Combined embedding
    """
    embeddings = []
    used_weights = []
    
    # Add text embedding
    if text_emb is not None and text_emb.size > 0:
        embeddings.append(text_emb * weights[0])
        used_weights.append(weights[0])
    
    # Add visual embedding if available
    if visual_emb is not None and visual_emb.size > 0:
        # Resize to match text embedding if needed
        if len(embeddings) > 0:
            target_size = embeddings[0].shape[0]
            if visual_emb.shape[0] != target_size:
                # Simple resize by truncation or padding
                if visual_emb.shape[0] > target_size:
                    visual_emb = visual_emb[:target_size]
                else:
                    padding = np.zeros(target_size - visual_emb.shape[0])
                    visual_emb = np.concatenate([visual_emb, padding])
        
        embeddings.append(visual_emb * weights[1])
        used_weights.append(weights[1])
    
    # Add metadata embedding
    if metadata_emb is not None and metadata_emb.size > 0:
        # Pad or truncate to match other embeddings
        if len(embeddings) > 0:
            target_size = embeddings[0].shape[0]
            if metadata_emb.shape[0] != target_size:
                if metadata_emb.shape[0] > target_size:
                    metadata_emb = metadata_emb[:target_size]
                else:
                    padding = np.zeros(target_size - metadata_emb.shape[0])
                    metadata_emb = np.concatenate([metadata_emb, padding])
        
        embeddings.append(metadata_emb * weights[2])
        used_weights.append(weights[2])
    
    if not embeddings:
        return np.zeros(384)  # Default size
    
    # Combine embeddings
    combined = sum(embeddings)
    
    # Normalize
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    
    return combined

def search_similar_scenes(scenes_or_query: Any, 
                         query_or_embeddings: Any = None,
                         top_k_or_metadata: Any = None,
                         top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for similar scenes using embedding similarity.
    Supports two calling patterns:
    1. search_similar_scenes(scenes, query_embedding, top_k) - for scene objects
    2. search_similar_scenes(query_embedding, scene_embeddings, scene_metadata, top_k) - for raw embeddings
    """
    # Pattern 1: scenes list with query embedding
    if isinstance(scenes_or_query, list) and len(scenes_or_query) > 0 and isinstance(scenes_or_query[0], dict):
        scenes = scenes_or_query
        query_embedding = query_or_embeddings
        k = top_k_or_metadata if top_k_or_metadata is not None else top_k
        
        similarities = []
        for i, scene in enumerate(scenes):
            scene_embedding = scene.get('embedding', {}).get('combined_embedding')
            if scene_embedding is not None and hasattr(scene_embedding, 'shape'):
                # Calculate cosine similarity
                min_dim = min(query_embedding.shape[0], scene_embedding.shape[0])
                query_norm = query_embedding[:min_dim]
                scene_norm = scene_embedding[:min_dim]
                
                # Normalize vectors
                query_norm = query_norm / (np.linalg.norm(query_norm) + 1e-8)
                scene_norm = scene_norm / (np.linalg.norm(scene_norm) + 1e-8)
                
                similarity = np.dot(query_norm, scene_norm)
            else:
                similarity = 0.0
            
            similarities.append({
                'scene': scene,
                'similarity_score': float(similarity)
            })
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarities[:k]
    
    # Pattern 2: raw embeddings
    else:
        query_embedding = scenes_or_query
        scene_embeddings = query_or_embeddings
        scene_metadata = top_k_or_metadata
        k = top_k
        
        if len(scene_embeddings) != len(scene_metadata):
            raise ValueError("Embeddings and metadata lists must have same length")
        
        similarities = []
        
        for i, scene_emb in enumerate(scene_embeddings):
            # Calculate cosine similarity
            if scene_emb is not None and scene_emb.size > 0:
                # Ensure same dimensionality
                min_dim = min(query_embedding.shape[0], scene_emb.shape[0])
                query_norm = query_embedding[:min_dim]
                scene_norm = scene_emb[:min_dim]
                
                # Normalize vectors
                query_norm = query_norm / (np.linalg.norm(query_norm) + 1e-8)
                scene_norm = scene_norm / (np.linalg.norm(scene_norm) + 1e-8)
                
                similarity = np.dot(query_norm, scene_norm)
            else:
                similarity = 0.0
            
            similarities.append({
                'scene_index': i,
                'similarity': float(similarity),
                'metadata': scene_metadata[i]
            })
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:k]

def process_natural_language_query(query: str) -> Optional[np.ndarray]:
    """
    Process natural language query and return embedding for similarity search.
    
    Args:
        query: Natural language query
        
    Returns:
        Query embedding vector or None if processing fails
    """
    try:
        # Create text embedding for the query
        query_embedding = create_text_embedding(query)
        return query_embedding
    except Exception as e:
        logging.error(f"Failed to process query '{query}': {e}")
        return None

def parse_natural_language_query(query: str) -> Dict[str, Any]:
    """
    Parse natural language query to extract structured search criteria.
    
    Args:
        query: Natural language query
        
    Returns:
        Parsed query criteria with extracted entities
    """
    query_lower = query.lower()
    
    # Extract actors mentioned
    # This is a simple implementation - could be enhanced with NER
    actors = []
    common_actors = ['abhishek', 'amitabh', 'shahrukh', 'salman', 'aamir', 'hrithik', 'ranbir']
    for actor in common_actors:
        if actor in query_lower:
            actors.append(actor)
    
    # Extract actions
    actions = []
    if 'danc' in query_lower:
        actions.append('dancing')
    if 'fight' in query_lower:
        actions.append('fighting')
    if 'run' in query_lower:
        actions.append('running')
    
    # Extract emotions
    emotions = []
    emotion_keywords = ['happy', 'sad', 'angry', 'surprised', 'romantic', 'emotional']
    for emotion in emotion_keywords:
        if emotion in query_lower:
            emotions.append(emotion)
    
    # Extract objects/settings
    objects = []
    if 'car' in query_lower:
        objects.append('car')
    if 'house' in query_lower or 'home' in query_lower:
        objects.append('building')
    
    return {
        'original_query': query,
        'actors': actors,
        'actions': actions,
        'emotions': emotions,
        'objects': objects,
        'query_embedding': create_text_embedding(query)
    }

# Safe versions of functions
safe_create_multimodal_embedding = safe_execute(create_multimodal_embedding, default={})
safe_search_similar_scenes = safe_execute(search_similar_scenes, default=[])
safe_process_natural_language_query = safe_execute(process_natural_language_query, default={})