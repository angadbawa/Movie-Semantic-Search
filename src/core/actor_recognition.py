from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import cv2
from .face_recognition import detect_and_encode_faces, compare_face_encodings
from ..utils.config import get_config
from ..utils.helpers import safe_execute
import logging

class ActorDatabase:
    """Database for storing and matching actor face encodings."""
    
    def __init__(self):
        self.actors = {}  # actor_name -> {'encodings': [...], 'metadata': {...}}
        
    def add_actor(self, actor_name: str, face_encodings: List[np.ndarray], metadata: Dict[str, Any] = None):
        """Add an actor to the database with their face encodings."""
        if actor_name not in self.actors:
            self.actors[actor_name] = {'encodings': [], 'metadata': metadata or {}}
        
        self.actors[actor_name]['encodings'].extend(face_encodings)
        logging.info(f"Added {len(face_encodings)} encodings for actor: {actor_name}")
    
    def identify_actor(self, face_encoding: np.ndarray, tolerance: float = 0.6) -> Tuple[Optional[str], float]:
        """
        Identify an actor from a face encoding.
        
        Args:
            face_encoding: Face encoding to match
            tolerance: Distance threshold for matching
            
        Returns:
            Tuple of (actor_name, confidence) or (None, 0.0)
        """
        best_match = None
        best_distance = float('inf')
        
        for actor_name, actor_data in self.actors.items():
            for stored_encoding in actor_data['encodings']:
                distance, is_match = compare_face_encodings(face_encoding, stored_encoding, tolerance)
                
                if is_match and distance < best_distance:
                    best_distance = distance
                    best_match = actor_name
        
        if best_match:
            confidence = 1.0 - (best_distance / tolerance)  # Convert distance to confidence
            return best_match, confidence
        
        return None, 0.0
    
    def get_actor_list(self) -> List[str]:
        """Get list of all actors in the database."""
        return list(self.actors.keys())

# Global actor database instance
_actor_db = ActorDatabase()

def get_actor_database() -> ActorDatabase:
    """Get the global actor database instance."""
    return _actor_db

def initialize_actor_database(actor_data: Dict[str, Dict[str, Any]]):
    """
    Initialize the actor database with known actors.
    
    Args:
        actor_data: Dictionary mapping actor names to their data
                   Format: {actor_name: {'image_paths': [...], 'metadata': {...}}}
    """
    global _actor_db
    _actor_db = ActorDatabase()
    
    for actor_name, data in actor_data.items():
        encodings = []
        
        # Process reference images for this actor
        for image_path in data.get('image_paths', []):
            try:
                image = cv2.imread(image_path)
                faces = detect_and_encode_faces(image)
                
                for face in faces:
                    if 'encoding' in face:
                        encodings.append(face['encoding'])
                        
            except Exception as e:
                logging.warning(f"Failed to process image {image_path} for {actor_name}: {e}")
        
        if encodings:
            _actor_db.add_actor(actor_name, encodings, data.get('metadata', {}))

def identify_actors_in_frame(image: np.ndarray, 
                           actor_db: Optional[ActorDatabase] = None,
                           confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Identify known actors in a frame.
    
    Args:
        image: Input frame
        actor_db: Actor database (uses global if None)
        confidence_threshold: Minimum confidence for identification
        
    Returns:
        List of identified actors with their locations
    """
    if actor_db is None:
        actor_db = get_actor_database()
    
    # Detect faces in the frame
    faces = detect_and_encode_faces(image)
    
    identified_actors = []
    
    for face in faces:
        if 'encoding' not in face:
            continue
        
        # Try to identify this face
        actor_name, confidence = actor_db.identify_actor(face['encoding'])
        
        if actor_name and confidence >= confidence_threshold:
            identified_actors.append({
                'actor_name': actor_name,
                'confidence': confidence,
                'bbox': face['bbox'],
                'center': face['center'],
                'face_data': face
            })
    
    return identified_actors

def analyze_shot_actors(shot_frames: List[np.ndarray], 
                       actor_db: Optional[ActorDatabase] = None,
                       sample_rate: int = 5) -> Dict[str, Any]:
    """
    Analyze actors present throughout a shot.
    
    Args:
        shot_frames: All frames in the shot
        actor_db: Actor database
        sample_rate: Sample every Nth frame
        
    Returns:
        Shot actor analysis
    """
    if not shot_frames:
        return {}
    
    if actor_db is None:
        actor_db = get_actor_database()
    
    # Sample frames for analysis
    sampled_indices = range(0, len(shot_frames), sample_rate)
    
    actor_appearances = {}  # actor_name -> [frame_indices]
    frame_actors = []  # List of actors per frame
    
    for i in sampled_indices:
        frame_actors_data = identify_actors_in_frame(shot_frames[i], actor_db)
        frame_actors.append({
            'frame_index': i,
            'actors': frame_actors_data
        })
        
        # Track actor appearances
        for actor_data in frame_actors_data:
            actor_name = actor_data['actor_name']
            if actor_name not in actor_appearances:
                actor_appearances[actor_name] = []
            actor_appearances[actor_name].append(i)
    
    # Calculate actor statistics
    actor_stats = {}
    for actor_name, frame_indices in actor_appearances.items():
        actor_stats[actor_name] = {
            'appearances': len(frame_indices),
            'frame_indices': frame_indices,
            'presence_ratio': len(frame_indices) / len(sampled_indices),
            'first_appearance': min(frame_indices),
            'last_appearance': max(frame_indices)
        }
    
    # Find co-occurring actors (actors appearing together)
    co_occurrences = {}
    for frame_data in frame_actors:
        actors_in_frame = [actor['actor_name'] for actor in frame_data['actors']]
        
        if len(actors_in_frame) > 1:
            # Create pairs of co-occurring actors
            for i, actor1 in enumerate(actors_in_frame):
                for actor2 in actors_in_frame[i+1:]:
                    pair = tuple(sorted([actor1, actor2]))
                    if pair not in co_occurrences:
                        co_occurrences[pair] = []
                    co_occurrences[pair].append(frame_data['frame_index'])
    
    return {
        'actors_present': list(actor_appearances.keys()),
        'actor_stats': actor_stats,
        'co_occurrences': co_occurrences,
        'frame_by_frame': frame_actors,
        'num_unique_actors': len(actor_appearances),
        'max_actors_per_frame': max(len(frame['actors']) for frame in frame_actors) if frame_actors else 0
    }

def create_actor_interaction_graph(shots_actors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create an interaction graph between actors across all shots.
    
    Args:
        shots_actors: List of actor analyses for each shot
        
    Returns:
        Actor interaction graph data
    """
    # Collect all co-occurrences across shots
    global_co_occurrences = {}
    actor_shot_appearances = {}
    
    for shot_idx, shot_data in enumerate(shots_actors):
        if not shot_data:
            continue
        
        # Track which shots each actor appears in
        for actor in shot_data.get('actors_present', []):
            if actor not in actor_shot_appearances:
                actor_shot_appearances[actor] = []
            actor_shot_appearances[actor].append(shot_idx)
        
        # Aggregate co-occurrences
        shot_co_occurrences = shot_data.get('co_occurrences', {})
        for pair, frame_indices in shot_co_occurrences.items():
            if pair not in global_co_occurrences:
                global_co_occurrences[pair] = {'shots': [], 'total_frames': 0}
            
            global_co_occurrences[pair]['shots'].append(shot_idx)
            global_co_occurrences[pair]['total_frames'] += len(frame_indices)
    
    # Calculate interaction strengths
    interaction_strengths = {}
    for pair, data in global_co_occurrences.items():
        actor1, actor2 = pair
        
        # Calculate interaction strength based on:
        # 1. Number of shots they appear together
        # 2. Total frames they're together
        # 3. Relative to their individual appearances
        
        actor1_shots = len(actor_shot_appearances.get(actor1, []))
        actor2_shots = len(actor_shot_appearances.get(actor2, []))
        shared_shots = len(data['shots'])
        
        # Interaction strength (0 to 1)
        strength = shared_shots / min(actor1_shots, actor2_shots) if min(actor1_shots, actor2_shots) > 0 else 0
        
        interaction_strengths[pair] = {
            'strength': strength,
            'shared_shots': shared_shots,
            'total_frames_together': data['total_frames'],
            'shot_indices': data['shots']
        }
    
    return {
        'actors': list(actor_shot_appearances.keys()),
        'actor_shot_appearances': actor_shot_appearances,
        'interactions': interaction_strengths,
        'total_actors': len(actor_shot_appearances),
        'total_interactions': len(interaction_strengths)
    }

def find_actors_dancing_together(shots_actors: List[Dict[str, Any]], 
                                shots_actions: List[Dict[str, Any]],
                                target_actors: List[str]) -> List[Dict[str, Any]]:
    """
    Find shots where specific actors are dancing together.
    
    Args:
        shots_actors: Actor analysis for each shot
        shots_actions: Action analysis for each shot  
        target_actors: List of actor names to search for
        
    Returns:
        List of shots where target actors are dancing together
    """
    matching_shots = []
    
    for shot_idx, (actor_data, action_data) in enumerate(zip(shots_actors, shots_actions)):
        if not actor_data or not action_data:
            continue
        
        # Check if shot has dancing
        has_dancing = action_data.get('has_dancing', False)
        if not has_dancing:
            continue
        
        # Check if target actors are present
        actors_present = set(actor_data.get('actors_present', []))
        target_actors_set = set(target_actors)
        
        if not target_actors_set.issubset(actors_present):
            continue  # Not all target actors are present
        
        # Check if target actors appear together in frames with dancing
        co_occurrences = actor_data.get('co_occurrences', {})
        dancing_segments = action_data.get('dancing_segments', [])
        
        # Check if any target actor pairs co-occur during dancing segments
        target_pairs = []
        for i, actor1 in enumerate(target_actors):
            for actor2 in target_actors[i+1:]:
                pair = tuple(sorted([actor1, actor2]))
                if pair in co_occurrences:
                    target_pairs.append(pair)
        
        if target_pairs:
            # Verify co-occurrence during dancing segments
            dancing_frames = set()
            for segment in dancing_segments:
                start_frame = segment.get('start_frame', 0)
                end_frame = segment.get('end_frame', 0)
                dancing_frames.update(range(start_frame, end_frame))
            
            # Check if co-occurrences overlap with dancing frames
            overlapping_pairs = []
            for pair in target_pairs:
                pair_frames = set(co_occurrences[pair])
                if pair_frames.intersection(dancing_frames):
                    overlapping_pairs.append(pair)
            
            if overlapping_pairs:
                matching_shots.append({
                    'shot_index': shot_idx,
                    'actors_present': list(actors_present),
                    'target_actors_together': overlapping_pairs,
                    'dancing_confidence': max(seg.get('dancing', {}).get('confidence', 0) 
                                            for seg in dancing_segments),
                    'actor_data': actor_data,
                    'action_data': action_data
                })
    
    return matching_shots

def create_actor_signature(shot_actors: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create actor signature for shot similarity comparison.
    
    Args:
        shot_actors: Shot actor analysis results
        
    Returns:
        Actor signature
    """
    if not shot_actors:
        return {}
    
    return {
        'actors_present': shot_actors.get('actors_present', []),
        'num_actors': shot_actors.get('num_unique_actors', 0),
        'co_occurring_pairs': list(shot_actors.get('co_occurrences', {}).keys()),
        'max_actors_per_frame': shot_actors.get('max_actors_per_frame', 0)
    }

# Global actor database instance
_actor_database = None

def initialize_actor_database(actor_config: Dict[str, Any]) -> None:
    """
    Initialize the global actor database with actor configurations.
    
    Args:
        actor_config: Dictionary mapping actor names to their image paths/encodings
    """
    global _actor_database
    _actor_database = ActorDatabase()
    
    for actor_name, actor_data in actor_config.items():
        if 'image_paths' in actor_data:
            # Load face encodings from image paths
            face_encodings = []
            for image_path in actor_data['image_paths']:
                try:
                    # Load image and extract face encodings
                    image = cv2.imread(image_path)
                    if image is not None:
                        faces = detect_and_encode_faces(image)
                        if faces:
                            face_encodings.extend([face['encoding'] for face in faces])
                except Exception as e:
                    logging.warning(f"Failed to load image {image_path} for actor {actor_name}: {e}")
            
            if face_encodings:
                _actor_database.add_actor(actor_name, face_encodings, actor_data.get('metadata', {}))
        
        elif 'encodings' in actor_data:
            # Use pre-computed encodings
            encodings = [np.array(enc) for enc in actor_data['encodings']]
            _actor_database.add_actor(actor_name, encodings, actor_data.get('metadata', {}))
    
    logging.info(f"Initialized actor database with {len(_actor_database.actors)} actors")

def get_actor_database() -> Optional[ActorDatabase]:
    """Get the global actor database instance."""
    return _actor_database

# Safe versions of functions
safe_identify_actors_in_frame = safe_execute(identify_actors_in_frame, default=[])
safe_analyze_shot_actors = safe_execute(analyze_shot_actors, default={})
safe_find_actors_dancing_together = safe_execute(find_actors_dancing_together, default=[])