from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import json
import logging

from .shot_segmentation import process_video_to_shots
from .scene_detection import process_shots_to_scenes
from .object_detection import detect_objects_in_frame, create_object_signature
from .face_recognition import detect_and_encode_faces, create_face_signature
from .action_recognition import analyze_shot_actions, create_action_signature
from .emotion_detection import analyze_shot_emotions, create_emotion_signature
from .actor_recognition import analyze_shot_actors, create_actor_signature, find_actors_dancing_together
from .audio_analysis import analyze_shot_audio, create_dialogue_signature
from .multimodal_embeddings import (
    create_multimodal_embedding, 
    process_natural_language_query,
    search_similar_scenes
)
from ..utils.config import get_config, get_path
from ..utils.helpers import safe_execute, batch_process
from ..processing.metadata_storage import MetadataStorage

class MultimodalMovieAnalyzer:
    """Complete multimodal movie analysis pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config() if config_path is None else config_path
        self.metadata_storage = MetadataStorage()
        
    def analyze_complete_movie(self, video_path: str, movie_id: str) -> Dict[str, Any]:
        """
        Run complete multimodal analysis on a movie.
        
        Args:
            video_path: Path to movie file
            movie_id: Unique identifier for the movie
            
        Returns:
            Complete analysis results
        """
        logging.info(f"Starting complete analysis for movie: {movie_id}")
        
        # Step 1: Shot segmentation
        logging.info("Step 1: Shot segmentation...")
        shots = process_video_to_shots(video_path)
        
        if not shots:
            logging.error("No shots detected")
            return {}
        
        # Step 2: Multimodal analysis of each shot
        logging.info("Step 2: Multimodal shot analysis...")
        enhanced_shots = []
        
        for i, shot in enumerate(shots):
            logging.info(f"Analyzing shot {i+1}/{len(shots)}")
            
            # Get representative frame (middle frame)
            shot_frames = self._load_shot_frames(shot.get('video_path'))
            if not shot_frames:
                continue
                
            representative_frame = shot_frames[len(shot_frames) // 2]
            
            # Analyze all modalities
            enhanced_shot = self._analyze_shot_multimodal(shot, shot_frames, representative_frame)
            enhanced_shots.append(enhanced_shot)
        
        # Step 3: Scene detection with enhanced features
        logging.info("Step 3: Enhanced scene detection...")
        scenes = self._detect_scenes_multimodal(enhanced_shots)
        
        # Step 4: Create actor interaction graph
        logging.info("Step 4: Actor interaction analysis...")
        interaction_graph = self._create_interaction_graph(enhanced_shots)
        
        # Step 5: Generate multimodal embeddings
        logging.info("Step 5: Multimodal embedding generation...")
        embeddings_data = self._generate_embeddings(enhanced_shots, scenes)
        
        # Step 6: Store metadata
        logging.info("Step 6: Storing metadata...")
        complete_analysis = {
            'movie_id': movie_id,
            'video_path': video_path,
            'shots': enhanced_shots,
            'scenes': scenes,
            'interaction_graph': interaction_graph,
            'embeddings': embeddings_data,
            'analysis_metadata': {
                'total_shots': len(enhanced_shots),
                'total_scenes': len(scenes),
                'total_actors': len(interaction_graph.get('actors', [])),
                'processing_complete': True
            }
        }
        
        self.metadata_storage.store_movie_analysis(movie_id, complete_analysis)
        
        logging.info(f"Complete analysis finished for {movie_id}")
        return complete_analysis
    
    def _analyze_shot_multimodal(self, shot: Dict[str, Any], 
                                shot_frames: List[np.ndarray],
                                representative_frame: np.ndarray) -> Dict[str, Any]:
        """Analyze a single shot across all modalities."""
        
        enhanced_shot = shot.copy()
        
        # Object detection (already done in shot segmentation, but enhance)
        objects_signature = create_object_signature(shot.get('objects', {}))
        
        # Face detection and recognition
        faces = detect_and_encode_faces(representative_frame)
        faces_signature = create_face_signature(faces)
        
        # Action recognition
        actions_analysis = analyze_shot_actions(shot_frames)
        actions_signature = create_action_signature(actions_analysis)
        
        # Emotion detection (requires faces)
        emotions_analysis = analyze_shot_emotions(shot_frames, [faces] * len(shot_frames))
        emotions_signature = create_emotion_signature(emotions_analysis)
        
        # Actor recognition
        actors_analysis = analyze_shot_actors(shot_frames)
        actors_signature = create_actor_signature(actors_analysis)
        
        # Audio analysis
        audio_analysis = {}
        if shot.get('video_path'):
            audio_analysis = analyze_shot_audio(shot['video_path'])
        dialogue_signature = create_dialogue_signature(audio_analysis)
        
        # Create multimodal embedding
        multimodal_data = {
            'objects': objects_signature,
            'faces': faces_signature,
            'actions': actions_signature,
            'emotions': emotions_signature,
            'actors': actors_signature,
            'dialogue': dialogue_signature
        }
        
        embedding_data = create_multimodal_embedding(multimodal_data, representative_frame)
        
        # Enhance shot data
        enhanced_shot.update({
            'multimodal_analysis': {
                'objects': objects_signature,
                'faces': faces_signature,
                'actions': actions_analysis,
                'emotions': emotions_analysis,
                'actors': actors_analysis,
                'dialogue': audio_analysis
            },
            'signatures': {
                'objects': objects_signature,
                'faces': faces_signature,
                'actions': actions_signature,
                'emotions': emotions_signature,
                'actors': actors_signature,
                'dialogue': dialogue_signature
            },
            'embedding': embedding_data,
            'representative_frame_path': self._save_representative_frame(representative_frame, shot['shot_id'])
        })
        
        return enhanced_shot
    
    def _detect_scenes_multimodal(self, enhanced_shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect scenes using multimodal similarity."""
        
        # Use enhanced similarity calculation
        scenes = []
        current_scene_shots = [0]  # Start with first shot
        
        similarity_threshold = get_config("scene_detection.similarity_threshold")
        
        for i in range(1, len(enhanced_shots)):
            # Calculate multimodal similarity with recent shots in current scene
            recent_shots = current_scene_shots[-3:]  # Look at last 3 shots
            
            max_similarity = 0.0
            for shot_idx in recent_shots:
                similarity = self._calculate_multimodal_similarity(
                    enhanced_shots[i], enhanced_shots[shot_idx]
                )
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity >= similarity_threshold:
                current_scene_shots.append(i)
            else:
                # End current scene, start new one
                if len(current_scene_shots) >= 2:  # Minimum scene length
                    scenes.append(self._create_scene_from_shots(enhanced_shots, current_scene_shots))
                current_scene_shots = [i]
        
        # Add final scene
        if current_scene_shots:
            scenes.append(self._create_scene_from_shots(enhanced_shots, current_scene_shots))
        
        return scenes
    
    def _calculate_multimodal_similarity(self, shot1: Dict[str, Any], shot2: Dict[str, Any]) -> float:
        """Calculate similarity between two shots using all modalities."""
        
        signatures1 = shot1.get('signatures', {})
        signatures2 = shot2.get('signatures', {})
        
        # Calculate similarity for each modality
        similarities = {}
        
        # Object similarity
        obj_sim = self._calculate_object_similarity(
            signatures1.get('objects', {}), signatures2.get('objects', {})
        )
        similarities['objects'] = obj_sim
        
        # Face similarity  
        face_sim = self._calculate_face_similarity(
            signatures1.get('faces', {}), signatures2.get('faces', {})
        )
        similarities['faces'] = face_sim
        
        # Action similarity
        action_sim = self._calculate_action_similarity(
            signatures1.get('actions', {}), signatures2.get('actions', {})
        )
        similarities['actions'] = action_sim
        
        # Emotion similarity
        emotion_sim = self._calculate_emotion_similarity(
            signatures1.get('emotions', {}), signatures2.get('emotions', {})
        )
        similarities['emotions'] = emotion_sim
        
        # Actor similarity
        actor_sim = self._calculate_actor_similarity(
            signatures1.get('actors', {}), signatures2.get('actors', {})
        )
        similarities['actors'] = actor_sim
        
        # Dialogue similarity
        dialogue_sim = self._calculate_dialogue_similarity(
            signatures1.get('dialogue', {}), signatures2.get('dialogue', {})
        )
        similarities['dialogue'] = dialogue_sim
        
        # Weighted combination
        weights = {
            'objects': 0.2,
            'faces': 0.15,
            'actions': 0.2,
            'emotions': 0.15,
            'actors': 0.2,
            'dialogue': 0.1
        }
        
        weighted_similarity = sum(
            similarities[modality] * weights[modality] 
            for modality in similarities
        )
        
        return weighted_similarity
    
    def search_scenes_by_query(self, movie_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for scenes using natural language query.
        
        Args:
            movie_id: Movie identifier
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            List of matching scenes with timestamps
        """
        # Load movie analysis
        movie_data = self.metadata_storage.load_movie_analysis(movie_id)
        if not movie_data:
            return []
        
        # Process query
        query_data = process_natural_language_query(query)
        
        # Handle specific queries like "Abhishek and Amitabh dancing together"
        if query_data['actors'] and 'dancing' in query_data['actions']:
            return self._search_actors_dancing_together(
                movie_data, query_data['actors'], top_k
            )
        
        # General multimodal search
        scenes = movie_data.get('scenes', [])
        scene_embeddings = []
        scene_metadata = []
        
        for scene in scenes:
            # Get representative embedding (average of shot embeddings)
            shot_embeddings = []
            for shot_idx in scene.get('shot_indices', []):
                if shot_idx < len(movie_data.get('shots', [])):
                    shot = movie_data['shots'][shot_idx]
                    embedding = shot.get('embedding', {}).get('combined_embedding')
                    if embedding is not None:
                        shot_embeddings.append(embedding)
            
            if shot_embeddings:
                scene_embedding = np.mean(shot_embeddings, axis=0)
                scene_embeddings.append(scene_embedding)
                scene_metadata.append(scene)
        
        # Search using query embedding
        query_embedding = query_data['query_embedding']
        results = search_similar_scenes(query_embedding, scene_embeddings, scene_metadata, top_k)
        
        # Add timestamps and additional info
        enhanced_results = []
        for result in results:
            scene = result['metadata']
            enhanced_result = {
                'scene_id': scene.get('scene_id'),
                'similarity_score': result['similarity'],
                'start_timestamp': self._get_scene_start_timestamp(scene, movie_data),
                'end_timestamp': self._get_scene_end_timestamp(scene, movie_data),
                'duration': scene.get('duration_seconds', 0),
                'description': self._generate_scene_description(scene, movie_data),
                'scene_data': scene
            }
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _search_actors_dancing_together(self, movie_data: Dict[str, Any], 
                                      target_actors: List[str], 
                                      top_k: int) -> List[Dict[str, Any]]:
        """Search for specific actors dancing together."""
        
        shots = movie_data.get('shots', [])
        shots_actors = [shot.get('multimodal_analysis', {}).get('actors', {}) for shot in shots]
        shots_actions = [shot.get('multimodal_analysis', {}).get('actions', {}) for shot in shots]
        
        # Find matching shots
        matching_shots = find_actors_dancing_together(shots_actors, shots_actions, target_actors)
        
        # Convert to scene format and add timestamps
        results = []
        for match in matching_shots[:top_k]:
            shot_idx = match['shot_index']
            shot = shots[shot_idx]
            
            result = {
                'scene_id': f"shot_{shot_idx}",
                'similarity_score': match['dancing_confidence'],
                'start_timestamp': self._get_shot_timestamp(shot),
                'end_timestamp': self._get_shot_timestamp(shot) + shot.get('duration_seconds', 0),
                'duration': shot.get('duration_seconds', 0),
                'description': f"{', '.join(target_actors)} dancing together",
                'actors_present': match['actors_present'],
                'dancing_confidence': match['dancing_confidence'],
                'shot_data': shot
            }
            results.append(result)
        
        return results
    
    def _load_shot_frames(self, video_path: str) -> List[np.ndarray]:
        """Load frames from shot video file."""
        import cv2
        
        if not video_path or not Path(video_path).exists():
            return []
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames
    
    def _save_representative_frame(self, frame: np.ndarray, shot_id: str) -> str:
        """Save representative frame for a shot."""
        import cv2
        
        output_dir = get_path('output_dir') / 'representative_frames'
        output_dir.mkdir(exist_ok=True)
        
        frame_path = output_dir / f"{shot_id}_representative.jpg"
        cv2.imwrite(str(frame_path), frame)
        
        return str(frame_path)
    
    def _generate_embeddings(self, enhanced_shots: List[Dict[str, Any]], 
                           scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate and organize all embeddings."""
        
        shot_embeddings = []
        scene_embeddings = []
        
        # Collect shot embeddings
        for shot in enhanced_shots:
            embedding = shot.get('embedding', {}).get('combined_embedding')
            if embedding is not None:
                shot_embeddings.append(embedding)
        
        # Create scene embeddings (average of constituent shots)
        for scene in scenes:
            shot_indices = scene.get('shot_indices', [])
            scene_shot_embeddings = []
            
            for idx in shot_indices:
                if idx < len(shot_embeddings):
                    scene_shot_embeddings.append(shot_embeddings[idx])
            
            if scene_shot_embeddings:
                scene_embedding = np.mean(scene_shot_embeddings, axis=0)
                scene_embeddings.append(scene_embedding)
        
        return {
            'shot_embeddings': shot_embeddings,
            'scene_embeddings': scene_embeddings,
            'embedding_dimension': len(shot_embeddings[0]) if shot_embeddings else 0
        }
    
    def get_movie_statistics(self, movie_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a movie."""
        
        movie_data = self.metadata_storage.load_movie_analysis(movie_id)
        if not movie_data:
            return {}
        
        shots = movie_data.get('shots', [])
        scenes = movie_data.get('scenes', [])
        interaction_graph = movie_data.get('interaction_graph', {})
        
        # Collect statistics
        stats = {
            'basic_stats': {
                'total_shots': len(shots),
                'total_scenes': len(scenes),
                'total_duration': sum(shot.get('duration_seconds', 0) for shot in shots),
                'avg_shot_duration': np.mean([shot.get('duration_seconds', 0) for shot in shots]) if shots else 0,
                'avg_scene_duration': np.mean([scene.get('duration_seconds', 0) for scene in scenes]) if scenes else 0
            },
            'content_stats': self._analyze_content_statistics(shots),
            'actor_stats': self._analyze_actor_statistics(interaction_graph),
            'emotion_stats': self._analyze_emotion_statistics(shots),
            'action_stats': self._analyze_action_statistics(shots)
        }
        
        return stats
    
    def search_scenes_by_query(self, movie_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search scenes in a movie using natural language query.
        
        Args:
            movie_id: Movie identifier
            query: Natural language search query
            top_k: Number of top results to return
            
        Returns:
            List of matching scenes with similarity scores
        """
        logging.info(f"Searching scenes in movie {movie_id} for query: '{query}'")
        
        # Load movie data
        movie_data = self.metadata_storage.load_movie_analysis(movie_id)
        if not movie_data:
            logging.error(f"Movie {movie_id} not found")
            return []
        
        scenes = movie_data.get('scenes', [])
        if not scenes:
            logging.warning(f"No scenes found for movie {movie_id}")
            return []
        
        # Process query to get embedding
        query_embedding = process_natural_language_query(query)
        if query_embedding is None:
            logging.error("Failed to process query")
            return []
        
        # Search for similar scenes
        results = search_similar_scenes(scenes, query_embedding, top_k)
        
        # Enhance results with additional information
        enhanced_results = []
        for result in results:
            scene = result['scene']
            enhanced_result = {
                'scene_id': scene.get('scene_id', 'unknown'),
                'similarity_score': result['similarity_score'],
                'start_timestamp': scene.get('start_timestamp', 0),
                'end_timestamp': scene.get('end_timestamp', 0),
                'duration': scene.get('duration_seconds', 0),
                'description': self._generate_scene_description(scene),
                'shot_count': scene.get('shot_count', 0)
            }
            
            # Add actor information if available
            actors_present = scene.get('signatures', {}).get('actors', {}).get('actors_present', [])
            if actors_present:
                enhanced_result['actors_present'] = actors_present
            
            # Add action information if query relates to dancing
            if 'danc' in query.lower():
                actions = scene.get('signatures', {}).get('actions', {})
                dancing_confidence = actions.get('dancing', {}).get('confidence', 0)
                enhanced_result['dancing_confidence'] = dancing_confidence
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _generate_scene_description(self, scene: Dict[str, Any]) -> str:
        """Generate a human-readable description of a scene."""
        signatures = scene.get('signatures', {})
        
        description_parts = []
        
        # Objects
        objects = signatures.get('objects', {}).get('unique_objects', [])
        if objects:
            description_parts.append(f"Objects: {', '.join(objects[:3])}")
        
        # Actors
        actors = signatures.get('actors', {}).get('actors_present', [])
        if actors:
            description_parts.append(f"Actors: {', '.join(actors[:2])}")
        
        # Actions
        actions = signatures.get('actions', {})
        if actions:
            top_actions = []
            for action, data in actions.items():
                if isinstance(data, dict) and data.get('confidence', 0) > 0.5:
                    top_actions.append(action)
            if top_actions:
                description_parts.append(f"Actions: {', '.join(top_actions[:2])}")
        
        # Emotions
        emotions = signatures.get('emotions', {})
        if emotions:
            dominant_emotion = max(emotions.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
            if dominant_emotion[1] > 0.3:
                description_parts.append(f"Emotion: {dominant_emotion[0]}")
        
        return "; ".join(description_parts) if description_parts else "Scene content"

# Safe versions of key methods
safe_analyze_complete_movie = safe_execute(MultimodalMovieAnalyzer().analyze_complete_movie, default={})
safe_search_scenes_by_query = safe_execute(MultimodalMovieAnalyzer().search_scenes_by_query, default=[])