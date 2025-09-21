from typing import Dict, Any, List, Optional
import json
import sqlite3
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
from ..utils.config import get_path

class MetadataStorage:
    """Storage system for movie analysis metadata."""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = get_path('output_dir') / 'movie_metadata.db'
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Movies table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS movies (
                    movie_id TEXT PRIMARY KEY,
                    title TEXT,
                    video_path TEXT,
                    analysis_date TIMESTAMP,
                    total_shots INTEGER,
                    total_scenes INTEGER,
                    duration_seconds REAL,
                    metadata_json TEXT
                )
            ''')
            
            # Shots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS shots (
                    shot_id TEXT PRIMARY KEY,
                    movie_id TEXT,
                    shot_index INTEGER,
                    start_frame INTEGER,
                    end_frame INTEGER,
                    duration_seconds REAL,
                    video_path TEXT,
                    representative_frame_path TEXT,
                    analysis_json TEXT,
                    embedding_vector BLOB,
                    FOREIGN KEY (movie_id) REFERENCES movies (movie_id)
                )
            ''')
            
            # Scenes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scenes (
                    scene_id TEXT PRIMARY KEY,
                    movie_id TEXT,
                    scene_index INTEGER,
                    shot_indices TEXT,  -- JSON array of shot indices
                    start_timestamp REAL,
                    end_timestamp REAL,
                    duration_seconds REAL,
                    description TEXT,
                    analysis_json TEXT,
                    embedding_vector BLOB,
                    FOREIGN KEY (movie_id) REFERENCES movies (movie_id)
                )
            ''')
            
            # Actors table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS actors (
                    actor_id TEXT PRIMARY KEY,
                    actor_name TEXT UNIQUE,
                    face_encodings BLOB,  -- Serialized face encodings
                    metadata_json TEXT
                )
            ''')
            
            # Actor appearances table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS actor_appearances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    movie_id TEXT,
                    actor_name TEXT,
                    shot_id TEXT,
                    scene_id TEXT,
                    confidence REAL,
                    bbox_json TEXT,  -- Bounding box coordinates
                    FOREIGN KEY (movie_id) REFERENCES movies (movie_id),
                    FOREIGN KEY (shot_id) REFERENCES shots (shot_id),
                    FOREIGN KEY (scene_id) REFERENCES scenes (scene_id)
                )
            ''')
            
            # Interactions table (for actor interaction graph)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS actor_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    movie_id TEXT,
                    actor1 TEXT,
                    actor2 TEXT,
                    interaction_strength REAL,
                    shared_shots INTEGER,
                    total_frames_together INTEGER,
                    shot_indices_json TEXT,
                    FOREIGN KEY (movie_id) REFERENCES movies (movie_id)
                )
            ''')
            
            # Search index for embeddings (for faster similarity search)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embedding_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    movie_id TEXT,
                    shot_id TEXT,
                    scene_id TEXT,
                    embedding_type TEXT,  -- 'shot' or 'scene'
                    embedding_vector BLOB,
                    metadata_json TEXT,
                    FOREIGN KEY (movie_id) REFERENCES movies (movie_id)
                )
            ''')
            
            conn.commit()
            logging.info(f"Database initialized at: {self.db_path}")
    
    def store_movie_analysis(self, movie_id: str, analysis_data: Dict[str, Any]):
        """Store complete movie analysis in database."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Store movie metadata
            movie_metadata = analysis_data.get('analysis_metadata', {})
            cursor.execute('''
                INSERT OR REPLACE INTO movies 
                (movie_id, video_path, analysis_date, total_shots, total_scenes, 
                 duration_seconds, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                movie_id,
                analysis_data.get('video_path', ''),
                datetime.now().isoformat(),
                movie_metadata.get('total_shots', 0),
                movie_metadata.get('total_scenes', 0),
                sum(shot.get('duration_seconds', 0) for shot in analysis_data.get('shots', [])),
                json.dumps(movie_metadata)
            ))
            
            # Store shots
            for shot in analysis_data.get('shots', []):
                embedding = shot.get('embedding', {}).get('combined_embedding')
                embedding_blob = self._serialize_embedding(embedding) if embedding is not None else None
                
                cursor.execute('''
                    INSERT OR REPLACE INTO shots
                    (shot_id, movie_id, shot_index, start_frame, end_frame,
                     duration_seconds, video_path, representative_frame_path,
                     analysis_json, embedding_vector)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    shot.get('shot_id'),
                    movie_id,
                    shot.get('shot_index', 0),
                    shot.get('start_frame', 0),
                    shot.get('end_frame', 0),
                    shot.get('duration_seconds', 0),
                    shot.get('video_path', ''),
                    shot.get('representative_frame_path', ''),
                    json.dumps(shot.get('multimodal_analysis', {})),
                    embedding_blob
                ))
            
            # Store scenes
            for scene in analysis_data.get('scenes', []):
                cursor.execute('''
                    INSERT OR REPLACE INTO scenes
                    (scene_id, movie_id, scene_index, shot_indices,
                     start_timestamp, end_timestamp, duration_seconds,
                     description, analysis_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    scene.get('scene_id'),
                    movie_id,
                    scene.get('scene_index', 0),
                    json.dumps(scene.get('shot_indices', [])),
                    scene.get('start_timestamp', 0),
                    scene.get('end_timestamp', 0),
                    scene.get('duration_seconds', 0),
                    scene.get('description', ''),
                    json.dumps(scene)
                ))
            
            # Store actor interactions
            interaction_graph = analysis_data.get('interaction_graph', {})
            interactions = interaction_graph.get('interactions', {})
            
            for (actor1, actor2), interaction_data in interactions.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO actor_interactions
                    (movie_id, actor1, actor2, interaction_strength,
                     shared_shots, total_frames_together, shot_indices_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    movie_id,
                    actor1,
                    actor2,
                    interaction_data.get('strength', 0),
                    interaction_data.get('shared_shots', 0),
                    interaction_data.get('total_frames_together', 0),
                    json.dumps(interaction_data.get('shot_indices', []))
                ))
            
            conn.commit()
            logging.info(f"Stored analysis for movie: {movie_id}")
    
    def load_movie_analysis(self, movie_id: str) -> Optional[Dict[str, Any]]:
        """Load complete movie analysis from database."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Load movie metadata
            cursor.execute('SELECT * FROM movies WHERE movie_id = ?', (movie_id,))
            movie_row = cursor.fetchone()
            
            if not movie_row:
                return None
            
            # Load shots
            cursor.execute('SELECT * FROM shots WHERE movie_id = ? ORDER BY shot_index', (movie_id,))
            shot_rows = cursor.fetchall()
            
            shots = []
            for row in shot_rows:
                embedding = self._deserialize_embedding(row[9]) if row[9] else None
                
                shot = {
                    'shot_id': row[0],
                    'shot_index': row[2],
                    'start_frame': row[3],
                    'end_frame': row[4],
                    'duration_seconds': row[5],
                    'video_path': row[6],
                    'representative_frame_path': row[7],
                    'multimodal_analysis': json.loads(row[8]) if row[8] else {},
                    'embedding': {'combined_embedding': embedding} if embedding is not None else {}
                }
                shots.append(shot)
            
            # Load scenes
            cursor.execute('SELECT * FROM scenes WHERE movie_id = ? ORDER BY scene_index', (movie_id,))
            scene_rows = cursor.fetchall()
            
            scenes = []
            for row in scene_rows:
                scene = {
                    'scene_id': row[0],
                    'scene_index': row[2],
                    'shot_indices': json.loads(row[3]) if row[3] else [],
                    'start_timestamp': row[4],
                    'end_timestamp': row[5],
                    'duration_seconds': row[6],
                    'description': row[7],
                    'analysis_data': json.loads(row[8]) if row[8] else {}
                }
                scenes.append(scene)
            
            # Load interaction graph
            cursor.execute('SELECT * FROM actor_interactions WHERE movie_id = ?', (movie_id,))
            interaction_rows = cursor.fetchall()
            
            interactions = {}
            actors = set()
            
            for row in interaction_rows:
                actor1, actor2 = row[2], row[3]
                actors.add(actor1)
                actors.add(actor2)
                
                interactions[(actor1, actor2)] = {
                    'strength': row[4],
                    'shared_shots': row[5],
                    'total_frames_together': row[6],
                    'shot_indices': json.loads(row[7]) if row[7] else []
                }
            
            interaction_graph = {
                'actors': list(actors),
                'interactions': interactions,
                'total_actors': len(actors),
                'total_interactions': len(interactions)
            }
            
            return {
                'movie_id': movie_id,
                'video_path': movie_row[2],
                'shots': shots,
                'scenes': scenes,
                'interaction_graph': interaction_graph,
                'analysis_metadata': json.loads(movie_row[6]) if movie_row[6] else {}
            }
    
    def get_movie_statistics(self, movie_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a movie."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Basic movie stats
            cursor.execute('''
                SELECT total_shots, total_scenes, duration_seconds, analysis_date
                FROM movies WHERE movie_id = ?
            ''', (movie_id,))
            
            movie_row = cursor.fetchone()
            if not movie_row:
                return {}
            
            # Actor statistics
            cursor.execute('''
                SELECT COUNT(DISTINCT actor_name) as unique_actors,
                       COUNT(*) as total_appearances
                FROM actor_appearances WHERE movie_id = ?
            ''', (movie_id,))
            
            actor_stats = cursor.fetchone()
            
            # Scene duration statistics
            cursor.execute('''
                SELECT AVG(duration_seconds) as avg_scene_duration,
                       MIN(duration_seconds) as min_scene_duration,
                       MAX(duration_seconds) as max_scene_duration
                FROM scenes WHERE movie_id = ?
            ''', (movie_id,))
            
            scene_stats = cursor.fetchone()
            
            return {
                'basic_stats': {
                    'total_shots': movie_row[0],
                    'total_scenes': movie_row[1],
                    'total_duration': movie_row[2],
                    'analysis_date': movie_row[3]
                },
                'actor_stats': {
                    'unique_actors': actor_stats[0] if actor_stats else 0,
                    'total_appearances': actor_stats[1] if actor_stats else 0
                },
                'scene_stats': {
                    'avg_duration': scene_stats[0] if scene_stats else 0,
                    'min_duration': scene_stats[1] if scene_stats else 0,
                    'max_duration': scene_stats[2] if scene_stats else 0
                }
            }
    
    def list_all_movies(self) -> List[Dict[str, Any]]:
        """List all movies in the database."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT movie_id, video_path, total_shots, total_scenes,
                       duration_seconds, analysis_date
                FROM movies
                ORDER BY analysis_date DESC
            ''')
            
            rows = cursor.fetchall()
            
            movies = []
            for row in rows:
                movies.append({
                    'movie_id': row[0],
                    'video_path': row[1],
                    'total_shots': row[2],
                    'total_scenes': row[3],
                    'duration_seconds': row[4],
                    'analysis_date': row[5]
                })
            
            return movies
    
    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize numpy array to bytes for database storage."""
        return embedding.tobytes()
    
    def _deserialize_embedding(self, embedding_bytes: bytes) -> np.ndarray:
        """Deserialize bytes back to numpy array."""
        return np.frombuffer(embedding_bytes, dtype=np.float32)