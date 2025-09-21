import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Optional, List

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def setup_basic_imports():
    """Import basic pipeline modules."""
    from src import (
        process_video_to_shots,
        process_shots_to_scenes,
        setup_logging,
        get_config,
        get_path,
        pipe
    )
    return {
        'process_video_to_shots': process_video_to_shots,
        'process_shots_to_scenes': process_shots_to_scenes,
        'setup_logging': setup_logging,
        'get_config': get_config,
        'get_path': get_path,
        'pipe': pipe
    }

def setup_multimodal_imports():
    """Import multimodal pipeline modules."""
    from src.core.multimodal_pipeline import MultimodalMovieAnalyzer
    from src.core.actor_recognition import initialize_actor_database
    from src.processing.metadata_storage import MetadataStorage
    from src.utils.helpers import setup_logging
    from src.utils.config import get_config
    
    return {
        'MultimodalMovieAnalyzer': MultimodalMovieAnalyzer,
        'initialize_actor_database': initialize_actor_database,
        'MetadataStorage': MetadataStorage,
        'setup_logging': setup_logging,
        'get_config': get_config
    }

def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup unified command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Movie Semantic Search - Unified Pipeline System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='pipeline', help='Pipeline type')
    
    # ===== BASIC PIPELINE =====
    basic_parser = subparsers.add_parser('basic', help='Basic 2-modality pipeline (objects + faces)')
    
    basic_parser.add_argument('--video', '-v', required=True, help='Path to input video file')
    basic_parser.add_argument('--mode', '-m', choices=['full', 'shots-only', 'scenes-only'], 
                             default='full', help='Processing mode')
    basic_parser.add_argument('--output-dir', '-o', help='Output directory')
    basic_parser.add_argument('--shots-dir', help='Directory containing shot metadata (for scenes-only mode)')
    basic_parser.add_argument('--max-frames', type=int, help='Maximum number of frames to process')
    
    # ===== MULTIMODAL PIPELINE =====
    multimodal_parser = subparsers.add_parser('multimodal', help='Advanced 6-modality pipeline')
    multimodal_subparsers = multimodal_parser.add_subparsers(dest='command', help='Multimodal commands')
    
    # Analyze command
    analyze_parser = multimodal_subparsers.add_parser('analyze', help='Analyze movie with multimodal pipeline')
    analyze_parser.add_argument('--video', '-v', required=True, help='Path to video file')
    analyze_parser.add_argument('--movie-id', '-m', required=True, help='Unique movie identifier')
    analyze_parser.add_argument('--title', '-t', help='Movie title (optional)')
    
    # Search command
    search_parser = multimodal_subparsers.add_parser('search', help='Search scenes by natural language query')
    search_parser.add_argument('--movie-id', '-m', required=True, help='Movie identifier')
    search_parser.add_argument('--query', '-q', required=True, help='Natural language search query')
    search_parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results to return')
    
    # Initialize actors command
    actors_parser = multimodal_subparsers.add_parser('init-actors', help='Initialize actor database')
    actors_parser.add_argument('--config', '-c', required=True, help='Actor configuration JSON file')
    
    # Statistics command
    stats_parser = multimodal_subparsers.add_parser('stats', help='Get movie statistics')
    stats_parser.add_argument('--movie-id', '-m', required=True, help='Movie identifier')
    
    # List movies command
    list_parser = multimodal_subparsers.add_parser('list', help='List all analyzed movies')
    
    # Database commands
    db_parser = multimodal_subparsers.add_parser('db', help='Database operations')
    db_subparsers = db_parser.add_subparsers(dest='db_command')
    
    # Export database
    export_parser = db_subparsers.add_parser('export', help='Export movie data')
    export_parser.add_argument('--movie-id', '-m', required=True, help='Movie identifier')
    export_parser.add_argument('--output', '-o', required=True, help='Output JSON file')
    
    # ===== GLOBAL OPTIONS =====
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--config-file', help='Path to configuration file')
    
    return parser

# ===== BASIC PIPELINE FUNCTIONS =====

def process_shots_mode(video_path: str, output_dir: Optional[str] = None, modules=None) -> list:
    """Process video into shots only."""
    logging.info("=== BASIC SHOT SEGMENTATION MODE ===")
    
    shots = modules['process_video_to_shots'](video_path, output_dir)
    
    if shots:
        logging.info(f"✅ Successfully processed {len(shots)} shots")
        for i, shot in enumerate(shots[:5]):  # Show first 5 shots
            logging.info(f"Shot {i+1}: {shot['frame_count']} frames, "
                        f"{shot['duration_seconds']:.1f}s, "
                        f"{len(shot.get('objects', {}).get('unique_objects', []))} objects")
    else:
        logging.error("❌ No shots were processed")
    
    return shots

def process_scenes_mode(shots_data: list, modules=None) -> list:
    """Process shots into scenes."""
    logging.info("=== BASIC SCENE DETECTION MODE ===")
    
    if not shots_data:
        logging.error("No shot data available for scene detection")
        return []
    
    scenes = modules['process_shots_to_scenes'](shots_data)
    
    if scenes:
        logging.info(f"✅ Successfully detected {len(scenes)} scenes")
        for i, scene in enumerate(scenes):
            logging.info(f"Scene {i+1}: {scene['shot_count']} shots, "
                        f"{scene['duration_seconds']:.1f}s, "
                        f"{len(scene['unique_objects'])} unique objects")
    else:
        logging.error("❌ No scenes were detected")
    
    return scenes

def process_full_pipeline(video_path: str, output_dir: Optional[str] = None, modules=None) -> tuple:
    """Run the complete basic processing pipeline."""
    logging.info("=== BASIC FULL PIPELINE MODE ===")
    
    # Process shots
    shots = process_shots_mode(video_path, output_dir, modules)
    
    if not shots:
        return [], []
    
    # Process scenes
    scenes = process_scenes_mode(shots, modules)
    
    return shots, scenes

def save_results(shots: list, scenes: list, output_dir: Path):
    """Save processing results to files."""
    # Save shots metadata
    shots_file = output_dir / "shots_metadata.json"
    with open(shots_file, 'w') as f:
        json.dump(shots, f, indent=2, default=str)
    logging.info(f"Saved shots metadata: {shots_file}")
    
    # Save scenes metadata
    if scenes:
        scenes_file = output_dir / "scenes_metadata.json"
        with open(scenes_file, 'w') as f:
            json.dump(scenes, f, indent=2, default=str)
        logging.info(f"Saved scenes metadata: {scenes_file}")

def handle_basic_pipeline(args):
    """Handle basic pipeline commands."""
    modules = setup_basic_imports()
    
    # Validate input video
    video_path = Path(args.video)
    if not video_path.exists():
        logging.error(f"Video file not found: {video_path}")
        return False
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = modules['get_path']('output_dir')
    
    logging.info(f"Input video: {video_path}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Processing mode: {args.mode}")
    
    try:
        if args.mode == "shots-only":
            shots = process_shots_mode(str(video_path), str(output_dir), modules)
            scenes = []
            
        elif args.mode == "scenes-only":
            if not args.shots_dir:
                logging.error("--shots-dir required for scenes-only mode")
                return False
            
            # Load shots metadata
            shots_file = Path(args.shots_dir) / "shots_metadata.json"
            if not shots_file.exists():
                logging.error(f"Shots metadata not found: {shots_file}")
                return False
            
            with open(shots_file, 'r') as f:
                shots = json.load(f)
            
            scenes = process_scenes_mode(shots, modules)
            
        else:  # full mode
            shots, scenes = process_full_pipeline(str(video_path), str(output_dir), modules)
        
        # Save results
        save_results(shots, scenes, output_dir)
        
        # Summary
        logging.info("=== BASIC PROCESSING COMPLETE ===")
        logging.info(f"📊 Total shots: {len(shots)}")
        logging.info(f"🎬 Total scenes: {len(scenes)}")
        logging.info(f"📁 Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        logging.error(f"Basic processing failed: {e}")
        return False

# ===== MULTIMODAL PIPELINE FUNCTIONS =====

def analyze_movie_command(args, modules):
    """Handle multimodal movie analysis command."""
    logging.info(f"Starting multimodal analysis for movie: {args.movie_id}")
    
    # Validate video file
    video_path = Path(args.video)
    if not video_path.exists():
        logging.error(f"Video file not found: {video_path}")
        return False
    
    # Initialize analyzer
    analyzer = modules['MultimodalMovieAnalyzer']()
    
    try:
        # Run complete analysis
        results = analyzer.analyze_complete_movie(str(video_path), args.movie_id)
        
        if results:
            logging.info("✅ Multimodal analysis completed successfully!")
            
            # Print summary
            stats = results.get('analysis_metadata', {})
            print(f"\n📊 Multimodal Analysis Summary:")
            print(f"   Movie ID: {args.movie_id}")
            print(f"   Total Shots: {stats.get('total_shots', 0)}")
            print(f"   Total Scenes: {stats.get('total_scenes', 0)}")
            print(f"   Total Actors: {len(results.get('interaction_graph', {}).get('actors', []))}")
            
            # Show top actors and their interactions
            interaction_graph = results.get('interaction_graph', {})
            if interaction_graph.get('actors'):
                print(f"\n🎭 Detected Actors:")
                for actor in interaction_graph['actors'][:5]:  # Top 5
                    print(f"   - {actor}")
            
            return True
        else:
            logging.error("❌ Multimodal analysis failed")
            return False
            
    except Exception as e:
        logging.error(f"Multimodal analysis failed with error: {e}")
        return False

def search_scenes_command(args, modules):
    """Handle multimodal scene search command."""
    logging.info(f"Searching scenes in movie {args.movie_id} for: '{args.query}'")
    
    # Initialize analyzer
    analyzer = modules['MultimodalMovieAnalyzer']()
    
    try:
        # Search scenes
        results = analyzer.search_scenes_by_query(args.movie_id, args.query, args.top_k)
        
        if results:
            print(f"\n🔍 Multimodal Search Results for: '{args.query}'")
            print(f"Found {len(results)} matching scenes:\n")
            
            for i, result in enumerate(results, 1):
                print(f"{i}. Scene: {result['scene_id']}")
                print(f"   Similarity: {result['similarity_score']:.3f}")
                print(f"   Timestamp: {result['start_timestamp']:.1f}s - {result['end_timestamp']:.1f}s")
                print(f"   Duration: {result['duration']:.1f}s")
                print(f"   Description: {result['description']}")
                
                # Show additional details for actor+dancing queries
                if 'actors_present' in result:
                    print(f"   Actors: {', '.join(result['actors_present'])}")
                if 'dancing_confidence' in result:
                    print(f"   Dancing Confidence: {result['dancing_confidence']:.3f}")
                
                print()
            
            return True
        else:
            print(f"❌ No scenes found matching: '{args.query}'")
            return False
            
    except Exception as e:
        logging.error(f"Multimodal search failed with error: {e}")
        return False

def init_actors_command(args, modules):
    """Handle actor database initialization."""
    logging.info(f"Initializing actor database from: {args.config}")
    
    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"Actor config file not found: {config_path}")
        return False
    
    try:
        # Load actor configuration
        with open(config_path, 'r') as f:
            actor_config = json.load(f)
        
        # Initialize database
        modules['initialize_actor_database'](actor_config)
        
        print(f"✅ Actor database initialized with {len(actor_config)} actors")
        for actor_name in actor_config.keys():
            print(f"   - {actor_name}")
        
        return True
        
    except Exception as e:
        logging.error(f"Actor database initialization failed: {e}")
        return False

def stats_command(args, modules):
    """Handle statistics command."""
    logging.info(f"Getting statistics for movie: {args.movie_id}")
    
    # Initialize storage
    storage = modules['MetadataStorage']()
    
    try:
        stats = storage.get_movie_statistics(args.movie_id)
        
        if stats:
            print(f"\n📊 Statistics for Movie: {args.movie_id}")
            
            # Basic stats
            basic = stats.get('basic_stats', {})
            print(f"\n📹 Basic Information:")
            print(f"   Total Shots: {basic.get('total_shots', 0)}")
            print(f"   Total Scenes: {basic.get('total_scenes', 0)}")
            print(f"   Total Duration: {basic.get('total_duration', 0):.1f} seconds")
            print(f"   Analysis Date: {basic.get('analysis_date', 'Unknown')}")
            
            # Actor stats
            actor = stats.get('actor_stats', {})
            print(f"\n🎭 Actor Information:")
            print(f"   Unique Actors: {actor.get('unique_actors', 0)}")
            print(f"   Total Appearances: {actor.get('total_appearances', 0)}")
            
            # Scene stats
            scene = stats.get('scene_stats', {})
            print(f"\n🎬 Scene Information:")
            print(f"   Average Scene Duration: {scene.get('avg_duration', 0):.1f}s")
            print(f"   Shortest Scene: {scene.get('min_duration', 0):.1f}s")
            print(f"   Longest Scene: {scene.get('max_duration', 0):.1f}s")
            
            return True
        else:
            print(f"❌ No statistics found for movie: {args.movie_id}")
            return False
            
    except Exception as e:
        logging.error(f"Statistics retrieval failed: {e}")
        return False

def list_movies_command(args, modules):
    """Handle list movies command."""
    logging.info("Listing all analyzed movies")
    
    # Initialize storage
    storage = modules['MetadataStorage']()
    
    try:
        movies = storage.list_all_movies()
        
        if movies:
            print(f"\n📚 Analyzed Movies ({len(movies)} total):")
            print()
            
            for movie in movies:
                print(f"🎬 {movie['movie_id']}")
                print(f"   Path: {movie['video_path']}")
                print(f"   Shots: {movie['total_shots']}, Scenes: {movie['total_scenes']}")
                print(f"   Duration: {movie['duration_seconds']:.1f}s")
                print(f"   Analyzed: {movie['analysis_date']}")
                print()
            
            return True
        else:
            print("❌ No movies found in database")
            return False
            
    except Exception as e:
        logging.error(f"Movie listing failed: {e}")
        return False

def export_movie_command(args, modules):
    """Handle movie data export command."""
    logging.info(f"Exporting movie data: {args.movie_id}")
    
    # Initialize storage
    storage = modules['MetadataStorage']()
    
    try:
        movie_data = storage.load_movie_analysis(args.movie_id)
        
        if movie_data:
            # Save to JSON file
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(movie_data, f, indent=2, default=str)
            
            print(f"✅ Movie data exported to: {output_path}")
            return True
        else:
            print(f"❌ Movie not found: {args.movie_id}")
            return False
            
    except Exception as e:
        logging.error(f"Export failed: {e}")
        return False

def handle_multimodal_pipeline(args):
    """Handle multimodal pipeline commands."""
    modules = setup_multimodal_imports()
    
    success = False
    
    if args.command == 'analyze':
        success = analyze_movie_command(args, modules)
    elif args.command == 'search':
        success = search_scenes_command(args, modules)
    elif args.command == 'init-actors':
        success = init_actors_command(args, modules)
    elif args.command == 'stats':
        success = stats_command(args, modules)
    elif args.command == 'list':
        success = list_movies_command(args, modules)
    elif args.command == 'db':
        if args.db_command == 'export':
            success = export_movie_command(args, modules)
        else:
            print("❌ Unknown database command")
    else:
        print("❌ Unknown multimodal command")
    
    return success

def main():
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    
    # Import setup_logging from appropriate module
    if args.pipeline == 'basic':
        modules = setup_basic_imports()
        modules['setup_logging'](level=log_level)
    elif args.pipeline == 'multimodal':
        modules = setup_multimodal_imports()
        modules['setup_logging'](level=log_level)
    else:
        logging.basicConfig(level=log_level)
    
    # Handle pipeline commands
    success = False
    
    if args.pipeline == 'basic':
        success = handle_basic_pipeline(args)
    elif args.pipeline == 'multimodal':
        success = handle_multimodal_pipeline(args)
    else:
        parser.print_help()
        success = False
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()