import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Optional, List

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.core.multimodal_pipeline import MultimodalMovieAnalyzer
from src.core.actor_recognition import initialize_actor_database
from src.processing.metadata_storage import MetadataStorage
from src.utils.helpers import setup_logging
from src.utils.config import get_config

def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Multimodal Movie Semantic Search System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze movie with multimodal pipeline')
    analyze_parser.add_argument('--video', '-v', required=True, help='Path to video file')
    analyze_parser.add_argument('--movie-id', '-m', required=True, help='Unique movie identifier')
    analyze_parser.add_argument('--title', '-t', help='Movie title (optional)')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search scenes by natural language query')
    search_parser.add_argument('--movie-id', '-m', required=True, help='Movie identifier')
    search_parser.add_argument('--query', '-q', required=True, help='Natural language search query')
    search_parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results to return')
    
    # Initialize actors command
    actors_parser = subparsers.add_parser('init-actors', help='Initialize actor database')
    actors_parser.add_argument('--config', '-c', required=True, help='Actor configuration JSON file')
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Get movie statistics')
    stats_parser.add_argument('--movie-id', '-m', required=True, help='Movie identifier')
    
    # List movies command
    list_parser = subparsers.add_parser('list', help='List all analyzed movies')
    
    # Database commands
    db_parser = subparsers.add_parser('db', help='Database operations')
    db_subparsers = db_parser.add_subparsers(dest='db_command')
    
    # Export database
    export_parser = db_subparsers.add_parser('export', help='Export movie data')
    export_parser.add_argument('--movie-id', '-m', required=True, help='Movie identifier')
    export_parser.add_argument('--output', '-o', required=True, help='Output JSON file')
    
    # Global options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--config-file', help='Path to configuration file')
    
    return parser

def analyze_movie_command(args):
    """Handle movie analysis command."""
    logging.info(f"Starting multimodal analysis for movie: {args.movie_id}")
    
    # Validate video file
    video_path = Path(args.video)
    if not video_path.exists():
        logging.error(f"Video file not found: {video_path}")
        return False
    
    # Initialize analyzer
    analyzer = MultimodalMovieAnalyzer()
    
    try:
        # Run complete analysis
        results = analyzer.analyze_complete_movie(str(video_path), args.movie_id)
        
        if results:
            logging.info("✅ Analysis completed successfully!")
            
            # Print summary
            stats = results.get('analysis_metadata', {})
            print(f"\n📊 Analysis Summary:")
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
            logging.error("❌ Analysis failed")
            return False
            
    except Exception as e:
        logging.error(f"Analysis failed with error: {e}")
        return False

def search_scenes_command(args):
    """Handle scene search command."""
    logging.info(f"Searching scenes in movie {args.movie_id} for: '{args.query}'")
    
    # Initialize analyzer
    analyzer = MultimodalMovieAnalyzer()
    
    try:
        # Search scenes
        results = analyzer.search_scenes_by_query(args.movie_id, args.query, args.top_k)
        
        if results:
            print(f"\n🔍 Search Results for: '{args.query}'")
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
        logging.error(f"Search failed with error: {e}")
        return False

def init_actors_command(args):
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
        initialize_actor_database(actor_config)
        
        print(f"✅ Actor database initialized with {len(actor_config)} actors")
        for actor_name in actor_config.keys():
            print(f"   - {actor_name}")
        
        return True
        
    except Exception as e:
        logging.error(f"Actor database initialization failed: {e}")
        return False

def stats_command(args):
    """Handle statistics command."""
    logging.info(f"Getting statistics for movie: {args.movie_id}")
    
    # Initialize storage
    storage = MetadataStorage()
    
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

def list_movies_command(args):
    """Handle list movies command."""
    logging.info("Listing all analyzed movies")
    
    # Initialize storage
    storage = MetadataStorage()
    
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

def export_movie_command(args):
    """Handle movie data export command."""
    logging.info(f"Exporting movie data: {args.movie_id}")
    
    # Initialize storage
    storage = MetadataStorage()
    
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

def main():
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)
    
    # Handle commands
    success = False
    
    if args.command == 'analyze':
        success = analyze_movie_command(args)
    elif args.command == 'search':
        success = search_scenes_command(args)
    elif args.command == 'init-actors':
        success = init_actors_command(args)
    elif args.command == 'stats':
        success = stats_command(args)
    elif args.command == 'list':
        success = list_movies_command(args)
    elif args.command == 'db':
        if args.db_command == 'export':
            success = export_movie_command(args)
        else:
            print("❌ Unknown database command")
    else:
        parser.print_help()
        success = False
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
