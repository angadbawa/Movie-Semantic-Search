import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from src import (
    process_video_to_shots,
    process_shots_to_scenes,
    setup_logging,
    get_config,
    get_path,
    pipe
)

def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Movie Semantic Search - Scene Segmentation and Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["full", "shots-only", "scenes-only"],
        default="full",
        help="Processing mode: full pipeline, shots only, or scenes only"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory (uses config default if not specified)"
    )
    
    parser.add_argument(
        "--shots-dir",
        type=str,
        help="Directory containing shot metadata (for scenes-only mode)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of frames to process"
    )
    
    return parser

def process_shots_mode(video_path: str, output_dir: Optional[str] = None) -> list:
    """Process video into shots only."""
    logging.info("=== SHOT SEGMENTATION MODE ===")
    
    shots = process_video_to_shots(video_path, output_dir)
    
    if shots:
        logging.info(f"✅ Successfully processed {len(shots)} shots")
        for i, shot in enumerate(shots[:5]):  # Show first 5 shots
            logging.info(f"Shot {i+1}: {shot['frame_count']} frames, "
                        f"{shot['duration_seconds']:.1f}s, "
                        f"{len(shot.get('objects', {}).get('unique_objects', []))} objects")
    else:
        logging.error("❌ No shots were processed")
    
    return shots

def process_scenes_mode(shots_data: list) -> list:
    """Process shots into scenes."""
    logging.info("=== SCENE DETECTION MODE ===")
    
    if not shots_data:
        logging.error("No shot data available for scene detection")
        return []
    
    scenes = process_shots_to_scenes(shots_data)
    
    if scenes:
        logging.info(f"✅ Successfully detected {len(scenes)} scenes")
        for i, scene in enumerate(scenes):
            logging.info(f"Scene {i+1}: {scene['shot_count']} shots, "
                        f"{scene['duration_seconds']:.1f}s, "
                        f"{len(scene['unique_objects'])} unique objects")
    else:
        logging.error("❌ No scenes were detected")
    
    return scenes

def process_full_pipeline(video_path: str, output_dir: Optional[str] = None) -> tuple:
    """Run the complete processing pipeline."""
    logging.info("=== FULL PIPELINE MODE ===")
    
    # Process shots
    shots = process_shots_mode(video_path, output_dir)
    
    if not shots:
        return [], []
    
    # Process scenes
    scenes = process_scenes_mode(shots)
    
    return shots, scenes

def save_results(shots: list, scenes: list, output_dir: Path):
    """Save processing results to files."""
    import json
    
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

def main():
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)
    
    # Validate input video
    video_path = Path(args.video)
    if not video_path.exists():
        logging.error(f"Video file not found: {video_path}")
        sys.exit(1)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = get_path('output_dir')
    
    logging.info(f"Input video: {video_path}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Processing mode: {args.mode}")
    
    try:
        if args.mode == "shots-only":
            shots = process_shots_mode(str(video_path), str(output_dir))
            scenes = []
            
        elif args.mode == "scenes-only":
            if not args.shots_dir:
                logging.error("--shots-dir required for scenes-only mode")
                sys.exit(1)
            
            # Load shots metadata
            shots_file = Path(args.shots_dir) / "shots_metadata.json"
            if not shots_file.exists():
                logging.error(f"Shots metadata not found: {shots_file}")
                sys.exit(1)
            
            import json
            with open(shots_file, 'r') as f:
                shots = json.load(f)
            
            scenes = process_scenes_mode(shots)
            
        else:  # full mode
            shots, scenes = process_full_pipeline(str(video_path), str(output_dir))
        
        # Save results
        save_results(shots, scenes, output_dir)
        
        # Summary
        logging.info("=== PROCESSING COMPLETE ===")
        logging.info(f"📊 Total shots: {len(shots)}")
        logging.info(f"🎬 Total scenes: {len(scenes)}")
        logging.info(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
