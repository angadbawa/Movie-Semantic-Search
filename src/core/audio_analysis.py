from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
from ..utils.config import get_config
from ..utils.helpers import safe_execute, memoize
import logging

try:
    import whisper
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False
    logging.warning("Whisper not available. Install with: pip install openai-whisper")

try:
    from transformers import pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available for text analysis")

@memoize
def load_whisper_model(model_size: str = "base"):
    """Load Whisper model for speech-to-text."""
    if not _WHISPER_AVAILABLE:
        raise ImportError("Whisper not available")
    
    logging.info(f"Loading Whisper model: {model_size}")
    return whisper.load_model(model_size)

@memoize
def load_sentiment_pipeline():
    """Load sentiment analysis pipeline."""
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers not available")
    
    return pipeline("sentiment-analysis", 
                   model="cardiffnlp/twitter-roberta-base-sentiment-latest")

@memoize
def load_summarization_pipeline():
    """Load text summarization pipeline."""
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers not available")
    
    return pipeline("summarization", 
                   model="facebook/bart-large-cnn")

def extract_audio_from_video(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio from video file.
    
    Args:
        video_path: Path to video file
        output_path: Output audio file path
        
    Returns:
        Path to extracted audio file
    """
    try:
        import moviepy.editor as mp
        
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = f"{video_name}_audio.wav"
        
        # Load video and extract audio
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        
        # Write audio file
        audio.write_audiofile(output_path, verbose=False, logger=None)
        
        # Clean up
        audio.close()
        video.close()
        
        logging.info(f"Extracted audio to: {output_path}")
        return output_path
        
    except Exception as e:
        logging.error(f"Audio extraction failed: {e}")
        return ""

def transcribe_audio(audio_path: str, model=None) -> Dict[str, Any]:
    """
    Transcribe audio to text using Whisper.
    
    Args:
        audio_path: Path to audio file
        model: Whisper model
        
    Returns:
        Transcription results with timestamps
    """
    if not _WHISPER_AVAILABLE:
        return {'text': '', 'segments': []}
    
    if model is None:
        model = load_whisper_model()
    
    try:
        # Transcribe with word-level timestamps
        result = model.transcribe(audio_path, word_timestamps=True)
        
        return {
            'text': result['text'],
            'language': result['language'],
            'segments': result['segments']
        }
        
    except Exception as e:
        logging.error(f"Audio transcription failed: {e}")
        return {'text': '', 'segments': []}

def analyze_dialogue_sentiment(text: str, sentiment_pipeline=None) -> Dict[str, Any]:
    """
    Analyze sentiment of dialogue text.
    
    Args:
        text: Dialogue text
        sentiment_pipeline: Sentiment analysis pipeline
        
    Returns:
        Sentiment analysis results
    """
    if not _TRANSFORMERS_AVAILABLE or not text.strip():
        return {'sentiment': 'neutral', 'confidence': 0.0}
    
    if sentiment_pipeline is None:
        sentiment_pipeline = load_sentiment_pipeline()
    
    try:
        # Split text into chunks if too long
        max_length = 512
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        
        sentiments = []
        for chunk in chunks:
            if chunk.strip():
                result = sentiment_pipeline(chunk)[0]
                sentiments.append({
                    'label': result['label'],
                    'score': result['score']
                })
        
        if not sentiments:
            return {'sentiment': 'neutral', 'confidence': 0.0}
        
        # Aggregate sentiments
        avg_score = np.mean([s['score'] for s in sentiments])
        
        # Map labels to standard format
        label_mapping = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral', 
            'LABEL_2': 'positive',
            'NEGATIVE': 'negative',
            'NEUTRAL': 'neutral',
            'POSITIVE': 'positive'
        }
        
        # Use most confident sentiment
        best_sentiment = max(sentiments, key=lambda x: x['score'])
        mapped_label = label_mapping.get(best_sentiment['label'], 'neutral')
        
        return {
            'sentiment': mapped_label,
            'confidence': float(avg_score),
            'all_sentiments': sentiments
        }
        
    except Exception as e:
        logging.error(f"Sentiment analysis failed: {e}")
        return {'sentiment': 'neutral', 'confidence': 0.0}

def summarize_dialogue(text: str, summarizer=None, max_length: int = 100) -> str:
    """
    Summarize dialogue text.
    
    Args:
        text: Dialogue text
        summarizer: Summarization pipeline
        max_length: Maximum summary length
        
    Returns:
        Summary text
    """
    if not _TRANSFORMERS_AVAILABLE or not text.strip():
        return ""
    
    if summarizer is None:
        summarizer = load_summarization_pipeline()
    
    try:
        # Only summarize if text is long enough
        if len(text.split()) < 20:
            return text
        
        # Truncate if too long for model
        max_input_length = 1024
        if len(text) > max_input_length:
            text = text[:max_input_length]
        
        summary = summarizer(text, max_length=max_length, min_length=10, do_sample=False)
        return summary[0]['summary_text']
        
    except Exception as e:
        logging.error(f"Text summarization failed: {e}")
        return text[:200] + "..." if len(text) > 200 else text

def analyze_shot_audio(shot_video_path: str, 
                      start_time: float = 0, 
                      end_time: Optional[float] = None) -> Dict[str, Any]:
    """
    Analyze audio content of a video shot.
    
    Args:
        shot_video_path: Path to shot video file
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        Audio analysis results
    """
    try:
        # Extract audio from shot
        audio_path = extract_audio_from_video(shot_video_path)
        
        if not audio_path:
            return {}
        
        # Transcribe audio
        transcription = transcribe_audio(audio_path)
        
        dialogue_text = transcription.get('text', '')
        
        if not dialogue_text.strip():
            return {
                'has_dialogue': False,
                'transcription': '',
                'sentiment': {'sentiment': 'neutral', 'confidence': 0.0},
                'summary': ''
            }
        
        # Analyze sentiment
        sentiment = analyze_dialogue_sentiment(dialogue_text)
        
        # Create summary
        summary = summarize_dialogue(dialogue_text)
        
        # Clean up temporary audio file
        try:
            Path(audio_path).unlink()
        except:
            pass
        
        return {
            'has_dialogue': True,
            'transcription': dialogue_text,
            'language': transcription.get('language', 'unknown'),
            'segments': transcription.get('segments', []),
            'sentiment': sentiment,
            'summary': summary,
            'dialogue_length': len(dialogue_text.split())
        }
        
    except Exception as e:
        logging.error(f"Shot audio analysis failed: {e}")
        return {}

def create_dialogue_signature(audio_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create dialogue signature for scene similarity.
    
    Args:
        audio_analysis: Audio analysis results
        
    Returns:
        Dialogue signature
    """
    if not audio_analysis:
        return {}
    
    return {
        'has_dialogue': audio_analysis.get('has_dialogue', False),
        'sentiment': audio_analysis.get('sentiment', {}).get('sentiment', 'neutral'),
        'dialogue_length': audio_analysis.get('dialogue_length', 0),
        'language': audio_analysis.get('language', 'unknown'),
        'summary_keywords': extract_keywords_from_summary(audio_analysis.get('summary', ''))
    }

def extract_keywords_from_summary(summary: str) -> List[str]:
    """Extract keywords from dialogue summary."""
    if not summary:
        return []
    
    # Simple keyword extraction (could be enhanced with NLP)
    import re
    
    # Remove common words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]+\b', summary.lower())
    keywords = [word for word in words if len(word) > 3 and word not in stop_words]
    
    # Return unique keywords
    return list(set(keywords))

def calculate_dialogue_similarity(dialogue1: Dict[str, Any], dialogue2: Dict[str, Any]) -> float:
    """
    Calculate similarity between dialogue signatures.
    
    Args:
        dialogue1: First dialogue signature
        dialogue2: Second dialogue signature
        
    Returns:
        Similarity score between 0 and 1
    """
    if not dialogue1 or not dialogue2:
        return 0.0
    
    # Dialogue presence similarity
    has_dialogue1 = dialogue1.get('has_dialogue', False)
    has_dialogue2 = dialogue2.get('has_dialogue', False)
    
    if not has_dialogue1 and not has_dialogue2:
        return 1.0  # Both have no dialogue
    
    if has_dialogue1 != has_dialogue2:
        return 0.0  # One has dialogue, one doesn't
    
    # Sentiment similarity
    sentiment1 = dialogue1.get('sentiment', 'neutral')
    sentiment2 = dialogue2.get('sentiment', 'neutral')
    sentiment_similarity = 1.0 if sentiment1 == sentiment2 else 0.0
    
    # Keyword similarity
    keywords1 = set(dialogue1.get('summary_keywords', []))
    keywords2 = set(dialogue2.get('summary_keywords', []))
    
    if not keywords1 and not keywords2:
        keyword_similarity = 1.0
    elif not keywords1 or not keywords2:
        keyword_similarity = 0.0
    else:
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        keyword_similarity = intersection / union if union > 0 else 0.0
    
    # Language similarity
    lang1 = dialogue1.get('language', 'unknown')
    lang2 = dialogue2.get('language', 'unknown')
    language_similarity = 1.0 if lang1 == lang2 else 0.0
    
    # Weighted combination
    return 0.4 * sentiment_similarity + 0.4 * keyword_similarity + 0.2 * language_similarity

def find_emotional_dialogue_scenes(shots_audio: List[Dict[str, Any]], 
                                 target_emotion: str = "positive") -> List[Dict[str, Any]]:
    """
    Find scenes with specific emotional dialogue.
    
    Args:
        shots_audio: List of audio analyses for shots
        target_emotion: Target emotion to search for
        
    Returns:
        List of shots with matching emotional dialogue
    """
    matching_shots = []
    
    for i, audio_data in enumerate(shots_audio):
        if not audio_data or not audio_data.get('has_dialogue'):
            continue
        
        sentiment_data = audio_data.get('sentiment', {})
        detected_emotion = sentiment_data.get('sentiment', 'neutral')
        confidence = sentiment_data.get('confidence', 0.0)
        
        if detected_emotion == target_emotion and confidence > 0.5:
            matching_shots.append({
                'shot_index': i,
                'emotion': detected_emotion,
                'confidence': confidence,
                'dialogue': audio_data.get('transcription', ''),
                'summary': audio_data.get('summary', ''),
                'audio_data': audio_data
            })
    
    return matching_shots

# Safe versions of functions
safe_extract_audio_from_video = safe_execute(extract_audio_from_video, default="")
safe_transcribe_audio = safe_execute(transcribe_audio, default={'text': '', 'segments': []})
safe_analyze_shot_audio = safe_execute(analyze_shot_audio, default={})