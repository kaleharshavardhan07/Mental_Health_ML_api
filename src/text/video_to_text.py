"""
Video to Text Extraction Module
Extracts text from video files using speech recognition and OCR
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import json


class VideoToTextExtractor:
    """Extracts text from video files using speech recognition"""
    
    def __init__(self, use_gpu: bool = None):
        """
        Initialize the video to text extractor
        
        Args:
            use_gpu: Whether to use GPU for processing (None for auto-detect)
        """
        self.device = "cuda" if use_gpu or (use_gpu is None and self._is_cuda_available()) else "cpu"
        self.temp_dir = Path(tempfile.gettempdir()) / "video_text_extraction"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for required dependencies
        self._check_dependencies()
        
        print(f"VideoToTextExtractor initialized on {self.device}")
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _check_dependencies(self) -> None:
        """Check for required dependencies"""
        try:
            import moviepy.editor
            print("✓ moviepy installed")
        except ImportError:
            print("✗ moviepy not installed. Install with: pip install moviepy")
        
        try:
            import speech_recognition
            print("✓ SpeechRecognition installed")
        except ImportError:
            print("✗ SpeechRecognition not installed. Install with: pip install SpeechRecognition")
        
        try:
            import pydub
            print("✓ pydub installed")
        except ImportError:
            print("✗ pydub not installed. Install with: pip install pydub")
    
    def extract_audio_from_video(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            output_path: Path for output audio file (optional)
            
        Returns:
            Path to extracted audio file
        """
        try:
            from moviepy.editor import VideoFileClip
            
            video = VideoFileClip(video_path)
            
            if output_path is None:
                output_path = str(self.temp_dir / f"audio_{Path(video_path).stem}.wav")
            
            # Extract audio and save as WAV
            if video.audio is not None:
                video.audio.write_audiofile(output_path, verbose=False, logger=None)
            else:
                raise ValueError("Video has no audio track")
            
            video.close()
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio from video: {str(e)}")
    
    def transcribe_audio(self, audio_path: str, language: str = "en-US") -> str:
        """
        Transcribe audio to text using speech recognition
        
        Args:
            audio_path: Path to audio file
            language: Language code for transcription
            
        Returns:
            Transcribed text
        """
        try:
            import speech_recognition as sr
            
            recognizer = sr.Recognizer()
            
            # Load audio file
            with sr.AudioFile(audio_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
            
            # Try Google Speech Recognition first (free, no API key needed)
            try:
                text = recognizer.recognize_google(audio_data, language=language)
                return text
            except sr.UnknownValueError:
                print("Speech recognition could not understand audio")
                return ""
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition: {e}")
                return ""
            
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe audio: {str(e)}")
    
    def extract_text_from_video(self, video_path: str, language: str = "en-US", 
                                keep_temp_files: bool = False) -> Dict[str, any]:
        """
        Extract text from video file
        
        Args:
            video_path: Path to video file
            language: Language code for transcription
            keep_temp_files: Whether to keep temporary files
            
        Returns:
            Dictionary with extracted text and metadata
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        result = {
            "success": True,
            "video_path": str(video_path),
            "text": "",
            "metadata": {
                "video_size_bytes": video_path.stat().st_size,
                "video_name": video_path.name
            }
        }
        
        try:
            # Step 1: Extract audio from video
            print(f"Extracting audio from {video_path.name}...")
            audio_path = self.extract_audio_from_video(str(video_path))
            
            # Step 2: Transcribe audio to text
            print("Transcribing audio to text...")
            text = self.transcribe_audio(audio_path, language=language)
            
            result["text"] = text
            result["metadata"]["audio_path"] = audio_path
            result["metadata"]["text_length"] = len(text)
            result["metadata"]["word_count"] = len(text.split()) if text else 0
            
            # Clean up temporary files
            if not keep_temp_files:
                audio_file = Path(audio_path)
                if audio_file.exists():
                    audio_file.unlink()
                    print(f"Cleaned up temporary audio file: {audio_file.name}")
            
            if not text:
                result["success"] = False
                result["error"] = "No text could be extracted from the video"
            
            return result
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            return result
    
    def extract_text_from_video_batch(self, video_paths: List[str], language: str = "en-US",
                                       keep_temp_files: bool = False) -> List[Dict[str, any]]:
        """
        Extract text from multiple video files
        
        Args:
            video_paths: List of paths to video files
            language: Language code for transcription
            keep_temp_files: Whether to keep temporary files
            
        Returns:
            List of result dictionaries
        """
        results = []
        for video_path in video_paths:
            print(f"\nProcessing: {video_path}")
            result = self.extract_text_from_video(
                video_path, 
                language=language, 
                keep_temp_files=keep_temp_files
            )
            results.append(result)
        return results


# Singleton extractor instance
_extractor = None


def get_extractor(use_gpu: bool = None) -> VideoToTextExtractor:
    """
    Get or create the singleton extractor instance
    
    Args:
        use_gpu: Whether to use GPU for processing
        
    Returns:
        VideoToTextExtractor instance
    """
    global _extractor
    if _extractor is None:
        _extractor = VideoToTextExtractor(use_gpu=use_gpu)
    return _extractor


def extract_text_from_video(video_path: str, language: str = "en-US",
                            keep_temp_files: bool = False) -> Dict[str, any]:
    """
    Convenience function to extract text from video
    
    Args:
        video_path: Path to video file
        language: Language code for transcription
        keep_temp_files: Whether to keep temporary files
        
    Returns:
        Result dictionary with extracted text
    """
    extractor = get_extractor()
    return extractor.extract_text_from_video(
        video_path, 
        language=language, 
        keep_temp_files=keep_temp_files
    )


def extract_text_from_video_batch(video_paths: List[str], language: str = "en-US",
                                   keep_temp_files: bool = False) -> List[Dict[str, any]]:
    """
    Convenience function to extract text from multiple videos
    
    Args:
        video_paths: List of paths to video files
        language: Language code for transcription
        keep_temp_files: Whether to keep temporary files
        
    Returns:
        List of result dictionaries
    """
    extractor = get_extractor()
    return extractor.extract_text_from_video_batch(
        video_paths, 
        language=language, 
        keep_temp_files=keep_temp_files
    )
