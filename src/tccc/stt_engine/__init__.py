"""
Speech-to-Text Engine module for TCCC.ai system.

This module provides real-time audio transcription with speaker diarization,
supporting streaming recognition with custom vocabulary and adaptive models.
"""

import os
import logging
import importlib.util
from typing import Dict, Any, Optional, Type

# Export common data structures
from .stt_engine import TranscriptionResult, TranscriptionSegment, Word, TranscriptionConfig, STTEngine

# Setup logger
from tccc.utils.logging import get_logger
logger = get_logger(__name__)

# Engine factory function
def create_stt_engine(engine_type: str = "auto", config: Optional[Dict[str, Any]] = None) -> STTEngine:
    """
    Create an STT engine instance based on the specified type and available implementations.
    
    Args:
        engine_type: Type of STT engine to create 
                    ("auto", "faster-whisper", "whisper", "mock")
        config: Optional configuration dictionary
        
    Returns:
        Initialized STT engine instance
    """
    # Use environment variable if engine_type is "auto"
    if engine_type == "auto":
        if os.environ.get("USE_MOCK_STT", "0") == "1":
            engine_type = "mock"
        elif os.environ.get("USE_FASTER_WHISPER", "1") == "1":
            engine_type = "faster-whisper"
        else:
            engine_type = "whisper"
    
    # Initialize empty config if none provided
    if config is None:
        config = {}
    
    # Create engine based on type
    if engine_type == "mock":
        # Mock implementation
        from .mock_stt import MockSTTEngine
        engine = MockSTTEngine()
        logger.info("Using Mock STT Engine")
        
    elif engine_type == "faster-whisper":
        # Check if faster-whisper is available
        faster_whisper_available = importlib.util.find_spec("faster_whisper") is not None
        
        if faster_whisper_available:
            try:
                # Use our updated Faster Whisper implementation
                from .faster_whisper_stt import FasterWhisperSTT
                
                # We'll need to adapt the FasterWhisperSTT to match our STT Engine interface
                class FasterWhisperSTTEngine(STTEngine):
                    """Adapter for FasterWhisperSTT that implements the STTEngine interface"""
                    
                    def __init__(self):
                        """Initialize with default settings"""
                        super().__init__()
                        self.faster_whisper = None
                    
                    def initialize(self, config: Dict[str, Any]) -> bool:
                        """Initialize using FasterWhisperSTT"""
                        self.config = config
                        self.faster_whisper = FasterWhisperSTT(config)
                        success = self.faster_whisper.initialize()
                        self.initialized = success
                        return success
                    
                    def transcribe_segment(self, audio, metadata=None):
                        """Use FasterWhisperSTT for transcription"""
                        if not self.initialized:
                            return {'error': 'STT Engine not initialized', 'text': ''}
                        
                        # Create transcription config from metadata
                        config = TranscriptionConfig()
                        if metadata:
                            if 'word_timestamps' in metadata:
                                config.word_timestamps = metadata['word_timestamps']
                        
                        # Transcribe using FasterWhisperSTT
                        result = self.faster_whisper.transcribe(audio, config)
                        
                        # Convert to dictionary
                        return self._result_to_dict(result)
                    
                    def _result_to_dict(self, result: TranscriptionResult) -> Dict[str, Any]:
                        """Convert TranscriptionResult to dictionary"""
                        segments = []
                        for segment in result.segments:
                            seg_dict = {
                                'text': segment.text,
                                'start_time': segment.start_time,
                                'end_time': segment.end_time,
                                'confidence': segment.confidence
                            }
                            
                            if segment.speaker is not None:
                                seg_dict['speaker'] = segment.speaker
                            
                            if segment.words:
                                seg_dict['words'] = [
                                    {
                                        'text': word.text,
                                        'start_time': word.start_time,
                                        'end_time': word.end_time,
                                        'confidence': word.confidence,
                                        'speaker': word.speaker
                                    }
                                    for word in segment.words
                                ]
                            
                            segments.append(seg_dict)
                        
                        return {
                            'text': result.text,
                            'segments': segments,
                            'is_partial': result.is_partial,
                            'language': result.language
                        }
                    
                    def get_status(self):
                        """Get status from FasterWhisperSTT"""
                        if not self.initialized or not self.faster_whisper:
                            return {'initialized': False}
                        
                        return self.faster_whisper.get_status()
                    
                    def shutdown(self):
                        """Shutdown FasterWhisperSTT"""
                        if self.faster_whisper:
                            return self.faster_whisper.shutdown()
                        return True
                
                engine = FasterWhisperSTTEngine()
                logger.info("Using Faster Whisper STT Engine")
            except ImportError as e:
                logger.warning(f"Failed to import Faster Whisper STT: {e}")
                # Fall back to standard implementation
                engine = STTEngine()
                logger.info("Falling back to standard STT Engine")
        else:
            logger.warning("Faster Whisper not available, falling back to standard implementation")
            engine = STTEngine()
            logger.info("Using standard STT Engine")
    
    else:
        # Default to standard Whisper implementation
        engine = STTEngine()
        logger.info("Using standard STT Engine")
    
    return engine

# Export factory function and classes
__all__ = [
    'STTEngine', 'TranscriptionResult', 'TranscriptionSegment', 
    'Word', 'TranscriptionConfig', 'create_stt_engine'
]