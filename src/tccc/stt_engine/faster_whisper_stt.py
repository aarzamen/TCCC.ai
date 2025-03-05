"""
Faster Whisper STT implementation for TCCC.ai system.

This module implements the Nexa AI's faster-whisper-5 speech-to-text engine,
optimized for Jetson hardware with improved accuracy and performance.
"""

import os
import time
import torch
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass

from tccc.utils.logging import get_logger
from tccc.stt_engine.stt_engine import TranscriptionResult, TranscriptionSegment, Word, TranscriptionConfig

logger = get_logger(__name__)

class FasterWhisperSTT:
    """
    Implements Nexa AI's faster-whisper-5 for speech recognition.
    
    This implementation provides significantly faster inference speed
    compared to standard Whisper, with optimizations for edge deployment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the faster-whisper STT engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.hardware_config = config.get('hardware', {})
        
        # Extract model settings
        self.model_size = self.model_config.get('size', 'tiny')
        self.model_path = self.model_config.get('path', f'models/faster-whisper-{self.model_size}')
        self.language = self.model_config.get('language', 'en')
        self.beam_size = self.model_config.get('beam_size', 5)
        self.compute_type = self.model_config.get('compute_type', 'float16')
        
        # Extract hardware acceleration settings
        self.enable_acceleration = self.hardware_config.get('enable_acceleration', True)
        self.cuda_device = self.hardware_config.get('cuda_device', 0)
        self.cpu_threads = self.hardware_config.get('cpu_threads', 4)
        
        # Internal state
        self.model = None
        self.initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize the faster-whisper model.
        
        Returns:
            Success status
        """
        try:
            from faster_whisper import WhisperModel
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() and self.enable_acceleration else "cpu"
            compute_type = self.compute_type
            
            # Adjust settings based on device
            if device == "cpu":
                # For CPU, use int8 for better performance
                compute_type = "int8"
            
            logger.info(f"Initializing faster-whisper model '{self.model_size}' on {device} with compute type {compute_type}")
            
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Load the model
            self.model = WhisperModel(
                model_size_or_path=self.model_path,
                device=device,
                compute_type=compute_type,
                download_root=os.path.dirname(self.model_path),
                cpu_threads=self.cpu_threads if device == "cpu" else 0
            )
            
            self.initialized = True
            logger.info(f"Faster-whisper model '{self.model_size}' initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize faster-whisper model: {e}")
            return False
    
    def transcribe(self, audio: np.ndarray, config: Optional[TranscriptionConfig] = None) -> TranscriptionResult:
        """
        Transcribe audio using the faster-whisper model.
        
        Args:
            audio: Audio data as numpy array
            config: Transcription configuration
            
        Returns:
            TranscriptionResult object
        """
        if not self.initialized or self.model is None:
            logger.error("Model not initialized")
            return TranscriptionResult(text="", is_partial=False)
        
        try:
            # Process the config
            if config is None:
                config = TranscriptionConfig()
            
            # Start timing
            start_time = time.time()
            
            # Convert to float32 if needed
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize if needed
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()
            
            # Transcribe with faster-whisper
            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=self.beam_size,
                word_timestamps=config.word_timestamps,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                task="transcribe",
                temperature=0.0,  # Use greedy decoding for deterministic results
            )
            
            # Convert to our result format
            full_text = ""
            result_segments = []
            
            for segment in segments:
                # Get word timestamps if available
                words = []
                if hasattr(segment, "words") and segment.words:
                    for word_data in segment.words:
                        word = Word(
                            text=word_data.word,
                            start_time=word_data.start,
                            end_time=word_data.end,
                            confidence=word_data.probability,
                            speaker=None
                        )
                        words.append(word)
                
                # Create segment
                ts_segment = TranscriptionSegment(
                    text=segment.text,
                    start_time=segment.start,
                    end_time=segment.end,
                    confidence=segment.avg_logprob,
                    words=words
                )
                
                result_segments.append(ts_segment)
                full_text += segment.text + " "
            
            # Create result
            result = TranscriptionResult(
                text=full_text.strip(),
                segments=result_segments,
                is_partial=False,
                language=info.language
            )
            
            # Log performance
            processing_time = time.time() - start_time
            audio_duration = len(audio) / 16000  # Assuming 16kHz
            real_time_factor = processing_time / audio_duration if audio_duration > 0 else 0
            
            logger.debug(f"Transcription completed in {processing_time:.2f}s, RTF: {real_time_factor:.2f}x")
            
            return result
            
        except Exception as e:
            logger.error(f"Faster-whisper transcription error: {e}")
            return TranscriptionResult(text="", is_partial=False)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the model.
        
        Returns:
            Status dictionary
        """
        status = {
            'initialized': self.initialized,
            'model_type': 'faster-whisper',
            'model_size': self.model_size,
            'language': self.language,
            'compute_type': self.compute_type,
            'acceleration': {
                'enabled': self.enable_acceleration,
                'device': "cuda" if torch.cuda.is_available() and self.enable_acceleration else "cpu",
                'cpu_threads': self.cpu_threads
            }
        }
        
        return status