"""
Faster Whisper STT implementation for TCCC.ai system.

This module implements Faster Whisper for speech-to-text conversion,
optimized for Jetson hardware with improved accuracy and performance.
Faster Whisper is a highly optimized implementation of OpenAI's Whisper
model that provides significantly faster inference.
"""

import os
import time
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import deque

from tccc.utils.logging import get_logger
from tccc.stt_engine.stt_engine import TranscriptionResult, TranscriptionSegment, Word, TranscriptionConfig

# Check if faster_whisper is available
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = get_logger(__name__)

class FasterWhisperSTT:
    """
    Implements Faster Whisper for speech recognition.
    
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
        self.model_size = self.model_config.get('size', 'small')
        self.model_path = self.model_config.get('path', f'models/faster-whisper-{self.model_size}')
        self.language = self.model_config.get('language', 'en')
        self.beam_size = self.model_config.get('beam_size', 5)
        self.compute_type = self.model_config.get('compute_type', 'float16')
        self.vad_filter = self.model_config.get('vad_filter', True)
        self.vad_parameters = self.model_config.get('vad_parameters', {
            'threshold': 0.5,
            'min_speech_duration_ms': 250,
            'max_speech_duration_s': 30.0,
            'min_silence_duration_ms': 500
        })
        
        # Medical domain settings
        self.use_medical_vocabulary = self.model_config.get('use_medical_vocabulary', True)
        self.vocabulary_path = self.model_config.get('vocabulary_path', 'config/vocabulary/custom_terms.txt')
        
        # Extract hardware acceleration settings
        self.enable_acceleration = self.hardware_config.get('enable_acceleration', True)
        self.cuda_device = self.hardware_config.get('cuda_device', 0)
        self.cpu_threads = self.hardware_config.get('cpu_threads', 4)
        
        # Internal state
        self.model = None
        self.initialized = False
        self.vocabulary = []
        
        # Recent transcriptions cache
        self.recent_segments = deque(maxlen=10)
        
        # Performance metrics
        self.metrics = {
            'total_audio_seconds': 0,
            'total_processing_time': 0,
            'transcript_count': 0,
            'error_count': 0,
            'avg_rtf': 0
        }
    
    def initialize(self) -> bool:
        """
        Initialize the faster-whisper model.
        
        Returns:
            Success status
        """
        if self.initialized:
            return True
            
        try:
            if not FASTER_WHISPER_AVAILABLE:
                logger.error("faster-whisper is not available. Please install it with 'pip install faster-whisper'")
                return False
                
            # Determine device and compute type
            device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() and self.enable_acceleration else "cpu"
            compute_type = self.compute_type
            
            # Adjust settings based on device
            if device == "cpu" and compute_type == "float16":
                # Fall back to int8 for better CPU performance
                compute_type = "int8"
                logger.info("Using int8 quantization for CPU inference")
            
            # Log initialization
            logger.info(f"Initializing faster-whisper model '{self.model_size}' on {device} with compute type {compute_type}")
            
            # Create model directory if it doesn't exist
            if not os.path.exists(self.model_path) and "/" in self.model_path:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Load medical vocabulary if enabled
            if self.use_medical_vocabulary:
                self._load_vocabulary()
            
            # Load the model
            self.model = WhisperModel(
                model_size_or_path=self.model_size,
                device=device,
                compute_type=compute_type,
                download_root=os.path.dirname(self.model_path) if "/" in self.model_path else None,
                cpu_threads=self.cpu_threads if device == "cpu" else 0
            )
            
            self.initialized = True
            logger.info(f"Faster-whisper model '{self.model_size}' initialized successfully on {device}")
            
            # Perform a warm-up inference
            self._warm_up()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize faster-whisper model: {str(e)}")
            return False
    
    def _load_vocabulary(self) -> bool:
        """
        Load custom medical vocabulary for improved recognition.
        
        Returns:
            Success status
        """
        try:
            if not os.path.exists(self.vocabulary_path):
                logger.warning(f"Vocabulary file not found: {self.vocabulary_path}")
                return False
                
            # Load vocabulary file
            with open(self.vocabulary_path, 'r') as f:
                lines = f.readlines()
                
            # Process each line
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                # Check if line has a replacement format (original -> corrected)
                if '->' in line:
                    original, corrected = [part.strip() for part in line.split('->')]
                    self.vocabulary.append(original)
                    
                # Check if line has an abbreviation format (ABBR = Full Form)
                elif '=' in line:
                    abbr, full_form = [part.strip() for part in line.split('=')]
                    self.vocabulary.append(abbr)
                    self.vocabulary.append(full_form)
                    
                # Otherwise, treat as a simple medical term
                else:
                    term = line.strip()
                    self.vocabulary.append(term)
                    
            logger.info(f"Loaded {len(self.vocabulary)} terms into medical vocabulary")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {str(e)}")
            return False
    
    def _warm_up(self) -> bool:
        """
        Warm up the model with a short dummy audio segment.
        
        Returns:
            Success status
        """
        try:
            # Create a short dummy audio segment (0.5 seconds of silence)
            dummy_audio = np.zeros(8000, dtype=np.float32)
            
            logger.info("Warming up model with dummy inference")
            
            # Perform transcription
            self.transcribe(dummy_audio)
            
            logger.info("Model warm-up completed")
            return True
            
        except Exception as e:
            logger.warning(f"Model warm-up failed: {str(e)}")
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
            
            # Set up transcription parameters
            transcription_params = {
                "language": self.language,
                "beam_size": self.beam_size,
                "word_timestamps": config.word_timestamps,
                "task": "transcribe",
                "temperature": 0.0  # Use greedy decoding for deterministic results
            }
            
            # Add VAD parameters if enabled
            if self.vad_filter:
                transcription_params["vad_filter"] = True
                transcription_params["vad_parameters"] = self.vad_parameters
            
            # Add vocabulary if available
            if self.use_medical_vocabulary and self.vocabulary:
                transcription_params["initial_prompt"] = " ".join(self.vocabulary[:20])  # Use top terms as prompt
            
            # Transcribe with faster-whisper
            segments, info = self.model.transcribe(audio, **transcription_params)
            
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
            
            # Update metrics
            processing_time = time.time() - start_time
            audio_duration = len(audio) / 16000  # Assuming 16kHz
            real_time_factor = processing_time / audio_duration if audio_duration > 0 else 0
            
            self.metrics['total_audio_seconds'] += audio_duration
            self.metrics['total_processing_time'] += processing_time
            self.metrics['transcript_count'] += 1
            
            # Update average RTF
            self.metrics['avg_rtf'] = (
                (self.metrics['avg_rtf'] * (self.metrics['transcript_count'] - 1) + real_time_factor) /
                self.metrics['transcript_count']
            )
            
            # Store segment for context
            self.recent_segments.append(result)
            
            logger.debug(f"Transcription completed in {processing_time:.2f}s, RTF: {real_time_factor:.2f}x")
            
            return result
            
        except Exception as e:
            logger.error(f"Faster-whisper transcription error: {str(e)}")
            self.metrics['error_count'] += 1
            return TranscriptionResult(text="", is_partial=False)
    
    def transcribe_file(self, file_path: str, config: Optional[TranscriptionConfig] = None) -> TranscriptionResult:
        """
        Transcribe audio from a file.
        
        Args:
            file_path: Path to audio file
            config: Transcription configuration
            
        Returns:
            TranscriptionResult object
        """
        try:
            # Load audio file
            import soundfile as sf
            audio, sample_rate = sf.read(file_path)
            
            # Convert to mono if needed
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
            # Transcribe
            return self.transcribe(audio, config)
            
        except Exception as e:
            logger.error(f"Error transcribing file {file_path}: {str(e)}")
            self.metrics['error_count'] += 1
            return TranscriptionResult(text="", is_partial=False)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the model.
        
        Returns:
            Status dictionary
        """
        # Calculate performance metrics
        avg_rtf = self.metrics['avg_rtf']
        total_seconds = self.metrics['total_audio_seconds']
        transcript_count = self.metrics['transcript_count']
        error_count = self.metrics['error_count']
        
        # Create status dictionary
        status = {
            'initialized': self.initialized,
            'model_type': 'faster-whisper',
            'model_size': self.model_size,
            'language': self.language,
            'compute_type': self.compute_type,
            'acceleration': {
                'enabled': self.enable_acceleration,
                'device': "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() and self.enable_acceleration else "cpu",
                'cpu_threads': self.cpu_threads
            },
            'vad_filter': self.vad_filter,
            'vocabulary': {
                'enabled': self.use_medical_vocabulary,
                'terms_count': len(self.vocabulary)
            },
            'performance': {
                'transcript_count': transcript_count,
                'error_count': error_count,
                'total_audio_seconds': total_seconds,
                'realtime_factor': avg_rtf,
                'accuracy_estimate': 1.0 - (error_count / (transcript_count + 1))
            }
        }
        
        return status
        
    def shutdown(self) -> bool:
        """
        Shutdown the STT engine, releasing resources.
        
        Returns:
            Success status
        """
        try:
            # Release model resources
            self.model = None
            self.initialized = False
            
            # Clear caches
            self.recent_segments.clear()
            self.vocabulary.clear()
            
            logger.info("Faster-whisper STT engine shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Error shutting down Faster-whisper STT engine: {str(e)}")
            return False