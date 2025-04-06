"""
Faster Whisper implementation for TCCC.ai STT engine.

This implementation uses CTranslate2 for CUDA-optimized inference
with the Whisper model on NVIDIA Jetson platforms.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, List
import torch

logger = logging.getLogger(__name__)

class FasterWhisperProcessor:
    """Adapter for the faster-whisper library with CUDA optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the faster-whisper processor.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        self.model = None
        self.initialized = False
        
        # Extract config parameters
        self.model_size = config.get('model', 'tiny.en')
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.compute_type = config.get('compute_type', 'float16')
        self.cpu_threads = config.get('cpu_threads', 4)
        self.num_workers = config.get('num_workers', 2)
        
        # Optional model path
        self.model_path = config.get('model_path', None)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
    def initialize(self):
        """Initialize the faster-whisper model.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Only import faster_whisper when needed
            from faster_whisper import WhisperModel
            
            logger.info(f"Initializing faster-whisper with model: {self.model_size}, device: {self.device}, compute_type: {self.compute_type}")
            
            # Check CUDA availability if using CUDA
            if self.device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                self.device = 'cpu'
            
            # Create the model
            self.model = WhisperModel(
                model_size_or_path=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.model_path,
                cpu_threads=self.cpu_threads,
                num_workers=self.num_workers
            )
            
            logger.info(f"faster-whisper model '{self.model_size}' loaded successfully on {self.device}")
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize faster-whisper: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def transcribe(self, audio: np.ndarray, language: str = 'en') -> Dict[str, Any]:
        """Transcribe audio using faster-whisper.
        
        Args:
            audio: Audio data as numpy array (float32 in range [-1, 1])
            language: Language code
            
        Returns:
            Dict with transcription results
        """
        if not self.initialized:
            logger.error("Model not initialized. Call initialize() first.")
            return {'text': '', 'segments': []}
        
        try:
            # Ensure audio is in the correct format (float32 in range [-1, 1])
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
                
            # Ensure audio is scaled to [-1, 1]
            if np.abs(audio).max() > 1.0:
                audio = audio / 32768.0  # Assuming 16-bit PCM
            
            # Log audio stats for debugging
            logger.debug(f"Audio stats: min={audio.min():.6f}, max={audio.max():.6f}, mean={np.abs(audio).mean():.6f}, " 
                        f"shape={audio.shape}, dtype={audio.dtype}")
            
            # Run transcription
            segments, info = self.model.transcribe(
                audio=audio,
                language=language,
                task="transcribe",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Convert segments to list for easier processing
            segments_list = list(segments)
            
            # Create result dictionary
            result = {
                'text': ' '.join(seg.text for seg in segments_list),
                'segments': []
            }
            
            # Convert segments to dictionaries
            for i, seg in enumerate(segments_list):
                segment_dict = {
                    'id': i,
                    'text': seg.text,
                    'start': seg.start,
                    'end': seg.end,
                    'confidence': seg.avg_logprob
                }
                result['segments'].append(segment_dict)
            
            # Add metadata if we have it
            result.update({
                'language': info.language,
                'language_probability': info.language_probability
            })
            
            logger.debug(f"Transcription successful: {result['text'][:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'text': '', 'segments': []}
