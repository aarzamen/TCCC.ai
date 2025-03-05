"""
Mock STT Engine implementation for testing without dependencies.
"""

import os
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)

class MockSTTEngine:
    """Mock implementation of the STT Engine for testing."""
    
    def __init__(self):
        """Initialize the mock STT engine."""
        self.initialized = False
        self.config = {}
        self.context = ""
        
        # Recent transcriptions
        self.recent_segments = []
        
        # Performance metrics
        self.metrics = {
            'total_audio_seconds': 0,
            'total_processing_time': 0,
            'transcript_count': 0,
            'error_count': 0,
            'avg_confidence': 0.92,
            'real_time_factor': 0.25
        }
        
        # Sample transcripts to cycle through
        self.sample_transcripts = [
            "The patient has a gunshot wound to the right leg with significant bleeding.",
            "I'm applying a tourniquet above the wound site to control hemorrhage.",
            "Hemorrhage is now controlled. Moving on to airway assessment.",
            "Patient's airway is clear. Breathing is rapid but adequate.",
            "Vital signs are: BP 110/70, pulse 120, respiratory rate 22."
        ]
        self.transcript_index = 0
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the STT engine with configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Success status
        """
        self.config = config
        self.initialized = True
        logger.info("Initialized MockSTTEngine")
        return True
    
    def transcribe_segment(self, audio: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Transcribe audio segment.
        
        Args:
            audio: Audio data as numpy array
            metadata: Additional metadata for transcription
            
        Returns:
            Dictionary with transcription result
        """
        if not self.initialized:
            return {'error': 'STT Engine not initialized', 'text': ''}
        
        # Track performance
        start_time = time.time()
        
        # Get audio duration (assuming 16kHz)
        audio_duration = len(audio) / 16000 if isinstance(audio, np.ndarray) else 1.0
        
        # Get next sample transcript
        transcript = self.sample_transcripts[self.transcript_index]
        self.transcript_index = (self.transcript_index + 1) % len(self.sample_transcripts)
        
        # Create a mock result
        result = {
            'text': transcript,
            'segments': [
                {
                    'text': transcript,
                    'start_time': 0.0,
                    'end_time': audio_duration,
                    'confidence': 0.92,
                    'words': [
                        {
                            'text': word,
                            'start_time': i * 0.5,
                            'end_time': i * 0.5 + 0.4,
                            'confidence': 0.92
                        }
                        for i, word in enumerate(transcript.split())
                    ]
                }
            ],
            'is_partial': False,
            'language': 'en'
        }
        
        # Track performance metrics
        processing_time = time.time() - start_time
        
        self.metrics['total_audio_seconds'] += audio_duration
        self.metrics['total_processing_time'] += processing_time
        self.metrics['transcript_count'] += 1
        
        # Add metrics to result
        result['metrics'] = {
            'audio_duration': audio_duration,
            'processing_time': processing_time,
            'real_time_factor': processing_time / audio_duration if audio_duration > 0 else 0
        }
        
        time.sleep(0.1)  # Simulate processing time
        return result
    
    def update_context(self, context: str) -> bool:
        """
        Update context for improved transcription accuracy.
        
        Args:
            context: Context string
            
        Returns:
            Success status
        """
        self.context = context
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the STT engine.
        
        Returns:
            Status dictionary
        """
        return {
            'initialized': self.initialized,
            'model': {
                'model_type': 'mock_whisper',
                'model_size': 'tiny-en',
                'language': 'en'
            },
            'metrics': self.metrics,
            'diarization': {
                'enabled': False,
                'initialized': False
            },
            'vocabulary': {
                'enabled': True,
                'medical_terms': 119,
                'abbreviations': 38
            }
        }