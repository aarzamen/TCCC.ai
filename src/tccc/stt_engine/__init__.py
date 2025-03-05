"""
Speech-to-Text Engine module for TCCC.ai system.

This module provides real-time audio transcription with speaker diarization,
supporting streaming recognition with custom vocabulary and adaptive models.
"""

import os
import importlib.util

# Export common data structures
from .stt_engine import TranscriptionResult, TranscriptionSegment, Word, TranscriptionConfig

# Check which STT engine to use
if os.environ.get("USE_MOCK_STT", "0") == "1":
    from .mock_stt import MockSTTEngine as STTEngine
else:
    # First, try to use Nexa AI's faster-whisper implementation
    faster_whisper_available = importlib.util.find_spec("faster_whisper") is not None
    if faster_whisper_available and os.environ.get("USE_FASTER_WHISPER", "1") == "1":
        try:
            from .faster_whisper_stt import FasterWhisperSTT
            # This is a wrapper to maintain the same interface
            from .stt_engine import STTEngine
            # Replace the model manager initialization in STTEngine
            def _init_model_manager_patch(self, config):
                self.model_manager = FasterWhisperSTT(config)
                return self.model_manager.initialize()
            # Patch the method
            STTEngine._initialize_model_manager = _init_model_manager_patch
        except ImportError:
            # Fall back to standard implementation
            from .stt_engine import STTEngine
    else:
        # Use the standard Whisper implementation
        from .stt_engine import STTEngine

__all__ = ['STTEngine', 'TranscriptionResult', 'TranscriptionSegment', 'Word', 'TranscriptionConfig']