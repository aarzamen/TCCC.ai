"""
Speech-to-Text Engine module for TCCC.ai system.

This module provides real-time audio transcription with speaker diarization,
supporting streaming recognition with custom vocabulary and adaptive models.
"""

import os

# Check for environment variable to use mock implementation
if os.environ.get("USE_MOCK_STT", "0") == "1":
    from .mock_stt import MockSTTEngine as STTEngine
else:
    from .stt_engine import STTEngine, TranscriptionResult, TranscriptionConfig

__all__ = ['STTEngine']