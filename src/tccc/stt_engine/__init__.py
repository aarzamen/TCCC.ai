"""
Speech-to-Text Engine module for TCCC.ai system.

This module provides real-time audio transcription with speaker diarization,
supporting streaming recognition with custom vocabulary and adaptive models.
"""

from .stt_engine import STTEngine, TranscriptionResult, TranscriptionConfig

__all__ = ['STTEngine', 'TranscriptionResult', 'TranscriptionConfig']