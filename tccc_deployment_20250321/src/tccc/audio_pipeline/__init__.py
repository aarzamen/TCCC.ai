"""
Audio Pipeline module for TCCC.ai system.

This module handles audio capture, processing, and streaming functionalities.
It provides real-time audio processing with noise reduction, enhancement,
and voice activity detection for optimal speech recognition.
"""

from .audio_pipeline import AudioPipeline, AudioProcessor, AudioSource, StreamBuffer

__all__ = ['AudioPipeline', 'AudioProcessor', 'AudioSource', 'StreamBuffer']