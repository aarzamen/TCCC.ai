"""
Utility module for TCCC.ai system.

This module provides common utilities like logging, configuration, and helper functions.
"""

from tccc.utils.logging import get_logger, configure_logging
from tccc.utils.config import Config

try:
    from tccc.utils.config_manager import ConfigManager
except ImportError:
    pass

try:
    from tccc.utils.module_adapter import AudioPipelineAdapter, STTEngineAdapter, ProcessingCoreAdapter
except ImportError:
    pass

try:
    from tccc.utils.audio_data_converter import (
        convert_audio_format,
        standardize_audio_for_stt,
        standardize_audio_for_pipeline,
        normalize_audio,
        get_audio_format_info,
        AUDIO_FORMAT_INT16,
        AUDIO_FORMAT_INT32,
        AUDIO_FORMAT_FLOAT32
    )
except ImportError:
    pass

# Import VAD manager
try:
    from tccc.utils.vad_manager import VADManager, get_vad_manager, VADMode, VADResult
except ImportError:
    pass

__all__ = [
    'get_logger',
    'configure_logging',
    'Config'
]

# Add optional components to __all__ if available
try:
    __all__.append('ConfigManager')
except NameError:
    pass

try:
    __all__.extend([
        'AudioPipelineAdapter', 
        'STTEngineAdapter', 
        'ProcessingCoreAdapter'
    ])
except NameError:
    pass

try:
    __all__.extend([
        'convert_audio_format',
        'standardize_audio_for_stt',
        'standardize_audio_for_pipeline',
        'normalize_audio',
        'get_audio_format_info',
        'AUDIO_FORMAT_INT16',
        'AUDIO_FORMAT_INT32',
        'AUDIO_FORMAT_FLOAT32'
    ])
except NameError:
    pass

try:
    __all__.extend([
        'VADManager',
        'get_vad_manager',
        'VADMode',
        'VADResult'
    ])
except NameError:
    pass