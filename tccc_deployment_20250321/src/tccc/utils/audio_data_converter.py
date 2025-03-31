"""
Audio Data Converter module for resolving data type inconsistencies between components.

This module provides standardized functions to convert audio data between different
numpy data types and formats, ensuring compatibility between the AudioPipeline and
STT Engine components.
"""

import numpy as np
from typing import Tuple, Union, Optional

# Define standard audio formats
AUDIO_FORMAT_INT16 = "int16"    # np.int16 format, typically in range [-32768, 32767]
AUDIO_FORMAT_INT32 = "int32"    # np.int32 format, typically in range [-2^31, 2^31-1]  
AUDIO_FORMAT_FLOAT32 = "float32"  # np.float32 format, typically in range [-1.0, 1.0]

# Define normalization factors for different formats
NORM_FACTORS = {
    AUDIO_FORMAT_INT16: 2**15,  # 32768
    AUDIO_FORMAT_INT32: 2**31,  # 2147483648
    AUDIO_FORMAT_FLOAT32: 1.0   # Float data is already normalized
}


def convert_audio_format(
    audio_data: np.ndarray,
    target_format: str,
    source_format: Optional[str] = None
) -> np.ndarray:
    """
    Convert audio data between different formats with proper normalization.
    
    Args:
        audio_data: Audio data as numpy array
        target_format: Target format (one of "int16", "int32", "float32")
        source_format: Source format (if None, will be inferred from data type)
        
    Returns:
        Converted audio data
    """
    # Infer source format if not provided
    if source_format is None:
        if audio_data.dtype == np.int16:
            source_format = AUDIO_FORMAT_INT16
        elif audio_data.dtype == np.int32:
            source_format = AUDIO_FORMAT_INT32
        elif audio_data.dtype == np.float32:
            source_format = AUDIO_FORMAT_FLOAT32
        else:
            raise ValueError(f"Unsupported audio data type: {audio_data.dtype}")
    
    # No conversion needed if already in target format
    if source_format == target_format:
        return audio_data
    
    # Get corresponding numpy dtypes for source and target formats
    np_dtypes = {
        AUDIO_FORMAT_INT16: np.int16,
        AUDIO_FORMAT_INT32: np.int32,
        AUDIO_FORMAT_FLOAT32: np.float32
    }
    
    # Normalize to float32 [-1.0, 1.0] range first (common intermediate format)
    if source_format in [AUDIO_FORMAT_INT16, AUDIO_FORMAT_INT32]:
        normalized = audio_data.astype(np.float32) / NORM_FACTORS[source_format]
    else:
        normalized = audio_data.astype(np.float32)  # Already normalized
    
    # Convert to target format
    if target_format in [AUDIO_FORMAT_INT16, AUDIO_FORMAT_INT32]:
        # Scale up to the appropriate integer range
        converted = (normalized * NORM_FACTORS[target_format]).astype(np_dtypes[target_format])
    else:
        # Keep as float32
        converted = normalized
    
    return converted


def normalize_audio(audio_data: np.ndarray, target_range: float = 1.0) -> np.ndarray:
    """
    Normalize audio data to a specified peak amplitude.
    
    Args:
        audio_data: Audio data as numpy array
        target_range: Target peak amplitude (default: 1.0)
        
    Returns:
        Normalized audio data
    """
    # Convert to float32 for normalization
    float_audio = audio_data.astype(np.float32)
    
    # Find peak amplitude
    peak = np.abs(float_audio).max()
    
    # Normalize only if peak is non-zero
    if peak > 0:
        float_audio = float_audio * (target_range / peak)
    
    return float_audio


def standardize_audio_for_stt(
    audio_data: np.ndarray,
    original_format: Optional[str] = None
) -> np.ndarray:
    """
    Standardize audio data for STT processing (convert to float32 [-1.0, 1.0]).
    
    Args:
        audio_data: Audio data as numpy array
        original_format: Original audio format (if None, will be inferred)
        
    Returns:
        Standardized audio data in float32 format
    """
    return convert_audio_format(
        audio_data,
        target_format=AUDIO_FORMAT_FLOAT32, 
        source_format=original_format
    )


def standardize_audio_for_pipeline(
    audio_data: np.ndarray, 
    original_format: Optional[str] = None
) -> np.ndarray:
    """
    Standardize audio data for pipeline processing (convert to int16).
    
    Args:
        audio_data: Audio data as numpy array
        original_format: Original audio format (if None, will be inferred)
        
    Returns:
        Standardized audio data in int16 format
    """
    return convert_audio_format(
        audio_data,
        target_format=AUDIO_FORMAT_INT16,
        source_format=original_format
    )


def ensure_audio_size(
    audio_data: np.ndarray,
    expected_size: int,
    mode: str = "pad_or_truncate"
) -> np.ndarray:
    """
    Ensure audio data has the expected size by padding or truncating.
    
    Args:
        audio_data: Audio data as numpy array
        expected_size: Expected size in samples
        mode: Mode for resizing ("pad", "truncate", or "pad_or_truncate")
        
    Returns:
        Resized audio data
    """
    current_size = len(audio_data)
    
    # No resizing needed if already the expected size
    if current_size == expected_size:
        return audio_data
    
    # Determine resize operation based on mode
    if mode == "pad" or (mode == "pad_or_truncate" and current_size < expected_size):
        # Pad with zeros
        padded_audio = np.zeros(expected_size, dtype=audio_data.dtype)
        padded_audio[:current_size] = audio_data
        return padded_audio
    
    elif mode == "truncate" or (mode == "pad_or_truncate" and current_size > expected_size):
        # Truncate to expected size
        return audio_data[:expected_size]
    
    else:
        raise ValueError(f"Unsupported resize mode: {mode}")


def get_audio_format_info(audio_data: np.ndarray) -> dict:
    """
    Get information about audio data format.
    
    Args:
        audio_data: Audio data as numpy array
        
    Returns:
        Dictionary with format information
    """
    info = {
        "dtype": str(audio_data.dtype),
        "shape": audio_data.shape,
        "min": float(audio_data.min()),
        "max": float(audio_data.max()),
        "mean": float(audio_data.mean()),
        "std": float(audio_data.std())
    }
    
    # Determine logical format
    if audio_data.dtype == np.int16:
        info["format"] = AUDIO_FORMAT_INT16
        info["range"] = "[-32768, 32767]"
    elif audio_data.dtype == np.int32:
        info["format"] = AUDIO_FORMAT_INT32
        info["range"] = "[-2147483648, 2147483647]"
    elif audio_data.dtype == np.float32:
        info["format"] = AUDIO_FORMAT_FLOAT32
        
        # Check if properly normalized
        abs_max = np.abs(audio_data).max()
        if abs_max <= 1.0:
            info["range"] = "[-1.0, 1.0]"
            info["normalized"] = True
        else:
            info["range"] = f"[{float(audio_data.min())}, {float(audio_data.max())}]"
            info["normalized"] = False
    else:
        info["format"] = "unknown"
        info["range"] = f"[{float(audio_data.min())}, {float(audio_data.max())}]"
    
    return info