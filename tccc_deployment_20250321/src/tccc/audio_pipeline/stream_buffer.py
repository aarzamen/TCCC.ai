"""
Enhanced StreamBuffer implementation for TCCC.ai audio pipeline.

This module provides a thread-safe buffer for audio streaming between components,
with support for different audio formats and automatic conversion.
"""

import queue
import numpy as np
import threading
import time
from typing import Optional, Dict, Any

from tccc.utils.audio_data_converter import (
    convert_audio_format,
    AUDIO_FORMAT_INT16,
    AUDIO_FORMAT_FLOAT32
)


class StreamBuffer:
    """
    Enhanced thread-safe buffer for streaming audio data between components.
    Supports different audio formats and automatic conversion.
    """
    
    def __init__(
        self, 
        buffer_size: int = 10,
        timeout_ms: int = 100,
        default_format: str = AUDIO_FORMAT_INT16,
        auto_convert: bool = True
    ):
        """
        Initialize stream buffer.
        
        Args:
            buffer_size: Number of chunks to buffer
            timeout_ms: Timeout for blocking operations in milliseconds
            default_format: Default audio format for data in the buffer
            auto_convert: Whether to automatically convert data to/from the default format
        """
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.timeout = timeout_ms / 1000.0
        self.default_format = default_format
        self.auto_convert = auto_convert
        self.closed = False
        
        # Maps supported formats to numpy dtypes
        self.format_dtypes = {
            AUDIO_FORMAT_INT16: np.int16,
            AUDIO_FORMAT_FLOAT32: np.float32
        }
        
        # Lock for thread-safe operations that can't use Queue directly
        self.lock = threading.RLock()
        
        # Metadata about the audio stream
        self.metadata = {
            "format": default_format,
            "sample_rate": 16000,  # Default sample rate
            "channels": 1,         # Default mono audio
            "chunk_size": 1024     # Default chunk size
        }
    
    def write(self, data: np.ndarray, format_hint: Optional[str] = None) -> int:
        """
        Write data to the buffer with optional format conversion.
        
        Args:
            data: Audio data to write
            format_hint: Format of the input data (if None, will be inferred)
            
        Returns:
            Number of bytes written
        """
        if self.closed:
            return 0
            
        try:
            # Convert data to the default format if auto_convert is enabled
            if self.auto_convert and format_hint is not None and format_hint != self.default_format:
                data = convert_audio_format(data, self.default_format, format_hint)
            
            # Put data in the queue
            self.buffer.put(data, block=True, timeout=self.timeout)
            return len(data.tobytes())
        except queue.Full:
            return 0
    
    def read(self, size: int = -1, target_format: Optional[str] = None, timeout_ms: Optional[int] = None) -> np.ndarray:
        """
        Read data from the buffer with optional format conversion.
        
        Args:
            size: Number of bytes to read (unused, included for compatibility)
            target_format: Target format for the output data
            timeout_ms: Optional timeout in milliseconds (overrides the default timeout)
            
        Returns:
            Audio data or empty array of the appropriate format if no data available
        """
        if self.closed:
            # Return empty array in the appropriate format
            dtype = self.format_dtypes.get(
                target_format or self.default_format, 
                self.format_dtypes[self.default_format]
            )
            return np.array([], dtype=dtype)
            
        try:
            # Use provided timeout or default
            timeout = self.timeout
            if timeout_ms is not None:
                timeout = timeout_ms / 1000.0
                
            # Get data from the queue
            data = self.buffer.get(block=True, timeout=timeout)
            
            # Convert to the requested format if needed
            if self.auto_convert and target_format is not None and target_format != self.default_format:
                data = convert_audio_format(data, target_format, self.default_format)
                
            return data
        except queue.Empty:
            # Return empty array in the appropriate format
            dtype = self.format_dtypes.get(
                target_format or self.default_format, 
                self.format_dtypes[self.default_format]
            )
            return np.array([], dtype=dtype)
    
    def peek(self, timeout_ms: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Peek at the next item in the buffer without removing it.
        
        Args:
            timeout_ms: Timeout for the peek operation in milliseconds
                        (None for no timeout, 0 for non-blocking)
            
        Returns:
            Next audio data item or None if empty
        """
        if self.closed or self.buffer.empty():
            return None
        
        # Use the lock to safely peek
        with self.lock:
            try:
                # If timeout is None, wait indefinitely
                if timeout_ms is None:
                    # Use the buffer's _queue to peek (internal implementation detail)
                    return self.buffer.queue[0] if not self.buffer.empty() else None
                
                # Non-blocking peek
                elif timeout_ms == 0:
                    return self.buffer.queue[0] if not self.buffer.empty() else None
                
                # Blocking peek with timeout
                else:
                    peek_timeout = timeout_ms / 1000.0
                    start_time = time.time()
                    
                    while time.time() - start_time < peek_timeout:
                        if not self.buffer.empty():
                            return self.buffer.queue[0]
                        time.sleep(0.001)  # Small sleep to prevent tight loop
                    
                    return None
            except Exception:
                return None
    
    def clear(self) -> int:
        """
        Clear all items from the buffer.
        
        Returns:
            Number of items cleared
        """
        count = 0
        
        # Use the lock to safely clear
        with self.lock:
            while not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                    count += 1
                except queue.Empty:
                    break
        
        return count
    
    def close(self) -> None:
        """Close the stream buffer."""
        self.closed = True
        self.clear()
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the audio stream.
        
        Returns:
            Dictionary with metadata
        """
        return self.metadata.copy()
    
    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Update metadata about the audio stream.
        
        Args:
            metadata: Dictionary with metadata to update
        """
        self.metadata.update(metadata)
    
    def set_format(self, format_name: str) -> None:
        """
        Set the default audio format for the buffer.
        
        Args:
            format_name: Name of the format (e.g., "int16", "float32")
        """
        if format_name in self.format_dtypes:
            self.default_format = format_name
            self.metadata["format"] = format_name
        else:
            raise ValueError(f"Unsupported audio format: {format_name}")
    
    def is_empty(self) -> bool:
        """
        Check if the buffer is empty.
        
        Returns:
            True if the buffer is empty, False otherwise
        """
        return self.buffer.empty()
    
    def is_full(self) -> bool:
        """
        Check if the buffer is full.
        
        Returns:
            True if the buffer is full, False otherwise
        """
        return self.buffer.full()
    
    def get_size(self) -> int:
        """
        Get the current number of items in the buffer.
        
        Returns:
            Number of items in the buffer
        """
        return self.buffer.qsize()
    
    def get_capacity(self) -> int:
        """
        Get the maximum capacity of the buffer.
        
        Returns:
            Maximum number of items the buffer can hold
        """
        return self.buffer.maxsize