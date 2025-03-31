"""
Audio Chunk Manager module for handling different chunk sizes between components.

This module provides utilities for audio chunk size management, resampling,
buffering, and segmentation to ensure compatibility between different components
of the TCCC.ai system.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from collections import deque
import logging

from tccc.utils.audio_data_converter import (
    convert_audio_format, 
    ensure_audio_size, 
    AUDIO_FORMAT_INT16,
    AUDIO_FORMAT_FLOAT32
)

logger = logging.getLogger(__name__)

# Common audio chunk sizes used in the system
CHUNK_SIZE_SMALL = 512       # Small chunks for low-latency processing
CHUNK_SIZE_DEFAULT = 1024    # Default audio pipeline chunk size
CHUNK_SIZE_MEDIUM = 2048     # Medium size for some processing steps
CHUNK_SIZE_LARGE = 4096      # Large size for FFT processing
CHUNK_SIZE_SECOND = 16000    # One second of audio at 16kHz
CHUNK_SIZE_STT = 16000       # Default STT processing segment


class AudioChunkBuffer:
    """
    Buffer for accumulating audio chunks to reach a target size.
    Useful for converting between different chunk size requirements.
    """
    
    def __init__(
        self, 
        target_size: int = CHUNK_SIZE_SECOND,
        input_size: int = CHUNK_SIZE_DEFAULT,
        format: str = AUDIO_FORMAT_INT16,
        overlap: bool = False,
        overlap_percent: float = 0.25,
        max_buffer_chunks: int = 30
    ):
        """
        Initialize the audio chunk buffer.
        
        Args:
            target_size: Target chunk size to output
            input_size: Expected input chunk size (for preallocating)
            format: Audio format to maintain in the buffer
            overlap: Whether to use overlapping chunks on output
            overlap_percent: How much to overlap between output chunks (0.0-1.0)
            max_buffer_chunks: Maximum number of chunks to store in the buffer
        """
        self.target_size = target_size
        self.input_size = input_size
        self.format = format
        self.overlap = overlap
        self.overlap_size = int(target_size * overlap_percent) if overlap else 0
        
        # Get numpy dtype corresponding to the format
        if format == AUDIO_FORMAT_INT16:
            self.dtype = np.int16
        elif format == AUDIO_FORMAT_FLOAT32:
            self.dtype = np.float32
        else:
            raise ValueError(f"Unsupported audio format: {format}")
        
        # Initialize buffer storage
        self.buffer = deque(maxlen=max_buffer_chunks) 
        self.accumulated_samples = 0
        
        # Store the overlap buffer
        self.overlap_buffer = np.array([], dtype=self.dtype)
        
        # Initialize buffer for pre-allocation
        self._preallocate_buffer()
    
    def _preallocate_buffer(self):
        """Pre-allocate the combined buffer for efficient processing."""
        # Create an empty array of the target size
        self._combined_buffer = np.zeros(self.target_size * 2, dtype=self.dtype)
        self._combined_size = 0
    
    def add_chunk(self, chunk: np.ndarray) -> None:
        """
        Add a chunk of audio to the buffer.
        
        Args:
            chunk: Audio chunk as numpy array
        """
        # Ensure chunk is in the correct format
        if chunk.dtype != self.dtype:
            chunk = convert_audio_format(chunk, self.format)
        
        # Add to buffer
        self.buffer.append(chunk)
        self.accumulated_samples += len(chunk)
    
    def get_chunk(self) -> Optional[np.ndarray]:
        """
        Get a chunk of audio of the target size if enough data is available.
        
        Returns:
            Audio chunk of target size, or None if not enough data
        """
        # Check if we have enough data
        if self.accumulated_samples < self.target_size:
            return None
        
        # Create combined buffer
        self._combined_size = 0
        
        # First add any overlap from previous chunk
        if len(self.overlap_buffer) > 0:
            overlap_len = len(self.overlap_buffer)
            self._combined_buffer[0:overlap_len] = self.overlap_buffer
            self._combined_size += overlap_len
        
        # Add chunks until we reach or exceed target size
        remaining = self.target_size - self._combined_size
        while remaining > 0 and len(self.buffer) > 0:
            chunk = self.buffer.popleft()
            chunk_len = len(chunk)
            
            # Add chunk to combined buffer
            self._combined_buffer[self._combined_size:self._combined_size + chunk_len] = chunk
            self._combined_size += chunk_len
            self.accumulated_samples -= chunk_len
            
            # Update remaining samples needed
            remaining = self.target_size - self._combined_size
        
        # If we don't have enough data, put everything back
        if self._combined_size < self.target_size:
            # Put back everything we took out
            temp_buffer = self._combined_buffer[0:self._combined_size].copy()
            self.buffer.appendleft(temp_buffer)
            self.accumulated_samples += self._combined_size
            
            # Reset overlap buffer
            self.overlap_buffer = np.array([], dtype=self.dtype)
            
            return None
        
        # Extract the target-sized chunk
        result = self._combined_buffer[0:self.target_size].copy()
        
        # Save overlap for next chunk if needed
        if self.overlap and self.overlap_size > 0:
            start_idx = self.target_size - self.overlap_size
            self.overlap_buffer = result[start_idx:].copy()
        else:
            self.overlap_buffer = np.array([], dtype=self.dtype)
        
        # If we have extra data beyond target size, put it back in buffer
        extra_size = self._combined_size - self.target_size
        if extra_size > 0:
            extra_data = self._combined_buffer[self.target_size:self._combined_size].copy()
            self.buffer.appendleft(extra_data)
            self.accumulated_samples += extra_size
        
        return result
    
    def get_all_chunks(self) -> List[np.ndarray]:
        """
        Get all available complete chunks of target size.
        
        Returns:
            List of audio chunks of target size
        """
        chunks = []
        while True:
            chunk = self.get_chunk()
            if chunk is None:
                break
            chunks.append(chunk)
        return chunks
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.accumulated_samples = 0
        self.overlap_buffer = np.array([], dtype=self.dtype)
        self._preallocate_buffer()
    
    def get_available_samples(self) -> int:
        """
        Get the number of samples available in the buffer.
        
        Returns:
            Number of available samples
        """
        return self.accumulated_samples
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the buffer state.
        
        Returns:
            Dictionary with buffer information
        """
        return {
            "target_size": self.target_size,
            "input_size": self.input_size,
            "format": self.format,
            "dtype": str(self.dtype),
            "overlap": self.overlap,
            "overlap_size": self.overlap_size,
            "accumulated_samples": self.accumulated_samples,
            "buffer_chunks": len(self.buffer),
            "overlap_buffer_size": len(self.overlap_buffer)
        }


class ChunkSizeAdapter:
    """
    Adapter for converting between different chunk sizes.
    Handles efficient conversion for both fixed and variable chunk sizes.
    """
    
    def __init__(
        self,
        source_size: int,
        target_size: int,
        format: str = AUDIO_FORMAT_INT16,
        sample_rate: int = 16000,
        allow_partial: bool = False,
        pad_partial: bool = True
    ):
        """
        Initialize the chunk size adapter.
        
        Args:
            source_size: Source chunk size
            target_size: Target chunk size
            format: Audio format to use
            sample_rate: Sample rate of the audio in Hz
            allow_partial: Whether to allow partial chunks as output
            pad_partial: Whether to pad partial chunks to the target size
        """
        self.source_size = source_size
        self.target_size = target_size
        self.format = format
        self.sample_rate = sample_rate
        self.allow_partial = allow_partial
        self.pad_partial = pad_partial
        
        # Configure chunk buffering based on relative sizes
        if source_size < target_size:
            # Need to accumulate chunks to reach target size
            self.buffer = AudioChunkBuffer(
                target_size=target_size,
                input_size=source_size,
                format=format
            )
            self.direct_conversion = False
        elif source_size > target_size:
            # Need to split chunks into smaller pieces
            self.buffer = []  # No need for buffer, we'll split directly
            self.remainder = np.array([], dtype=np.int16 if format == AUDIO_FORMAT_INT16 else np.float32)
            self.direct_conversion = False
        else:
            # Same size, direct pass-through
            self.direct_conversion = True
    
    def process(self, chunk: np.ndarray) -> List[np.ndarray]:
        """
        Process a chunk and convert to the target size.
        
        Args:
            chunk: Input audio chunk
            
        Returns:
            List of chunks in the target size
        """
        # If sizes match, pass through directly
        if self.direct_conversion:
            # Just ensure correct size and format
            processed_chunk = ensure_audio_size(chunk, self.target_size, mode="pad_or_truncate")
            return [processed_chunk]
        
        # If source is smaller than target, buffer and accumulate
        if self.source_size < self.target_size:
            self.buffer.add_chunk(chunk)
            return self.buffer.get_all_chunks()
        
        # If source is larger than target, split into smaller chunks
        chunks = []
        
        # First, handle any remainder from previous calls
        if len(self.remainder) > 0:
            # Combine remainder with the start of this chunk
            combined = np.concatenate([self.remainder, chunk])
            
            # Reset remainder
            self.remainder = np.array([], dtype=chunk.dtype)
            
            # Use the combined chunk instead
            chunk = combined
        
        # Calculate how many complete target-sized chunks we can get
        num_complete_chunks = len(chunk) // self.target_size
        
        # Extract the complete chunks
        for i in range(num_complete_chunks):
            start = i * self.target_size
            end = start + self.target_size
            chunks.append(chunk[start:end])
        
        # Handle remainder
        remainder_start = num_complete_chunks * self.target_size
        if remainder_start < len(chunk):
            remainder = chunk[remainder_start:]
            
            # If partial chunks are allowed, output the remainder too
            if self.allow_partial:
                if self.pad_partial:
                    # Pad to target size
                    padded = ensure_audio_size(remainder, self.target_size, mode="pad")
                    chunks.append(padded)
                else:
                    # Output as-is (smaller than target size)
                    chunks.append(remainder)
            else:
                # Save remainder for next call
                self.remainder = remainder
        
        return chunks
    
    def flush(self) -> List[np.ndarray]:
        """
        Flush any remaining audio in the adapter.
        
        Returns:
            List of remaining chunks
        """
        chunks = []
        
        # If source is smaller than target and we have buffer
        if self.source_size < self.target_size:
            # Get any remaining chunks that reach target size
            chunks.extend(self.buffer.get_all_chunks())
            
            # If partial chunks are allowed, get remainder
            if self.allow_partial:
                remaining_samples = self.buffer.get_available_samples()
                if remaining_samples > 0:
                    # Combine all remaining chunks
                    combined = np.concatenate([chunk for chunk in self.buffer.buffer])
                    
                    if self.pad_partial:
                        # Pad to target size
                        padded = ensure_audio_size(combined, self.target_size, mode="pad")
                        chunks.append(padded)
                    else:
                        # Output as-is (smaller than target size)
                        chunks.append(combined)
            
            # Clear the buffer
            self.buffer.clear()
        
        # If source is larger than target and we have remainder
        elif self.source_size > self.target_size and len(self.remainder) > 0:
            if self.allow_partial:
                if self.pad_partial:
                    # Pad to target size
                    padded = ensure_audio_size(self.remainder, self.target_size, mode="pad")
                    chunks.append(padded)
                else:
                    # Output as-is (smaller than target size)
                    chunks.append(self.remainder)
            
            # Clear the remainder
            self.remainder = np.array([], dtype=self.remainder.dtype)
        
        return chunks
    
    def clear(self) -> None:
        """Clear all internal buffers."""
        if self.source_size < self.target_size:
            self.buffer.clear()
        elif self.source_size > self.target_size:
            self.remainder = np.array([], dtype=self.remainder.dtype)


class AudioChunkProcessor:
    """
    Processes audio chunks with configurable chunk sizes.
    Provides a common interface for both streaming and batch processing
    with consistent chunk size handling.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        process_fn=None
    ):
        """
        Initialize the audio chunk processor.
        
        Args:
            config: Configuration dictionary
            process_fn: Function to process each chunk (takes chunk as input)
        """
        # Extract chunk size configuration
        audio_config = config.get('audio', {})
        self.input_chunk_size = audio_config.get('chunk_size', CHUNK_SIZE_DEFAULT)
        self.process_chunk_size = audio_config.get('process_chunk_size', self.input_chunk_size)
        self.output_chunk_size = audio_config.get('output_chunk_size', self.input_chunk_size)
        
        # Extract format configuration
        self.input_format = audio_config.get('format', AUDIO_FORMAT_INT16)
        self.process_format = audio_config.get('process_format', self.input_format)
        self.output_format = audio_config.get('output_format', self.input_format)
        
        # Sample rate
        self.sample_rate = audio_config.get('sample_rate', 16000)
        
        # Create adapters for input and output conversion
        self.input_adapter = ChunkSizeAdapter(
            source_size=self.input_chunk_size,
            target_size=self.process_chunk_size,
            format=self.process_format,
            sample_rate=self.sample_rate
        )
        
        self.output_adapter = ChunkSizeAdapter(
            source_size=self.process_chunk_size,
            target_size=self.output_chunk_size,
            format=self.output_format,
            sample_rate=self.sample_rate,
            allow_partial=True,
            pad_partial=True
        )
        
        # Store processing function
        self.process_fn = process_fn
        
        # Initialize stats
        self.stats = {
            'chunks_processed': 0,
            'samples_processed': 0,
            'start_time': 0,
            'processing_time': 0
        }
    
    def process_chunk(self, chunk: np.ndarray) -> List[np.ndarray]:
        """
        Process a single audio chunk with proper size handling.
        
        Args:
            chunk: Input audio chunk
            
        Returns:
            List of processed chunks
        """
        # Convert to process format if needed
        if chunk.dtype != (np.int16 if self.process_format == AUDIO_FORMAT_INT16 else np.float32):
            chunk = convert_audio_format(chunk, self.process_format)
        
        # Adapt input chunk size to processing chunk size
        adapted_chunks = self.input_adapter.process(chunk)
        
        # Process each adapted chunk
        processed_chunks = []
        for adapted_chunk in adapted_chunks:
            if self.process_fn:
                # Apply processing function
                processed_chunk = self.process_fn(adapted_chunk)
            else:
                # Pass through if no processing function
                processed_chunk = adapted_chunk
            
            # Convert processed chunk to output chunk size
            output_chunks = self.output_adapter.process(processed_chunk)
            processed_chunks.extend(output_chunks)
            
            # Update stats
            self.stats['chunks_processed'] += 1
            self.stats['samples_processed'] += len(adapted_chunk)
        
        return processed_chunks
    
    def flush(self) -> List[np.ndarray]:
        """
        Flush any remaining audio in the adapters.
        
        Returns:
            List of remaining processed chunks
        """
        # Flush input adapter
        input_chunks = self.input_adapter.flush()
        
        # Process each flushed chunk
        processed_chunks = []
        for chunk in input_chunks:
            if self.process_fn:
                # Apply processing function
                processed_chunk = self.process_fn(chunk)
            else:
                # Pass through if no processing function
                processed_chunk = chunk
            
            # Add to processed chunks
            processed_chunks.append(processed_chunk)
            
            # Update stats
            self.stats['chunks_processed'] += 1
            self.stats['samples_processed'] += len(chunk)
        
        # Flush output adapter with all processed chunks
        output_chunks = []
        for chunk in processed_chunks:
            output_chunks.extend(self.output_adapter.process(chunk))
        
        # Final flush of output adapter
        output_chunks.extend(self.output_adapter.flush())
        
        return output_chunks
    
    def clear(self) -> None:
        """Clear all internal buffers."""
        self.input_adapter.clear()
        self.output_adapter.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the processor.
        
        Returns:
            Dictionary with processor statistics
        """
        return {
            'chunks_processed': self.stats['chunks_processed'],
            'samples_processed': self.stats['samples_processed'],
            'processing_seconds': self.stats['samples_processed'] / self.sample_rate,
            'input_chunk_size': self.input_chunk_size,
            'process_chunk_size': self.process_chunk_size,
            'output_chunk_size': self.output_chunk_size
        }
    
    def set_process_fn(self, process_fn) -> None:
        """
        Set the processing function.
        
        Args:
            process_fn: Function to process each chunk
        """
        self.process_fn = process_fn
    
    def set_chunk_sizes(self, input_size: int = None, process_size: int = None, output_size: int = None) -> None:
        """
        Update chunk size configuration.
        
        Args:
            input_size: New input chunk size
            process_size: New processing chunk size
            output_size: New output chunk size
        """
        # Update sizes if provided
        if input_size is not None:
            self.input_chunk_size = input_size
        
        if process_size is not None:
            self.process_chunk_size = process_size
        
        if output_size is not None:
            self.output_chunk_size = output_size
        
        # Recreate adapters with new sizes
        self.clear()
        
        self.input_adapter = ChunkSizeAdapter(
            source_size=self.input_chunk_size,
            target_size=self.process_chunk_size,
            format=self.process_format,
            sample_rate=self.sample_rate
        )
        
        self.output_adapter = ChunkSizeAdapter(
            source_size=self.process_chunk_size,
            target_size=self.output_chunk_size,
            format=self.output_format,
            sample_rate=self.sample_rate,
            allow_partial=True,
            pad_partial=True
        )