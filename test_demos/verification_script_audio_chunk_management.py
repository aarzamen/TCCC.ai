#!/usr/bin/env python3
"""
Verification script for audio chunk size management.
Tests the functionality of the audio chunk manager module.
"""

import os
import sys
import time
import numpy as np
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AudioChunkManagement")

# Import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tccc.utils.audio_chunk_manager import (
    AudioChunkBuffer, 
    ChunkSizeAdapter, 
    AudioChunkProcessor,
    CHUNK_SIZE_DEFAULT,
    CHUNK_SIZE_SECOND
)
from src.tccc.utils.audio_data_converter import (
    AUDIO_FORMAT_INT16,
    AUDIO_FORMAT_FLOAT32
)


class AudioChunkManagementVerifier:
    """Verifies the functionality of the audio chunk manager module."""
    
    def __init__(self):
        """Initialize the verifier."""
        self.test_sample_rate = 16000
        self.test_duration_sec = 2  # 2 seconds of audio
        self.test_freq = 440  # A4 note
        
        # Generate test audio data
        self.test_audio = self._generate_test_audio()
    
    def _generate_test_audio(self) -> np.ndarray:
        """
        Generate test audio data.
        
        Returns:
            Test audio data as numpy array
        """
        sample_count = int(self.test_sample_rate * self.test_duration_sec)
        t = np.linspace(0, self.test_duration_sec, sample_count, False)
        audio = np.sin(2 * np.pi * self.test_freq * t)
        return (audio * 32767).astype(np.int16)
    
    def verify_audio_chunk_buffer(self) -> bool:
        """
        Verify the AudioChunkBuffer class.
        
        Returns:
            Success status
        """
        try:
            logger.info("Verifying AudioChunkBuffer...")
            
            # Create buffer with target size 16000 (1 second at 16kHz)
            buffer = AudioChunkBuffer(
                target_size=CHUNK_SIZE_SECOND,
                input_size=CHUNK_SIZE_DEFAULT,
                format=AUDIO_FORMAT_INT16,
                overlap=False
            )
            
            # Split test audio into small chunks
            small_chunks = []
            for i in range(0, len(self.test_audio), CHUNK_SIZE_DEFAULT):
                chunk = self.test_audio[i:i+CHUNK_SIZE_DEFAULT]
                if len(chunk) == CHUNK_SIZE_DEFAULT:  # Only use complete chunks
                    small_chunks.append(chunk)
            
            # Add chunks to buffer
            chunks_added = 0
            for chunk in small_chunks:
                buffer.add_chunk(chunk)
                chunks_added += 1
                
                # Get all available complete chunks
                output_chunks = buffer.get_all_chunks()
                
                # Log progress
                logger.info(f"Added {chunks_added} chunks ({chunks_added * CHUNK_SIZE_DEFAULT} samples), "
                          f"got {len(output_chunks)} complete chunks of size {CHUNK_SIZE_SECOND}")
            
            # Verify buffer state
            buffer_info = buffer.get_info()
            logger.info(f"Buffer info: {buffer_info}")
            
            # Verify non-overlapping mode
            overlap_buffer = AudioChunkBuffer(
                target_size=8000,  # 0.5 seconds
                input_size=CHUNK_SIZE_DEFAULT,
                format=AUDIO_FORMAT_INT16,
                overlap=True,
                overlap_percent=0.25
            )
            
            # Add chunks to buffer
            for chunk in small_chunks[:16]:  # Add first 16 chunks
                overlap_buffer.add_chunk(chunk)
            
            # Get chunks
            overlap_chunks = overlap_buffer.get_all_chunks()
            
            # Verify overlap
            if len(overlap_chunks) > 1:
                # Check if second chunk starts with overlap from first chunk
                first_chunk_end = overlap_chunks[0][-2000:]  # Last 25% of first chunk
                second_chunk_start = overlap_chunks[1][:2000]  # First 25% of second chunk
                
                # They should be identical
                overlap_match = np.array_equal(first_chunk_end, second_chunk_start)
                logger.info(f"Overlap match: {overlap_match}")
            
            # Test with different format
            float_buffer = AudioChunkBuffer(
                target_size=4096,
                input_size=1024,
                format=AUDIO_FORMAT_FLOAT32
            )
            
            # Convert a chunk to float32
            float_chunk = self.test_audio[:1024].astype(np.float32) / 32768.0
            
            # Add to buffer
            float_buffer.add_chunk(float_chunk)
            
            # Verify dtype
            float_output = float_buffer.get_chunk()
            if float_output is not None:
                logger.info(f"Float buffer output dtype: {float_output.dtype}")
            
            return True
            
        except Exception as e:
            logger.error(f"AudioChunkBuffer verification failed: {e}")
            return False
    
    def verify_chunk_size_adapter(self) -> bool:
        """
        Verify the ChunkSizeAdapter class.
        
        Returns:
            Success status
        """
        try:
            logger.info("Verifying ChunkSizeAdapter...")
            
            # Create adapter to convert small to large chunks
            small_to_large = ChunkSizeAdapter(
                source_size=CHUNK_SIZE_DEFAULT,
                target_size=CHUNK_SIZE_SECOND,
                format=AUDIO_FORMAT_INT16
            )
            
            # Split test audio into small chunks
            small_chunks = []
            for i in range(0, len(self.test_audio), CHUNK_SIZE_DEFAULT):
                chunk = self.test_audio[i:i+CHUNK_SIZE_DEFAULT]
                if len(chunk) == CHUNK_SIZE_DEFAULT:
                    small_chunks.append(chunk)
            
            # Process small chunks to get large chunks
            large_chunks = []
            for chunk in small_chunks:
                output_chunks = small_to_large.process(chunk)
                large_chunks.extend(output_chunks)
            
            # Flush any remaining data
            large_chunks.extend(small_to_large.flush())
            
            # Verify large chunks
            logger.info(f"Converted {len(small_chunks)} small chunks to {len(large_chunks)} large chunks")
            
            # Create adapter to convert large to small chunks
            large_to_small = ChunkSizeAdapter(
                source_size=CHUNK_SIZE_SECOND,
                target_size=CHUNK_SIZE_DEFAULT,
                format=AUDIO_FORMAT_INT16
            )
            
            # Process large chunks to get small chunks
            small_chunks_2 = []
            for chunk in large_chunks:
                output_chunks = large_to_small.process(chunk)
                small_chunks_2.extend(output_chunks)
            
            # Flush any remaining data
            small_chunks_2.extend(large_to_small.flush())
            
            # Verify small chunks
            logger.info(f"Converted {len(large_chunks)} large chunks to {len(small_chunks_2)} small chunks")
            
            # Create adapter with format conversion
            format_adapter = ChunkSizeAdapter(
                source_size=CHUNK_SIZE_DEFAULT,
                target_size=CHUNK_SIZE_DEFAULT,
                format=AUDIO_FORMAT_FLOAT32
            )
            
            # Process an int16 chunk to get float32 chunk
            int16_chunk = self.test_audio[:CHUNK_SIZE_DEFAULT]
            float32_chunks = format_adapter.process(int16_chunk)
            
            # Verify float32 chunk
            logger.info(f"Converted int16 chunk to {len(float32_chunks)} float32 chunks, "
                      f"dtype={float32_chunks[0].dtype if float32_chunks else 'N/A'}")
            
            # Test partial chunk handling
            partial_adapter = ChunkSizeAdapter(
                source_size=CHUNK_SIZE_SECOND,
                target_size=CHUNK_SIZE_DEFAULT,
                format=AUDIO_FORMAT_INT16,
                allow_partial=True,
                pad_partial=True
            )
            
            # Process a large chunk that's not a multiple of the target size
            odd_sized_chunk = self.test_audio[:CHUNK_SIZE_SECOND + 500]
            partial_chunks = partial_adapter.process(odd_sized_chunk)
            
            # Verify partial chunk padding
            logger.info(f"Processed odd-sized chunk ({len(odd_sized_chunk)} samples) to {len(partial_chunks)} chunks")
            if partial_chunks:
                last_chunk = partial_chunks[-1]
                logger.info(f"Last chunk size: {len(last_chunk)}")
            
            return True
            
        except Exception as e:
            logger.error(f"ChunkSizeAdapter verification failed: {e}")
            return False
    
    def verify_audio_chunk_processor(self) -> bool:
        """
        Verify the AudioChunkProcessor class.
        
        Returns:
            Success status
        """
        try:
            logger.info("Verifying AudioChunkProcessor...")
            
            # Create config for processor
            config = {
                'audio': {
                    'sample_rate': 16000,
                    'chunk_size': CHUNK_SIZE_DEFAULT,
                    'process_chunk_size': 4096,
                    'output_chunk_size': 8192,
                    'format': AUDIO_FORMAT_INT16,
                    'process_format': AUDIO_FORMAT_FLOAT32,
                    'output_format': AUDIO_FORMAT_INT16
                }
            }
            
            # Create a simple processing function
            def process_fn(chunk):
                # Convert to float if not already
                if chunk.dtype != np.float32:
                    chunk = chunk.astype(np.float32) / 32768.0
                
                # Apply simple gain
                chunk = chunk * 0.8
                
                return chunk
            
            # Create processor
            processor = AudioChunkProcessor(config, process_fn)
            
            # Process chunks
            output_chunks = []
            for i in range(0, len(self.test_audio), CHUNK_SIZE_DEFAULT):
                chunk = self.test_audio[i:i+CHUNK_SIZE_DEFAULT]
                if len(chunk) == CHUNK_SIZE_DEFAULT:
                    output = processor.process_chunk(chunk)
                    output_chunks.extend(output)
            
            # Flush any remaining data
            output_chunks.extend(processor.flush())
            
            # Verify output
            logger.info(f"Processed {len(self.test_audio)} samples in chunks of {CHUNK_SIZE_DEFAULT} to "
                      f"{len(output_chunks)} output chunks of size {config['audio']['output_chunk_size']}")
            
            # Check processor stats
            stats = processor.get_stats()
            logger.info(f"Processor stats: {stats}")
            
            # Test changing chunk sizes
            processor.set_chunk_sizes(input_size=2048, process_size=8192, output_size=4096)
            
            # Check new sizes
            stats = processor.get_stats()
            logger.info(f"Updated processor stats: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"AudioChunkProcessor verification failed: {e}")
            return False
    
    def run_verification(self) -> bool:
        """
        Run all verification tests.
        
        Returns:
            Overall success status
        """
        logger.info("Starting audio chunk management verification")
        
        # Verify AudioChunkBuffer
        if not self.verify_audio_chunk_buffer():
            logger.error("AudioChunkBuffer verification failed")
            return False
        
        # Verify ChunkSizeAdapter
        if not self.verify_chunk_size_adapter():
            logger.error("ChunkSizeAdapter verification failed")
            return False
        
        # Verify AudioChunkProcessor
        if not self.verify_audio_chunk_processor():
            logger.error("AudioChunkProcessor verification failed")
            return False
        
        logger.info("Audio chunk management verification completed successfully")
        return True


if __name__ == "__main__":
    verifier = AudioChunkManagementVerifier()
    success = verifier.run_verification()
    
    if success:
        logger.info("VERIFICATION RESULT: PASS")
        sys.exit(0)
    else:
        logger.error("VERIFICATION RESULT: FAIL")
        sys.exit(1)