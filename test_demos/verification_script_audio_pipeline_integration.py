#!/usr/bin/env python3
"""
Verification script for audio pipeline integration with STT engine.
Tests data type conversion and format compatibility between components.
"""

import os
import sys
import time
import wave
import numpy as np
import logging
from typing import Dict, Any, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AudioPipelineIntegration")

# Import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tccc.audio_pipeline.audio_pipeline import AudioPipeline
from src.tccc.stt_engine.stt_engine import STTEngine
from src.tccc.utils.audio_data_converter import (
    convert_audio_format, 
    standardize_audio_for_stt,
    standardize_audio_for_pipeline,
    get_audio_format_info,
    AUDIO_FORMAT_INT16,
    AUDIO_FORMAT_FLOAT32
)


class AudioPipelineIntegrationVerifier:
    """Verifies the integration between AudioPipeline and STTEngine."""
    
    def __init__(self):
        """Initialize the verifier."""
        self.audio_pipeline = None
        self.stt_engine = None
        self.test_files = [
            "test_data/test_speech.wav", 
            "test_data/sample_call.wav"
        ]
        
    def load_components(self) -> bool:
        """
        Load audio pipeline and STT engine components.
        
        Returns:
            bool: Success status
        """
        try:
            # Set up audio pipeline with test configuration
            self.audio_pipeline = AudioPipeline()
            audio_config = {
                'audio': {
                    'sample_rate': 16000,
                    'channels': 1,
                    'format': 'int16',
                    'chunk_size': 1024
                },
                'io': {
                    'input_sources': [
                        {
                            'name': 'test_file',
                            'type': 'file',
                            'path': self.test_files[0],
                            'loop': False
                        }
                    ],
                    'default_input': 'test_file',
                    'stream_output': {
                        'buffer_size': 10,
                        'timeout_ms': 100
                    }
                },
                'noise_reduction': {
                    'enabled': True,
                    'strength': 0.5,
                    'threshold_db': -20
                },
                'vad': {
                    'enabled': True,
                    'sensitivity': 2
                }
            }
            
            # Initialize audio pipeline
            pipeline_init = self.audio_pipeline.initialize(audio_config)
            
            # Set up STT engine with test configuration
            self.stt_engine = STTEngine()
            stt_config = {
                'model': {
                    'type': 'whisper',
                    'size': 'tiny',
                    'language': 'en'
                },
                'hardware': {
                    'enable_acceleration': False
                },
                'transcription': {
                    'confidence_threshold': 0.5,
                    'word_timestamps': True
                },
                'vocabulary': {
                    'enabled': True,
                    'path': 'config/vocabulary/custom_terms.txt'
                }
            }
            
            # Initialize STT engine
            stt_init = self.stt_engine.initialize(stt_config)
            
            return pipeline_init and stt_init
            
        except Exception as e:
            logger.error(f"Failed to load components: {e}")
            return False
            
    def run_format_conversion_tests(self) -> bool:
        """
        Run tests for audio format conversion.
        
        Returns:
            bool: Success status
        """
        try:
            logger.info("Running audio format conversion tests")
            
            # Create test audio data in different formats
            # 1. Create a simple sine wave in int16 format
            sample_rate = 16000
            duration = 0.1  # 100ms
            freq = 440  # A4 note
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            int16_audio = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            
            # 2. Convert to float32 format [-1.0, 1.0]
            float32_audio = int16_audio.astype(np.float32) / 32768.0
            
            # 3. Test conversion int16 -> float32
            converted_float = convert_audio_format(int16_audio, AUDIO_FORMAT_FLOAT32)
            float_conversion_ok = np.allclose(converted_float, float32_audio, atol=1e-5)
            logger.info(f"Int16 to Float32 conversion {'OK' if float_conversion_ok else 'FAILED'}")
            
            # 4. Test conversion float32 -> int16
            converted_int = convert_audio_format(float32_audio, AUDIO_FORMAT_INT16)
            int_conversion_ok = np.allclose(converted_int, int16_audio, atol=1)
            logger.info(f"Float32 to Int16 conversion {'OK' if int_conversion_ok else 'FAILED'}")
            
            # 5. Test standardization functions
            standardized_stt = standardize_audio_for_stt(int16_audio)
            stt_format_ok = standardized_stt.dtype == np.float32
            logger.info(f"Standardize for STT {'OK' if stt_format_ok else 'FAILED'}")
            
            standardized_pipeline = standardize_audio_for_pipeline(float32_audio)
            pipeline_format_ok = standardized_pipeline.dtype == np.int16
            logger.info(f"Standardize for Pipeline {'OK' if pipeline_format_ok else 'FAILED'}")
            
            # 6. Test audio format info
            int16_info = get_audio_format_info(int16_audio)
            float32_info = get_audio_format_info(float32_audio)
            
            logger.info(f"Int16 format info: {int16_info['format']}, range: {int16_info['range']}")
            logger.info(f"Float32 format info: {float32_info['format']}, range: {float32_info['range']}")
            
            return float_conversion_ok and int_conversion_ok and stt_format_ok and pipeline_format_ok
            
        except Exception as e:
            logger.error(f"Format conversion tests failed: {e}")
            return False
    
    def verify_stream_buffer_formats(self) -> bool:
        """
        Verify that the stream buffer correctly handles different audio formats.
        
        Returns:
            bool: Success status
        """
        try:
            logger.info("Verifying stream buffer format handling")
            
            # Get stream buffer from audio pipeline
            stream_buffer = self.audio_pipeline.get_audio_stream()
            
            # Create test audio data
            sample_rate = 16000
            duration = 0.1  # 100ms
            freq = 440  # A4 note
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            int16_audio = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            float32_audio = int16_audio.astype(np.float32) / 32768.0
            
            # Test writing int16 data
            stream_buffer.clear()
            stream_buffer.write(int16_audio, format_hint=AUDIO_FORMAT_INT16)
            
            # Read as float32
            read_data = stream_buffer.read(target_format=AUDIO_FORMAT_FLOAT32)
            read_as_float_ok = read_data.dtype == np.float32
            logger.info(f"Read as float32 {'OK' if read_as_float_ok else 'FAILED'}")
            
            # Test writing float32 data
            stream_buffer.clear()
            stream_buffer.write(float32_audio, format_hint=AUDIO_FORMAT_FLOAT32)
            
            # Read as int16
            read_data = stream_buffer.read(target_format=AUDIO_FORMAT_INT16)
            read_as_int_ok = read_data.dtype == np.int16
            logger.info(f"Read as int16 {'OK' if read_as_int_ok else 'FAILED'}")
            
            # Test metadata
            metadata = stream_buffer.get_metadata()
            logger.info(f"Stream buffer metadata: {metadata}")
            
            # Update format and check metadata
            stream_buffer.set_format(AUDIO_FORMAT_FLOAT32)
            metadata = stream_buffer.get_metadata()
            format_update_ok = metadata["format"] == AUDIO_FORMAT_FLOAT32
            logger.info(f"Format update {'OK' if format_update_ok else 'FAILED'}")
            
            return read_as_float_ok and read_as_int_ok and format_update_ok
            
        except Exception as e:
            logger.error(f"Stream buffer format verification failed: {e}")
            return False
    
    def run_audio_to_stt_flow_test(self, test_file: str) -> bool:
        """
        Test the complete flow from audio capture to STT processing.
        
        Args:
            test_file: Path to the test file
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Testing audio to STT flow with {test_file}")
            
            # Configure audio pipeline to use test file
            self.audio_pipeline.stop_capture()
            
            # Update audio source configuration
            test_config = {
                'io': {
                    'input_sources': [
                        {
                            'name': 'test_file',
                            'type': 'file',
                            'path': test_file,
                            'loop': False
                        }
                    ],
                    'default_input': 'test_file'
                }
            }
            
            # Start capture
            if not self.audio_pipeline.start_capture('test_file'):
                logger.error("Failed to start audio capture")
                return False
            
            logger.info("Audio capture started, waiting for data...")
            
            # Process audio chunks and send to STT
            transcription = ""
            max_chunks = 50  # Limit the number of chunks to process
            chunk_count = 0
            
            start_time = time.time()
            
            while chunk_count < max_chunks:
                # Get audio data from pipeline
                pipeline_audio = self.audio_pipeline.get_audio(timeout_ms=500)
                
                if pipeline_audio is None or len(pipeline_audio) == 0:
                    time.sleep(0.1)
                    continue
                
                # Log audio format info
                audio_info = get_audio_format_info(pipeline_audio)
                logger.debug(f"Pipeline audio: {audio_info}")
                
                # Convert to format required by STT
                stt_audio = standardize_audio_for_stt(pipeline_audio)
                
                # Process with STT Engine
                result = self.stt_engine.transcribe_segment(
                    stt_audio, 
                    metadata={"is_partial": True}
                )
                
                # Extract text from result
                if result and isinstance(result, dict) and "text" in result:
                    text = result["text"]
                    if text:
                        transcription += text + " "
                        logger.info(f"Transcription: {text}")
                
                chunk_count += 1
                time.sleep(0.1)  # Small delay to avoid tight loop
                
                # Stop after 5 seconds regardless
                if time.time() - start_time > 5:
                    break
            
            # Stop capture
            self.audio_pipeline.stop_capture()
            
            logger.info(f"Final transcription: {transcription}")
            logger.info(f"Processed {chunk_count} chunks")
            
            return len(transcription.strip()) > 0
            
        except Exception as e:
            logger.error(f"Audio to STT flow test failed: {e}")
            return False
    
    def run_verification(self) -> bool:
        """
        Run all verification tests.
        
        Returns:
            bool: Overall success status
        """
        logger.info("Starting audio pipeline integration verification")
        
        # Load components
        if not self.load_components():
            logger.error("Failed to load components")
            return False
        
        # Run format conversion tests
        if not self.run_format_conversion_tests():
            logger.error("Format conversion tests failed")
            return False
        
        # Verify stream buffer format handling
        if not self.verify_stream_buffer_formats():
            logger.error("Stream buffer format verification failed")
            return False
        
        # Run audio to STT flow test
        for test_file in self.test_files:
            if not os.path.exists(test_file):
                logger.warning(f"Test file not found: {test_file}")
                continue
                
            if not self.run_audio_to_stt_flow_test(test_file):
                logger.error(f"Audio to STT flow test failed for {test_file}")
                return False
        
        logger.info("Audio pipeline integration verification completed successfully")
        return True


if __name__ == "__main__":
    verifier = AudioPipelineIntegrationVerifier()
    success = verifier.run_verification()
    
    if success:
        logger.info("VERIFICATION RESULT: PASS")
        sys.exit(0)
    else:
        logger.error("VERIFICATION RESULT: FAIL")
        sys.exit(1)