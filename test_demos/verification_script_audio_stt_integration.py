#!/usr/bin/env python3
"""
TCCC.ai Audio Pipeline to STT Engine Integration Verification Script

This script provides focused testing of the integration between the Audio Pipeline
and STT Engine components:

1. Initializes both modules with minimal configuration
2. Tests the data flow from Audio Pipeline to STT Engine
3. Verifies event format compatibility
4. Validates audio segment processing

Example usage:
    python verification_script_audio_stt_integration.py
"""

import os
import sys
import time
import json
import logging
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the required components
from tccc.audio_pipeline.audio_pipeline import AudioPipeline
from tccc.stt_engine.stt_engine import STTEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AudioSTTIntegration")

class AudioSTTIntegrationVerifier:
    """
    Verifies the integration between Audio Pipeline and STT Engine
    """
    
    def __init__(self):
        """Initialize the verifier"""
        self.audio_pipeline = None
        self.stt_engine = None
        self.test_results = {
            "module_initialization": False,
            "audio_segment_generation": False,
            "audio_segment_format": False,
            "stt_processing": False,
            "event_format": False,
            "complete_flow": False
        }

    def initialize_components(self):
        """Initialize both components with minimal configuration"""
        logger.info("Initializing components...")
        
        try:
            # Initialize Audio Pipeline
            self.audio_pipeline = AudioPipeline()
            audio_config = {
                "audio": {
                    "sample_rate": 16000,
                    "channels": 1,
                    "format": "int16",
                    "chunk_size": 1024
                },
                "io": {
                    "input_sources": [
                        {
                            "name": "test_file",
                            "type": "file",
                            "path": "test_data/test_speech.wav"
                        }
                    ],
                    "default_input": "test_file"
                }
            }
            audio_init = self.audio_pipeline.initialize(audio_config)
            
            # Initialize STT Engine
            self.stt_engine = STTEngine()
            stt_config = {
                "model": {
                    "type": "whisper",
                    "size": "medium"
                },
                "hardware": {
                    "enable_acceleration": False  # Disable for testing
                }
            }
            stt_init = self.stt_engine.initialize(stt_config)
            
            # Update result
            self.test_results["module_initialization"] = audio_init and stt_init
            logger.info(f"Component initialization: {'SUCCESS' if self.test_results['module_initialization'] else 'FAILURE'}")
            
            return self.test_results["module_initialization"]
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False

    def test_audio_segment_generation(self):
        """Test Audio Pipeline's ability to generate audio segments"""
        logger.info("Testing audio segment generation...")
        
        try:
            # Start audio capture
            self.audio_pipeline.start_capture()
            
            # Wait for processing to start
            time.sleep(1)
            
            # Try to get audio segment
            audio_segment = None
            
            # Test method availability
            if hasattr(self.audio_pipeline, 'get_audio_segment'):
                audio_segment = self.audio_pipeline.get_audio_segment()
                logger.info("Using get_audio_segment() method")
            elif hasattr(self.audio_pipeline, 'get_audio'):
                audio_segment = self.audio_pipeline.get_audio()
                logger.info("Using get_audio() method")
            else:
                logger.error("No compatible method found for getting audio segments")
                return False
            
            # Verify we got audio data
            if audio_segment is not None and len(audio_segment) > 0:
                self.test_results["audio_segment_generation"] = True
                logger.info(f"Audio segment generation: SUCCESS (size: {len(audio_segment)})")
            else:
                logger.error("Failed to generate audio segment")
                
            # Stop audio capture
            self.audio_pipeline.stop_capture()
            
            return self.test_results["audio_segment_generation"]
            
        except Exception as e:
            logger.error(f"Error testing audio segment generation: {e}")
            # Stop audio capture in case of error
            try:
                self.audio_pipeline.stop_capture()
            except:
                pass
            return False

    def test_audio_segment_format(self):
        """Test that audio segments have the correct format"""
        logger.info("Testing audio segment format...")
        
        try:
            # Start audio capture
            self.audio_pipeline.start_capture()
            
            # Wait for processing to start
            time.sleep(1)
            
            # Get audio segment
            audio_segment = None
            if hasattr(self.audio_pipeline, 'get_audio_segment'):
                audio_segment = self.audio_pipeline.get_audio_segment()
            elif hasattr(self.audio_pipeline, 'get_audio'):
                audio_segment = self.audio_pipeline.get_audio()
            
            # Stop audio capture
            self.audio_pipeline.stop_capture()
            
            if audio_segment is None:
                logger.error("Failed to get audio segment")
                return False
            
            # Check audio segment format
            if isinstance(audio_segment, np.ndarray):
                is_valid_dtype = audio_segment.dtype in [np.int16, np.float32, np.int32]
                has_data = len(audio_segment) > 0
                
                if is_valid_dtype and has_data:
                    self.test_results["audio_segment_format"] = True
                    logger.info(f"Audio segment format: SUCCESS (dtype: {audio_segment.dtype}, size: {len(audio_segment)})")
                else:
                    logger.error(f"Invalid audio segment format: dtype={audio_segment.dtype}, size={len(audio_segment)}")
            else:
                logger.error(f"Unexpected audio segment type: {type(audio_segment)}")
            
            return self.test_results["audio_segment_format"]
            
        except Exception as e:
            logger.error(f"Error testing audio segment format: {e}")
            # Stop audio capture in case of error
            try:
                self.audio_pipeline.stop_capture()
            except:
                pass
            return False

    def test_stt_processing(self):
        """Test STT Engine's ability to process audio segments"""
        logger.info("Testing STT processing...")
        
        try:
            # Start audio capture
            self.audio_pipeline.start_capture()
            
            # Wait for processing to start
            time.sleep(1)
            
            # Get audio segment
            audio_segment = None
            if hasattr(self.audio_pipeline, 'get_audio_segment'):
                audio_segment = self.audio_pipeline.get_audio_segment()
            elif hasattr(self.audio_pipeline, 'get_audio'):
                audio_segment = self.audio_pipeline.get_audio()
            
            # Stop audio capture
            self.audio_pipeline.stop_capture()
            
            if audio_segment is None:
                logger.error("Failed to get audio segment")
                return False
            
            # Process with STT Engine
            transcription = self.stt_engine.transcribe_segment(audio_segment)
            
            # Check transcription result
            if transcription and isinstance(transcription, dict):
                has_text = "text" in transcription and transcription["text"]
                has_segments = "segments" in transcription and transcription["segments"]
                
                if has_text and has_segments:
                    self.test_results["stt_processing"] = True
                    logger.info(f"STT processing: SUCCESS (text: '{transcription['text']}')")
                else:
                    logger.error(f"Invalid transcription result: {transcription}")
            else:
                logger.error(f"Unexpected transcription type: {type(transcription)}")
            
            return self.test_results["stt_processing"]
            
        except Exception as e:
            logger.error(f"Error testing STT processing: {e}")
            # Stop audio capture in case of error
            try:
                self.audio_pipeline.stop_capture()
            except:
                pass
            return False

    def test_event_format(self):
        """Test that event formats are compatible"""
        logger.info("Testing event format compatibility...")
        
        try:
            # Start audio capture
            self.audio_pipeline.start_capture()
            
            # Wait for processing to start
            time.sleep(1)
            
            # Get audio segment
            audio_segment = None
            if hasattr(self.audio_pipeline, 'get_audio_segment'):
                audio_segment = self.audio_pipeline.get_audio_segment()
            elif hasattr(self.audio_pipeline, 'get_audio'):
                audio_segment = self.audio_pipeline.get_audio()
            
            # Stop audio capture
            self.audio_pipeline.stop_capture()
            
            if audio_segment is None:
                logger.error("Failed to get audio segment")
                return False
            
            # Add metadata for testing
            metadata = {
                "source": "audio_pipeline",
                "timestamp": time.time(),
                "session_id": "test_session",
                "format": "PCM16",
                "sample_rate": 16000
            }
            
            # Process with STT Engine
            transcription = self.stt_engine.transcribe_segment(audio_segment, metadata)
            
            # Check transcription result format
            if transcription and isinstance(transcription, dict):
                has_text = "text" in transcription and transcription["text"]
                has_metadata = "metadata" in transcription
                
                if has_text and has_metadata:
                    # Check if metadata was properly handled
                    preserved_fields = 0
                    for key in ["source", "timestamp", "session_id"]:
                        if key in metadata and key in transcription.get("metadata", {}):
                            if metadata[key] == transcription["metadata"][key]:
                                preserved_fields += 1
                    
                    if preserved_fields > 0:
                        self.test_results["event_format"] = True
                        logger.info(f"Event format compatibility: SUCCESS (preserved {preserved_fields} metadata fields)")
                    else:
                        logger.warning("Metadata not properly preserved in transcription")
                else:
                    logger.error(f"Invalid transcription format: {list(transcription.keys())}")
            else:
                logger.error(f"Unexpected transcription type: {type(transcription)}")
            
            return self.test_results["event_format"]
            
        except Exception as e:
            logger.error(f"Error testing event format: {e}")
            # Stop audio capture in case of error
            try:
                self.audio_pipeline.stop_capture()
            except:
                pass
            return False

    def test_complete_flow(self):
        """Test the complete flow from Audio Pipeline to STT Engine"""
        logger.info("Testing complete flow...")
        
        try:
            # Start audio capture
            self.audio_pipeline.start_capture()
            
            # Process multiple segments
            segments_processed = 0
            max_segments = 5
            
            logger.info(f"Processing {max_segments} audio segments...")
            
            for i in range(max_segments):
                # Wait for audio processing
                time.sleep(0.5)
                
                # Get audio segment
                audio_segment = None
                if hasattr(self.audio_pipeline, 'get_audio_segment'):
                    audio_segment = self.audio_pipeline.get_audio_segment()
                elif hasattr(self.audio_pipeline, 'get_audio'):
                    audio_segment = self.audio_pipeline.get_audio()
                
                if audio_segment is None or len(audio_segment) == 0:
                    logger.warning(f"No audio segment available on attempt {i+1}")
                    continue
                
                # Add metadata
                metadata = {
                    "source": "audio_pipeline",
                    "timestamp": time.time(),
                    "session_id": "test_session",
                    "sequence": i
                }
                
                # Process with STT Engine
                transcription = self.stt_engine.transcribe_segment(audio_segment, metadata)
                
                if transcription and "text" in transcription:
                    segments_processed += 1
                    logger.info(f"Processed segment {i+1}: '{transcription['text']}'")
                else:
                    logger.warning(f"Failed to transcribe segment {i+1}")
            
            # Stop audio capture
            self.audio_pipeline.stop_capture()
            
            # Check results
            self.test_results["complete_flow"] = segments_processed >= 3
            logger.info(f"Complete flow test: {'SUCCESS' if self.test_results['complete_flow'] else 'FAILURE'} " +
                       f"(processed {segments_processed}/{max_segments} segments)")
            
            return self.test_results["complete_flow"]
            
        except Exception as e:
            logger.error(f"Error testing complete flow: {e}")
            # Stop audio capture in case of error
            try:
                self.audio_pipeline.stop_capture()
            except:
                pass
            return False

    def run_verification(self):
        """Run all verification tests"""
        logger.info("Starting Audio Pipeline to STT Engine integration verification...")
        
        # Initialize components
        if not self.initialize_components():
            logger.error("Component initialization failed, stopping verification")
            return False
        
        # Run tests
        self.test_audio_segment_generation()
        self.test_audio_segment_format()
        self.test_stt_processing()
        self.test_event_format()
        self.test_complete_flow()
        
        # Summarize results
        self.print_summary()
        
        # Calculate overall success
        return all(self.test_results.values())

    def print_summary(self):
        """Print a summary of verification results"""
        logger.info("\n" + "="*50)
        logger.info("Audio Pipeline to STT Engine Integration Verification Results")
        logger.info("="*50)
        
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        
        overall = all(self.test_results.values())
        logger.info("-"*50)
        logger.info(f"Overall Integration Status: {'✓ PASS' if overall else '❌ FAIL'}")
        logger.info("="*50)

def main():
    """Main entry point"""
    verifier = AudioSTTIntegrationVerifier()
    success = verifier.run_verification()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())