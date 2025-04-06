#!/usr/bin/env python3
"""
Test script for the TCCC STT Engine.

This script tests the speech-to-text functionality by:
1. Capturing audio from the microphone
2. Processing it through the STT engine
3. Displaying the transcription results
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
import logging
import asyncio
import threading
import queue

# Add project source to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root / 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("STTEngineTest")

# Import TCCC modules
from tccc.audio_pipeline.audio_pipeline import AudioPipeline
from tccc.stt_engine.stt_engine import STTEngine
from tccc.utils.config import Config

class TranscriptionCollector:
    def __init__(self):
        self.results = []
        self.lock = threading.Lock()
        
    def add_result(self, text, metadata=None):
        with self.lock:
            self.results.append({
                'text': text,
                'timestamp': time.time(),
                'metadata': metadata or {}
            })
            
    def get_results(self):
        with self.lock:
            return self.results.copy()

class STTTester:
    def __init__(self, config_path, device_index=None):
        self.config_path = config_path
        self.device_index = device_index
        self.audio_pipeline = None
        self.stt_engine = None
        self.loop = None
        self.collector = TranscriptionCollector()
        self.audio_queue = queue.Queue()
        self.running = False
        self.audio_thread = None
        self.sequence_num = 0
        
    def load_config(self):
        """Load configuration from file"""
        try:
            config = Config.from_yaml(self.config_path)
            audio_config = config.get('audio_pipeline', {})
            stt_config = config.get('stt_engine', {})
            
            # Override device index if specified
            if self.device_index is not None:
                audio_config['microphone']['device_index'] = self.device_index
                logger.info(f"Using device index: {self.device_index}")
            
            logger.info(f"Audio configuration: {audio_config}")
            logger.info(f"STT configuration: {stt_config}")
            
            return audio_config, stt_config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    async def initialize(self):
        """Initialize the audio pipeline and STT engine"""
        try:
            audio_config, stt_config = self.load_config()
            
            # Initialize audio pipeline
            logger.info("Initializing Audio Pipeline...")
            self.audio_pipeline = AudioPipeline()
            await self.audio_pipeline.initialize(audio_config)
            logger.info("Audio Pipeline initialized successfully")
            
            # Initialize STT engine
            logger.info("Initializing STT Engine...")
            self.stt_engine = STTEngine()
            await self.stt_engine.initialize(stt_config)
            self.stt_engine.set_system_reference(self)
            logger.info("STT Engine initialized successfully")
            
            # Register audio callback
            self.audio_pipeline.register_audio_callback(self.process_audio)
            
            return True
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False
    
    def process_audio(self, audio_data):
        """Process audio data from pipeline"""
        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            else:
                audio_array = audio_data
                
            # Add sequence number
            self.sequence_num += 1
            
            # Add to queue for STT processing
            audio_metadata = {
                'sequence': self.sequence_num,
                'timestamp': time.time(),
                'format': 'int16',
                'channels': 1,  # Assuming mono
                'sample_rate': 16000  # Assuming 16kHz
            }
            
            # Calculate audio level
            level = np.abs(audio_array).mean() / 32768.0
            if level > 0.01:  # Only process if audio level is above threshold
                # Enqueue audio for processing
                self.stt_engine.enqueue_audio(audio_array, audio_metadata)
                logger.debug(f"Enqueued audio chunk {self.sequence_num}, level: {level:.4f}")
            else:
                logger.debug(f"Skipping low-level audio: {level:.4f}")
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
    
    def process_event(self, event_data):
        """Process transcription event from STT engine"""
        try:
            if not event_data:
                return
            
            text = event_data.get('data', {}).get('text', '')
            if text and len(text.strip()) > 0:
                # Extract metadata
                metadata = {
                    'sequence': event_data.get('sequence'),
                    'confidence': event_data.get('data', {}).get('confidence'),
                    'is_partial': event_data.get('data', {}).get('is_partial', False)
                }
                
                # Add to collector
                self.collector.add_result(text, metadata)
                
                # Log the result
                is_partial = "(partial) " if metadata.get('is_partial') else ""
                logger.info(f"Transcription {is_partial}[{metadata.get('sequence')}]: {text}")
                
        except Exception as e:
            logger.error(f"Error processing transcription event: {e}")
    
    async def start(self, duration=30):
        """Start the STT testing process"""
        try:
            logger.info("Starting audio capture...")
            self.running = True
            self.audio_pipeline.start_capture()
            
            logger.info(f"Test running for {duration} seconds...")
            try:
                await asyncio.sleep(duration)
            except asyncio.CancelledError:
                logger.info("Test cancelled")
            
            logger.info("Stopping audio capture...")
            self.audio_pipeline.stop_capture()
            self.running = False
            
            # Print transcription summary
            results = self.collector.get_results()
            logger.info(f"Test completed with {len(results)} transcription results")
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}: {result['text']}")
            
            return results
        except Exception as e:
            logger.error(f"Error during test: {e}")
            self.running = False
            return []
    
    async def shutdown(self):
        """Clean shutdown of all components"""
        try:
            if self.audio_pipeline:
                self.audio_pipeline.stop_capture()
            
            if self.stt_engine:
                await self.stt_engine.shutdown()
                
            logger.info("All components shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

async def run_test(args):
    """Run the STT engine test"""
    tester = STTTester(args.config, args.device)
    
    try:
        # Initialize components
        if not await tester.initialize():
            logger.error("Initialization failed")
            return 1
        
        # Run the test
        results = await tester.start(args.duration)
        
        # Shut down
        await tester.shutdown()
        
        # Report results
        if results:
            logger.info(f"Test successful with {len(results)} transcription results")
            return 0
        else:
            logger.error("Test completed without any transcription results")
            return 1
    except Exception as e:
        logger.error(f"Test failed: {e}")
        await tester.shutdown()
        return 1

def main():
    """Main entry point for the STT engine test"""
    parser = argparse.ArgumentParser(description="Test TCCC STT Engine")
    parser.add_argument("--config", default=str(project_root / "config/jetson_mvp.yaml"), 
                      help="Path to config file")
    parser.add_argument("--device", type=int, default=0,
                      help="Audio device index to use for testing")
    parser.add_argument("--duration", type=int, default=30,
                      help="Test duration in seconds")
    args = parser.parse_args()
    
    # Create and run the event loop
    try:
        asyncio.run(run_test(args))
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
