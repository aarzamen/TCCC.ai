#!/usr/bin/env python3
"""
Optimized Audio-to-STT Pipeline with Model Caching

This script demonstrates the complete audio capture to transcription pipeline
using the model cache manager for improved performance. It supports both 
microphone and file input modes, with battlefield audio enhancement.
"""

import os
import sys
import time
import argparse
import logging
import threading
from pathlib import Path
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AudioToSTT")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run optimized audio-to-STT pipeline")
    parser.add_argument("--file", action="store_true", help="Use file input instead of microphone")
    parser.add_argument("--input-file", type=str, default="test_data/test_speech.wav", 
                        help="Path to input audio file (if --file is used)")
    parser.add_argument("--device-id", type=int, default=0, 
                        help="Microphone device ID (default: 0, Razer Seiren V3 Mini)")
    parser.add_argument("--model", type=str, choices=["tiny", "tiny.en", "base", "small", "medium"], 
                        default="tiny.en", help="STT model size (default: tiny.en)")
    parser.add_argument("--battlefield", action="store_true", 
                        help="Enable battlefield audio enhancement mode")
    parser.add_argument("--output-file", type=str, default="", 
                        help="Save transcription to file (optional)")
    parser.add_argument("--display", action="store_true", 
                        help="Show transcription on display")
    return parser.parse_args()

class AudioSTTPipeline:
    """
    Integrated Audio Capture to STT Pipeline.
    """
    
    def __init__(self, config):
        """Initialize the pipeline."""
        self.config = config
        self.audio_pipeline = None
        self.stt_engine = None
        self.initialized = False
        self.running = False
        self.segment_queue = queue.Queue(maxsize=10)
        self.transcription_queue = queue.Queue(maxsize=20)
        self.vad_manager = None
        self.display_interface = None
        
    def initialize(self):
        """Initialize all components of the pipeline."""
        try:
            from tccc.audio_pipeline.audio_pipeline import AudioPipeline
            from tccc.stt_engine.stt_engine import STTEngine
            
            # Initialize Audio Pipeline
            logger.info("Initializing Audio Pipeline")
            self.audio_pipeline = AudioPipeline()
            
            # Set up audio configuration
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
                            "name": "microphone",
                            "type": "device",
                            "device_id": self.config.device_id
                        },
                        {
                            "name": "file_input",
                            "type": "file",
                            "path": self.config.input_file
                        }
                    ],
                    "default_input": "file_input" if self.config.file else "microphone"
                },
                "processing": {
                    "enable_vad": True,
                    "battlefield_mode": self.config.battlefield
                }
            }
            
            # Initialize audio pipeline
            audio_init = self.audio_pipeline.initialize(audio_config)
            if not audio_init:
                logger.error("Failed to initialize audio pipeline")
                return False
            
            # Initialize STT Engine
            logger.info("Initializing STT Engine")
            self.stt_engine = STTEngine()
            
            # Set up STT configuration
            stt_config = {
                "model": {
                    "type": "whisper",
                    "size": self.config.model,
                    "use_model_cache": True  # Use the model cache manager
                },
                "hardware": {
                    "enable_acceleration": True  # Use GPU if available
                },
                "streaming": {
                    "enabled": True,
                    "max_context_length_sec": 60
                },
                "vocabulary": {
                    "enabled": True,
                    "path": "config/vocabulary/custom_terms.txt"
                }
            }
            
            # Initialize STT engine
            stt_init = self.stt_engine.initialize(stt_config)
            if not stt_init:
                logger.error("Failed to initialize STT engine")
                return False
            
            # Initialize display if requested
            if self.config.display:
                logger.info("Initializing display")
                try:
                    from tccc.display.display_interface import DisplayInterface
                    
                    self.display_interface = DisplayInterface()
                    display_config = {
                        "width": 1280,
                        "height": 720,
                        "headless": False,
                        "font_size": 16
                    }
                    
                    display_init = self.display_interface.initialize(display_config)
                    if not display_init:
                        logger.warning("Failed to initialize display, continuing without display")
                        self.display_interface = None
                except ImportError:
                    logger.warning("Display module not available, continuing without display")
                    self.display_interface = None
            
            # Mark as initialized
            self.initialized = True
            logger.info("Audio-to-STT pipeline initialized successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Error importing required modules: {e}")
            return False
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            return False
    
    def start(self):
        """Start the pipeline processing."""
        if not self.initialized:
            logger.error("Pipeline not initialized")
            return False
        
        try:
            # Start audio capture
            logger.info("Starting audio capture")
            self.audio_pipeline.start_capture()
            
            # Start worker threads
            self.running = True
            self.audio_thread = threading.Thread(target=self._audio_worker)
            self.stt_thread = threading.Thread(target=self._stt_worker)
            self.display_thread = threading.Thread(target=self._display_worker)
            
            self.audio_thread.daemon = True
            self.stt_thread.daemon = True
            self.display_thread.daemon = True
            
            self.audio_thread.start()
            self.stt_thread.start()
            self.display_thread.start()
            
            logger.info("Pipeline started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting pipeline: {e}")
            return False
    
    def stop(self):
        """Stop the pipeline processing."""
        if not self.running:
            return
        
        try:
            # Stop processing
            self.running = False
            
            # Stop audio capture
            logger.info("Stopping audio capture")
            self.audio_pipeline.stop_capture()
            
            # Wait for threads to complete
            self.audio_thread.join(timeout=2)
            self.stt_thread.join(timeout=2)
            self.display_thread.join(timeout=2)
            
            # Clean up resources
            if self.stt_engine:
                self.stt_engine.shutdown()
            
            if self.display_interface:
                self.display_interface.shutdown()
            
            logger.info("Pipeline stopped")
            
        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")
    
    def _audio_worker(self):
        """Audio processing worker thread."""
        logger.info("Audio worker started")
        
        while self.running:
            try:
                # Get audio segment from pipeline
                audio_segment = None
                
                # Try different methods based on what's available
                if hasattr(self.audio_pipeline, 'get_audio_segment'):
                    audio_segment = self.audio_pipeline.get_audio_segment()
                elif hasattr(self.audio_pipeline, 'get_audio'):
                    audio_segment = self.audio_pipeline.get_audio()
                
                # Check if we got a valid segment
                if audio_segment is not None and len(audio_segment) > 0:
                    # Put segment in queue for STT processing
                    try:
                        self.segment_queue.put(audio_segment, block=False)
                    except queue.Full:
                        # Queue full, discard oldest segments
                        try:
                            _ = self.segment_queue.get_nowait()
                            self.segment_queue.put(audio_segment, block=False)
                        except:
                            pass
                
                # Sleep to prevent tight loop
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in audio worker: {e}")
        
        logger.info("Audio worker stopped")
    
    def _stt_worker(self):
        """STT processing worker thread."""
        logger.info("STT worker started")
        
        while self.running:
            try:
                # Get audio segment from queue
                try:
                    audio_segment = self.segment_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Process with STT engine
                metadata = {
                    "battlefield_audio": self.config.battlefield,
                    "word_timestamps": True
                }
                
                # Transcribe segment
                result = self.stt_engine.transcribe_segment(audio_segment, metadata)
                
                # Check if we got a valid result
                if result and "text" in result and result["text"]:
                    # Put result in display queue
                    try:
                        self.transcription_queue.put(result, block=False)
                    except queue.Full:
                        # Queue full, discard oldest transcription
                        try:
                            _ = self.transcription_queue.get_nowait()
                            self.transcription_queue.put(result, block=False)
                        except:
                            pass
                    
                    # Print the transcription
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] {result['text']}")
                    
                    # Save to file if requested
                    if self.config.output_file:
                        with open(self.config.output_file, 'a') as f:
                            f.write(f"[{timestamp}] {result['text']}\n")
                
            except Exception as e:
                logger.error(f"Error in STT worker: {e}")
        
        logger.info("STT worker stopped")
    
    def _display_worker(self):
        """Display worker thread."""
        if not self.display_interface:
            return
            
        logger.info("Display worker started")
        
        # Recent transcriptions for context
        recent_texts = []
        max_recent = 5
        
        while self.running:
            try:
                # Get transcription from queue
                try:
                    result = self.transcription_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Add to recent transcriptions
                text = result["text"]
                recent_texts.append(text)
                if len(recent_texts) > max_recent:
                    recent_texts.pop(0)
                
                # Format display text
                display_text = "\n".join(recent_texts)
                
                # Update display
                self.display_interface.display_text(display_text, clear=True)
                
            except Exception as e:
                logger.error(f"Error in display worker: {e}")
        
        logger.info("Display worker stopped")

def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Create test directory if needed
    if args.file:
        os.makedirs(os.path.dirname(args.input_file), exist_ok=True)
        
        # Check if the file exists
        if not os.path.exists(args.input_file):
            logger.warning(f"Input file not found: {args.input_file}")
            logger.info("Creating a test file with spoken audio")
            
            # Create a test WAV file with speech
            from verify_audio_stt_e2e import create_test_wav_file
            create_test_wav_file(args.input_file)
    
    # Print configuration
    print("=" * 60)
    print(" Optimized Audio-to-STT Pipeline".center(60))
    print("=" * 60)
    print(f"Input Source: {'File' if args.file else 'Microphone'}")
    if args.file:
        print(f"Input File: {args.input_file}")
    else:
        print(f"Microphone Device ID: {args.device_id}")
    print(f"STT Model: {args.model}")
    print(f"Battlefield Mode: {'Enabled' if args.battlefield else 'Disabled'}")
    if args.output_file:
        print(f"Output File: {args.output_file}")
    if args.display:
        print("Display: Enabled")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = AudioSTTPipeline(args)
    
    if not pipeline.initialize():
        logger.error("Failed to initialize pipeline")
        return 1
    
    # Start processing
    if not pipeline.start():
        logger.error("Failed to start pipeline")
        return 1
    
    try:
        print("\nTranscribing... Press Ctrl+C to stop\n")
        
        # Keep running until interrupted
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Stop processing
        pipeline.stop()
        
        print("\nPipeline stopped")
        
        # Print STT engine status if available
        if pipeline.stt_engine:
            try:
                status = pipeline.stt_engine.get_status()
                
                print("\n" + "=" * 60)
                print(" STT Engine Status ".center(60))
                print("=" * 60)
                
                print(f"Model: {status.get('model', {}).get('model_size', 'unknown')}")
                
                if 'performance' in status:
                    perf = status['performance']
                    print(f"Transcripts: {perf.get('transcript_count', 0)}")
                    print(f"Real-time factor: {perf.get('real_time_factor', 0):.2f}x")
                
                if 'caching' in status:
                    cache = status['caching']
                    print(f"Caching: {'Enabled' if cache.get('enabled', False) else 'Disabled'}")
                    print(f"Cache hits: {cache.get('hits', 0)}")
                    print(f"Cache misses: {cache.get('misses', 0)}")
                
                print("=" * 60)
            except Exception as e:
                logger.error(f"Error getting STT status: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())