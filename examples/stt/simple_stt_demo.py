#!/usr/bin/env python3
"""
Simple STT demo that shows real-time transcription from microphone input.
This script focuses on showing clear audio processing and transcription results.
"""

import os
import sys
import time
import numpy as np
import wave
import argparse
from datetime import datetime

# Set up paths
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

# Parse arguments
parser = argparse.ArgumentParser(description="Simple STT demo")
parser.add_argument("--mock", action="store_true", help="Use mock STT engine instead of real engine")
args = parser.parse_args()

# Set environment variables based on args
if args.mock:
    os.environ["USE_MOCK_STT"] = "1"
    print("Using mock STT engine (simulated responses)")
else:
    os.environ["USE_MOCK_STT"] = "0"
    print("Using real STT engine (Faster Whisper)")

# TCCC imports
from tccc.audio_pipeline import AudioPipeline
from tccc.stt_engine import create_stt_engine
from tccc.utils.logging import get_logger

# Disable verbose logging
os.environ["TCCC_LOG_LEVEL"] = "ERROR"
logger = get_logger(__name__)

def simple_stt_demo():
    """Run a simple STT demo showing clear real-time transcription."""
    print("\n===== TCCC Real-Time Transcription Demo =====")
    
    # Initialize STT engine
    print("\nInitializing STT engine...")
    engine_type = "mock" if args.mock else "faster-whisper"
    stt_engine = create_stt_engine(engine_type)
    
    # Basic config
    stt_config = {
        "model": {
            "type": engine_type,
            "size": "tiny",
            "compute_type": "int8",
            "vad_filter": False,  # Disable VAD filter to process all audio
            "language": "en"
        },
        "hardware": {
            "enable_acceleration": True,
            "cpu_threads": 4
        }
    }
    
    # Initialize STT engine
    stt_engine.initialize(stt_config)
    print("STT engine initialized!")
    
    # Initialize audio pipeline
    print("\nInitializing Audio Pipeline...")
    audio_pipeline = AudioPipeline()
    
    # Configure audio pipeline
    audio_config = {
        "io": {
            "input_sources": [
                {
                    "name": "microphone",
                    "type": "microphone",
                    "device_id": 0
                }
            ],
            "default_input": "microphone"
        },
        "processing": {
            "sample_rate": 16000,
            "channels": 1,
            "enable_vad": False,  # Disable VAD in pipeline too
            "noise_reduction": True
        }
    }
    
    # Initialize audio pipeline
    audio_pipeline.initialize(audio_config)
    
    # Start capture
    print("\nStarting microphone capture...")
    audio_pipeline.start_capture("microphone")
    
    # Set up timestamp for transcript output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_file = f"transcript_{timestamp}.txt"
    
    # Create output files
    with open(transcript_file, "w") as f:
        f.write("=== Transcription Results ===\n\n")
    
    # Main loop
    print("\n\033[1m===== START SPEAKING NOW =====\033[0m")
    print("Say simple phrases clearly into the microphone")
    print("Press Ctrl+C to exit\n")
    
    last_text = ""
    quiet_count = 0
    try:
        while True:
            # Get audio from pipeline
            audio_stream = audio_pipeline.get_audio_stream()
            if audio_stream:
                audio_data = audio_stream.read()
                
                if audio_data is not None and len(audio_data) > 0:
                    # Display audio level
                    if isinstance(audio_data, np.ndarray):
                        level = np.max(np.abs(audio_data)) / 32767.0 * 100
                    else:
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        level = np.max(np.abs(audio_array)) / 32767.0 * 100
                    
                    # Only process if audio level is significant
                    if level > 2.0:  # Threshold to avoid processing silence
                        quiet_count = 0
                        
                        # Process through STT
                        result = stt_engine.transcribe_segment(audio_data)
                        
                        # Display result if we have text
                        if result and 'text' in result and result['text'].strip():
                            text = result['text'].strip()
                            
                            # Only show if different from last text
                            if text != last_text:
                                last_text = text
                                
                                # Display with timestamp and audio level
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                print(f"\033[32m[{timestamp}]\033[0m \033[1m{text}\033[0m (level: {level:.1f}%)")
                                
                                # Save to transcript file
                                with open(transcript_file, "a") as f:
                                    f.write(f"[{timestamp}] {text}\n")
                    else:
                        # Count quiet periods
                        quiet_count += 1
                        if quiet_count % 10 == 0:  # Show level periodically during silence
                            sys.stdout.write(f"\r\033[90mAudio level: {level:.1f}% (waiting for speech)\033[0m")
                            sys.stdout.flush()
            
            # Short sleep to prevent CPU overuse
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")
    finally:
        # Clean up
        print("\nShutting down...")
        audio_pipeline.stop_capture()
        if hasattr(audio_pipeline, 'shutdown'):
            audio_pipeline.shutdown()
        stt_engine.shutdown()
        
        print(f"\nTranscript saved to: {transcript_file}")
        print("\nDemo completed.")

if __name__ == "__main__":
    simple_stt_demo()