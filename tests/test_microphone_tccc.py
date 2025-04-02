#!/usr/bin/env python3
"""
Test script to verify microphone capture using TCCC audio pipeline.
This script uses the TCCC components to test the audio pipeline with a real microphone.
"""

import os
import sys
import time
import numpy as np
import wave
from datetime import datetime

# Set up paths
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

# TCCC imports
from tccc.audio_pipeline import AudioPipeline
from tccc.utils.logging import get_logger

# Setup logging
logger = get_logger(__name__)

# Configuration
RECORD_SECONDS = 5
OUTPUT_FILE = "tccc_mic_test.wav"

def test_microphone_capture():
    """Test microphone capture using TCCC's AudioPipeline."""
    print("\n===== TCCC Microphone Test =====")
    
    # Initialize audio pipeline
    audio_pipeline = AudioPipeline()
    
    # Configure audio pipeline
    audio_config = {
        "io": {
            "input_sources": [
                {
                    "name": "microphone",
                    "type": "microphone",
                    "device_id": 0  # Razer Seiren V3 Mini
                }
            ],
            "default_input": "microphone"
        },
        "processing": {
            "sample_rate": 16000,
            "channels": 1,
            "enable_vad": True,
            "noise_reduction": True
        }
    }
    
    # Initialize with config
    print("Initializing audio pipeline...")
    audio_pipeline.initialize(audio_config)
    
    # Get available sources
    sources = audio_pipeline.get_available_sources()
    print("\nAvailable audio sources:")
    for source in sources:
        print(f"- {source['name']} ({source['type']})")
    
    # Check if microphone source is available
    mic_source = None
    for source in sources:
        if source['type'] == 'microphone':
            mic_source = source['name']
            break
    
    if not mic_source:
        print("ERROR: No microphone source found!")
        audio_pipeline.shutdown()
        return
    
    # Start capture
    print(f"\nStarting audio capture from '{mic_source}'...")
    success = audio_pipeline.start_capture(mic_source)
    if not success:
        print("ERROR: Failed to start audio capture!")
        audio_pipeline.shutdown()
        return
    
    # Set up wave file for recording
    wf = wave.open(OUTPUT_FILE, 'wb')
    wf.setnchannels(1)  # Mono
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(16000)  # 16kHz
    
    print(f"Recording for {RECORD_SECONDS} seconds...")
    print("Audio level meter:")
    
    # Record for specified duration
    start_time = time.time()
    frames = []
    
    try:
        while time.time() - start_time < RECORD_SECONDS:
            # Get audio stream
            audio_stream = audio_pipeline.get_audio_stream()
            if audio_stream:
                # Read audio data
                audio_data = audio_stream.read()
                if audio_data is not None and len(audio_data) > 0:
                    # Convert to bytes for saving
                    if isinstance(audio_data, np.ndarray):
                        if audio_data.dtype != np.int16:
                            audio_data = (audio_data * 32767).astype(np.int16)
                        audio_bytes = audio_data.tobytes()
                    else:
                        audio_bytes = audio_data
                    
                    # Save to frames for WAV file
                    frames.append(audio_bytes)
                    
                    # Display audio level
                    if isinstance(audio_data, np.ndarray):
                        level = np.max(np.abs(audio_data)) / 32767.0 * 100
                    else:
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        level = np.max(np.abs(audio_array)) / 32767.0 * 100
                    
                    # Display level meter
                    bars = int(level / 5)
                    sys.stdout.write("\r[" + "#" * bars + " " * (20 - bars) + "] " + f"{level:.1f}%")
                    sys.stdout.flush()
            
            # Sleep to prevent tight loop
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\nRecording stopped by user.")
    finally:
        # Write recorded audio to file
        if frames:
            print("\n\nSaving recording...")
            for frame in frames:
                wf.writeframes(frame)
            wf.close()
            
            # Print file info
            file_size = os.path.getsize(OUTPUT_FILE) / 1024
            print(f"File saved: {OUTPUT_FILE} ({file_size:.1f} KB)")
            print(f"Full path: {os.path.abspath(OUTPUT_FILE)}")
        else:
            print("\n\nNo audio data captured!")
            wf.close()
            
        # Clean up
        audio_pipeline.stop_capture()
        # Check if shutdown method exists (compatibility with older versions)
        if hasattr(audio_pipeline, 'shutdown'):
            audio_pipeline.shutdown()
        print("\nTest completed.")

if __name__ == "__main__":
    test_microphone_capture()