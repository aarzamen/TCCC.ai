#!/usr/bin/env python3
"""
Test script to verify real microphone input with the STT engine.
This script captures audio from a microphone and uses the real STT engine.
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

# Force use of real STT
os.environ["USE_MOCK_STT"] = "0"

# TCCC imports
from tccc.audio_pipeline import AudioPipeline
from tccc.stt_engine import create_stt_engine
from tccc.utils.logging import get_logger

# Setup logging
logger = get_logger(__name__)

# Configuration
RECORD_SECONDS = 10
OUTPUT_FILE = "stt_real_test.wav"

def test_real_stt():
    """Test microphone capture with real STT engine."""
    print("\n===== TCCC Real STT Test =====")
    
    # Initialize components
    print("Initializing STT Engine...")
    stt_engine = create_stt_engine("faster-whisper")
    
    # Basic config - normally we would load this from a file
    stt_config = {
        "model": {
            "type": "faster-whisper",
            "size": "tiny",
            "compute_type": "int8",
            "vad_filter": True,
            "language": "en"
        },
        "hardware": {
            "enable_acceleration": True,
            "cpu_threads": 4
        }
    }
    
    # Initialize STT engine
    success = stt_engine.initialize(stt_config)
    if not success:
        print("Failed to initialize STT engine!")
        return
    
    print("STT engine initialized successfully!")
    status = stt_engine.get_status()
    print(f"STT Status: {status}")
    
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
        stt_engine.shutdown()
        return
    
    # Start capture
    print(f"\nStarting audio capture from '{mic_source}'...")
    success = audio_pipeline.start_capture(mic_source)
    if not success:
        print("ERROR: Failed to start audio capture!")
        audio_pipeline.shutdown()
        stt_engine.shutdown()
        return
    
    # Set up wave file for recording
    wf = wave.open(OUTPUT_FILE, 'wb')
    wf.setnchannels(1)  # Mono
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(16000)  # 16kHz
    
    print(f"Recording for {RECORD_SECONDS} seconds...")
    print("Speak into the microphone...")
    
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
                    
                    # Transcribe the segment
                    transcription = stt_engine.transcribe_segment(audio_data)
                    
                    # Display level meter and transcription if available
                    bars = int(level / 5)
                    status_line = f"[{'#' * bars}{' ' * (20 - bars)}] {level:.1f}%"
                    
                    if transcription and transcription.get('text', '').strip():
                        status_line += f" | {transcription['text']}"
                    
                    sys.stdout.write(f"\r{status_line}")
                    sys.stdout.flush()
            
            # Sleep to prevent tight loop
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\nRecording stopped by user.")
    finally:
        print("\n\nFinishing recording and transcription...")
        
        # Write recorded audio to file
        if frames:
            for frame in frames:
                wf.writeframes(frame)
            wf.close()
            
            # Print file info
            file_size = os.path.getsize(OUTPUT_FILE) / 1024
            print(f"File saved: {OUTPUT_FILE} ({file_size:.1f} KB)")
            print(f"Full path: {os.path.abspath(OUTPUT_FILE)}")
            
            # Do a final transcription of the full audio
            print("\nPerforming final transcription of the full recording...")
            try:
                with open(OUTPUT_FILE, 'rb') as f:
                    # Skip WAV header (44 bytes)
                    f.seek(44)
                    audio_data = np.frombuffer(f.read(), dtype=np.int16)
                    
                    # Transcribe
                    start_time = time.time()
                    result = stt_engine.transcribe_segment(audio_data)
                    elapsed_time = time.time() - start_time
                    
                    if result and 'text' in result:
                        print(f"\nFinal transcription (took {elapsed_time:.2f}s):")
                        print("-" * 80)
                        print(result['text'])
                        print("-" * 80)
                    else:
                        print("No transcription result returned.")
            except Exception as e:
                print(f"Error performing final transcription: {e}")
        else:
            print("\n\nNo audio data captured!")
            wf.close()
            
        # Clean up
        audio_pipeline.stop_capture()
        if hasattr(audio_pipeline, 'shutdown'):
            audio_pipeline.shutdown()
        stt_engine.shutdown()
        print("\nTest completed.")

if __name__ == "__main__":
    test_real_stt()