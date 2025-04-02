#!/usr/bin/env python3
"""
STT demo with mock engine, to clearly demonstrate the microphone
and STT functionality without real model initialization issues.
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

# Force mock for reliable demo
os.environ["USE_MOCK_STT"] = "1"

# TCCC imports
from tccc.audio_pipeline import AudioPipeline
from tccc.stt_engine import create_stt_engine

# Main function
def main():
    """
    Run a simple demonstration of the microphone and STT functionality.
    """
    print("\n===== Tactical Combat Casualty Care - Speech Recognition Demo =====\n")
    print("This demo shows the TCCC system capturing audio from the microphone")
    print("and processing it through the speech recognition engine.\n")
    
    # Initialize STT engine with mock (for reliable demo)
    print("Initializing STT engine...")
    stt_engine = create_stt_engine("mock")
    stt_engine.initialize({})
    
    # Configure audio pipeline
    print("Setting up audio pipeline...")
    audio_pipeline = AudioPipeline()
    audio_config = {
        "io": {
            "input_sources": [
                {"name": "microphone", "type": "microphone", "device_id": 0}
            ],
            "default_input": "microphone"
        },
        "processing": {
            "sample_rate": 16000,
            "channels": 1,
            "enable_vad": True
        }
    }
    
    # Initialize pipeline
    audio_pipeline.initialize(audio_config)
    
    # Create output file for recording
    output_wav = "tccc_demo_recording.wav"
    wf = wave.open(output_wav, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    
    # Start capture
    print("Starting audio capture...\n")
    audio_pipeline.start_capture("microphone")
    
    # Instructions
    print("=" * 60)
    print("RECORDING ACTIVE - Speak into the microphone")
    print("Say medical phrases clearly to test the system")
    print("Examples: 'The patient has a gunshot wound', 'applying tourniquet'")
    print("Press Ctrl+C to end recording")
    print("=" * 60 + "\n")
    
    # Processing loop
    start_time = time.time()
    try:
        while True:
            # Get audio from pipeline
            audio_stream = audio_pipeline.get_audio_stream()
            if audio_stream:
                audio_data = audio_stream.read()
                
                if audio_data is not None and len(audio_data) > 0:
                    # Convert to numpy array if needed
                    if isinstance(audio_data, np.ndarray):
                        if audio_data.dtype != np.int16:
                            audio_data = (audio_data * 32767).astype(np.int16)
                        audio_bytes = audio_data.tobytes()
                    else:
                        audio_bytes = audio_data
                        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    
                    # Write to wave file
                    wf.writeframes(audio_bytes)
                    
                    # Calculate audio level
                    level = np.max(np.abs(audio_data)) / 32767.0 * 100
                    
                    # Only process audio that has content
                    if level > 5.0:  # 5% threshold to avoid processing silence
                        # Process through STT
                        result = stt_engine.transcribe_segment(audio_data)
                        
                        # Display result
                        if result and 'text' in result and result['text'].strip():
                            text = result['text'].strip()
                            
                            # Show transcription with timestamp
                            elapsed = time.time() - start_time
                            timestamp = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
                            
                            # Format the output
                            print(f"[{timestamp}] {text}")
                    
                    # Show audio level
                    bars = int(level / 5)
                    level_meter = f"[{'#' * min(bars, 20)}{' ' * (20 - min(bars, 20))}] {level:.1f}%"
                    sys.stdout.write(f"\r{level_meter}")
                    sys.stdout.flush()
            
            # Sleep to prevent CPU overuse
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\nRecording stopped.")
    finally:
        # Clean up
        print("Finishing...")
        audio_pipeline.stop_capture()
        wf.close()
        
        # Print summary
        duration = time.time() - start_time
        file_size = os.path.getsize(output_wav) / 1024  # KB
        print(f"\nRecording saved: {output_wav} ({file_size:.1f} KB)")
        print(f"Duration: {int(duration // 60)} min {int(duration % 60)} sec")
        print("\nDemo completed.")

if __name__ == "__main__":
    main()