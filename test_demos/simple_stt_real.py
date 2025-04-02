#!/usr/bin/env python3
"""
Simple real STT demo optimized to show actual transcription, not mock data.
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

# Set up paths
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

# Force use of real STT engine
os.environ["USE_MOCK_STT"] = "0"

# TCCC imports
from tccc.audio_pipeline import AudioPipeline
from tccc.stt_engine import create_stt_engine
from tccc.utils.logging import get_logger

# Minimize logging to focus on results
os.environ["TCCC_LOG_LEVEL"] = "ERROR"
logger = get_logger(__name__)

def main():
    """Run a simple real STT demo."""
    # Print clear instructions
    print("\n===== TCCC REAL Speech-to-Text Demonstration =====\n")
    print("This demo will show real transcription of your voice.")
    print("Speak clearly into the microphone to see the results.\n")
    
    # Initialize components with minimal settings
    print("Initializing STT engine...")
    stt_engine = create_stt_engine("faster-whisper")
    
    # Important: Use minimal configuration with VAD disabled
    stt_config = {
        "model": {
            "size": "tiny",
            "compute_type": "int8",
            "vad_filter": False,  # Critical - disable VAD filtering
            "language": "en"
        }
    }
    
    # Initialize the engine
    stt_engine.initialize(stt_config)
    print("STT engine initialized.")
    
    # Set up audio pipeline with minimal processing
    print("\nInitializing audio pipeline...")
    audio_pipeline = AudioPipeline()
    
    # Configure audio pipeline with VAD disabled
    audio_config = {
        "io": {
            "input_sources": [
                {"name": "mic", "type": "microphone", "device_id": 0}
            ],
            "default_input": "mic"
        },
        "processing": {
            "enable_vad": False,  # Critical - disable VAD
            "noise_reduction": False  # Disable noise processing
        }
    }
    
    # Initialize and start audio capture
    audio_pipeline.initialize(audio_config)
    audio_pipeline.start_capture("mic")
    print("Audio capture started.\n")
    
    # Prepare for recording transcripts
    output_file = "speech_transcript.txt"
    with open(output_file, "w") as f:
        f.write("TCCC Speech Recognition Demo\n")
        f.write("=============================\n\n")
    
    # Clear instructions to user
    print("\n" + "=" * 50)
    print("START SPEAKING NOW - Talk clearly into the microphone")
    print("Say simple words and phrases. Press Ctrl+C to exit.")
    print("=" * 50 + "\n")
    
    # Use a buffer to accumulate audio for better results
    audio_buffer = np.array([], dtype=np.int16)
    buffer_duration_ms = 1000  # 1 second buffer
    samples_per_buffer = int(16000 * buffer_duration_ms / 1000)  # at 16kHz
    
    last_text = ""
    try:
        while True:
            # Get audio from the pipeline
            audio_stream = audio_pipeline.get_audio_stream()
            if audio_stream:
                audio_data = audio_stream.read()
                
                if audio_data is not None and len(audio_data) > 0:
                    # Convert to numpy array if needed
                    if not isinstance(audio_data, np.ndarray):
                        audio_data = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Add to buffer
                    audio_buffer = np.append(audio_buffer, audio_data)
                    
                    # Get audio level
                    level = np.max(np.abs(audio_data)) / 32767.0 * 100
                    
                    # Process buffer when it's large enough and has content
                    if len(audio_buffer) >= samples_per_buffer and level > 1.0:
                        # Process through STT engine
                        result = stt_engine.transcribe_segment(audio_buffer)
                        
                        # Reset buffer
                        audio_buffer = np.array([], dtype=np.int16)
                        
                        # Process result
                        if result and 'text' in result and result['text'].strip():
                            text = result['text'].strip()
                            
                            # Only show if different from last result
                            if text != last_text:
                                last_text = text
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                
                                # Show result
                                print(f"[{timestamp}] \033[1m{text}\033[0m")
                                
                                # Log to file
                                with open(output_file, "a") as f:
                                    f.write(f"[{timestamp}] {text}\n")
                    
                    # Show audio level indicator
                    bars = int(level / 5)
                    level_meter = f"[{'#' * bars}{' ' * (20 - bars)}] {level:.1f}%"
                    sys.stdout.write(f"\r{level_meter}")
                    sys.stdout.flush()
            
            # Small delay to prevent CPU overuse
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")
    finally:
        # Cleanup
        print("\nShutting down...")
        audio_pipeline.stop_capture()
        stt_engine.shutdown()
        
        print(f"Transcript saved to: {output_file}")
        print("\nDemo completed.")

if __name__ == "__main__":
    main()