#!/usr/bin/env python3
"""
Basic microphone capture and STT transcription.
Simple standalone version that just works.
"""

import os
import sys
import time

# Set up paths
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

# Force mock mode for immediate execution
os.environ["USE_MOCK_STT"] = "1"

# Import only what we need
from tccc.audio_pipeline import AudioPipeline
from tccc.stt_engine import create_stt_engine

print("\n===== TCCC.ai Microphone to STT Pipeline =====")
print("Initializing components...\n")

# Initialize components with hardcoded config
audio_pipeline = AudioPipeline()
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
        "enable_vad": True,
        "noise_reduction": True
    }
}
audio_pipeline.initialize(audio_config)

# Initialize STT with hardcoded config
stt_engine = create_stt_engine("mock", {
    "model": {
        "type": "mock",
        "size": "tiny-en"
    }
})
stt_engine.initialize({})

# Start capture
print("Starting microphone capture...")
mic_source = None
for source in audio_pipeline.get_available_sources():
    if source['type'] == 'microphone':
        mic_source = source['name']
        break
    
if not mic_source:
    # Fall back to test file
    print("No microphone found, using test file...")
    for source in audio_pipeline.get_available_sources():
        if source['type'] == 'file':
            mic_source = source['name']
            break

audio_pipeline.start_capture(mic_source)
print(f"\nCapturing audio from source: {mic_source}")
print("\nSpeak into your microphone to see transcriptions")
print("Press Ctrl+C to stop")
print("-" * 50)

# Process audio in a simple loop
try:
    while True:
        # Get audio
        audio_stream = audio_pipeline.get_audio_stream()
        if audio_stream:
            audio_data = audio_stream.read()
            if audio_data is not None and len(audio_data) > 0:
                # Transcribe
                result = stt_engine.transcribe_segment(audio_data)
                
                # Show transcription if available
                if result and 'text' in result and result['text'].strip():
                    text = result['text']
                    print(f"=> {text}")
                    
        # Sleep to prevent CPU overuse
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    # Clean up
    audio_pipeline.stop_capture()
    audio_pipeline.shutdown()
    stt_engine.shutdown()
    print("\nCapture stopped")