#!/usr/bin/env python3
"""
Simple test script for STT with a mock implementation.
This avoids dependency on external model files and focuses on testing
the integration and caching mechanisms.
"""

import os
import sys
import time
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("STT-Mock-Test")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def create_test_audio(duration_sec=3.0, sample_rate=16000):
    """Create a synthetic audio signal for testing."""
    # Create time array
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    
    # Create a simple tone with frequency sweep
    frequencies = np.linspace(300, 3000, len(t))
    audio = 0.5 * np.sin(2 * np.pi * frequencies * t / sample_rate)
    
    # Add some noise
    audio += 0.01 * np.random.normal(size=len(audio))
    
    return audio.astype(np.float32)

def main():
    """Main function."""
    print("\n" + "=" * 60)
    print(" STT Mock Test ".center(60, "="))
    print("=" * 60 + "\n")
    
    # Check if we can import the necessary modules
    try:
        from tccc.stt_engine.stt_engine import STTEngine
        print("✓ Successfully imported STTEngine")
    except ImportError as e:
        print(f"✗ Failed to import STTEngine: {e}")
        return 1
    
    # Enable mock mode for testing
    os.environ["USE_MOCK_STT"] = "1"
    print("Set USE_MOCK_STT=1 for testing")
    
    # Create STT engine
    print("\nInitializing STT Engine...")
    stt_engine = STTEngine()
    
    # Configure STT
    stt_config = {
        "model": {
            "type": "whisper",
            "size": "tiny.en",
            "use_model_cache": True
        }
    }
    
    # Initialize STT engine
    init_success = stt_engine.initialize(stt_config)
    print(f"STT Engine initialized: {init_success}")
    
    if not init_success:
        print("Error initializing STT engine")
        return 1
    
    # Create test audio
    print("\nCreating test audio...")
    audio = create_test_audio()
    print(f"Created test audio: {len(audio)/16000:.1f}s, shape={audio.shape}")
    
    # Transcribe audio
    print("\nTranscribing audio...")
    start_time = time.time()
    result = stt_engine.transcribe_segment(audio)
    transcription_time = time.time() - start_time
    
    # Print result
    print(f"Transcription completed in {transcription_time:.2f}s")
    if "text" in result:
        print(f"Result: \"{result['text']}\"")
    else:
        print("No text in result")
    
    # Print status
    print("\nSTT Engine Status:")
    status = stt_engine.get_status()
    
    # Print model info
    print(f"Model type: {status.get('model_type', 'unknown')}")
    print(f"Model size: {status.get('model_size', 'unknown')}")
    
    # Print performance info if available
    if 'performance' in status:
        perf = status['performance']
        print(f"Transcripts: {perf.get('transcript_count', 0)}")
        print(f"Errors: {perf.get('error_count', 0)}")
        print(f"RTF: {perf.get('realtime_factor', 0):.2f}")
    
    # Shutdown
    print("\nShutting down STT engine...")
    stt_engine.shutdown()
    
    print("\nTest completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())