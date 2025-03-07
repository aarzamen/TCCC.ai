#!/usr/bin/env python3
"""
Test STT Engine verification directly
"""

import os
import sys
import logging
import time
import yaml
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Force using actual implementation
os.environ["USE_MOCK_STT"] = "0"
os.environ["USE_FASTER_WHISPER"] = "1"

# Import modules
from tccc.stt_engine import create_stt_engine
from tccc.utils.config import Config

def main():
    """Main function"""
    print("Testing STT Engine verification directly")
    
    # Load configuration
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'stt_engine.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from: {config_path}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Create a modified config for verification that uses smaller models
    verification_config = config.copy()
    verification_config['model']['size'] = 'tiny.en'  # Use smallest model for verification
    verification_config['diarization']['enabled'] = False  # Disable diarization for faster verification
    
    # Use the factory function to create the engine
    engine_type = "faster-whisper"
    print(f"Creating STT engine with type: {engine_type}")
    engine = create_stt_engine(engine_type, verification_config)
    
    # Initialize engine
    print("Initializing engine...")
    result = engine.initialize(verification_config)
    print(f"Engine initialization result: {result}")
    
    # Check status
    print("Getting engine status...")
    status = engine.get_status()
    print(f"Engine status: {status}")
    
    # Create a simple audio signal
    print("Creating test audio...")
    sample_rate = 16000
    duration = 3  # seconds
    audio = create_sine_wave(duration, sample_rate)
    
    # Transcribe audio
    print("Transcribing audio...")
    start_time = time.time()
    transcription = engine.transcribe_segment(audio)
    elapsed_time = time.time() - start_time
    
    print(f"Transcription time: {elapsed_time:.2f}s")
    print(f"Transcription result: {transcription}")
    
    print("Test complete!")
    return 0

def create_sine_wave(duration, sample_rate):
    """
    Create a sine wave audio signal.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Numpy array with audio data
    """
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create sine wave with frequency sweep
    frequencies = np.linspace(200, 1000, len(t))
    audio = 0.5 * np.sin(2 * np.pi * frequencies * t / sample_rate)
    
    # Add some noise
    audio += 0.01 * np.random.normal(size=len(audio))
    
    return audio.astype(np.float32)

if __name__ == "__main__":
    sys.exit(main())