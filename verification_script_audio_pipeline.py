#!/usr/bin/env python3
"""
Verification script for Audio Pipeline module.

This script manually tests key components of the Audio Pipeline implementation
without relying on unit tests.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from tccc.audio_pipeline import AudioPipeline
from tccc.utils.logging import get_logger
from tccc.utils.config import Config

# Set up logging
logger = get_logger(__name__)

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 30)
    print(f" {title} ".center(30, "="))
    print("=" * 30 + "\n")

def main():
    """Main verification function."""
    print("Starting Audio Pipeline verification...")
    
    # Load configuration
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'audio_pipeline.yaml')
        config = Config.load_yaml(config_path)
        print(f"Loaded configuration from: {config_path}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Create a test file source if needed
    test_file_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    os.makedirs(test_file_dir, exist_ok=True)
    
    test_file_path = os.path.join(test_file_dir, 'sample_call.wav')
    if not os.path.exists(test_file_path):
        print(f"Creating test audio file: {test_file_path}")
        create_test_wav_file(test_file_path)
    
    # Update configuration for our test file
    for source in config['io']['input_sources']:
        if source['type'] == 'file':
            source['path'] = test_file_path
            source['loop'] = True  # Loop for continuous testing
    
    # Initialize Audio Pipeline
    print_separator("Initialization")
    pipeline = AudioPipeline()
    result = pipeline.initialize(config)
    print(f"Initialization result: {result}")
    
    if not result:
        print("Initialization failed, exiting")
        return 1
    
    # Check available sources
    print_separator("Available Sources")
    sources = pipeline.get_available_sources()
    for i, source in enumerate(sources):
        print(f"{i+1}. {source['name']} ({source['type']}): {source['sample_rate']}Hz, {source['channels']} channels")
    
    # Start audio capture
    print_separator("Starting Capture")
    source_name = next((source['name'] for source in config['io']['input_sources'] if source['type'] == 'file'), None)
    if not source_name:
        print("No file source found in configuration")
        return 1
    
    result = pipeline.start_capture(source_name)
    print(f"Start capture result: {result}")
    
    # Get audio stream
    stream = pipeline.get_audio_stream()
    print(f"Got audio stream: buffer size={stream.buffer.maxsize}")
    
    # Process for a few seconds
    print_separator("Processing")
    print("Processing audio for 5 seconds...")
    
    start_time = time.time()
    chunks_read = 0
    
    while time.time() - start_time < 5:
        data = stream.read()
        if len(data) > 0:
            chunks_read += 1
            if chunks_read % 10 == 0:
                print(f"Read {chunks_read} chunks of audio data")
        time.sleep(0.01)
    
    # Modify quality parameters
    print_separator("Quality Parameters")
    new_params = {
        'noise_reduction': {
            'strength': 0.5
        },
        'enhancement': {
            'target_level_db': -12
        }
    }
    result = pipeline.set_quality_parameters(new_params)
    print(f"Set quality parameters result: {result}")
    
    # Check status
    print_separator("Status")
    status = pipeline.get_status()
    print(f"Pipeline status: {'running' if status['running'] else 'stopped'}")
    print(f"Active source: {status['active_source']}")
    print(f"Chunks processed: {status['stats']['chunks_processed']}")
    print(f"Speech chunks: {status['stats']['speech_chunks']}")
    print(f"Average processing time: {status['stats']['average_processing_ms']:.2f}ms")
    
    # Stop capture
    print_separator("Stopping Capture")
    result = pipeline.stop_capture()
    print(f"Stop capture result: {result}")
    
    print("\nVerification complete!")
    return 0

def create_test_wav_file(file_path, duration=3.0, sample_rate=16000):
    """
    Create a test WAV file with a sine wave.
    
    Args:
        file_path: Path to save the WAV file
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    """
    try:
        import wave
        import struct
        
        # Create a simple sine wave
        frequency = 440.0  # A440
        num_samples = int(duration * sample_rate)
        
        # Generate sine wave samples
        samples = []
        for i in range(num_samples):
            # Generate a sine wave with some harmonics for a more complex sound
            t = float(i) / sample_rate
            value = 0.5 * np.sin(2.0 * np.pi * frequency * t)
            value += 0.25 * np.sin(2.0 * np.pi * frequency * 2 * t)  # First harmonic
            value += 0.125 * np.sin(2.0 * np.pi * frequency * 3 * t)  # Second harmonic
            
            # Add some noise
            value += 0.05 * np.random.normal()
            
            # Convert to 16-bit PCM
            value = int(value * 32767)
            value = max(-32768, min(32767, value))  # Clip
            samples.append(value)
        
        # Write WAV file
        with wave.open(file_path, 'w') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            
            # Convert samples to bytes
            sample_bytes = struct.pack(f"<{len(samples)}h", *samples)
            wf.writeframes(sample_bytes)
            
        print(f"Created test WAV file: {file_path}")
        print(f"  Duration: {duration}s, Sample rate: {sample_rate}Hz")
        return True
        
    except Exception as e:
        print(f"Error creating test WAV file: {e}")
        return False

if __name__ == "__main__":
    sys.exit(main())