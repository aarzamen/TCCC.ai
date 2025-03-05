#!/usr/bin/env python3
"""
Verification script for STT Engine module.

This script manually tests key components of the STT Engine implementation
without relying on unit tests.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from tccc.stt_engine import STTEngine
from tccc.utils.logging import get_logger
from tccc.utils import ConfigManager

# Set up logging
logger = get_logger(__name__)

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 30)
    print(f" {title} ".center(30, "="))
    print("=" * 30 + "\n")

def main():
    """Main verification function."""
    print("Starting STT Engine verification...")
    
    # Load configuration
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'stt_engine.yaml')
        config_manager = ConfigManager()
        config = ConfigManager.load_yaml(config_path)
        print(f"Loaded configuration from: {config_path}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Create a test file if needed
    test_file_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    os.makedirs(test_file_dir, exist_ok=True)
    
    test_file_path = os.path.join(test_file_dir, 'test_speech.wav')
    if not os.path.exists(test_file_path):
        print(f"Creating test audio file: {test_file_path}")
        create_test_wav_file(test_file_path)
    
    # Initialize STT Engine
    print_separator("Initialization")
    engine = STTEngine()
    
    # Create a modified config for verification that uses smaller models
    verification_config = config.copy()
    verification_config['model']['size'] = 'tiny'  # Use smallest model for verification
    verification_config['diarization']['enabled'] = False  # Disable diarization for faster verification
    
    result = engine.initialize(verification_config)
    print(f"Initialization result: {result}")
    
    if not result:
        print("Initialization failed, exiting")
        return 1
    
    # Test getting status
    print_separator("Status")
    status = engine.get_status()
    print(f"Engine initialized: {status['initialized']}")
    if 'model' in status:
        print(f"Model type: {status['model']['model_type']}")
        print(f"Model size: {status['model']['model_size']}")
    
    # Load test audio
    print_separator("Loading Test Audio")
    try:
        import wave
        import soundfile as sf
        audio, sample_rate = sf.read(test_file_path)
        print(f"Loaded audio: {len(audio)} samples, {sample_rate}Hz")
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            print(f"Resampling from {sample_rate}Hz to 16000Hz")
            # Simple resampling (for verification only)
            audio = np.interp(
                np.linspace(0, len(audio), int(len(audio) * 16000 / sample_rate)), 
                np.arange(len(audio)), 
                audio
            )
            sample_rate = 16000
        
    except Exception as e:
        print(f"Error loading audio: {e}")
        # Create synthetic audio (sine wave) as fallback
        print("Creating synthetic audio instead")
        sample_rate = 16000
        duration = 3  # seconds
        audio = create_sine_wave(duration, sample_rate)
    
    # Test context update
    print_separator("Context Update")
    context = "Patient is a 45-year-old male with history of hypertension and diabetes."
    result = engine.update_context(context)
    print(f"Context update result: {result}")
    
    # Test transcription
    print_separator("Transcription")
    start_time = time.time()
    result = engine.transcribe_segment(audio)
    end_time = time.time()
    
    print(f"Transcription time: {end_time - start_time:.2f}s")
    print(f"Transcription result:")
    print(f"  Text: {result['text']}")
    print(f"  Segments: {len(result['segments'])}")
    print(f"  Language: {result.get('language', 'unknown')}")
    print(f"  Real-time factor: {result['metrics']['real_time_factor']:.2f}x")
    
    # Print first segment details
    if result['segments']:
        segment = result['segments'][0]
        print(f"\nFirst segment:")
        print(f"  Text: {segment['text']}")
        print(f"  Time: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s")
        print(f"  Confidence: {segment['confidence']:.2f}")
        
        # Print word details if available
        if 'words' in segment and segment['words']:
            words = segment['words']
            print(f"\nWords in first segment (showing first 5):")
            for i, word in enumerate(words[:5]):
                print(f"  {i+1}. '{word['text']}' ({word['start_time']:.2f}s - {word['end_time']:.2f}s, conf: {word['confidence']:.2f})")
    
    # Get final status
    print_separator("Final Status")
    status = engine.get_status()
    
    # Print metrics
    metrics = status['metrics']
    print(f"Metrics:")
    print(f"  Total audio processed: {metrics['total_audio_seconds']:.2f}s")
    print(f"  Total processing time: {metrics['total_processing_time']:.2f}s")
    print(f"  Transcripts generated: {metrics['transcript_count']}")
    print(f"  Errors: {metrics['error_count']}")
    print(f"  Average confidence: {metrics['avg_confidence']:.2f}")
    print(f"  Average real-time factor: {metrics['real_time_factor']:.2f}x")
    
    print("\nVerification complete!")
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

def create_test_wav_file(file_path, duration=3.0, sample_rate=16000):
    """
    Create a test WAV file with a speech-like sine wave.
    
    Args:
        file_path: Path to save the WAV file
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    """
    try:
        import wave
        import struct
        
        # Create audio data (sine wave with frequency modulation to mimic speech)
        audio = create_sine_wave(duration, sample_rate)
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Write WAV file
        with wave.open(file_path, 'w') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            
            # Convert samples to bytes
            sample_bytes = audio_int16.tobytes()
            wf.writeframes(sample_bytes)
            
        print(f"Created test WAV file: {file_path}")
        print(f"  Duration: {duration}s, Sample rate: {sample_rate}Hz")
        return True
        
    except Exception as e:
        print(f"Error creating test WAV file: {e}")
        return False

if __name__ == "__main__":
    sys.exit(main())