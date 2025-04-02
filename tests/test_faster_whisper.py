#!/usr/bin/env python3
"""
Test Faster Whisper functionality
"""

import os
import numpy as np
from faster_whisper import WhisperModel

def main():
    """Main function"""
    # Load model
    model_path = "models/stt"
    print(f"Loading Faster Whisper model from {model_path}...")
    
    # Initialize model
    model = WhisperModel(
        model_size_or_path="tiny.en",
        device="cpu",
        compute_type="int8",
        download_root=model_path
    )
    print("Model loaded successfully!")
    
    # Create a dummy audio segment (1 second of silence)
    sample_rate = 16000
    dummy_audio = np.zeros(sample_rate, dtype=np.float32)
    
    # Transcribe the dummy audio
    print("Transcribing dummy audio...")
    segments, info = model.transcribe(dummy_audio, language="en")
    
    # Print results
    print(f"Detected language: {info.language}")
    
    print("Segments:")
    for segment in segments:
        print(f"  {segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
    
    print("Test complete!")

if __name__ == "__main__":
    main()