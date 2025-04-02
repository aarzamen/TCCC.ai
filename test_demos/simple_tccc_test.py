#!/usr/bin/env python3
"""
Simple TCCC Speech Test.

This script tests speech recognition with the smallest model for speed.
"""

import os
import sys
import time
import numpy as np
import soundfile as sf
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60 + "\n")

def record_audio(duration=10.0, fs=16000, device=0):
    """Record audio from the microphone."""
    try:
        import sounddevice as sd
        
        print(f"Recording {duration} seconds of audio...")
        print("Starting in:")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        print("Recording now! Please speak...")
        
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, 
                          dtype='float32', device=device)
        sd.wait()
        
        print("Recording complete!")
        return recording.flatten()
        
    except Exception as e:
        print(f"Error recording audio: {e}")
        return np.zeros(int(duration * fs), dtype=np.float32)

def transcribe_with_whisper(audio_file):
    """Transcribe audio with the Whisper model directly."""
    try:
        import whisper
        
        print("Loading Whisper model (tiny.en)...")
        model = whisper.load_model("tiny.en")
        
        print("Transcribing audio...")
        result = model.transcribe(audio_file)
        
        return result
        
    except Exception as e:
        print(f"Error transcribing with Whisper: {e}")
        return {"text": f"Error: {str(e)}"}

def main():
    """Main function."""
    print_header("TCCC Simple Speech Test")
    
    # Record audio
    print("Please speak a brief TCCC assessment when recording starts")
    audio = record_audio(duration=15.0)
    
    # Save audio to file
    audio_file = "tccc_quick_test.wav"
    sf.write(audio_file, audio, 16000)
    print(f"Audio saved to {audio_file}")
    
    # Transcribe with Whisper
    result = transcribe_with_whisper(audio_file)
    
    # Print results
    print_header("Transcription Results")
    print(f"Transcription: {result['text']}")
    
    # Save transcription to file
    with open("tccc_quick_transcription.txt", "w") as f:
        f.write(result["text"])
    
    print(f"Transcription saved to tccc_quick_transcription.txt")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())