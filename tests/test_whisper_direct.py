#!/usr/bin/env python3
"""
Test the Whisper transcription directly with a recorded audio file.
This bypasses the STT engine wrapper for a direct test.
"""

import sys
import os
import numpy as np
import soundfile as sf

# Try to import faster_whisper
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("faster_whisper module not available")

def main():
    """Test the Whisper transcription directly with a recorded audio file."""
    if not WHISPER_AVAILABLE:
        print("Faster Whisper not available. Please install it with:")
        print("pip install faster-whisper")
        return False
    
    # Load the audio file
    audio_file = "medical_terms.wav"
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        return False
    
    print(f"Loading audio file: {audio_file}")
    audio_data, sample_rate = sf.read(audio_file)
    
    # Initialize the Whisper model
    print("Initializing Whisper model...")
    model_size = "tiny.en"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    # Transcribe the audio
    print("Transcribing audio...")
    segments, info = model.transcribe(audio_data, beam_size=1)
    
    # Print the transcription
    print("\n===== Transcription =====")
    transcript = ""
    for segment in segments:
        print(f"{segment.text}")
        transcript += segment.text + " "
    
    print("\n===== Detected Language =====")
    print(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
    
    # Try to extract medical terms from the transcript
    print("\n===== Medical Term Detection =====")
    medical_terms = ["injury", "blast", "wound", "trauma", "bleeding", 
                  "BP", "blood pressure", "pulse", "heart rate", 
                  "tourniquet", "needle", "morphine", "medic"]
    
    for term in medical_terms:
        if term.lower() in transcript.lower():
            print(f"Detected medical term: {term}")
    
    return True

if __name__ == "__main__":
    main()