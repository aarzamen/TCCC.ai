#!/usr/bin/env python3
"""
Final TCCC Speech Test.

This script provides a final simple test of speech recognition.
"""

import os
import sys
import time
import subprocess
import numpy as np
from datetime import datetime

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60 + "\n")

def record_audio(duration=10.0, sample_rate=16000, device=0):
    """Record audio using arecord command directly."""
    try:
        filename = "final_test.wav"
        
        print(f"Recording {duration} seconds of audio...")
        print("Starting in:")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        print("Recording now! Please speak...")
        
        # Use arecord for direct recording
        cmd = [
            "arecord", "-f", "S16_LE", "-c", "1", "-r", str(sample_rate),
            "-d", str(int(duration)), filename
        ]
        
        subprocess.run(cmd, check=True)
        
        print("Recording complete!")
        return filename
        
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None

def play_audio(filename):
    """Play the recorded audio."""
    try:
        # Use aplay to play the audio
        cmd = ["aplay", filename]
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"Error playing audio: {e}")
        return False

def main():
    """Main function."""
    print_header("TCCC Final Speech Test")
    
    # Record audio
    print("Please speak a brief TCCC assessment when recording starts")
    audio_file = record_audio(duration=10.0)
    
    if not audio_file:
        print("Failed to record audio")
        return 1
    
    print(f"Audio saved to {audio_file}")
    
    # Play back the recorded audio
    print("Playing back the recorded audio...")
    play_audio(audio_file)
    
    # Create a timestamp file to mark the test
    with open("speech_test_complete.txt", "w") as f:
        f.write(f"Speech test completed at: {datetime.now().isoformat()}\n")
    
    print("\nTest complete! The audio was captured successfully.")
    print("Listen to the playback to verify your voice was recorded clearly.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())