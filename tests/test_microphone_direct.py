#!/usr/bin/env python3
"""
Simple test script to verify microphone capture is working on Jetson.
This script directly captures audio from the microphone without using any TCCC components.
"""

import sys
import time
import pyaudio
import numpy as np
import wave
import os

# Configuration
DEVICE_ID = 0  # Razer Seiren Mini (can be changed via command line)
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RECORD_SECONDS = 5
OUTPUT_FILE = "test_recording.wav"

def record_and_save():
    """Record audio from the microphone and save to a file."""
    print("\n===== Microphone Direct Test =====")
    
    # List available audio devices
    p = pyaudio.PyAudio()
    print("\nAvailable audio input devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:  # Only show input devices
            print(f"Device {i}: {info['name']}")
    
    device_id = DEVICE_ID
    if len(sys.argv) > 1:
        try:
            device_id = int(sys.argv[1])
        except ValueError:
            print(f"Invalid device ID, using default: {DEVICE_ID}")
    
    try:
        # Get device info
        device_info = p.get_device_info_by_index(device_id)
        print(f"\nUsing device {device_id}: {device_info['name']}")
        
        # Open recording stream
        print(f"Starting recording ({RECORD_SECONDS} seconds)...")
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_id,
            frames_per_buffer=CHUNK_SIZE
        )
        
        # Record audio
        frames = []
        for i in range(0, int(SAMPLE_RATE / CHUNK_SIZE * RECORD_SECONDS)):
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)
            
            # Show audio levels
            audio_data = np.frombuffer(data, dtype=np.int16)
            level = np.max(np.abs(audio_data)) / 32767.0 * 100
            bars = int(level / 5)
            sys.stdout.write("\r[" + "#" * bars + " " * (20 - bars) + "] " + f"{level:.1f}%")
            sys.stdout.flush()
        
        print("\n\nRecording complete!")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        
        # Save the recorded audio to a WAV file
        print(f"Saving recording to {OUTPUT_FILE}...")
        wf = wave.open(OUTPUT_FILE, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        # Print file info
        file_size = os.path.getsize(OUTPUT_FILE) / 1024
        print(f"File saved ({file_size:.1f} KB)")
        print(f"Full path: {os.path.abspath(OUTPUT_FILE)}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        p.terminate()
        print("\nTest completed.")

if __name__ == "__main__":
    record_and_save()