#!/usr/bin/env python3
"""
Direct microphone test with the Razer Seiren V3 Mini
"""
import pyaudio
import wave
import numpy as np
import sys
import time

# Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
OUTPUT_FILE = "razer_mic_test.wav"
DEVICE_INDEX = 0  # Razer Seiren V3 Mini

# Initialize PyAudio
audio = pyaudio.PyAudio()

# List available input devices
print("\nAvailable input devices:")
info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        name = audio.get_device_info_by_host_api_device_index(0, i).get('name')
        print(f"Device {i}: {name}")
        
        # Auto-select Razer if found
        if "Razer" in name:
            DEVICE_INDEX = i
            print(f"  *** SELECTED: Razer Seiren V3 Mini (Device {i}) ***")

print(f"\nUsing device index: {DEVICE_INDEX}")

try:
    # Open audio stream
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=DEVICE_INDEX,
        frames_per_buffer=CHUNK
    )

    print("\n= = = = = = = = = = = = = = = = = = = = = = = = = = = = =")
    print("Recording from Razer Seiren V3 Mini for 5 seconds")
    print("Speak into the microphone now!")
    print("= = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n")

    # Buffer to store recorded data
    frames = []

    # Record audio
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        
        # Convert to numpy array for level calculation
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Calculate audio level (RMS)
        level = np.sqrt(np.mean(np.square(audio_data))) / 32767.0
        
        # Display audio level meter
        meter_length = 50
        filled_length = int(level * meter_length)
        meter = '█' * filled_length + '░' * (meter_length - filled_length)
        
        # Print level in dB and with meter
        db_level = 20 * np.log10(level + 1e-9)
        sys.stdout.write(f"\rLevel: [{meter}] {level*100:.1f}% ({db_level:.1f} dB)")
        sys.stdout.flush()
        
    print("\n\nRecording complete!")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Save the recorded audio to a WAV file
    print(f"Saving to {OUTPUT_FILE}...")
    
    with wave.open(OUTPUT_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    print(f"Recording saved to {OUTPUT_FILE}")
    
    # Play back the recorded audio
    print("\nPlaying back recording...")
    
    # Open the file for playback
    play_stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True
    )
    
    # Read and play back the file
    with wave.open(OUTPUT_FILE, 'rb') as wf:
        data = wf.readframes(CHUNK)
        while data:
            play_stream.write(data)
            data = wf.readframes(CHUNK)
    
    # Stop and close the playback stream
    play_stream.stop_stream()
    play_stream.close()
    
    print("Playback complete!")

except Exception as e:
    print(f"Error: {e}")

finally:
    # Terminate PyAudio
    audio.terminate()
    print("\nTest complete. Check the output file for the recording quality.")