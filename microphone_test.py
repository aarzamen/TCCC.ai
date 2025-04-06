#!/usr/bin/env python3
"""
Simple microphone test utility to verify audio capture is working.
This will directly capture from the microphone and show audio levels.
"""

import os
import sys
import time
import numpy as np
import pyaudio
import wave
import argparse
from datetime import datetime

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_device_info():
    """Print information about available audio devices"""
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    
    print(f"\nAudio devices ({device_count} found):")
    print("-" * 50)
    
    for i in range(device_count):
        dev_info = p.get_device_info_by_index(i)
        is_input = dev_info['maxInputChannels'] > 0
        is_output = dev_info['maxOutputChannels'] > 0
        
        device_type = []
        if is_input:
            device_type.append("INPUT")
        if is_output:
            device_type.append("OUTPUT")
            
        print(f"Device {i}: {dev_info['name']} [{', '.join(device_type)}]")
        if is_input:
            print(f"  Input channels: {dev_info['maxInputChannels']}")
            print(f"  Default sample rate: {dev_info['defaultSampleRate']}")
    
    # Show default devices
    default_input = p.get_default_input_device_info()
    print(f"\nDefault Input Device: {default_input['index']} - {default_input['name']}")
    
    p.terminate()
    return default_input['index']

def test_microphone(device_id=None, duration=10, rate=16000, chunk_size=1024, save_audio=True):
    """Test microphone functionality with direct PyAudio access"""
    p = pyaudio.PyAudio()
    
    # Get device info
    if device_id is None:
        device_id = p.get_default_input_device_info()['index']
        print(f"Using default input device: {device_id}")
    
    device_info = p.get_device_info_by_index(device_id)
    print(f"\nTesting microphone: {device_info['name']} (Device {device_id})")
    
    # Create diagnostic directory
    os.makedirs("microphone_diagnostic", exist_ok=True)
    
    # Setup recording
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        input=True,
        input_device_index=device_id,
        frames_per_buffer=chunk_size
    )
    
    print(f"\n🔊 Recording audio for {duration} seconds...")
    print("Speak into the microphone to test levels")
    print("Audio levels: [....................] 0%")
    
    # Record audio
    frames = []
    frame_count = 0
    max_level_seen = 0
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)
            
            # Calculate audio levels
            audio_data = np.frombuffer(data, dtype=np.int16)
            max_level = np.abs(audio_data).max()
            mean_level = np.abs(audio_data).mean()
            
            # Update max level
            max_level_seen = max(max_level_seen, max_level)
            
            # Calculate level as percentage of max int16 value
            level_pct = min(100, max_level / 32768.0 * 100)
            
            # Draw a progress bar with levels
            bars = int(level_pct / 5)  # 20 bars for 100%
            progress_bar = "█" * bars + "." * (20 - bars)
            
            frame_count += 1
            
            # Only update display every few frames to avoid flickering
            if frame_count % 5 == 0:
                # Use different colors based on level
                if level_pct < 1:
                    status = "🔴 SILENT"
                elif level_pct < 5:
                    status = "🟡 VERY QUIET"
                elif level_pct < 20:
                    status = "🟢 QUIET"
                else:
                    status = "🟢 GOOD"
                    
                # Clear line and update
                print(f"\rAudio levels: [{progress_bar}] {level_pct:.1f}% {status} (mean: {mean_level:.1f})", end="")
    
    except KeyboardInterrupt:
        print("\n\nRecording stopped by user")
    finally:
        print("\n\nRecording complete!")
        
        # Close stream
        stream.stop_stream()
        stream.close()
        
        # Terminate PyAudio
        p.terminate()
        
        # Save audio file if requested
        if save_audio and len(frames) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wave_file_path = f"microphone_diagnostic/test_recording_{timestamp}.wav"
            
            with wave.open(wave_file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(rate)
                wf.writeframes(b''.join(frames))
            
            print(f"Audio saved to: {wave_file_path}")
            
            # Save a diagnostic info file
            info_file_path = f"microphone_diagnostic/test_info_{timestamp}.txt"
            with open(info_file_path, 'w') as f:
                f.write(f"Device: {device_info['name']} (index {device_id})\n")
                f.write(f"Channels: 1\n")
                f.write(f"Sample rate: {rate}\n")
                f.write(f"Max level recorded: {max_level_seen} ({max_level_seen/32768.0*100:.1f}%)\n")
                f.write(f"Recording duration: {time.time() - start_time:.1f} seconds\n")
            
            print(f"Diagnostic info saved to: {info_file_path}")
        
        return max_level_seen, frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test microphone functionality")
    parser.add_argument("--device", type=int, help="Device index to use (run without this option first to see available devices)")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration in seconds")
    parser.add_argument("--no-save", action="store_true", help="Don't save the recorded audio")
    
    args = parser.parse_args()
    
    # First get device info
    default_device = get_device_info()
    
    # If device not specified, use default
    device_to_use = args.device if args.device is not None else default_device
    
    # Run the test
    max_level, frames = test_microphone(
        device_id=device_to_use,
        duration=args.duration,
        save_audio=not args.no_save
    )
    
    # Provide feedback based on results
    if max_level < 500:
        print("\n⚠️ WARNING: Very low audio levels detected! Check if:")
        print("  - The microphone is connected properly")
        print("  - The microphone is not muted (check system volume settings)")
        print("  - You are speaking loud enough into the microphone")
    elif max_level < 3000:
        print("\n⚠️ Audio was detected but levels are low. Consider:")
        print("  - Increasing the microphone input volume in system settings")
        print("  - Positioning the microphone closer to the audio source")
    else:
        print("\n✅ Good audio levels detected. Your microphone appears to be working correctly.")
