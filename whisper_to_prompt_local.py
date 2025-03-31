#!/usr/bin/env python3
"""
Voice-to-text input for Claude using local Whisper model.
Records from microphone, transcribes with faster-whisper, copies to clipboard.
"""

import os
import sys
import time
import argparse
import numpy as np
import sounddevice as sd
import pyperclip
import soundfile as sf
from pathlib import Path

# Set up proper path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster_whisper module not found")
    print("Please install it with: pip install faster-whisper")
    sys.exit(1)

# Default parameters
SAMPLE_RATE = 16000
RECORD_SECONDS = 30
MODEL_SIZE = "tiny.en"  # Options: tiny.en, base.en, small.en, medium.en, large-v2
TEMP_FILE = "/tmp/whisper_recording.wav"
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'stt'))

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def record_audio(device_id=0, duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
    """Record audio from the microphone."""
    print(f"Recording for {duration} seconds (or press Ctrl+C to stop)...")
    
    # Start recording
    audio_data = sd.rec(int(duration * sample_rate), 
                      samplerate=sample_rate, 
                      channels=1, 
                      dtype='float32',
                      device=device_id)
    
    # Show progress with option to stop early
    try:
        for i in range(duration):
            print(f"\rRecording: {i+1}/{duration} seconds", end="")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\rRecording stopped early.                ")
    
    print("\rRecording complete!                      ")
    sd.wait()  # Wait for recording to complete
    
    # Save to temporary file
    sf.write(TEMP_FILE, audio_data, sample_rate)
    print(f"Audio saved to {TEMP_FILE}")
    
    return TEMP_FILE

def transcribe_with_local_whisper(audio_file, model_size=MODEL_SIZE):
    """Transcribe audio using local Whisper model."""
    print(f"Transcribing with local faster-whisper model ({model_size})...")
    
    # Always use the model name directly, don't try to construct a local path
    # This will use the default Hugging Face model cache
    model_path = model_size
    
    try:
        # Try to use GPU first with half precision
        try:
            print("Attempting to use GPU...")
            model = WhisperModel(model_path, device="cuda", compute_type="float16")
            device_used = "GPU (float16)"
        except:
            try:
                print("GPU with float16 failed, trying GPU with float32...")
                model = WhisperModel(model_path, device="cuda", compute_type="float32")
                device_used = "GPU (float32)"
            except:
                print("GPU failed, falling back to CPU...")
                model = WhisperModel(model_path, device="cpu", compute_type="int8")
                device_used = "CPU (int8)"
        
        print(f"Using {device_used} for inference")
        
        # Transcribe audio
        print("Beginning transcription...")
        start_time = time.time()
        segments, info = model.transcribe(audio_file, beam_size=5, language="en")
        
        # Collect all segments
        transcript = ""
        for segment in segments:
            transcript += segment.text + " "
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Transcription complete in {duration:.2f} seconds")
        print(f"Result: {len(transcript)} characters")
        
        return transcript.strip()
        
    except Exception as e:
        print(f"Error in transcription: {e}")
        return f"Error: {e}"

def list_audio_devices():
    """List available audio devices."""
    devices = sd.query_devices()
    
    print_separator("Available Audio Devices")
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Input device
            print(f"Input Device {i}: {device['name']}")
    
    return 0

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import sounddevice
        import soundfile
        import pyperclip
        import faster_whisper
        return True
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("\nPlease install all required packages with:")
        print("pip install sounddevice soundfile pyperclip faster-whisper")
        return False

def main():
    """Main function."""
    if not check_dependencies():
        return 1
        
    parser = argparse.ArgumentParser(description="Voice-to-text input for Claude using local Whisper")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--device", type=int, default=0, help="Audio device ID (default: 0)")
    parser.add_argument("--duration", type=int, default=RECORD_SECONDS, 
                        help=f"Maximum recording duration in seconds (default: {RECORD_SECONDS})")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                        help=f"Audio sample rate (default: {SAMPLE_RATE})")
    parser.add_argument("--model", default=MODEL_SIZE, 
                        help=f"Whisper model size (default: {MODEL_SIZE})")
    parser.add_argument("--continuous", action="store_true", 
                        help="Run in continuous mode, transcribing multiple recordings")
    
    args = parser.parse_args()
    
    # Handle device listing
    if args.list_devices:
        return list_audio_devices()
    
    print_separator("Voice-to-Text for Claude (Local Whisper)")
    print(f"Using audio device ID: {args.device}")
    print(f"Maximum recording duration: {args.duration} seconds (Ctrl+C to stop early)")
    print(f"Using Whisper model: {args.model}")
    
    if args.continuous:
        print("\nRunning in continuous mode. Press Ctrl+C during idle time to exit.")
        
        session_num = 1
        try:
            while True:
                print(f"\nSession {session_num} - Starting recording in 3 seconds...")
                # Count down instead of waiting for input
                for i in range(3, 0, -1):
                    print(f"{i}...")
                    time.sleep(1)
                print("Recording now!")
                
                # Record and transcribe
                audio_file = record_audio(args.device, args.duration, args.sample_rate)
                transcript = transcribe_with_local_whisper(audio_file, args.model)
                
                if transcript:
                    # Show transcript
                    print("\nTranscript:")
                    print("-" * 80)
                    print(transcript)
                    print("-" * 80)
                    
                    # Copy to clipboard - try multiple methods
                    copied = False
                    try:
                        # Try pyperclip first
                        pyperclip.copy(transcript)
                        copied = True
                    except Exception as e:
                        try:
                            # Try direct command line utility as fallback
                            with open('/tmp/transcript.txt', 'w') as f:
                                f.write(transcript)
                            os.system('cat /tmp/transcript.txt | xclip -selection clipboard 2>/dev/null || cat /tmp/transcript.txt | xsel -b 2>/dev/null')
                            copied = True
                        except Exception as e2:
                            print(f"\nCouldn't copy to clipboard: {e}")
                            print("Please manually copy the transcript above.")
                    
                    if copied:
                        print("\nTranscript copied to clipboard! Paste it into your Claude conversation.")
                        # Also save to a file for backup
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        with open(f"transcript_{timestamp}.txt", 'w') as f:
                            f.write(transcript)
                
                session_num += 1
                print("\nReady for next recording...")
                # Add a small pause between recordings
                print("Waiting 5 seconds before next recording...")
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\nContinuous mode ended by user.")
    else:
        try:
            print("\nStarting recording in 3 seconds...")
            # Count down instead of waiting for input
            for i in range(3, 0, -1):
                print(f"{i}...")
                time.sleep(1)
            print("Recording now!")
            
            # Record audio
            audio_file = record_audio(args.device, args.duration, args.sample_rate)
            
            # Transcribe audio
            transcript = transcribe_with_local_whisper(audio_file, args.model)
            
            if not transcript:
                print("Error: No transcript generated")
                return 1
            
            # Show transcript
            print("\nTranscript:")
            print("-" * 80)
            print(transcript)
            print("-" * 80)
            
            # Copy to clipboard
            try:
                pyperclip.copy(transcript)
                print("\nTranscript copied to clipboard! Paste it into your Claude conversation.")
            except Exception as e:
                print(f"\nCouldn't copy to clipboard: {e}")
                print("Please manually copy the transcript above.")
            
            print_separator("Done")
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return 1
        except Exception as e:
            print(f"\nError: {e}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())