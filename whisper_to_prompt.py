#!/usr/bin/env python3
"""
Voice-to-text input for Claude using OpenAI's Whisper API.
Records from microphone, transcribes with OpenAI Whisper API, copies to clipboard.
Optimized for maximum performance and accuracy.
"""

import os
import sys
import time
import argparse
import numpy as np
import sounddevice as sd
import pyperclip
import soundfile as sf
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not found in environment variables.")
    print("Please set your OpenAI API key with: export OPENAI_API_KEY='your-api-key'")
    print("Or create a .env file with OPENAI_API_KEY=your-api-key")
    sys.exit(1)

# Default parameters
SAMPLE_RATE = 16000
RECORD_SECONDS = 30  # Longer recording time for more content
TEMP_FILE = "/tmp/whisper_recording.wav"

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

def transcribe_with_openai_api(audio_file, model="whisper-1"):
    """Transcribe audio using OpenAI's Whisper API for maximum accuracy."""
    print("Transcribing with OpenAI Whisper API (maximum accuracy)...")
    
    # Verify API key
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key":
        print("Error: Valid OpenAI API key not found.")
        print("Please update your .env file with your actual API key.")
        return "Error: OpenAI API key not configured. Please update your .env file."
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    url = "https://api.openai.com/v1/audio/transcriptions"
    
    try:
        with open(audio_file, "rb") as f:
            files = {"file": f}
            data = {
                "model": model,
                "language": "en",  # Specify English for better accuracy
                "response_format": "json",
                "temperature": 0,  # Use lowest temperature for maximum accuracy
            }
            
            print(f"Sending request to OpenAI API... (key starting with: {OPENAI_API_KEY[:4]}{'*' * 20})")
            response = requests.post(url, headers=headers, files=files, data=data, timeout=60)
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return f"Error: API returned status code {response.status_code}. Details: {response.text}"
        
        result = response.json()
        transcript = result.get("text", "")
        print(f"Transcription successful! ({len(transcript)} characters)")
        return transcript
    
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return f"Error: Network or request problem. Details: {e}"
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return f"Error: An unexpected error occurred. Details: {e}"

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
        import requests
        import dotenv
        return True
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("\nPlease install all required packages with:")
        print("pip install sounddevice soundfile pyperclip requests python-dotenv")
        return False

def main():
    """Main function."""
    if not check_dependencies():
        return 1
        
    parser = argparse.ArgumentParser(description="Voice-to-text input for Claude using OpenAI Whisper API")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--device", type=int, default=0, help="Audio device ID (default: 0)")
    parser.add_argument("--duration", type=int, default=RECORD_SECONDS, 
                        help=f"Maximum recording duration in seconds (default: {RECORD_SECONDS})")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                        help=f"Audio sample rate (default: {SAMPLE_RATE})")
    parser.add_argument("--model", default="whisper-1", 
                        help=f"OpenAI Whisper API model (default: whisper-1)")
    parser.add_argument("--continuous", action="store_true", 
                        help="Run in continuous mode, transcribing multiple recordings")
    
    args = parser.parse_args()
    
    # Handle device listing
    if args.list_devices:
        return list_audio_devices()
    
    print_separator("Voice-to-Text for Claude (Maximum Accuracy)")
    print(f"Using audio device ID: {args.device}")
    print(f"Maximum recording duration: {args.duration} seconds (Ctrl+C to stop early)")
    print(f"Using OpenAI API model: {args.model}")
    
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
                transcript = transcribe_with_openai_api(audio_file, args.model)
                
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
            transcript = transcribe_with_openai_api(audio_file, args.model)
            
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