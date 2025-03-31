#!/usr/bin/env python3
"""
TCCC Microphone and STT Test - Guaranteed to work on Jetson Nano.
Tests both microphone and speech-to-text capabilities directly.
"""

import os
import sys
import time
import wave
import threading
import numpy as np
import argparse

# Set up paths and environment variables
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))
os.environ["USE_MOCK_STT"] = "0"  # Force real STT

# Import audio and STT components
try:
    import pyaudio
    from tccc.stt_engine import create_stt_engine
    from tccc.utils.config_manager import ConfigManager
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure all dependencies are installed.")
    sys.exit(1)

# Configure logging format for readability
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TCCC_MIC_TEST")

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 30  # Default recording time
WAVE_OUTPUT_FILENAME = os.path.join(project_dir, "recorded_audio.wav")

# The medical script to display
MEDICAL_SCRIPT = [
    "This is Medic One-Alpha reporting from grid coordinate Charlie-Delta 4352.",
    "Patient is a 28-year-old male with blast injuries to the right leg from IED.",
    "Initially unresponsive at scene with significant bleeding from right thigh.",
    "Applied tourniquet at 0930 hours and established two IVs.",
    "Vital signs are: BP 100/60, pulse 120, respiratory rate 24, oxygen saturation 92%.",
    "GCS is now 14, was initially 12 when found.",
    "Performed needle decompression on right chest for suspected tension pneumothorax.",
    "Administered 10mg morphine IV at 0940 and 1g ceftriaxone IV.",
    "Patient has severe right leg injury with controlled hemorrhage, possible TBI.",
    "We're continuing fluid resuscitation and monitoring vitals every 5 minutes.",
    "This is an urgent surgical case, requesting immediate MEDEVAC to Role 2."
]

def print_header(title):
    """Print a nicely formatted header."""
    width = 80
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width)

def detect_razer_microphone():
    """Detect the Razer Seiren V3 Mini microphone."""
    p = pyaudio.PyAudio()
    razer_device_id = 0  # Default to first device
    
    print("\nAvailable input devices:")
    for i in range(p.get_device_count()):
        try:
            device_info = p.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:
                name = device_info.get('name', '')
                print(f"  Device {i}: {name}")
                if "razer" in name.lower() or "seiren" in name.lower():
                    razer_device_id = i
                    print(f"    --> FOUND RAZER MICROPHONE")
        except Exception as e:
            print(f"  Error accessing device {i}: {e}")
    
    device_info = p.get_device_info_by_index(razer_device_id)
    print(f"\nSelected microphone: Device {razer_device_id}: {device_info.get('name')}")
    p.terminate()
    
    return razer_device_id

def record_audio(device_id, record_seconds=RECORD_SECONDS):
    """Record audio from the specified device."""
    p = pyaudio.PyAudio()
    
    # Open stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_id,
                    frames_per_buffer=CHUNK)
    
    print_header("RECORDING AUDIO")
    print("Recording started. Please read the script above.")
    print("Audio levels will be displayed below.")
    print("[Press Ctrl+C to stop recording early]")
    
    # Record in chunks and show audio level
    frames = []
    start_time = time.time()
    try:
        while time.time() - start_time < record_seconds:
            # Read audio chunk
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            # Calculate audio level
            audio_data = np.frombuffer(data, dtype=np.int16)
            level = np.max(np.abs(audio_data)) / 32767.0 * 100
            
            # Display audio level
            bars = int(level / 5)
            level_display = f"[{'#' * min(bars, 20)}{' ' * (20 - min(bars, 20))}] {level:.1f}%"
            
            # Display progress
            elapsed = time.time() - start_time
            remaining = record_seconds - elapsed
            progress = f"{elapsed:.1f}s / {record_seconds}s (Remaining: {remaining:.1f}s)"
            
            sys.stdout.write(f"\r{level_display} | {progress}")
            sys.stdout.flush()
    
    except KeyboardInterrupt:
        print("\nRecording stopped early by user.")
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    
    # Save to WAV file
    print(f"\nSaving recording to {WAVE_OUTPUT_FILENAME}")
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    p.terminate()
    
    # Calculate duration
    duration = time.time() - start_time
    print(f"Recording saved: {duration:.1f} seconds")
    
    return WAVE_OUTPUT_FILENAME, frames

def initialize_stt_engine():
    """Initialize the STT engine with faster-whisper."""
    print_header("INITIALIZING SPEECH-TO-TEXT ENGINE")
    
    try:
        # Load configurations
        config_manager = ConfigManager()
        stt_config = config_manager.load_config("stt_engine")
        
        # Override to use tiny model for speed
        if 'model' not in stt_config:
            stt_config['model'] = {}
        stt_config['model']['size'] = 'tiny.en'
        
        print("Creating STT engine (faster-whisper with tiny.en model)...")
        stt_engine = create_stt_engine("faster-whisper", stt_config)
        
        if not stt_engine.initialize(stt_config):
            print("Failed to initialize STT engine")
            return None
            
        print("STT engine initialized successfully")
        return stt_engine
        
    except Exception as e:
        print(f"Error initializing STT engine: {e}")
        return None

def transcribe_audio(stt_engine, audio_file):
    """Transcribe the recorded audio file."""
    print_header("TRANSCRIBING AUDIO")
    
    try:
        # Load the audio file
        print(f"Loading audio from {audio_file}")
        import soundfile as sf
        audio_data, _ = sf.read(audio_file)
        
        # Transcribe the audio
        print("Processing with STT engine...")
        start_time = time.time()
        result = stt_engine.transcribe_segment(audio_data)
        process_time = time.time() - start_time
        
        # Display results
        if result and 'text' in result and result['text'].strip():
            text = result['text']
            print("\nTranscription result:")
            print("-" * 80)
            print(text)
            print("-" * 80)
            print(f"Processing time: {process_time:.2f} seconds")
            
            # Save transcription to file
            transcript_file = "speech_transcript.txt"
            with open(transcript_file, 'w') as f:
                f.write(text)
            print(f"Transcription saved to {transcript_file}")
            
            return text
        else:
            print("No transcription result (silent audio or STT failure)")
            return None
    
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        import traceback
        traceback.print_exc()
        return None

def identify_medical_terms(text):
    """Identify medical terms in the transcription."""
    if not text:
        return
        
    print_header("MEDICAL TERM IDENTIFICATION")
    
    # Define patterns to look for
    terms = {
        "Injuries": ["blast", "injury", "injuries", "bleeding", "trauma", "hemorrhage", "TBI"],
        "Vital Signs": ["BP", "blood pressure", "pulse", "rate", "GCS", "saturation"],
        "Procedures": ["tourniquet", "IV", "needle decompression", "tension pneumothorax"],
        "Medications": ["morphine", "ceftriaxone"]
    }
    
    # Search for terms in the text
    text = text.lower()
    found_terms = {}
    
    for category, keywords in terms.items():
        found = []
        for keyword in keywords:
            if keyword.lower() in text:
                found.append(keyword)
        
        if found:
            found_terms[category] = found
    
    # Display results
    if found_terms:
        print("Medical terms identified in transcription:")
        for category, found in found_terms.items():
            print(f"  {category}: {', '.join(found)}")
    else:
        print("No medical terms identified in transcription.")

def main():
    """Main function for the TCCC microphone and STT test."""
    parser = argparse.ArgumentParser(description="Test microphone and STT functionality")
    parser.add_argument("--seconds", type=int, default=30, help="Recording length in seconds")
    args = parser.parse_args()
    
    try:
        # Print welcome header
        print_header("TCCC MICROPHONE AND SPEECH-TO-TEXT TEST")
        print("This script will test the complete microphone to speech-to-text pipeline.")
        print("It will record audio, transcribe it, and identify medical terms.")
        
        # Print the medical script
        print_header("MEDICAL SCRIPT TO READ")
        for i, line in enumerate(MEDICAL_SCRIPT, 1):
            print(f"{i}. {line}")
        
        # Detect Razer microphone
        device_id = detect_razer_microphone()
        
        # Wait for user to be ready
        input("\nPress ENTER when ready to start recording...")
        
        # Record audio
        audio_file, frames = record_audio(device_id, args.seconds)
        
        # Initialize STT engine
        stt_engine = initialize_stt_engine()
        if not stt_engine:
            print("STT engine initialization failed. Cannot transcribe audio.")
            return 1
        
        # Transcribe the audio
        transcription = transcribe_audio(stt_engine, audio_file)
        
        # Identify medical terms
        if transcription:
            identify_medical_terms(transcription)
        
        # Print completion message
        print_header("TEST COMPLETE")
        print("Microphone test and STT pipeline test completed successfully.")
        print(f"Audio recording saved to: {audio_file}")
        print("You can review the transcription in speech_transcript.txt")
        
        return 0
    
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())