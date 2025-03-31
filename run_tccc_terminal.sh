#!/bin/bash
# Terminal-based version of the TCCC test that works without a GUI

echo -e "\033[1;36m"
echo "┌──────────────────────────────────────────────────────────────────────────────┐"
echo "│                   TCCC.ai MEDICAL TRANSCRIPTION TEST                         │"
echo "│                                                                              │"
echo "│  This will test the microphone transcription functionality in terminal       │"
echo "│  mode - works without a graphical display.                                   │"
echo "│                                                                              │"
echo "└──────────────────────────────────────────────────────────────────────────────┘"
echo -e "\033[0m"

# Create a simple Python test script
TEST_SCRIPT="/tmp/mic_test_terminal.py"
cat > "$TEST_SCRIPT" << 'EOF'
#!/usr/bin/env python3
import os
import sys
import time
import wave
import threading
import re
from datetime import datetime

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("PyAudio not available - audio recording disabled")

# Configure paths
project_dir = "/home/ama/tccc-project"
sys.path.insert(0, os.path.join(project_dir, "src"))

# Try to import components
try:
    from tccc.stt_engine import create_stt_engine
    from tccc.utils.config_manager import ConfigManager
    STT_AVAILABLE = True
    
    # Load STT configuration
    config_manager = ConfigManager()
    stt_config = config_manager.load_config("stt_engine")
    
    # Initialize STT Engine (faster-whisper)
    print("Initializing STT Engine (faster-whisper)...")
    stt_engine = create_stt_engine("faster-whisper", stt_config)
    if not stt_engine.initialize(stt_config):
        print("Failed to initialize STT Engine")
        STT_AVAILABLE = False
except ImportError:
    STT_AVAILABLE = False
    print("STT engine not available - transcription disabled")

# Initialize audio recording if available
if PYAUDIO_AVAILABLE:
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 5  # Record in 5-second chunks
    WAVE_OUTPUT_FILENAME = os.path.join(project_dir, "tccc_mic_test.wav")

# Status variables
recording = False
recorded_frames = []
transcript = "Speak into the microphone..."
status_message = "Ready"
recording_thread = None
transcription_thread = None
continue_transcription = True

def record_audio_chunk():
    """Record a 5-second chunk of audio."""
    global recording, recorded_frames, status_message
    
    if not PYAUDIO_AVAILABLE:
        print("PyAudio not available - cannot record audio")
        return
    
    try:
        p = pyaudio.PyAudio()
        
        # List available audio devices
        print("Available audio devices:")
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:
                print(f"  Device {i}: {device_info.get('name')}")
        
        # Open stream
        stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=0,  # Use the first device (often the Razer Seiren)
                frames_per_buffer=CHUNK)
        
        print("\033[32mRecording 5-second audio chunk...\033[0m")
        status_message = "Recording..."
        recorded_frames = []
        
        # Record for RECORD_SECONDS
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            if not recording:
                break
            data = stream.read(CHUNK, exception_on_overflow=False)
            recorded_frames.append(data)
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save the recorded data as a WAV file
        if recorded_frames:
            with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(recorded_frames))
            
            status_message = f"Recording saved to {WAVE_OUTPUT_FILENAME}"
            print(f"Recording saved to {WAVE_OUTPUT_FILENAME}")
            
            # Transcribe the audio if STT is available
            if STT_AVAILABLE:
                process_audio_file(WAVE_OUTPUT_FILENAME)
        
    except Exception as e:
        status_message = f"Error: {str(e)}"
        print(f"\033[31mError recording audio: {e}\033[0m")

def process_audio_file(audio_file):
    """Process audio file with STT engine."""
    global transcript
    
    try:
        print("Transcribing audio...")
        result = stt_engine.transcribe_file(audio_file)
        
        if result and 'text' in result and result['text'].strip():
            text = result['text']
            print(f"\033[36mTranscribed text: {text}\033[0m")
            
            # Detect medical keywords
            extract_medical_keywords(text)
            
    except Exception as e:
        print(f"\033[31mError in transcription: {e}\033[0m")

def extract_medical_keywords(text):
    """Extract medical keywords from transcription."""
    text = text.lower()
    
    # Injuries
    injuries = []
    if any(kw in text for kw in ["injury", "wound", "trauma", "bleeding", "fracture", "blast"]):
        injury_type = "blast injury" if "blast" in text else "bleeding" if "bleeding" in text else "trauma"
        location = "leg" if "leg" in text else "chest" if "chest" in text else "unknown"
        injuries.append(f"{injury_type} to {location}")
    
    # Vital signs
    vitals = []
    if "bp" in text or "blood pressure" in text:
        match = re.search(r"BP (?:is |of )?(\d+/\d+)", text)
        if match:
            bp = match.group(1)
            vitals.append(f"BP: {bp}")
    
    # Medications
    medications = []
    if "morphine" in text:
        dosage = "10mg" if "10" in text else "unknown dosage"
        medications.append(f"Morphine ({dosage})")
    
    # Procedures
    procedures = []
    if "tourniquet" in text:
        procedures.append("Tourniquet application")
    elif "needle decompression" in text:
        procedures.append("Needle decompression")
    
    # Print results
    if injuries or vitals or medications or procedures:
        print("\n\033[1;33m=== MEDICAL ENTITIES DETECTED ===\033[0m")
        
        if injuries:
            print("\033[1;31mINJURIES:\033[0m")
            for item in injuries:
                print(f"  - {item}")
        
        if vitals:
            print("\033[1;32mVITAL SIGNS:\033[0m")
            for item in vitals:
                print(f"  - {item}")
        
        if procedures:
            print("\033[1;34mPROCEDURES:\033[0m")
            for item in procedures:
                print(f"  - {item}")
        
        if medications:
            print("\033[1;35mMEDICATIONS:\033[0m")
            for item in medications:
                print(f"  - {item}")
        
        print("\033[1;33m=================================\033[0m\n")

def continuous_recording():
    """Record audio continuously in 5-second chunks."""
    global recording
    
    while continue_transcription:
        if recording:
            record_audio_chunk()
        else:
            time.sleep(0.1)

# Main function
def main():
    global recording, continue_transcription
    
    print("\n\033[1;36mTCCC.ai Terminal-based Microphone Test\033[0m")
    print("\nThis test will record from your microphone and transcribe what you say.")
    print("It will detect medical terms like injuries, vital signs, and medications.")
    print("\nCommands:")
    print("  'start' - Begin recording")
    print("  'stop'  - Stop recording")
    print("  'quit'  - Exit the program")
    print("\nTest by reading this medical scenario:")
    print("\033[33m----------------------------------------------------------------------\033[0m")
    print("\033[33mMedic: This is Medic One-Alpha reporting. Patient has blast injuries\033[0m")
    print("\033[33mto the right leg from an IED. Applied tourniquet at 0930 hours.\033[0m")
    print("\033[33mVital signs are: BP 100/60, pulse 120. Administered 10mg morphine IV.\033[0m")
    print("\033[33m----------------------------------------------------------------------\033[0m")
    
    # Start background thread for recording
    recording_thread = threading.Thread(target=continuous_recording)
    recording_thread.daemon = True
    recording_thread.start()
    
    try:
        while True:
            command = input("\n\033[36mEnter command (start/stop/quit): \033[0m").strip().lower()
            
            if command == "start":
                recording = True
                print("\033[32mRecording started. Speak into the microphone...\033[0m")
            elif command == "stop":
                recording = False
                print("\033[32mRecording stopped.\033[0m")
            elif command == "quit":
                break
            else:
                print("\033[31mUnknown command. Use 'start', 'stop', or 'quit'.\033[0m")
    
    except KeyboardInterrupt:
        print("\nProgram interrupted.")
    finally:
        continue_transcription = False
        recording = False
        print("\n\033[1;36mTCCC.ai Microphone Test complete.\033[0m")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[31mError: {e}\033[0m")
EOF

# Make the script executable
chmod +x "$TEST_SCRIPT"

# Go to the project directory
cd /home/ama/tccc-project

# Run the simple microphone test
echo "Starting terminal-based microphone test..."
python "$TEST_SCRIPT"

# Show completion message
echo -e "\n\033[1;33mTest complete.\033[0m"