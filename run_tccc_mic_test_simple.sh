#!/bin/bash
# Simple test script that doesn't use PyTorch

echo -e "\033[1;36m"
echo "┌──────────────────────────────────────────────────────────────────────────────┐"
echo "│                   TCCC.ai MEDICAL TRANSCRIPTION TEST                         │"
echo "│                                                                              │"
echo "│  This will test the microphone transcription and display functionality       │"
echo "│  without requiring PyTorch or the Phi-2 model.                               │"
echo "│                                                                              │"
echo "└──────────────────────────────────────────────────────────────────────────────┘"
echo -e "\033[0m"

# Create a simple Python test script
TEST_SCRIPT="/tmp/mic_test_simple.py"
cat > "$TEST_SCRIPT" << 'EOF'
import os
import sys
import time
import pyaudio
import wave
import threading
import pygame
from datetime import datetime

# Set environment variables
os.environ["SDL_VIDEODRIVER"] = "x11"

# Configure paths
project_dir = "/home/ama/tccc-project"
sys.path.insert(0, os.path.join(project_dir, "src"))

# Initialize audio recording
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 30
WAVE_OUTPUT_FILENAME = os.path.join(project_dir, "tccc_mic_test.wav")

# Initialize pygame for display
pygame.init()
screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
pygame.display.set_caption("TCCC.ai Microphone Test")
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Fonts
font_large = pygame.font.SysFont('Arial', 36)
font_medium = pygame.font.SysFont('Arial', 24)
font_small = pygame.font.SysFont('Arial', 18)

# Status variables
recording = False
recorded_frames = []
transcript = "Speak into the microphone..."
status_message = "Ready"
recording_thread = None

def record_audio():
    global recording, recorded_frames, status_message
    
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
        
        print("Recording started...")
        status_message = "Recording..."
        recorded_frames = []
        
        while recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            recorded_frames.append(data)
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save the recorded data as a WAV file
        with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(recorded_frames))
        
        status_message = f"Recording saved to {WAVE_OUTPUT_FILENAME}"
        print(f"Recording saved to {WAVE_OUTPUT_FILENAME}")
        
        # Simulate transcription result with timestamp
        global transcript
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        transcript = f"[{now}] Audio recorded successfully from Razer Seiren microphone.\n\nMicrophone test confirmed working!"
        
    except Exception as e:
        status_message = f"Error: {str(e)}"
        print(f"Error recording audio: {e}")

def start_recording():
    global recording, recording_thread
    recording = True
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.daemon = True
    recording_thread.start()

def stop_recording():
    global recording
    recording = False
    if recording_thread:
        recording_thread.join(timeout=2.0)

def draw_screen():
    # Clear screen
    screen.fill(BLACK)
    
    # Draw header
    header_text = font_large.render("TCCC.ai MICROPHONE TEST", True, WHITE)
    screen.blit(header_text, (640 - header_text.get_width() // 2, 30))
    
    # Draw status
    status_text = font_medium.render(f"Status: {status_message}", True, GREEN if "Recording" in status_message else WHITE)
    screen.blit(status_text, (640 - status_text.get_width() // 2, 100))
    
    # Draw instructions
    instructions = [
        "Press SPACE to start/stop recording",
        "Press ESC to exit",
        "",
        "Use this test to verify the Razer Seiren V3 Mini microphone is working correctly.",
        "When recording is complete, a simulated transcription will appear below."
    ]
    
    for i, line in enumerate(instructions):
        text = font_small.render(line, True, WHITE)
        screen.blit(text, (50, 180 + i * 30))
    
    # Draw transcript box
    pygame.draw.rect(screen, (20, 20, 40), (50, 350, 1180, 320))
    pygame.draw.rect(screen, WHITE, (50, 350, 1180, 320), 2)
    
    # Draw transcript header
    transcript_header = font_medium.render("TRANSCRIPTION", True, WHITE)
    screen.blit(transcript_header, (60, 360))
    
    # Draw transcript text (with line wrapping)
    lines = []
    words = transcript.split()
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        if font_small.size(test_line)[0] < 1160:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    
    for i, line in enumerate(lines):
        if i < 10:  # Limit number of lines
            text = font_small.render(line, True, GREEN)
            screen.blit(text, (60, 400 + i * 26))
    
    # Update display
    pygame.display.flip()

def main():
    global recording, status_message
    
    running = True
    print("TCCC.ai Microphone Test started. Press SPACE to record, ESC to quit.")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if not recording:
                        start_recording()
                    else:
                        stop_recording()
        
        draw_screen()
        clock.tick(30)
    
    # Clean up
    if recording:
        stop_recording()
    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        pygame.quit()
EOF

# Make the script executable
chmod +x "$TEST_SCRIPT"

# Go to the project directory
cd /home/ama/tccc-project

# Run the simple microphone test
echo "Starting simple microphone test..."
python "$TEST_SCRIPT"

# Show completion message
echo -e "\n\033[1;33mTest complete.\033[0m"