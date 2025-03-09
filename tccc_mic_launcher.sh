#!/bin/bash
# TCCC Microphone Launcher
# Auto-launches full microphone capture with display integration
# For use on the Jetson Nano with Razer Seiren V3 Mini microphone

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# Print banner
echo -e "${BLUE}${BOLD}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                 TCCC SPEECH CAPTURE SYSTEM                      ║"
echo "║                                                                 ║"
echo "║  Tactical Combat Casualty Care - Speech Recognition System      ║"
echo "║  Razer Seiren V3 Mini + Jetson Nano + WaveShare Display        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# Set working directory to script location
cd "$(dirname "$0")"

# Activate virtual environment if present
if [ -d "venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${RESET}"
    source venv/bin/activate
fi

# Check for microphone
echo -e "${YELLOW}Checking for Razer Seiren V3 Mini microphone...${RESET}"
arecord -l | grep "Razer"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Razer microphone detected!${RESET}"
    MIC_DEVICE=0
else
    echo -e "${YELLOW}Razer microphone not detected, using default device${RESET}"
    MIC_DEVICE=0
fi

# Check for display
echo -e "${YELLOW}Checking for WaveShare display...${RESET}"
if [ -n "$DISPLAY" ]; then
    echo -e "${GREEN}Display detected: $DISPLAY${RESET}"
else
    echo -e "${YELLOW}Setting display to :0${RESET}"
    export DISPLAY=:0
fi

# Check for dependencies
echo -e "${YELLOW}Checking dependencies...${RESET}"
python3 -c "import numpy; import scipy; import pyaudio; import tkinter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Missing dependencies. Installing required packages...${RESET}"
    pip install numpy scipy pyaudio
fi

# Force use of actual model, not mock
export USE_MOCK_STT=0
export USE_MOCK_LLM=0

# Launch the application
echo -e "${GREEN}${BOLD}Launching TCCC Speech Capture System...${RESET}"
echo -e "${CYAN}Using Razer Seiren V3 Mini (device $MIC_DEVICE)${RESET}"
echo -e "${CYAN}Using WaveShare display at 1280x800 resolution${RESET}"
echo -e "${CYAN}Using actual Whisper STT engine (not mock)${RESET}"

# Execute the speech capture program with proper parameters
python3 microphone_to_text.py --enhancement auto

# Return to terminal on exit
echo
echo -e "${GREEN}${BOLD}TCCC Speech Capture complete!${RESET}"
echo -e "${CYAN}Transcription saved to improved_transcription.txt${RESET}"
echo -e "${CYAN}Audio saved to improved_audio.wav${RESET}"
echo -e "${CYAN}High-quality audio saved to highquality_audio.wav${RESET}"
echo

# If we activated a venv, deactivate it
if [ -d "venv" ]; then
    deactivate
fi