#!/bin/bash
# TCCC Full System Demo
# This script launches the full TCCC system with speech recognition,
# analysis, and display components active.

# Colors for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

# Print banner
echo -e "${BLUE}${BOLD}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                TCCC COMPLETE SYSTEM DEMO                        ║"
echo "║                                                                 ║"
echo "║  Tactical Combat Casualty Care - Full System Demonstration      ║"
echo "║  Speech Recognition + Analysis + Display                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# Define the base directory
BASE_DIR=$(dirname "$(readlink -f "$0")")
cd "$BASE_DIR"

# Activate virtual environment if present
if [ -d "venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${RESET}"
    source venv/bin/activate
fi

# Ensure we're using the real models, not mocks
export USE_MOCK_STT=0
export USE_MOCK_LLM=0

# Configure display environment
export DISPLAY=:0
export WAVESHARE_DISPLAY=1
export RESOLUTION=1280x800

# Verify components
echo -e "${YELLOW}Checking for required components...${RESET}"

# Check for microphone (Razer Seiren V3 Mini)
echo -n "Razer Seiren V3 Mini microphone: "
if arecord -l | grep -q "Razer"; then
    echo -e "${GREEN}DETECTED${RESET}"
    MIC_PRESENT=1
else
    echo -e "${RED}NOT FOUND${RESET}"
    MIC_PRESENT=0
    echo -e "${YELLOW}Warning: Using default audio device instead${RESET}"
fi

# Check for display
echo -n "WaveShare display: "
if xrandr | grep -q "HDMI"; then
    echo -e "${GREEN}DETECTED${RESET}"
    DISPLAY_PRESENT=1
else
    echo -e "${RED}NOT FOUND${RESET}"
    DISPLAY_PRESENT=0
    echo -e "${YELLOW}Warning: Will attempt to use available display${RESET}"
fi

# Check for STT model
echo -n "Faster-Whisper STT model: "
if [ -d "models/stt" ] && ls models/stt/*.bin >/dev/null 2>&1; then
    echo -e "${GREEN}FOUND${RESET}"
    STT_PRESENT=1
else
    echo -e "${RED}NOT FOUND${RESET}"
    STT_PRESENT=0
    echo -e "${YELLOW}Warning: Will attempt to download model if needed${RESET}"
fi

# Check for LLM model
echo -n "Phi-2 LLM model: "
if [ -d "models/llm" ] && ls models/llm/phi* >/dev/null 2>&1; then
    echo -e "${GREEN}FOUND${RESET}"
    LLM_PRESENT=1
else
    echo -e "${RED}NOT FOUND${RESET}"
    LLM_PRESENT=0
    echo -e "${YELLOW}Warning: Will use more basic model${RESET}"
fi

echo
echo -e "${GREEN}${BOLD}Starting TCCC Complete System Demo...${RESET}"
echo

# Launch the complete pipeline
# This includes:
# 1. Microphone capture w/enhancement
# 2. Speech-to-text processing  
# 3. LLM analysis
# 4. Display interface for visualization
python test_system_integration.py --use_display --real_mic --use_phi2 --use_whisper --battlemode

# Return to terminal on exit
echo
echo -e "${GREEN}${BOLD}TCCC Demo complete!${RESET}"
echo

# If we activated a venv, deactivate it
if [ -d "venv" ]; then
    deactivate
fi