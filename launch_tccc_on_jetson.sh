#!/bin/bash
# Launch TCCC system directly on Jetson's physical display
# This script will ensure the system runs on the Jetson's console even when started via SSH

# Colors for terminal output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

echo -e "${GREEN}${BOLD}Launching TCCC System on Jetson's physical display...${RESET}"
echo -e "${CYAN}This script automatically launches the complete TCCC system on the Jetson's console${RESET}"
echo -e "${CYAN}The system will continue running after you close this SSH session${RESET}"

# Set up environment
export DISPLAY=:0
export USE_MOCK_STT=0
export USE_MOCK_LLM=0
export WAVESHARE_DISPLAY=1
export RESOLUTION=1280x800

# Get the absolute path to the current directory
BASE_DIR=$(dirname "$(readlink -f "$0")")
cd "$BASE_DIR"

# Create the command to run
RUN_CMD="cd $BASE_DIR && source venv/bin/activate && python test_system_integration.py --use_display --real_mic --use_phi2 --use_whisper --battlemode"

# Launch the command in a terminal on the Jetson's display
echo -e "${GREEN}Executing command on Jetson display: ${CYAN}$RUN_CMD${RESET}"

# Try multiple terminal emulators in case some aren't available
if command -v lxterminal >/dev/null 2>&1; then
    # Launch using LXTerminal (common on Jetson)
    DISPLAY=:0 setsid lxterminal -e "bash -c '$RUN_CMD; exec bash'" &
    echo "Launched with lxterminal"
elif command -v xterm >/dev/null 2>&1; then
    # Fall back to xterm
    DISPLAY=:0 setsid xterm -fs 14 -bg black -fg green -title "TCCC SYSTEM" -e "bash -c '$RUN_CMD; exec bash'" &
    echo "Launched with xterm"
elif command -v gnome-terminal >/dev/null 2>&1; then
    # Try gnome-terminal
    DISPLAY=:0 setsid gnome-terminal -- bash -c "$RUN_CMD; exec bash" &
    echo "Launched with gnome-terminal"
else
    # Last resort - try direct display command
    echo "No suitable terminal found, trying direct execution"
    DISPLAY=:0 setsid bash -c "$RUN_CMD" &
fi

# Wait a moment to ensure the command starts
sleep 2

echo -e "${GREEN}${BOLD}TCCC System launched on Jetson's physical display${RESET}"
echo -e "${CYAN}The system will continue running even after you close this SSH session${RESET}"
echo -e "${CYAN}To observe the system, look at the Jetson's physical monitor${RESET}"
echo

# Verify process is running
ps aux | grep "test_system_integration.py" | grep -v grep
if [ $? -eq 0 ]; then
    echo -e "${GREEN}System successfully launched and running!${RESET}"
else
    echo -e "${RED}Warning: System may not have launched successfully.${RESET}"
fi