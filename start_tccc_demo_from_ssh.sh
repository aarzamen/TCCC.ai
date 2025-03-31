#!/bin/bash
# TCCC Demo Launcher for SSH sessions
# This script launches the full TCCC demo on the Jetson's physical console
# even when called from an SSH session

# Colors for terminal
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# Banner
echo -e "${CYAN}${BOLD}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           TCCC SYSTEM LAUNCHER (SSH TO CONSOLE)                 ║"
echo "║                                                                 ║"
echo "║  This script launches the TCCC system on the Jetson's console   ║"
echo "║  even when called from an SSH session                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# Function to check if running in SSH
is_ssh() {
  if [ -n "$SSH_CLIENT" ] || [ -n "$SSH_TTY" ]; then
    return 0  # true, is SSH
  else
    return 1  # false, not SSH
  fi
}

# Set the base directory
BASE_DIR=$(dirname "$(readlink -f "$0")")
cd "$BASE_DIR"

# Display information
echo -e "${YELLOW}Detecting environment...${RESET}"

if is_ssh; then
  echo -e "${YELLOW}Running via SSH session${RESET}"
  echo -e "${YELLOW}Will launch TCCC on Jetson's physical console (DISPLAY=:0)${RESET}"
  CONSOLE_LAUNCH=true
else
  echo -e "${GREEN}Running directly on Jetson console${RESET}"
  CONSOLE_LAUNCH=false
fi

# Check for X server
if DISPLAY=:0 xset q &>/dev/null; then
  echo -e "${GREEN}X Server is running on display :0${RESET}"
  XSERVER_RUNNING=true
else
  echo -e "${YELLOW}Warning: X Server not detected on display :0${RESET}"
  echo -e "${YELLOW}Will attempt launch anyway${RESET}"
  XSERVER_RUNNING=false
fi

# Launch command based on environment
if $CONSOLE_LAUNCH; then
  echo -e "${GREEN}Launching TCCC on physical display...${RESET}"
  
  # Try multiple terminal types that might be installed
  LAUNCH_SUCCESS=false
  
  # Try lxterminal (common on Jetson)
  if command -v lxterminal >/dev/null 2>&1; then
    echo "Using lxterminal..."
    DISPLAY=:0 setsid lxterminal --geometry=120x40 -t "TCCC COMPLETE SYSTEM" -e "$BASE_DIR/tccc_full_demo_console.sh" &
    LAUNCH_SUCCESS=true
  
  # Try xterm
  elif command -v xterm >/dev/null 2>&1; then
    echo "Using xterm..."
    DISPLAY=:0 setsid xterm -geometry 120x40 -bg black -fg cyan -title "TCCC COMPLETE SYSTEM" -e "$BASE_DIR/tccc_full_demo_console.sh" &
    LAUNCH_SUCCESS=true
  
  # Try gnome-terminal
  elif command -v gnome-terminal >/dev/null 2>&1; then
    echo "Using gnome-terminal..."
    DISPLAY=:0 setsid gnome-terminal --geometry=120x40 --title="TCCC COMPLETE SYSTEM" -- "$BASE_DIR/tccc_full_demo_console.sh" &
    LAUNCH_SUCCESS=true
  
  # Direct execution
  else
    echo "No suitable terminal found, trying direct execution..."
    DISPLAY=:0 setsid "$BASE_DIR/tccc_full_demo_console.sh" &
    LAUNCH_SUCCESS=true
  fi
  
  if $LAUNCH_SUCCESS; then
    echo -e "${GREEN}${BOLD}TCCC System launched on Jetson's physical display${RESET}"
    echo -e "${CYAN}Look at the Jetson's monitor to see the system running${RESET}"
    echo -e "${CYAN}You can safely close this SSH session - the demo will continue running${RESET}"
  else
    echo -e "${RED}Failed to launch TCCC on physical display${RESET}"
    exit 1
  fi

else
  # Running directly on console
  echo -e "${GREEN}Running TCCC directly...${RESET}"
  "$BASE_DIR/tccc_full_demo_console.sh"
fi