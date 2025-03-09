#!/bin/bash
# Install TCCC desktop shortcuts for easy access on the Jetson Nano

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RESET='\033[0m'

echo -e "${GREEN}Installing TCCC desktop shortcuts...${RESET}"

# Define the base directory
BASE_DIR=$(dirname "$(readlink -f "$0")")
cd "$BASE_DIR"

# Create local applications directory if it doesn't exist
mkdir -p ~/.local/share/applications/

# Install TCCC Microphone shortcut
cp TCCC_Microphone.desktop ~/.local/share/applications/
echo -e "${GREEN}Installed TCCC Microphone shortcut${RESET}"

# Install TCCC Full Demo shortcut
cp TCCC_Full_Demo.desktop ~/.local/share/applications/
echo -e "${GREEN}Installed TCCC Full System Demo shortcut${RESET}"

# Install RAG Explorer shortcut
cp TCCC_RAG_Explorer.desktop ~/.local/share/applications/ 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Installed TCCC RAG Explorer shortcut${RESET}"
else
    echo -e "${YELLOW}RAG Explorer shortcut not found (optional)${RESET}"
fi

# Update desktop database
update-desktop-database ~/.local/share/applications/ 2>/dev/null

echo -e "${GREEN}Desktop shortcuts installed successfully!${RESET}"
echo "You can now find TCCC applications in your application menu or on the desktop."