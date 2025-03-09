#!/bin/bash
# Configure WaveShare display for TCCC applications
# This sets up the correct resolution and display parameters

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
RESET='\033[0m'

echo -e "${GREEN}Configuring WaveShare display for TCCC...${RESET}"

# Detect connected displays
echo "Detecting displays..."
xrandr_output=$(xrandr)
echo "$xrandr_output"

# Look for HDMI connection (WaveShare display)
if echo "$xrandr_output" | grep -q "HDMI"; then
    echo -e "${GREEN}WaveShare display detected on HDMI!${RESET}"
    
    # Extract the exact HDMI connection name
    hdmi_connection=$(echo "$xrandr_output" | grep "HDMI" | grep " connected" | cut -d " " -f1)
    
    if [ -n "$hdmi_connection" ]; then
        echo "Setting WaveShare display to 1280x800 resolution..."
        
        # Set the resolution to 1280x800 (optimal for WaveShare display)
        xrandr --output "$hdmi_connection" --mode 1280x800
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Successfully configured WaveShare display at 1280x800${RESET}"
            
            # Optional: Make this the primary display if there are multiple displays
            if echo "$xrandr_output" | grep -q "eDP"; then
                echo "Setting HDMI as primary display..."
                xrandr --output "$hdmi_connection" --primary
            fi
            
            # Create environment variable configuration for TCCC scripts
            echo "# WaveShare Display Configuration" > ~/.tccc_display
            echo "export WAVESHARE_DISPLAY=1" >> ~/.tccc_display
            echo "export DISPLAY=:0" >> ~/.tccc_display
            echo "export DISPLAY_RESOLUTION=1280x800" >> ~/.tccc_display
            
            echo -e "${GREEN}Created display configuration at ~/.tccc_display${RESET}"
            echo "Add 'source ~/.tccc_display' to your ~/.bashrc to make it permanent"
        else
            echo -e "${RED}Failed to set resolution. Using current settings.${RESET}"
        fi
    else
        echo -e "${YELLOW}HDMI connection found but couldn't determine exact name${RESET}"
    fi
else
    echo -e "${YELLOW}No WaveShare display detected on HDMI${RESET}"
    echo "Continuing with default display configuration"
fi

# Ensure DISPLAY environment variable is set for X applications
if [ -z "$DISPLAY" ]; then
    echo -e "${YELLOW}DISPLAY environment variable not set, using :0${RESET}"
    export DISPLAY=:0
    echo "export DISPLAY=:0" >> ~/.tccc_display
fi

echo -e "${GREEN}Display configuration complete${RESET}"
echo "TCCC applications will use these display settings automatically"