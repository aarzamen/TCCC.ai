#!/bin/bash
#
# TCCC.ai MVP Startup Script
# --------------------------

# Source the virtual environment
source venv/bin/activate

# Source audio environment variables
if [ -f "~/tccc_audio_env.sh" ]; then
    source ~/tccc_audio_env.sh
fi

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Set power mode to balanced
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 1
fi

# Start TCCC system with display integration
echo "Starting TCCC.ai MVP system with display..."

# Check if HDMI display is connected
if xrandr 2>/dev/null | grep -q " connected"; then
    echo "HDMI display detected, enabling display interface"
    export TCCC_ENABLE_DISPLAY=1
    
    # Check for WaveShare 6.25" display specifically (in either orientation)
    if xrandr | grep -q "1560x720"; then
        echo "WaveShare 6.25\" display detected in landscape mode (1560x720)"
        export TCCC_DISPLAY_RESOLUTION="1560x720"
        export TCCC_DISPLAY_TYPE="waveshare_6_25"
    elif xrandr | grep -q "720x1560"; then
        echo "WaveShare 6.25\" display detected in portrait mode (720x1560)"
        echo "Note: Landscape mode (1560x720) is recommended for best experience"
        export TCCC_DISPLAY_RESOLUTION="720x1560"
        export TCCC_DISPLAY_TYPE="waveshare_6_25"
    else
        # Check general display resolution
        display_res=1560x720
        if [ -n "" ]; then
            echo "Display resolution: "
            export TCCC_DISPLAY_RESOLUTION=
        else
            echo "Using default display resolution: 720x1560 (WaveShare 6.25\")"
            export TCCC_DISPLAY_RESOLUTION="720x1560"
            export TCCC_DISPLAY_TYPE="waveshare_6_25"
        fi
    fi
else
    echo "No HDMI display detected, running in headless mode"
    export TCCC_ENABLE_DISPLAY=0
fi

# Run the main system module
# Pass the specific MVP config file name
python -m tccc.system --config jetson_mvp.yaml --log-level DEBUG

EXIT_CODE=$?

# This script can be run with:
# ./start_tccc_mvp.sh
