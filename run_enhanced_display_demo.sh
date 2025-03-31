#!/bin/bash
# TCCC Enhanced Display Demo launcher script
# Launches the enhanced display demo with vital signs visualization

# Set script to exit on error
set -e

# Change to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "ERROR: Virtual environment not found. Please set up the environment first."
    exit 1
fi

# Check for required dependencies
echo "Checking required dependencies..."
python -c "import pygame" 2>/dev/null || {
    echo "Installing pygame..."
    pip install pygame
}

# Check if running on Jetson
if [ -f "/etc/nv_tegra_release" ]; then
    echo "Running on Jetson hardware - optimizing for performance"
    export TCCC_DISPLAY_RESOLUTION="1280x720"
    export TCCC_OPTIMIZE_FOR_JETSON="1"
else
    echo "Running on standard hardware"
fi

# Run the display demo
echo "Launching TCCC Enhanced Display Demo..."
python tccc_enhanced_display_demo.py "$@"

# Deactivate virtual environment
deactivate

exit 0