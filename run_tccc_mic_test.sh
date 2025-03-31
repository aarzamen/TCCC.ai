#!/bin/bash
# Direct execution script for TCCC microphone test
# Runs in the current terminal instead of opening a new window

echo -e "\033[1;36m"
echo "┌──────────────────────────────────────────────────────────────────────────────┐"
echo "│                  TCCC.ai JETSON NANO MICROPHONE TEST                         │"
echo "│                                                                              │"
echo "│  This terminal will guide you through testing the microphone and display.    │"
echo "│  The system will listen to your speech, transcribe it, extract medical       │"
echo "│  information, and display results on the WaveShare screen.                   │"
echo "│                                                                              │"
echo "│  • Once the system is ready, you'll see 'SYSTEM IS ACTIVE' message           │"
echo "│  • Read the medical scenario script displayed below                          │"
echo "│  • The system will process your speech and update the WaveShare display      │"
echo "│  • Press Ctrl+C when finished to stop the test                               │"
echo "└──────────────────────────────────────────────────────────────────────────────┘"
echo -e "\033[0m"

# Make sure the Python script is executable
chmod +x /home/ama/tccc-project/tccc_mic_to_display.py

# Go to the project directory
cd /home/ama/tccc-project

# Run the microphone test
python tccc_mic_to_display.py

# Show completion message
echo -e "\n\033[1;33mTest complete.\033[0m"