#!/bin/bash
# Execute the microphone test script with proper environment setup

echo -e "\e[1;36m"
echo "┌───────────────────────────────────────────────────────┐"
echo "│          TCCC.ai MICROPHONE TEST LAUNCHER             │"
echo "│                                                       │"
echo "│  This script will test:                               │"
echo "│  - Razer Seiren V3 Mini microphone                    │"
echo "│  - Speech-to-text transcription (faster-whisper)      │"
echo "│  - Medical entity extraction                          │"
echo "│                                                       │"
echo "└───────────────────────────────────────────────────────┘"
echo -e "\e[0m"

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Ensure the test script is executable 
chmod +x microphone_test_complete.py

# Set display to use hardware or software rendering
export SDL_VIDEODRIVER=x11
export PYTHONUNBUFFERED=1

# Check for the Razer microphone explicitly
echo -e "\e[1;33mDetecting Razer Seiren V3 Mini microphone...\e[0m"
arecord -l | grep -i "razer\|seiren" 
if [ $? -eq 0 ]; then
  echo -e "\e[1;32mRazer microphone detected!\e[0m"
else
  echo -e "\e[1;31mWarning: Razer microphone not explicitly detected. Using default device.\e[0m"
fi

echo -e "\e[1;33mStarting microphone test in 2 seconds...\e[0m"
sleep 2

# Run the test script
python microphone_test_complete.py

# Show final message
echo -e "\e[1;36m"
echo "┌───────────────────────────────────────────────────────┐"
echo "│                  TEST COMPLETED                       │"
echo "└───────────────────────────────────────────────────────┘"
echo -e "\e[0m"

# Check if a recording file was created
LATEST_RECORDING=$(ls -t tccc_recording_*.wav 2>/dev/null | head -1)
if [ -n "$LATEST_RECORDING" ]; then
  echo -e "\e[1;32mRecording saved to: $LATEST_RECORDING\e[0m"
  echo -e "\e[1;33mPlay recording with: aplay $LATEST_RECORDING\e[0m"
fi