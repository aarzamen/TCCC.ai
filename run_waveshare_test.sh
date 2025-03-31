#!/bin/bash
# Launch script optimized for Waveshare display on Jetson Nano

echo -e "\e[1;36m"
echo "┌───────────────────────────────────────────────────────┐"
echo "│          TCCC.ai WAVESHARE MICROPHONE TEST            │"
echo "│                                                       │"
echo "│  This script will test the complete pipeline:         │"
echo "│  - Razer Seiren V3 Mini microphone                    │"
echo "│  - Speech-to-text transcription (faster-whisper)      │"
echo "│  - Medical entity extraction                          │"
echo "│  - WaveShare display (1280x800)                       │"
echo "│                                                       │"
echo "└───────────────────────────────────────────────────────┘"
echo -e "\e[0m"

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Ensure the test script is executable 
chmod +x microphone_test_complete.py

# Set environment variables for the WaveShare display
export SDL_VIDEODRIVER=x11
export SDL_FBDEV=/dev/fb0
export DISPLAY=:0
export PYTHONUNBUFFERED=1
export TCCC_DISPLAY_TYPE=waveshare

# Adjust backlight brightness to maximum if supported
if [ -f /sys/class/backlight/waveshare/brightness ]; then
  echo -e "\e[1;33mSetting WaveShare display brightness to maximum...\e[0m"
  sudo sh -c "echo 255 > /sys/class/backlight/waveshare/brightness" 2>/dev/null || true
fi

# Clean swap to free up memory
echo -e "\e[1;33mCleaning swap space to improve performance...\e[0m"
sudo swapoff -a
sudo swapon -a
free -h

# Check for the Razer microphone explicitly
echo -e "\e[1;33mDetecting Razer Seiren V3 Mini microphone...\e[0m"
arecord -l | grep -i "razer\|seiren" 
if [ $? -eq 0 ]; then
  echo -e "\e[1;32mRazer microphone detected!\e[0m"
else
  echo -e "\e[1;31mWarning: Razer microphone not explicitly detected. Using default device.\e[0m"
fi

# Check for WaveShare display
echo -e "\e[1;33mVerifying WaveShare display...\e[0m"
if [ -e /dev/fb0 ]; then
  echo -e "\e[1;32mWaveShare display detected on /dev/fb0\e[0m"
else
  echo -e "\e[1;31mWarning: WaveShare display not detected on /dev/fb0\e[0m"
fi

echo -e "\e[1;33mStarting microphone test in 3 seconds...\e[0m"
echo -e "\e[1;32mPress SPACE BAR to start/stop recording\e[0m"
echo -e "\e[1;32mPress ESCAPE to exit the test\e[0m"
sleep 3

# Run the test script with proper environment
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
  echo "Audio file duration:"
  soxi -d "$LATEST_RECORDING" 2>/dev/null || echo "soxi not available to get duration"
  echo -e "\e[1;33mPlay recording with: aplay $LATEST_RECORDING\e[0m"
fi