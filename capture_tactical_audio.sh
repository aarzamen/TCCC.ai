#!/bin/bash
# Script to capture tactical audio from the Razer Seiren V3 Mini microphone

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}====== TCCC Tactical Audio Capture ======${NC}"
echo "This script will record audio from your Razer Seiren V3 Mini microphone"
echo "and save it for TCCC processing."

# Create output directory if it doesn't exist
OUTPUT_DIR="tactical_audio_data"
mkdir -p $OUTPUT_DIR

# Get current timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FILENAME="${OUTPUT_DIR}/tactical_recording_${TIMESTAMP}.wav"

# Get recording duration
echo -e "${BLUE}How many seconds would you like to record? (default: 30)${NC}"
read -p "> " DURATION
DURATION=${DURATION:-30}

# Get a description for the recording
echo -e "${BLUE}Please provide a brief description of the tactical scenario:${NC}"
read -p "> " DESCRIPTION

# Prepare for recording
echo -e "${YELLOW}Preparing to record from Razer Seiren V3 Mini...${NC}"
echo "Recording will last for ${DURATION} seconds"
echo "Please begin your tactical assessment/report when prompted"

# Countdown
echo "Starting in:"
for i in {3..1}
do
   echo "$i..."
   sleep 1
done

# Start recording
echo -e "${GREEN}RECORDING NOW - Please speak...${NC}"
arecord -D hw:0,0 -f S16_LE -c1 -r48000 -d $DURATION $FILENAME

# Save metadata
echo "Tactical Audio Recording" > "${FILENAME%.wav}.txt"
echo "Date: $(date)" >> "${FILENAME%.wav}.txt"
echo "Duration: ${DURATION} seconds" >> "${FILENAME%.wav}.txt"
echo "Description: ${DESCRIPTION}" >> "${FILENAME%.wav}.txt"

# Confirm success
echo -e "${GREEN}Recording complete!${NC}"
echo "Saved to: $FILENAME"
echo "Duration: ${DURATION} seconds"
echo ""
echo -e "${BLUE}Would you like to play back the recording? (y/n)${NC}"
read -p "> " PLAYBACK

if [[ $PLAYBACK == "y" || $PLAYBACK == "Y" ]]; then
    echo "Playing back recording..."
    aplay $FILENAME
fi

echo -e "${YELLOW}====== Capture Complete ======${NC}"
echo "The audio is now ready for processing by the TCCC system."
echo "Run ./launch_tccc_mvp.sh to process this audio."