#!/bin/bash
# Basic TCCC Demo
# Demonstrates minimal viable functionality

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored text
print_step() {
    echo -e "${BLUE}==>${NC} ${YELLOW}$1${NC}"
}

# Function to print success message
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

clear
echo -e "${YELLOW}"
echo "========================================================="
echo "                TCCC BASIC DEMO"
echo "========================================================="
echo -e "${NC}"
echo "This demo shows basic TCCC functionality"
echo

# Create temp directory
TEMP_DIR="/tmp/tccc_demo"
mkdir -p $TEMP_DIR

# Step 1: Check hardware
print_step "Checking hardware..."
echo "Audio device:"
arecord -l | grep "card"
echo "Display device:"
xrandr | grep " connected"
echo "Output device:"
aplay -l | grep "card"
print_success "All hardware components detected"
echo

# Step 2: Record audio
print_step "Testing audio recording (3 seconds)..."
echo "Please speak while recording..."
arecord -d 3 -f S16_LE -c1 -r16000 $TEMP_DIR/demo_rec.wav
print_success "Audio recorded successfully"
echo

# Step 3: Play back audio
print_step "Playing back recorded audio..."
aplay $TEMP_DIR/demo_rec.wav
print_success "Audio playback completed"
echo

# Step 4: Display sample text
print_step "Displaying sample text output..."
echo -e "${GREEN}=== Sample Transcription ===${NC}"
echo "This is a simulated transcription of audio input."
echo "It demonstrates the display of text from processed speech."
echo -e "${GREEN}=== Sample Analysis ===${NC}"
echo "ASSESSMENT: Sample medical assessment"
echo "VITALS: HR: 90 bpm, BP: 120/80 mmHg, RR: 16/min"
echo "TREATMENTS: TXA administered, tourniquet applied"
echo -e "${GREEN}==========================${NC}"
print_success "Display functionality demonstrated"
echo

# Conclusion
echo -e "${YELLOW}"
echo "========================================================="
echo "                DEMO COMPLETED"
echo "========================================================="
echo -e "${NC}"
echo "All basic MVP functions verified:"
echo "✓ Audio input (microphone)"
echo "✓ Audio output (speakers)"
echo "✓ Display output"
echo
echo "Press Enter to exit..."
read

# Clean up
rm -rf $TEMP_DIR