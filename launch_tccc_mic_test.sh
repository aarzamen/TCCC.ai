#!/bin/bash
# Auto-launch script for TCCC microphone test on Jetson Nano
# This launches a terminal window with the test script

# Path to the Python script
SCRIPT_PATH="/home/ama/tccc-project/tccc_mic_to_display.py"

# Make sure script is executable
chmod +x "$SCRIPT_PATH"

# Define terminal window geometry
TERM_WIDTH=120
TERM_HEIGHT=40

# Create temporary script that will be executed in terminal
TMP_SCRIPT="/tmp/tccc_mic_test_runner.sh"
cat > "$TMP_SCRIPT" << 'EOF'
#!/bin/bash
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

# Run the microphone test script
cd /home/ama/tccc-project
python tccc_mic_to_display.py

# Keep terminal open after script ends
echo -e "\n\033[1;33mTest complete. Press Enter to close this window.\033[0m"
read
EOF

# Make the script executable
chmod +x "$TMP_SCRIPT"

# Launch the terminal
xterm -fa 'Monospace' -fs 10 -geometry ${TERM_WIDTH}x${TERM_HEIGHT} -title "TCCC.ai Microphone Test" -bg black -fg white -e "$TMP_SCRIPT" &

echo "Launching TCCC Microphone Test terminal..."
exit 0